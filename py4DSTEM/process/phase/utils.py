from typing import Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom

#: Symbols for the polar representation of all optical aberrations up to the fifth order.
polar_symbols = (
    "C10",
    "C12",
    "phi12",
    "C21",
    "phi21",
    "C23",
    "phi23",
    "C30",
    "C32",
    "phi32",
    "C34",
    "phi34",
    "C41",
    "phi41",
    "C43",
    "phi43",
    "C45",
    "phi45",
    "C50",
    "C52",
    "phi52",
    "C54",
    "phi54",
    "C56",
    "phi56",
)

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {
    "defocus": "C10",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "coma": "C21",
    "coma_angle": "phi21",
    "Cs": "C30",
    "C5": "C50",
}


class ComplexProbe:
    """
    Complex Probe Class.

    Simplified version of CTF and Probe from abTEM:
    https://github.com/abTEM/abTEM/blob/master/abtem/transfer.py
    https://github.com/abTEM/abTEM/blob/master/abtem/waves.py

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions this contrast transfer function will be applied to [eV].
    semiangle_cutoff: float
        The semiangle cutoff describes the sharp Fourier space cutoff due to the objective aperture [mrad].
    gpts : Tuple[int,int]
        Number of grid points describing the wave functions.
    sampling : Tuple[float,float]
        Lateral sampling of wave functions in Å
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'
    rolloff: float, optional
        Tapers the cutoff edge over the given angular range [mrad].
    focal_spread: float, optional
        The 1/e width of the focal spread due to chromatic aberration and lens current instability [Å].
    angular_spread: float, optional
        The 1/e width of the angular deviations due to source size [mrad].
    gaussian_spread: float, optional
        The 1/e width image deflections due to vibrations and thermal magnetic noise [Å].
    phase_shift : float, optional
        A constant phase shift [radians].
    parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
        and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        energy: float,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        semiangle_cutoff: float = np.inf,
        rolloff: float = 2.,
        vacuum_probe_intensity: np.ndarray = None,
        device: str = "cpu",
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        gaussian_spread: float = 0.0,
        phase_shift: float = 0.0,
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        # Should probably be abstracted away in a device.py similar to:
        # https://github.com/abTEM/abTEM/blob/master/abtem/device.py
        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()):
                raise ValueError("{} not a recognized parameter".format(key))

        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._focal_spread = focal_spread
        self._angular_spread = angular_spread
        self._gaussian_spread = gaussian_spread
        self._phase_shift = phase_shift
        self._energy = energy
        self._wavelength = electron_wavelength_angstrom(energy)
        self._gpts = gpts
        self._sampling = sampling

        self._parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self.set_parameters(parameters)

    def set_parameters(self, parameters: dict):
        """
        Set the phase of the phase aberration.
        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = value

            elif symbol == "defocus":
                self._parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        return parameters

    def evaluate_aperture(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray] = None
    ) -> Union[float, np.ndarray]:

        xp = self._xp
        semiangle_cutoff = self._semiangle_cutoff / 1000

        if self._vacuum_probe_intensity is not None:
            vacuum_probe_intensity = xp.asarray(
                self._vacuum_probe_intensity, dtype=xp.float32
            )
            vacuum_probe_amplitude = xp.sqrt(xp.maximum(vacuum_probe_intensity, 0))
            return xp.fft.ifftshift(vacuum_probe_amplitude)

        if self._semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)


        if self._rolloff > 0.0 :
            rolloff = self._rolloff / 1000.0  # * semiangle_cutoff
            array = 0.5 * (
                1 + xp.cos(np.pi * (alpha - semiangle_cutoff + rolloff) / rolloff)
            )
            array[alpha > semiangle_cutoff] = 0.0
            array = xp.where(
                alpha > semiangle_cutoff - rolloff,
                array,
                xp.ones_like(alpha, dtype=xp.float32),
            )
        else:
            array = xp.array(alpha < semiangle_cutoff).astype(xp.float32)
        return array

    def evaluate_temporal_envelope(
        self, alpha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(
            -((0.5 * xp.pi / self._wavelength * self._focal_spread * alpha**2) ** 2)
        ).astype(xp.float32)

    def evaluate_gaussian_envelope(
        self, alpha: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(
            -0.5 * self._gaussian_spread**2 * alpha**2 / self._wavelength**2
        )

    def evaluate_spatial_envelope(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        p = self._parameters
        dchi_dk = (
            2
            * xp.pi
            / self._wavelength
            * (
                (p["C12"] * xp.cos(2.0 * (phi - p["phi12"])) + p["C10"]) * alpha
                + (
                    p["C23"] * xp.cos(3.0 * (phi - p["phi23"]))
                    + p["C21"] * xp.cos(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + (
                    p["C34"] * xp.cos(4.0 * (phi - p["phi34"]))
                    + p["C32"] * xp.cos(2.0 * (phi - p["phi32"]))
                    + p["C30"]
                )
                * alpha**3
                + (
                    p["C45"] * xp.cos(5.0 * (phi - p["phi45"]))
                    + p["C43"] * xp.cos(3.0 * (phi - p["phi43"]))
                    + p["C41"] * xp.cos(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + (
                    p["C56"] * xp.cos(6.0 * (phi - p["phi56"]))
                    + p["C54"] * xp.cos(4.0 * (phi - p["phi54"]))
                    + p["C52"] * xp.cos(2.0 * (phi - p["phi52"]))
                    + p["C50"]
                )
                * alpha**5
            )
        )

        dchi_dphi = (
            -2
            * xp.pi
            / self._wavelength
            * (
                1 / 2.0 * (2.0 * p["C12"] * xp.sin(2.0 * (phi - p["phi12"]))) * alpha
                + 1
                / 3.0
                * (
                    3.0 * p["C23"] * xp.sin(3.0 * (phi - p["phi23"]))
                    + 1.0 * p["C21"] * xp.sin(1.0 * (phi - p["phi21"]))
                )
                * alpha**2
                + 1
                / 4.0
                * (
                    4.0 * p["C34"] * xp.sin(4.0 * (phi - p["phi34"]))
                    + 2.0 * p["C32"] * xp.sin(2.0 * (phi - p["phi32"]))
                )
                * alpha**3
                + 1
                / 5.0
                * (
                    5.0 * p["C45"] * xp.sin(5.0 * (phi - p["phi45"]))
                    + 3.0 * p["C43"] * xp.sin(3.0 * (phi - p["phi43"]))
                    + 1.0 * p["C41"] * xp.sin(1.0 * (phi - p["phi41"]))
                )
                * alpha**4
                + 1
                / 6.0
                * (
                    6.0 * p["C56"] * xp.sin(6.0 * (phi - p["phi56"]))
                    + 4.0 * p["C54"] * xp.sin(4.0 * (phi - p["phi54"]))
                    + 2.0 * p["C52"] * xp.sin(2.0 * (phi - p["phi52"]))
                )
                * alpha**5
            )
        )

        return xp.exp(
            -xp.sign(self._angular_spread)
            * (self._angular_spread / 2 / 1000) ** 2
            * (dchi_dk**2 + dchi_dphi**2)
        )

    def evaluate_chi(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        p = self._parameters

        alpha2 = alpha**2
        alpha = xp.array(alpha)

        array = xp.zeros(alpha.shape, dtype=np.float32)
        if any([p[symbol] != 0.0 for symbol in ("C10", "C12", "phi12")]):
            array += (
                1 / 2 * alpha2 * (p["C10"] + p["C12"] * xp.cos(2 * (phi - p["phi12"])))
            )

        if any([p[symbol] != 0.0 for symbol in ("C21", "phi21", "C23", "phi23")]):
            array += (
                1
                / 3
                * alpha2
                * alpha
                * (
                    p["C21"] * xp.cos(phi - p["phi21"])
                    + p["C23"] * xp.cos(3 * (phi - p["phi23"]))
                )
            )

        if any(
            [p[symbol] != 0.0 for symbol in ("C30", "C32", "phi32", "C34", "phi34")]
        ):
            array += (
                1
                / 4
                * alpha2**2
                * (
                    p["C30"]
                    + p["C32"] * xp.cos(2 * (phi - p["phi32"]))
                    + p["C34"] * xp.cos(4 * (phi - p["phi34"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C41", "phi41", "C43", "phi43", "C45", "phi41")
            ]
        ):
            array += (
                1
                / 5
                * alpha2**2
                * alpha
                * (
                    p["C41"] * xp.cos((phi - p["phi41"]))
                    + p["C43"] * xp.cos(3 * (phi - p["phi43"]))
                    + p["C45"] * xp.cos(5 * (phi - p["phi45"]))
                )
            )

        if any(
            [
                p[symbol] != 0.0
                for symbol in ("C50", "C52", "phi52", "C54", "phi54", "C56", "phi56")
            ]
        ):
            array += (
                1
                / 6
                * alpha2**3
                * (
                    p["C50"]
                    + p["C52"] * xp.cos(2 * (phi - p["phi52"]))
                    + p["C54"] * xp.cos(4 * (phi - p["phi54"]))
                    + p["C56"] * xp.cos(6 * (phi - p["phi56"]))
                )
            )

        array = 2 * xp.pi / self._wavelength * array + self._phase_shift
        return array

    def evaluate_aberrations(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        xp = self._xp
        return xp.exp(-1.0j * self.evaluate_chi(alpha, phi))

    def evaluate(
        self, alpha: Union[float, np.ndarray], phi: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        array = self.evaluate_aberrations(alpha, phi)

        if self._semiangle_cutoff < np.inf or self._vacuum_probe_intensity is not None:
            array *= self.evaluate_aperture(alpha, phi)

        if self._focal_spread > 0.0:
            array *= self.evaluate_temporal_envelope(alpha)

        if self._angular_spread > 0.0:
            array *= self.evaluate_spatial_envelope(alpha, phi)

        if self._gaussian_spread > 0.0:
            array *= self.evaluate_gaussian_envelope(alpha)

        return array

    def _evaluate_ctf(self):
        alpha, phi = self.get_scattering_angles()

        array = self.evaluate(alpha, phi)
        return array

    def get_scattering_angles(self):
        kx, ky = self.get_spatial_frequencies()
        alpha, phi = self.polar_coordinates(
            kx * self._wavelength, ky * self._wavelength
        )
        return alpha, phi

    def get_spatial_frequencies(self):
        xp = self._xp
        kx, ky = spatial_frequencies(self._gpts, self._sampling)
        kx = xp.asarray(kx)
        ky = xp.asarray(ky)
        return kx, ky

    def polar_coordinates(self, x, y):
        """Calculate a polar grid for a given Cartesian grid."""
        xp = self._xp
        alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
        phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
        return alpha, phi

    def build(self):
        """Builds complex probe in the center of the region of interest."""
        xp = self._xp
        array = xp.fft.fftshift(xp.fft.ifft2(self._evaluate_ctf()))
        # if self._vacuum_probe_intensity is not None:
        array = array / xp.sqrt((xp.abs(array) ** 2).sum())
        self._array = array
        return self

    def visualize(self, **kwargs):
        """Plots the probe amplitude."""
        xp = self._xp
        asnumpy = self._asnumpy

        cmap = kwargs.get("cmap", "Greys_r")
        kwargs.pop("cmap", None)

        plt.imshow(
            asnumpy(xp.abs(self._array) ** 2),
            cmap=cmap,
            **kwargs,
        )
        return self


def spatial_frequencies(gpts: Tuple[int, int], sampling: Tuple[float, float]):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    return tuple(
        np.fft.fftfreq(n, d).astype(np.float32) for n, d in zip(gpts, sampling)
    )


def fourier_translation_operator(
    positions: np.ndarray, shape: tuple, xp=np
) -> np.ndarray:
    """
    Create an array representing one or more phase ramp(s) for shifting another array.

    Parameters
    ----------
    positions : array of xy-positions
        Positions to calculate fourier translation operators for
    shape : two int
        Array dimensions to be fourier-shifted
    xp: Callable
        Array computing module

    Returns
    -------
    Fourier translation operators
    """

    positions_shape = positions.shape

    if len(positions_shape) == 1:
        positions = positions[None]

    kx, ky = spatial_frequencies(shape, (1.0, 1.0))
    kx = kx.reshape((1, -1, 1))
    ky = ky.reshape((1, 1, -1))
    kx = xp.asarray(kx)
    ky = xp.asarray(ky)
    positions = xp.asarray(positions)
    x = positions[:, 0].reshape((-1,) + (1, 1))
    y = positions[:, 1].reshape((-1,) + (1, 1))

    result = xp.exp(-2.0j * np.pi * kx * x) * xp.exp(-2.0j * np.pi * ky * y)

    if len(positions_shape) == 1:
        return result[0]
    else:
        return result


def fft_shift(array, positions, xp=np):
    """
    Fourier-shift array using positions.

    Parameters
    ----------
    array: np.ndarray
        Array to be shifted
    positions: array of xy-positions
        Positions to fourier-shift array with
    xp: Callable
        Array computing module

    Returns
    -------
        Fourier-shifted array
    """
    return xp.fft.ifft2(
        xp.fft.fft2(array)
        * fourier_translation_operator(positions, array.shape[-2:], xp)
    )


def calculate_center_of_mass(
    intensities: np.ndarray,
    fit_function: str = "plane",
    plot_center_of_mass: bool = True,
    scan_sampling: Tuple[float, float] = (1.0, 1.0),
    reciprocal_sampling: Tuple[float, float] = (1.0, 1.0),
    scan_units: Tuple[str, str] = ("pixels", "pixels"),
    device: str = "cpu",
    **kwargs,
):
    """
    Common preprocessing function to compute and fit diffraction intensities CoM

    Parameters
    ----------
    intensities: (Rx,Ry,Qx,Qy) xp.ndarray
        Raw intensities array stored on device, with dtype xp.float32
    fit_function: str, optional
        2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
    plot_center_of_mass: bool, optional
        If True, the computed and normalized CoM arrays will be displayed
    scan_sampling: Tuple[float,float], optional
        Real-space scan sampling in `scan_units`
    reciprocal_sampling: Tuple[float,float], optional
        Reciprocal-space sampling in `A^-1`
    scan_units: Tuple[str,str], optional
        Real-space scan sampling units
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'

    Returns
    --------
    com_normalized_x: (Rx,Ry) xp.ndarray
        Normalized horizontal center of mass gradient
    com_normalized_y: (Rx,Ry) xp.ndarray
        Normalized vertical center of mass gradient

    Displays
    --------
    com_measured_x/y and com_normalized_x/y, optional
        Measured and normalized CoM gradients
    """

    if device == "cpu":
        xp = np
        asnumpy = np.asarray
        if isinstance(intensities, np.ndarray):
            intensities = asnumpy(intensities)
        else:
            intensities = cp.asnumpy(intensities)
    elif device == "gpu":
        xp = cp
        asnumpy = cp.asnumpy
        intensities = xp.asarray(intensities)
    else:
        raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

    intensities_shape = np.array(intensities.shape)
    intensities_sum = xp.sum(intensities, axis=(-2, -1))

    # Coordinates
    kx = xp.arange(intensities_shape[-2], dtype=xp.float32)
    ky = xp.arange(intensities_shape[-1], dtype=xp.float32)
    kya, kxa = xp.meshgrid(ky, kx)

    # calculate CoM
    com_measured_x = (
        xp.sum(intensities * kxa[None, None], axis=(-2, -1)) / intensities_sum
    )
    com_measured_y = (
        xp.sum(intensities * kya[None, None], axis=(-2, -1)) / intensities_sum
    )

    # Fit function to center of mass
    # TO-DO: allow py4DSTEM.process.calibration.fit_origin to accept xp.ndarrays
    or_fits = fit_origin(
        (asnumpy(com_measured_x), asnumpy(com_measured_y)),
        fitfunction=fit_function,
    )
    com_fitted_x = xp.asarray(or_fits[0])
    com_fitted_y = xp.asarray(or_fits[1])

    # fix CoM units
    com_normalized_x = (com_measured_x - com_fitted_x) * reciprocal_sampling[0]
    com_normalized_y = (com_measured_y - com_fitted_y) * reciprocal_sampling[1]

    # Optionally, plot
    if plot_center_of_mass:

        figsize = kwargs.get("figsize", (8, 8))
        cmap = kwargs.get("cmap", "RdBu_r")
        kwargs.pop("cmap", None)
        kwargs.pop("figsize", None)

        extent = [
            0,
            scan_sampling[1] * intensities_shape[1],
            scan_sampling[0] * intensities_shape[0],
            0,
        ]

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=(0.25, 0.5))

        for ax, arr, title in zip(
            grid,
            [
                com_measured_x,
                com_measured_y,
                com_normalized_x,
                com_normalized_y,
            ],
            ["CoM_x", "CoM_y", "Normalized CoM_x", "Normalized CoM_y"],
        ):
            ax.imshow(asnumpy(arr), extent=extent, cmap=cmap, **kwargs)
            ax.set_xlabel(f"x [{scan_units[0]}]")
            ax.set_ylabel(f"y [{scan_units[1]}]")
            ax.set_title(title)

    return asnumpy(com_normalized_x), asnumpy(com_normalized_y)


def center_of_mass_relative_rotation(
    com_normalized_x: np.ndarray,
    com_normalized_y: np.ndarray,
    rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
    plot_rotation: bool = True,
    maximize_divergence: bool = False,
    device: str = "cpu",
    **kwargs,
):
    """
    Solves for the relative rotation between scan directions
    and the reciprocal coordinate system. We do this by minimizing the curl of the
    CoM gradient vector field or, alternatively, maximizing the divergence.

    Parameters
    ----------
    com_normalized_x: (Rx,Ry) xp.ndarray
        Normalized horizontal center of mass gradient
    com_normalized_y: (Rx,Ry) xp.ndarray
        Normalized vertical center of mass gradient
    rotation_angles_deg: ndarray, optional
        Array of angles in degrees to perform curl minimization over
    plot_rotation: bool, optional
        If True, the CoM curl minimization search result will be displayed
    maximize_divergence: bool, optional
        If True, the divergence of the CoM gradient vector field is maximized
    device: str, optional
        Device to perform calculations on. Must be either 'cpu' or 'gpu'

    Returns
    --------
    self.com_x: np.ndarray
        Corrected horizontal center of mass gradient, as a numpy array
    self.com_y: np.ndarray
        Corrected vertical center of mass gradient, as a numpy array
    rotation_best_deg: float
        Rotation angle which minimizes CoM curl, in degrees
    rotation_best_transpose: bool
        Whether diffraction intensities need to be transposed to minimize CoM curl

    Displays
    --------
    rotation_curl/div vs rotation_angles_deg, optional
        Vector calculus quantity being minimized/maximized
    rotation_best_deg
        Summary statistics
    """

    if device == "cpu":
        xp = np
        asnumpy = np.asarray
        if isinstance(com_normalized_x, np.ndarray):
            com_normalized_x = asnumpy(com_normalized_x)
        else:
            com_normalized_x = cp.asnumpy(com_normalized_x)
        if isinstance(com_normalized_y, np.ndarray):
            com_normalized_y = asnumpy(com_normalized_y)
        else:
            com_normalized_y = cp.asnumpy(com_normalized_y)
    elif device == "gpu":
        xp = cp
        asnumpy = cp.asnumpy
        com_normalized_x = xp.asarray(com_normalized_x)
        com_normalized_y = xp.asarray(com_normalized_y)
    else:
        raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

    rotation_angles_deg = xp.asarray(rotation_angles_deg)
    rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

    # Untransposed
    com_measured_x = (
        xp.cos(rotation_angles_rad) * com_normalized_x[None]
        - xp.sin(rotation_angles_rad) * com_normalized_y[None]
    )
    com_measured_y = (
        xp.sin(rotation_angles_rad) * com_normalized_x[None]
        + xp.cos(rotation_angles_rad) * com_normalized_y[None]
    )

    if maximize_divergence:
        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
        rotation_div = xp.mean(xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1))
    else:
        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
        rotation_curl = xp.mean(xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1))

    # Transposed
    com_measured_x = (
        xp.cos(rotation_angles_rad) * com_normalized_y[None]
        - xp.sin(rotation_angles_rad) * com_normalized_x[None]
    )
    com_measured_y = (
        xp.sin(rotation_angles_rad) * com_normalized_y[None]
        + xp.cos(rotation_angles_rad) * com_normalized_x[None]
    )

    if maximize_divergence:
        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
        rotation_div_transpose = xp.mean(
            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
        )
    else:
        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
        rotation_curl_transpose = xp.mean(
            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
        )

    rotation_angles_rad = asnumpy(xp.squeeze(rotation_angles_rad))
    rotation_angles_deg = asnumpy(rotation_angles_deg)

    # Find lowest curl/ maximum div value
    if maximize_divergence:
        # Maximize Divergence
        ind_max = xp.argmax(rotation_div).item()
        ind_trans_max = xp.argmax(rotation_div_transpose).item()

        if rotation_div[ind_max] >= rotation_div_transpose[ind_trans_max]:
            rotation_best_deg = rotation_angles_deg[ind_max]
            rotation_best_rad = rotation_angles_rad[ind_max]
            rotation_best_transpose = False
        else:
            rotation_best_deg = rotation_angles_deg[ind_trans_max]
            rotation_best_rad = rotation_angles_rad[ind_trans_max]
            rotation_best_transpose = True
    else:
        # Minimize Curl
        ind_min = xp.argmin(rotation_curl).item()
        ind_trans_min = xp.argmin(rotation_curl_transpose).item()

        if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
            rotation_best_deg = rotation_angles_deg[ind_min]
            rotation_best_rad = rotation_angles_rad[ind_min]
            rotation_best_transpose = False
        else:
            rotation_best_deg = rotation_angles_deg[ind_trans_min]
            rotation_best_rad = rotation_angles_rad[ind_trans_min]
            rotation_best_transpose = True

    # Print summary
    print(("Best fit rotation = " f"{str(np.round(rotation_best_deg))} degrees."))
    if rotation_best_transpose:
        print("Diffraction intensities should be transposed.")
    else:
        print("No need to transpose diffraction intensities.")

    # Plot Curl/Div rotation
    if plot_rotation:

        figsize = kwargs.get("figsize", (8, 2))
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            rotation_angles_deg,
            asnumpy(rotation_div) if maximize_divergence else asnumpy(rotation_curl),
            label="CoM",
        )
        ax.plot(
            rotation_angles_deg,
            asnumpy(rotation_div_transpose)
            if maximize_divergence
            else asnumpy(rotation_curl_transpose),
            label="CoM after transpose",
        )
        y_r = ax.get_ylim()
        ax.plot(
            np.ones(2) * rotation_best_deg,
            y_r,
            color=(0, 0, 0, 1),
        )

        ax.legend(loc="best")
        ax.set_xlabel("Rotation [degrees]")
        if maximize_divergence:
            ax.set_ylabel("Mean Absolute Divergence")
            ax.set_aspect(
                np.ptp(rotation_angles_deg)
                / np.maximum(
                    np.ptp(rotation_div),
                    np.ptp(rotation_div_transpose),
                )
                / 4
            )
        else:
            ax.set_ylabel("Mean Absolute Curl")
            ax.set_aspect(
                np.ptp(rotation_angles_deg)
                / np.maximum(
                    np.ptp(rotation_curl),
                    np.ptp(rotation_curl_transpose),
                )
                / 4
            )
        fig.tight_layout()

    # Calculate corrected CoM
    if rotation_best_transpose:
        com_x = (
            xp.cos(rotation_best_rad) * com_normalized_y
            - xp.sin(rotation_best_rad) * com_normalized_x
        )
        com_y = (
            xp.sin(rotation_best_rad) * com_normalized_y
            + xp.cos(rotation_best_rad) * com_normalized_x
        )
    else:
        com_x = (
            xp.cos(rotation_best_rad) * com_normalized_x
            - xp.sin(rotation_best_rad) * com_normalized_y
        )
        com_y = (
            xp.sin(rotation_best_rad) * com_normalized_x
            + xp.cos(rotation_best_rad) * com_normalized_y
        )

    com_x = asnumpy(com_x)
    com_y = asnumpy(com_y)

    return com_x, com_y, rotation_best_deg, rotation_best_transpose

def subdivide_into_batches(num_items: int, num_batches: int = None, max_batch: int = None):
    """
    Split an n integer into m (almost) equal integers, such that the sum of smaller integers equals n.

    Parameters
    ----------
    n: int
        The integer to split.
    m: int
        The number integers n will be split into.

    Returns
    -------
    list of int
    """
    if (num_batches is not None) & (max_batch is not None):
        raise RuntimeError()

    if num_batches is None:
        if max_batch is not None:
            num_batches = (num_items + (-num_items % max_batch)) // max_batch
        else:
            raise RuntimeError()

    if num_items < num_batches:
        raise RuntimeError('num_batches may not be larger than num_items')

    elif num_items % num_batches == 0:
        return [num_items // num_batches] * num_batches
    else:
        v = []
        zp = num_batches - (num_items % num_batches)
        pp = num_items // num_batches
        for i in range(num_batches):
            if i >= zp:
                v = [pp + 1] + v
            else:
                v = [pp] + v
        return v

def generate_batches(num_items: int, num_batches: int = None, max_batch: int = None, start=0):
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end
