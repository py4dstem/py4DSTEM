import functools
from typing import Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    import cupy as cp
    from cupyx.scipy.fft import rfft
except ImportError:
    cp = None
    from scipy.fft import dstn, idstn

from py4DSTEM.process.utils import get_CoM
from py4DSTEM.process.utils.cross_correlate import align_and_shift_images
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from scipy.ndimage import gaussian_filter, uniform_filter1d
from skimage.restoration import unwrap_phase

# fmt: off

#: Symbols for the polar representation of all optical aberrations up to the fifth order.
polar_symbols = (
        "C10", "C12", "phi12",
        "C21", "phi21", "C23", "phi23",
        "C30", "C32", "phi32", "C34", "phi34",
        "C41", "phi41", "C43", "phi43", "C45", "phi45",
        "C50", "C52", "phi52", "C54", "phi54", "C56", "phi56",
)

#: Aliases for the most commonly used optical aberrations.
polar_aliases = {
        "defocus": "C10", "astigmatism": "C12", "astigmatism_angle": "phi12",
        "coma": "C21", "coma_angle": "phi21",
        "Cs": "C30",
        "C5": "C50",
}

# fmt: on

### Probe functions


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
    vacuum_probe_intensity: np.ndarray, optional
        Squared of corner-centered aperture amplitude to use, instead of semiangle_cutoff + rolloff
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
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        device: str = "cpu",
        focal_spread: float = 0.0,
        angular_spread: float = 0.0,
        gaussian_spread: float = 0.0,
        phase_shift: float = 0.0,
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):
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
            return vacuum_probe_amplitude

        if self._semiangle_cutoff == xp.inf:
            return xp.ones_like(alpha)

        if self._rolloff > 0.0:
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
        kx = xp.asarray(kx, dtype=xp.float32)
        ky = xp.asarray(ky, dtype=xp.float32)
        return kx, ky

    def polar_coordinates(self, x, y):
        """Calculate a polar grid for a given Cartesian grid."""
        xp = self._xp
        alpha = xp.sqrt(x.reshape((-1, 1)) ** 2 + y.reshape((1, -1)) ** 2)
        phi = xp.arctan2(x.reshape((-1, 1)), y.reshape((1, -1)))
        return alpha, phi

    def build(self):
        """Builds corner-centered complex probe in the center of the region of interest."""
        xp = self._xp
        array = xp.fft.ifft2(self._evaluate_ctf())
        array = array / xp.sqrt((xp.abs(array) ** 2).sum())
        self._array = array
        return self

    def visualize(self, **kwargs):
        """Plots the probe intensity."""
        xp = self._xp
        asnumpy = self._asnumpy

        cmap = kwargs.get("cmap", "Greys_r")
        kwargs.pop("cmap", None)

        plt.imshow(
            asnumpy(xp.abs(xp.fft.ifftshift(self._array)) ** 2),
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


### FFT-shift functions


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
    kx = xp.asarray(kx, dtype=xp.float32)
    ky = xp.asarray(ky, dtype=xp.float32)
    positions = xp.asarray(positions, dtype=xp.float32)
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
    translation_operator = fourier_translation_operator(positions, array.shape[-2:], xp)
    fourier_array = xp.fft.fft2(array)

    if len(translation_operator.shape) == 3 and len(fourier_array.shape) == 3:
        shifted_fourier_array = fourier_array[None] * translation_operator[:, None]
    else:
        shifted_fourier_array = fourier_array * translation_operator

    return xp.fft.ifft2(shifted_fourier_array)


### Batching functions


def subdivide_into_batches(
    num_items: int, num_batches: int = None, max_batch: int = None
):
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
        raise RuntimeError("num_batches may not be larger than num_items")

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


def generate_batches(
    num_items: int, num_batches: int = None, max_batch: int = None, start=0
):
    for batch in subdivide_into_batches(num_items, num_batches, max_batch):
        end = start + batch
        yield start, end

        start = end


#### Affine transformation functions


class AffineTransform:
    """
    Affine Transform Class.

    Simplified version of AffineTransform from tike:
    https://github.com/AdvancedPhotonSource/tike/blob/f9004a32fda5e49fa63b987e9ffe3c8447d59950/src/tike/ptycho/position.py

    AffineTransform() -> Identity

    Parameters
    ----------
    scale0: float
        x-scaling
    scale1: float
        y-scaling
    shear1: float
        \\gamma shear
    angle: float
        \\theta rotation angle
    t0: float
        x-translation
    t1: float
        y-translation
    dilation: float
        Isotropic expansion (multiplies scale0 and scale1)
    """

    def __init__(
        self,
        scale0: float = 1.0,
        scale1: float = 1.0,
        shear1: float = 0.0,
        angle: float = 0.0,
        t0: float = 0.0,
        t1: float = 0.0,
        dilation: float = 1.0,
    ):
        self.scale0 = scale0 * dilation
        self.scale1 = scale1 * dilation
        self.shear1 = shear1
        self.angle = angle
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def fromarray(self, T: np.ndarray):
        """
        Return an Affine Transfrom from a 2x2 matrix.
        Use decomposition method from Graphics Gems 2 Section 7.1
        """
        R = T[:2, :2].copy()
        scale0 = np.linalg.norm(R[0])
        if scale0 <= 0:
            return AffineTransform()
        R[0] /= scale0
        shear1 = R[0] @ R[1]
        R[1] -= shear1 * R[0]
        scale1 = np.linalg.norm(R[1])
        if scale1 <= 0:
            return AffineTransform()
        R[1] /= scale1
        shear1 /= scale1
        angle = np.arccos(R[0, 0])

        if T.shape[0] > 2:
            t0, t1 = T[2]
        else:
            t0 = t1 = 0.0

        return AffineTransform(
            scale0=float(scale0),
            scale1=float(scale1),
            shear1=float(shear1),
            angle=float(angle),
            t0=t0,
            t1=t1,
        )

    def asarray(self):
        """
        Return an 2x2 matrix of scale, shear, rotation.
        This matrix is scale @ shear @ rotate from left to right.
        """
        cosx = np.cos(self.angle)
        sinx = np.sin(self.angle)
        return (
            np.array(
                [
                    [self.scale0, 0.0],
                    [0.0, self.scale1],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [1.0, 0.0],
                    [self.shear1, 1.0],
                ],
                dtype="float32",
            )
            @ np.array(
                [
                    [+cosx, -sinx],
                    [+sinx, +cosx],
                ],
                dtype="float32",
            )
        )

    def asarray3(self):
        """
        Return an 3x2 matrix of scale, shear, rotation, translation.
        This matrix is scale @ shear @ rotate from left to right.
        Expects a homogenous (z) coordinate of 1.
        """
        T = np.empty((3, 2), dtype="float32")
        T[2] = (self.t0, self.t1)
        T[:2, :2] = self.asarray()
        return T

    def astuple(self):
        """Return the constructor parameters in a tuple."""
        return (
            self.scale0,
            self.scale1,
            self.shear1,
            self.angle,
            self.t0,
            self.t1,
        )

    def __call__(self, x: np.ndarray, origin=(0, 0), xp=np):
        origin = xp.asarray(origin, dtype=xp.float32)
        tf_matrix = self.asarray()
        tf_matrix = xp.asarray(tf_matrix, dtype=xp.float32)
        tf_translation = xp.array((self.t0, self.t1)) + origin
        return ((x - origin) @ tf_matrix) + tf_translation

    def __str__(self):
        return (
            "AffineTransform( \n"
            f"  scale0 = {self.scale0:.4f}, scale1 = {self.scale1:.4f}, \n"
            f"  shear1 = {self.shear1:.4f}, angle = {self.angle:.4f}, \n"
            f"  t0 = {self.t0:.4f}, t1 = {self.t1:.4f}, \n"
            ")"
        )


def estimate_global_transformation(
    positions0: np.ndarray,
    positions1: np.ndarray,
    origin: Tuple[int, int] = (0, 0),
    translation_allowed: bool = True,
    xp=np,
):
    """Use least squares to estimate the global affine transformation."""
    origin = xp.asarray(origin, dtype=xp.float32)

    try:
        if translation_allowed:
            a = xp.pad(positions0 - origin, ((0, 0), (0, 1)), constant_values=1)
        else:
            a = positions0 - origin

        b = positions1 - origin
        aT = a.conj().swapaxes(-1, -2)
        x = xp.linalg.inv(aT @ a) @ aT @ b

        tf = AffineTransform.fromarray(x)

    except xp.linalg.LinAlgError:
        tf = AffineTransform()

    error = xp.linalg.norm(tf(positions0, origin=origin, xp=xp) - positions1)

    return tf, error


def estimate_global_transformation_ransac(
    positions0: np.ndarray,
    positions1: np.ndarray,
    origin: Tuple[int, int] = (0, 0),
    translation_allowed: bool = True,
    min_sample: int = 64,
    max_error: float = 16,
    min_consensus: float = 0.75,
    max_iter: int = 20,
    xp=np,
):
    """Use RANSAC to estimate the global affine transformation."""
    best_fitness = np.inf  # small fitness is good
    transform = AffineTransform()

    # Choose a subset
    for subset in np.random.choice(
        a=len(positions0),
        size=(max_iter, min_sample),
        replace=True,
    ):
        # Fit to subset
        subset = np.unique(subset)
        candidate_model, _ = estimate_global_transformation(
            positions0=positions0[subset],
            positions1=positions1[subset],
            origin=origin,
            translation_allowed=translation_allowed,
            xp=xp,
        )

        # Determine inliars and outliars
        position_error = xp.linalg.norm(
            candidate_model(positions0, origin=origin, xp=xp) - positions1,
            axis=-1,
        )
        inliars = position_error <= max_error

        # Check if consensus reached
        if xp.sum(inliars) / len(inliars) >= min_consensus:
            # Refit with consensus inliars
            candidate_model, fitness = estimate_global_transformation(
                positions0=positions0[inliars],
                positions1=positions1[inliars],
                origin=origin,
                translation_allowed=translation_allowed,
                xp=xp,
            )
            if fitness < best_fitness:
                best_fitness = fitness
                transform = candidate_model

    return transform, best_fitness


def fourier_ring_correlation(
    image_1,
    image_2,
    pixel_size=None,
    bin_size=None,
    sigma=None,
    align_images=False,
    upsample_factor=8,
    device="cpu",
    plot_frc=True,
    frc_color="red",
    half_bit_color="blue",
):
    """
    Computes fourier ring correlation (FRC) of 2 arrays.
    Arrays must bet the same size.

    Parameters
     ----------
    image1: ndarray
        first image for FRC
    image2: ndarray
        second image for FRC
    pixel_size: tuple
        size of pixels in A (x,y)
    bin_size: float, optional
        size of bins for ring profile
    sigma: float, optional
        standard deviation for Gaussian kernel
    align_images: bool
        if True, aligns images using DFT upsampling of cross correlation.
    upsample factor: int
        if align_images, upsampling for correlation. Must be greater than 2.
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    plot_frc: bool, optional
        if True, plots frc
    frc_color: str, optional
        color of FRC line in plot
    half_bit_color: str, optional
        color of half-bit line

    Returns
    --------
    q_frc: ndarray
        spatial frequencies of FRC
    frc: ndarray
        fourier ring correlation
    half_bit: ndarray
        half-bit criteria
    """
    if device == "cpu":
        xp = np
    elif device == "gpu":
        xp = cp

    if align_images:
        image_2 = align_and_shift_images(
            image_1,
            image_2,
            upsample_factor=upsample_factor,
            device=device,
        )

    fft_image_1 = xp.fft.fft2(image_1)
    fft_image_2 = xp.fft.fft2(image_2)

    cc_mixed = xp.real(fft_image_1 * xp.conj(fft_image_2))
    cc_image_1 = xp.abs(fft_image_1) ** 2
    cc_image_2 = xp.abs(fft_image_2) ** 2

    # take 1D profile
    q_frc, cc_mixed_1D, n = return_1D_profile(
        cc_mixed,
        pixel_size=pixel_size,
        sigma=sigma,
        bin_size=bin_size,
        device=device,
    )
    _, cc_image_1_1D, _ = return_1D_profile(
        cc_image_1, pixel_size=pixel_size, sigma=sigma, bin_size=bin_size, device=device
    )
    _, cc_image_2_1D, _ = return_1D_profile(
        cc_image_2,
        pixel_size=pixel_size,
        sigma=sigma,
        bin_size=bin_size,
        device=device,
    )

    frc = cc_mixed_1D / ((cc_image_1_1D * cc_image_2_1D) ** 0.5)
    half_bit = 2 / xp.sqrt(n / 2)

    ind_max = xp.argmax(n)
    q_frc = q_frc[1:ind_max]
    frc = frc[1:ind_max]
    half_bit = half_bit[1:ind_max]

    if plot_frc:
        fig, ax = plt.subplots()
        if device == "gpu":
            ax.plot(q_frc.get(), frc.get(), label="FRC", color=frc_color)
            ax.plot(q_frc.get(), half_bit.get(), label="half bit", color=half_bit_color)
            ax.set_xlim([0, q_frc.get().max()])
        else:
            ax.plot(q_frc, frc, label="FRC", color=frc_color)
            ax.plot(q_frc, half_bit, label="half bit", color=half_bit_color)
            ax.set_xlim([0, q_frc.max()])
        ax.legend()
        ax.set_ylim([0, 1])

        if pixel_size is None:
            ax.set_xlabel(r"Spatial frequency (pixels)")
        else:
            ax.set_xlabel(r"Spatial frequency ($\AA$)")
        ax.set_ylabel("FRC")

    return q_frc, frc, half_bit


def return_1D_profile(
    intensity, pixel_size=None, bin_size=None, sigma=None, device="cpu"
):
    """
    Return 1D radial profile from corner centered array

    Parameters
     ----------
    intensity: ndarray
        Array for computing 1D profile
    pixel_size: tuple
        Size of pixels in A (x,y)
    bin_size: float, optional
        Size of bins for ring profile
    sigma: float, optional
        standard deviation for Gaussian kernel
    device: str, optional
        calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Returns
    --------
    q_bins: ndarray
        spatial frequencies of bins
    I_bins: ndarray
        Intensity of bins
    n: ndarray
        Number of pixels in each bin
    """
    if device == "cpu":
        xp = np
    elif device == "gpu":
        xp = cp

    if pixel_size is None:
        pixel_size = (1, 1)

    x = xp.fft.fftfreq(intensity.shape[0], pixel_size[0])
    y = xp.fft.fftfreq(intensity.shape[1], pixel_size[1])
    q = xp.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    q = q.ravel()

    intensity = intensity.ravel()

    if bin_size is None:
        bin_size = q[1] - q[0]

    q_bins = xp.arange(0, q.max() + bin_size, bin_size)

    inds = q / bin_size
    inds_f = xp.floor(inds).astype("int")
    d_ind = inds - inds_f

    nf = xp.bincount(inds_f, weights=(1 - d_ind), minlength=q_bins.shape[0])
    nc = xp.bincount(inds_f + 1, weights=(d_ind), minlength=q_bins.shape[0])
    n = nf + nc

    I_bins0 = xp.bincount(
        inds_f, weights=intensity * (1 - d_ind), minlength=q_bins.shape[0]
    )
    I_bins1 = xp.bincount(
        inds_f + 1, weights=intensity * (d_ind), minlength=q_bins.shape[0]
    )

    I_bins = (I_bins0 + I_bins1) / n
    if sigma is not None:
        I_bins = gaussian_filter(I_bins, sigma)

    return q_bins, I_bins, n


def fourier_rotate_real_volume(array, angle, axes=(0, 1), xp=np):
    """
    Rotates a 3D array using three Fourier-based shear operators.

    Parameters
     ----------
    array: ndarray
        3D array to rotate
    angle: float
        Angle in deg to rotate array by
    axes: tuple, Optional
        Axes defining plane in which to rotate about
    xp: Callable, optional
        Array computing module

    Returns
    --------
    output_arr: ndarray
        Fourier-rotated array
    """
    input_arr = xp.asarray(array, dtype=array.dtype)
    array_shape = np.array(input_arr.shape)
    ndim = input_arr.ndim

    if ndim != 3:
        raise ValueError("input array should be 3D")

    axes = list(axes)

    if len(axes) != 2:
        raise ValueError("axes should contain exactly two values")

    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError("axes should contain only integer values")

    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError("invalid rotation plane specified")

    axes.sort()
    rotation_ax = np.setdiff1d([0, 1, 2], axes)[0]
    plane_dims = array_shape[axes]

    qx = xp.fft.fftfreq(plane_dims[0], 1)
    qy = xp.fft.fftfreq(plane_dims[1], 1)
    qxa, qya = xp.meshgrid(qx, qy, indexing="ij")

    x = xp.arange(plane_dims[0]) - plane_dims[0] / 2
    y = xp.arange(plane_dims[1]) - plane_dims[1] / 2
    xa, ya = xp.meshgrid(x, y, indexing="ij")

    theta_90 = round(angle / 90)
    theta_rest = (angle + 45) % 90 - 45

    theta = np.deg2rad(theta_rest)
    a = np.tan(-theta / 2)
    b = np.sin(theta)

    xOp = xp.exp(-2j * np.pi * qxa * ya * a)
    yOp = xp.exp(-2j * np.pi * qya * xa * b)

    output_arr = input_arr.copy()

    # 90 degree rotation
    if abs(theta_90) > 0:
        if plane_dims[0] == plane_dims[1]:
            output_arr = xp.rot90(output_arr, theta_90, axes=axes)
        else:
            if plane_dims[0] > plane_dims[1]:
                xx = np.arange(plane_dims[1]) + (plane_dims[0] - plane_dims[1]) // 2
                if rotation_ax == 0:
                    output_arr[:, xx, :] = xp.rot90(
                        output_arr[:, xx, :], theta_90, axes=axes
                    )
                    output_arr[:, : xx[0], :] = 0
                    output_arr[:, xx[-1] :, :] = 0
                else:
                    output_arr[xx, :, :] = xp.rot90(
                        output_arr[xx, :, :], theta_90, axes=axes
                    )
                    output_arr[: xx[0], :, :] = 0
                    output_arr[xx[-1] :, :, :] = 0
            else:
                yy = np.arange(plane_dims[0]) + (plane_dims[1] - plane_dims[0]) // 2
                if rotation_ax == 2:
                    output_arr[:, yy, :] = xp.rot90(
                        output_arr[:, yy, :], theta_90, axes=axes
                    )
                    output_arr[:, : yy[0], :] = 0
                    output_arr[:, yy[-1] :, :] = 0
                else:
                    output_arr[:, :, yy] = xp.rot90(
                        output_arr[:, :, yy], theta_90, axes=axes
                    )
                    output_arr[:, :, : yy[0]] = 0
                    output_arr[:, :, yy[-1] :] = 0

    # small rotation
    if rotation_ax == 0:
        output_arr = xp.fft.ifft(xp.fft.fft(output_arr, axis=1) * xOp[None, :], axis=1)
        output_arr = xp.fft.ifft(xp.fft.fft(output_arr, axis=2) * yOp[None, :], axis=2)
        output_arr = xp.fft.ifft(xp.fft.fft(output_arr, axis=1) * xOp[None, :], axis=1)
        output_arr = xp.real(output_arr)

    elif rotation_ax == 1:
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=0) * xOp[:, None, :], axis=0
        )
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=2) * yOp[:, None, :], axis=2
        )
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=0) * xOp[:, None, :], axis=0
        )
        output_arr = np.real(output_arr)

    else:
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=0) * xOp[:, :, None], axis=0
        )
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=1) * yOp[:, :, None], axis=1
        )
        output_arr = xp.fft.ifft(
            xp.fft.fft(output_arr, axis=0) * xOp[:, :, None], axis=0
        )
        output_arr = xp.real(output_arr)

    return output_arr


### Divergence Projection Functions


def compute_divergence(vector_field, spacings, xp=np):
    """Computes divergence of vector_field"""
    num_dims = len(spacings)
    div = xp.zeros_like(vector_field[0])

    for i in range(num_dims):
        div += xp.gradient(vector_field[i], spacings[i], axis=i)

    return div


def compute_gradient(scalar_field, spacings, xp=np):
    """Computes gradient of scalar_field"""
    num_dims = len(spacings)
    grad = xp.zeros((num_dims,) + scalar_field.shape)

    for i in range(num_dims):
        grad[i] = xp.gradient(scalar_field, spacings[i], axis=i)

    return grad


def array_slice(axis, ndim, start, end, step=1):
    """Returns array slice along dynamic axis"""
    return (slice(None),) * (axis % ndim) + (slice(start, end, step),)


def make_array_rfft_compatible(array_nd, axis=0, xp=np):
    """Expand array to be rfft compatible"""
    array_shape = np.array(array_nd.shape)
    d = array_nd.ndim
    n = array_shape[axis]
    array_shape[axis] = (n + 1) * 2

    dtype = array_nd.dtype
    padded_array = xp.zeros(array_shape, dtype=dtype)

    padded_array[array_slice(axis, d, 1, n + 1)] = -array_nd
    padded_array[array_slice(axis, d, None, -n - 1, -1)] = array_nd

    return padded_array


def dst_I(array_nd, xp=np):
    """1D rfft-based DST-I"""
    d = array_nd.ndim
    for axis in range(d):
        crop_slice = array_slice(axis, d, 1, -1)
        array_nd = rfft(
            make_array_rfft_compatible(array_nd, axis=axis, xp=xp), axis=axis
        )[crop_slice].imag

    return array_nd


def idst_I(array_nd, xp=np):
    """1D rfft-based iDST-I"""
    scaling = np.prod((np.array(array_nd.shape) + 1) * 2)
    return dst_I(array_nd, xp=xp) / scaling


def preconditioned_laplacian(num_exterior, spacing=1, xp=np):
    """DST-I eigenvalues"""
    n = num_exterior - 1
    evals_1d = 2 - 2 * xp.cos(np.pi * xp.arange(1, num_exterior) / num_exterior)

    op = (
        xp.repeat(evals_1d, n**2)
        + xp.tile(evals_1d, n**2)
        + xp.tile(xp.repeat(evals_1d, n), n)
    )

    return -op / spacing**2


def preconditioned_poisson_solver(rhs_interior, spacing=1, xp=np):
    """DST-I based poisson solver"""
    nx, ny, nz = rhs_interior.shape
    if nx != ny or nx != nz:
        raise ValueError()

    op = preconditioned_laplacian(nx + 1, spacing=spacing, xp=xp)
    if xp is np:
        dst_rhs = dstn(rhs_interior, type=1).ravel()
        dst_u = (dst_rhs / op).reshape((nx, ny, nz))
        sol = idstn(dst_u, type=1)
    else:
        dst_rhs = dst_I(rhs_interior, xp=xp).ravel()
        dst_u = (dst_rhs / op).reshape((nx, ny, nz))
        sol = idst_I(dst_u, xp=xp)

    return sol


def project_vector_field_divergence(vector_field, spacings=(1, 1, 1), xp=np):
    """
    Returns solenoidal part of vector field using projection:

    f - \\grad{p}
    s.t. \\laplacian{p} = \\div{f}
    """

    div_v = compute_divergence(vector_field, spacings, xp=xp)
    p = preconditioned_poisson_solver(div_v, spacings[0], xp=xp)
    grad_p = compute_gradient(p, spacings, xp=xp)
    return vector_field - grad_p


# Nesterov acceleration functions
# https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/


@functools.cache
def nesterov_lambda(one_indexed_iter_num):
    if one_indexed_iter_num == 0:
        return 0
    return (1 + np.sqrt(1 + 4 * nesterov_lambda(one_indexed_iter_num - 1) ** 2)) / 2


def nesterov_gamma(zero_indexed_iter_num):
    one_indexed_iter_num = zero_indexed_iter_num + 1
    return (1 - nesterov_lambda(one_indexed_iter_num)) / nesterov_lambda(
        one_indexed_iter_num + 1
    )


def cartesian_to_polar_transform_2Ddata(
    im_cart,
    xy_center,
    num_theta_bins=90,
    radius_max=None,
    corner_centered=False,
    xp=np,
):
    """
    Quick cartesian to polar conversion.
    """

    # coordinates
    if radius_max is None:
        if corner_centered:
            radius_max = np.min(np.array(im_cart.shape) // 2)
        else:
            radius_max = np.sqrt(np.sum(np.array(im_cart.shape) ** 2)) // 2

    r = xp.arange(radius_max)
    t = xp.linspace(
        0,
        2.0 * np.pi,
        num_theta_bins,
        endpoint=False,
    )
    ra, ta = xp.meshgrid(r, t)

    # resampling coordinates
    x = ra * xp.cos(ta) + xy_center[0]
    y = ra * xp.sin(ta) + xy_center[1]

    xf = xp.floor(x).astype("int")
    yf = xp.floor(y).astype("int")
    dx = x - xf
    dy = y - yf

    mode = "wrap" if corner_centered else "clip"

    # resample image
    im_polar = (
        im_cart.ravel()[
            xp.ravel_multi_index(
                (xf, yf),
                im_cart.shape,
                mode=mode,
            )
        ]
        * (1 - dx)
        * (1 - dy)
        + im_cart.ravel()[
            xp.ravel_multi_index(
                (xf + 1, yf),
                im_cart.shape,
                mode=mode,
            )
        ]
        * (dx)
        * (1 - dy)
        + im_cart.ravel()[
            xp.ravel_multi_index(
                (xf, yf + 1),
                im_cart.shape,
                mode=mode,
            )
        ]
        * (1 - dx)
        * (dy)
        + im_cart.ravel()[
            xp.ravel_multi_index(
                (xf + 1, yf + 1),
                im_cart.shape,
                mode=mode,
            )
        ]
        * (dx)
        * (dy)
    )

    return im_polar


def polar_to_cartesian_transform_2Ddata(
    im_polar,
    xy_size,
    xy_center,
    corner_centered=False,
    xp=np,
):
    """
    Quick polar to cartesian conversion.
    """

    # coordinates
    sx, sy = xy_size
    cx, cy = xy_center

    if corner_centered:
        x = xp.fft.fftfreq(sx, d=1 / sx)
        y = xp.fft.fftfreq(sy, d=1 / sy)
    else:
        x = xp.arange(sx)
        y = xp.arange(sy)

    xa, ya = xp.meshgrid(x, y, indexing="ij")
    ra = xp.hypot(xa - cx, ya - cy)
    ta = xp.arctan2(ya - cy, xa - cx)

    t = xp.linspace(0, 2 * np.pi, im_polar.shape[0], endpoint=False)
    t_step = t[1] - t[0]

    # resampling coordinates
    t_ind = ta / t_step
    r_ind = ra.copy()
    tf = xp.floor(t_ind).astype("int")
    rf = xp.floor(r_ind).astype("int")

    # resample image
    im_cart = im_polar.ravel()[
        xp.ravel_multi_index(
            (tf, rf),
            im_polar.shape,
            mode=("wrap", "clip"),
        )
    ]

    return im_cart


def regularize_probe_amplitude(
    probe_init,
    width_max_pixels=2.0,
    nearest_angular_neighbor_averaging=5,
    enforce_constant_intensity=True,
    corner_centered=False,
):
    """
    Fits sigmoid for each angular direction.

    Parameters
    --------
    probe_init: np.array
        2D complex image of the probe in Fourier space.
    width_max_pixels: float
        Maximum edge width of the probe in pixels.
    nearest_angular_neighbor_averaging: int
        Number of nearest angular neighbor pixels to average to make aperture less jagged.
    enforce_constant_intensity: bool
        Set to true to make intensity inside the aperture constant.
    corner_centered: bool
        If True, the probe is assumed to be corner-centered

    Returns
    --------
    probe_corr: np.ndarray
        2D complex image of the corrected probe in Fourier space.
    coefs_all: np.ndarray
        coefficients for the sigmoid fits
    """

    # Get probe intensity
    probe_amp = np.abs(probe_init)
    probe_angle = np.angle(probe_init)
    probe_int = probe_amp**2

    # Center of mass for probe intensity
    xy_center = get_CoM(probe_int, device="cpu", corner_centered=corner_centered)

    # Convert intensity to polar coordinates
    polar_int = cartesian_to_polar_transform_2Ddata(
        probe_int,
        xy_center=xy_center,
        corner_centered=corner_centered,
        xp=np,
    )

    # Fit corrected probe intensity
    radius = np.arange(polar_int.shape[1])

    # estimate initial parameters
    sub = polar_int > (np.max(polar_int) * 0.5)
    sig_0 = np.mean(polar_int[sub])
    rad_0 = np.max(np.argwhere(np.sum(sub, axis=0)))
    width = width_max_pixels * 0.5

    # init
    def step_model(radius, sig_0, rad_0, width):
        return sig_0 * np.clip((rad_0 - radius) / width, 0.0, 1.0)

    coefs_all = np.zeros((polar_int.shape[0], 3))
    coefs_all[:, 0] = sig_0
    coefs_all[:, 1] = rad_0
    coefs_all[:, 2] = width

    # bounds
    lb = (0.0, 0.0, 1e-4)
    ub = (np.inf, np.inf, width_max_pixels)

    # refine parameters, generate polar image
    polar_fit = np.zeros_like(polar_int)
    for a0 in range(polar_int.shape[0]):
        coefs_all[a0, :] = curve_fit(
            step_model,
            radius,
            polar_int[a0, :],
            p0=coefs_all[a0, :],
            xtol=1e-12,
            bounds=(lb, ub),
        )[0]
        polar_fit[a0, :] = step_model(radius, *coefs_all[a0, :])

    # Compute best-fit constant intensity inside probe, update bounds
    sig_0 = np.median(coefs_all[:, 0])
    coefs_all[:, 0] = sig_0
    lb = (sig_0 - 1e-8, 0.0, 1e-4)
    ub = (sig_0 + 1e-8, np.inf, width_max_pixels)

    # refine parameters, generate polar image
    polar_int_corr = np.zeros_like(polar_int)
    for a0 in range(polar_int.shape[0]):
        coefs_all[a0, :] = curve_fit(
            step_model,
            radius,
            polar_int[a0, :],
            p0=coefs_all[a0, :],
            xtol=1e-12,
            bounds=(lb, ub),
        )[0]
        # polar_int_corr[a0, :] = step_model(radius, *coefs_all[a0, :])

    # make aperture less jagged, using moving mean
    coefs_all = np.apply_along_axis(
        uniform_filter1d,
        0,
        coefs_all,
        size=nearest_angular_neighbor_averaging,
        mode="wrap",
    )
    for a0 in range(polar_int.shape[0]):
        polar_int_corr[a0, :] = step_model(radius, *coefs_all[a0, :])

    # Convert back to cartesian coordinates
    int_corr = polar_to_cartesian_transform_2Ddata(
        polar_int_corr,
        xy_size=probe_init.shape,
        xy_center=xy_center,
        corner_centered=corner_centered,
    )

    amp_corr = np.sqrt(np.maximum(int_corr, 0))

    # Assemble output probe
    if not enforce_constant_intensity:
        max_coeff = np.sqrt(coefs_all[:, 0]).max()
        amp_corr = amp_corr / max_coeff * probe_amp

    probe_corr = amp_corr * np.exp(1j * probe_angle)

    return probe_corr, polar_int, polar_int_corr, coefs_all


def aberrations_basis_function(
    probe_size,
    probe_sampling,
    energy,
    max_angular_order,
    max_radial_order,
    xp=np,
):
    """ """

    # Add constant phase shift in basis
    mn = [[-1, 0, 0]]

    for m in range(1, max_radial_order):
        n_max = np.minimum(max_angular_order, m + 1)
        for n in range(0, n_max + 1):
            if (m + n) % 2:
                mn.append([m, n, 0])
                if n > 0:
                    mn.append([m, n, 1])

    aberrations_mn = np.array(mn)
    aberrations_mn = aberrations_mn[np.argsort(aberrations_mn[:, 1]), :]

    sub = aberrations_mn[:, 1] > 0
    aberrations_mn[sub, :] = aberrations_mn[sub, :][
        np.argsort(aberrations_mn[sub, 0]), :
    ]
    aberrations_mn[~sub, :] = aberrations_mn[~sub, :][
        np.argsort(aberrations_mn[~sub, 0]), :
    ]
    aberrations_num = aberrations_mn.shape[0]

    sx, sy = probe_size
    dx, dy = probe_sampling
    wavelength = electron_wavelength_angstrom(energy)

    qx = xp.fft.fftfreq(sx, dx)
    qy = xp.fft.fftfreq(sy, dy)
    qr2 = qx[:, None] ** 2 + qy[None, :] ** 2
    alpha = xp.sqrt(qr2) * wavelength
    theta = xp.arctan2(qy[None, :], qx[:, None])

    # Aberration basis
    aberrations_basis = xp.ones((alpha.size, aberrations_num))

    # Skip constant to avoid dividing by zero in normalization
    for a0 in range(1, aberrations_num):
        m, n, a = aberrations_mn[a0]
        if n == 0:
            # Radially symmetric basis
            aberrations_basis[:, a0] = (alpha ** (m + 1) / (m + 1)).ravel()

        elif a == 0:
            # cos coef
            aberrations_basis[:, a0] = (
                alpha ** (m + 1) * xp.cos(n * theta) / (m + 1)
            ).ravel()
        else:
            # sin coef
            aberrations_basis[:, a0] = (
                alpha ** (m + 1) * xp.sin(n * theta) / (m + 1)
            ).ravel()

    # global scaling
    aberrations_basis *= 2 * np.pi / wavelength

    return aberrations_basis, aberrations_mn


def fit_aberration_surface(
    complex_probe,
    probe_sampling,
    energy,
    max_angular_order,
    max_radial_order,
    xp=np,
):
    """ """
    probe_amp = xp.abs(complex_probe)
    probe_angle = xp.angle(complex_probe)

    if xp is np:
        probe_angle = probe_angle.astype(np.float64)
        unwrapped_angle = unwrap_phase(probe_angle, wrap_around=True).astype(xp.float32)
    else:
        probe_angle = xp.asnumpy(probe_angle).astype(np.float64)
        unwrapped_angle = unwrap_phase(probe_angle, wrap_around=True)
        unwrapped_angle = xp.asarray(unwrapped_angle).astype(xp.float32)

    raveled_basis, _ = aberrations_basis_function(
        complex_probe.shape,
        probe_sampling,
        energy,
        max_angular_order,
        max_radial_order,
        xp=xp,
    )

    raveled_weights = probe_amp.ravel()

    Aw = raveled_basis * raveled_weights[:, None]
    bw = unwrapped_angle.ravel() * raveled_weights
    coeff = xp.linalg.lstsq(Aw, bw, rcond=None)[0]

    fitted_angle = xp.tensordot(raveled_basis, coeff, axes=1).reshape(probe_angle.shape)

    return fitted_angle, coeff


def rotate_point(origin, point, angle):
    """
    Rotate a point (x1, y1) counterclockwise by a given angle around
    a given origin (x0, y0).

    Parameters
    --------
    origin: 2-tuple of floats
        (x0, y0)
    point: 2-tuple of floats
        (x1, y1)
    angle: float (radians)

    Returns
    --------
    rotated points (2-tuple)

    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
