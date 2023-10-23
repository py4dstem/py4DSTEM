"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from py4DSTEM.visualize import show, show_complex
from scipy.ndimage import rotate

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np

from emdfile import Array, Custom, Metadata, _read_metadata, tqdmnd
from py4DSTEM.data import Calibration
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.phase.iterative_ptychographic_constraints import (
    PtychographicConstraints,
)
from py4DSTEM.process.phase.utils import AffineTransform, polar_aliases
from py4DSTEM.process.utils import (
    electron_wavelength_angstrom,
    fourier_resample,
    get_shifted_ar,
)

warnings.simplefilter(action="always", category=UserWarning)


class PhaseReconstruction(Custom):
    """
    Base phase reconstruction class.
    Defines various common functions and properties for subclasses to inherit.
    """

    def attach_datacube(self, datacube: DataCube):
        """
        Attaches a datacube to a class initialized without one.

        Parameters
        ----------
        datacube: Datacube
            Input 4D diffraction pattern intensities

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """
        self._datacube = datacube
        return self

    def reinitialize_parameters(self, device: str = None, verbose: bool = None):
        """
        Reinitializes common parameters. This is useful when loading a previously-saved
        reconstruction (which set device='cpu' and verbose=True for compatibility) ,
        using different initialization parameters.

        Parameters
        ----------
        device: str, optional
            If not None, imports and assigns appropriate device modules
        verbose: bool, optional
            If not None, sets the verbosity to verbose

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if device is not None:
            if device == "cpu":
                self._xp = np
                self._asnumpy = np.asarray
                from scipy.ndimage import gaussian_filter

                self._gaussian_filter = gaussian_filter
                from scipy.special import erf

                self._erf = erf
            elif device == "gpu":
                self._xp = cp
                self._asnumpy = cp.asnumpy
                from cupyx.scipy.ndimage import gaussian_filter

                self._gaussian_filter = gaussian_filter
                from cupyx.scipy.special import erf

                self._erf = erf
            else:
                raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")
            self._device = device

        if verbose is not None:
            self._verbose = verbose

        return self

    def set_save_defaults(
        self,
        save_datacube: bool = False,
        save_exit_waves: bool = False,
        save_iterations: bool = True,
        save_iterations_frequency: int = 1,
    ):
        """
        Sets the class defaults for saving reconstructions to file.

        Parameters
        ----------
        save_datacube: bool, optional
            If True, self._datacube saved to file
        save_exit_waves: bool, optional
            If True, self._exit_waves saved to file
        save_iterations: bool, optional
            If True, self.probe_iterations and self.object_iterations saved to file
        save_iterations: int, optional
            If save_iterations is True, controls the frequency of saved iterations

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """
        self._save_datacube = save_datacube
        self._save_exit_waves = save_exit_waves
        self._save_iterations = save_iterations
        self._save_iterations_frequency = save_iterations_frequency
        return self

    def _preprocess_datacube_and_vacuum_probe(
        self,
        datacube,
        diffraction_intensities_shape=None,
        reshaping_method="fourier",
        probe_roi_shape=None,
        vacuum_probe_intensity=None,
        dp_mask=None,
        com_shifts=None,
    ):
        """
        Datacube preprocessing step, to set the reciprocal- and real-space sampling.
        Let the measured diffraction intensities have size (Rx,Ry,Qx,Qy), with reciprocal-space
        samping (dkx,dky). This sets a real-space sampling which is inversely proportional to
        the maximum scattering wavevector (Qx*dkx,Qy*dky).

        Often, it is beneficial to resample the measured diffraction intensities using a different
        reciprocal-space sampling (dkx',dky'), e.g. downsampling to save memory. This is achieved
        by specifying a diffraction_intensities_shape (Sx,Sy) which is different than (Qx,Qy).
        Note this does not affect the maximum scattering wavevector (Qx*dkx,Qy*dky) = (Sx*dkx',Sy*dky'),
        and thus the real-space sampling stays fixed.

        The real space sampling, (dx, dy), combined with the resampled diffraction_intensities_shape,
        sets the real-space probe region of interest (ROI) extent (dx*Sx, dy*Sy).
        Occasionally, one may also want to specify a larger probe ROI extent, e.g when the probe
        does not comfortably fit without self-ovelap artifacts, or when the scan step sizes are much
        smaller than the real-space sampling (dx,dy). This can be achieved by specifying a
        probe_roi_shape, which is larger than diffraction_intensities_shape, which will result in
        zero-padding of the diffraction intensities.

        Parameters
        ----------
        datacube: Datacube
            Input 4D diffraction pattern intensities
        diffraction_intensities_shape: (int,int), optional
            Resampled diffraction intensities shape.
            If None, no resamping is performed
        reshaping method: str, optional
            Reshaping method to use, one of 'bin', 'bilinear' or 'fourier' (default)
        probe_roi_shape, (int,int), optional
            Padded diffraction intensities shape.
            If None, no padding is performed
        vacuum_probe_intensity, np.ndarray, optional
            If not None, the vacuum probe intensity is also resampled and padded
        dp_mask, np.ndarray, optional
            If not None, dp_mask is also resampled and padded
        com_shifts, np.ndarray, optional
            If not None, com_shifts are multiplied by resampling factor

        Returns
        --------
        datacube: Datacube
            Resampled and Padded datacube
        """
        if com_shifts is not None:
            if np.isscalar(com_shifts[0]):
                com_shifts = (
                    np.ones(self._datacube.Rshape) * com_shifts[0],
                    np.ones(self._datacube.Rshape) * com_shifts[1],
                )

        if diffraction_intensities_shape is not None:
            Qx, Qy = datacube.shape[-2:]
            Sx, Sy = diffraction_intensities_shape

            resampling_factor_x = Sx / Qx
            resampling_factor_y = Sy / Qy

            if resampling_factor_x != resampling_factor_y:
                raise ValueError(
                    "Datacube calibration can only handle uniform Q-sampling."
                )

            if com_shifts is not None:
                com_shifts = (
                    com_shifts[0] * resampling_factor_x,
                    com_shifts[1] * resampling_factor_x,
                )

            if reshaping_method == "bin":
                bin_factor = int(1 / resampling_factor_x)
                if bin_factor < 1:
                    raise ValueError(
                        f"Calculated binning factor {bin_factor} is less than 1."
                    )

                datacube = datacube.bin_Q(N=bin_factor)
                if vacuum_probe_intensity is not None:
                    vacuum_probe_intensity = vacuum_probe_intensity[
                        ::bin_factor, ::bin_factor
                    ]
                if dp_mask is not None:
                    dp_mask = dp_mask[::bin_factor, ::bin_factor]
            else:
                datacube = datacube.resample_Q(
                    N=resampling_factor_x, method=reshaping_method
                )
                if vacuum_probe_intensity is not None:
                    vacuum_probe_intensity = fourier_resample(
                        vacuum_probe_intensity,
                        output_size=diffraction_intensities_shape,
                        force_nonnegative=True,
                    )
                if dp_mask is not None:
                    dp_mask = fourier_resample(
                        dp_mask,
                        output_size=diffraction_intensities_shape,
                        force_nonnegative=True,
                    )

        if probe_roi_shape is not None:
            Qx, Qy = datacube.shape[-2:]
            Sx, Sy = probe_roi_shape
            datacube = datacube.pad_Q(output_size=probe_roi_shape)

            if vacuum_probe_intensity is not None or dp_mask is not None:
                pad_kx = Sx - Qx
                pad_kx = (pad_kx // 2, pad_kx // 2 + pad_kx % 2)

                pad_ky = Sy - Qy
                pad_ky = (pad_ky // 2, pad_ky // 2 + pad_ky % 2)

            if vacuum_probe_intensity is not None:
                vacuum_probe_intensity = np.pad(
                    vacuum_probe_intensity, pad_width=(pad_kx, pad_ky), mode="constant"
                )

            if dp_mask is not None:
                dp_mask = np.pad(dp_mask, pad_width=(pad_kx, pad_ky), mode="constant")

        return datacube, vacuum_probe_intensity, dp_mask, com_shifts

    def _extract_intensities_and_calibrations_from_datacube(
        self,
        datacube: DataCube,
        require_calibrations: bool = False,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
    ):
        """
        Method to extract intensities and calibrations from datacube.

        Parameters
        ----------
        datacube: DataCube
            Input 4D diffraction pattern intensities
        require_calibrations: bool
            If False, warning is issued instead of raising an error
        force_scan_sampling: float, optional
            Override DataCube real space scan pixel size calibrations, in Angstrom
        force_angular_sampling: float, optional
            Override DataCube reciprocal pixel size calibration, in mrad
        force_reciprocal_sampling: float, optional
            Override DataCube reciprocal pixel size calibration, in A^-1

        Assigns
        --------
        self._grid_scan_shape: Tuple[int,int]
            Real-space scan size
        self._scan_sampling: Tuple[float,float]
            Real-space scan step sizes in 'A' or 'pixels'
        self._scan_units: Tuple[str,str]
            Real-space scan units
        self._angular_sampling: Tuple[float,float]
            Reciprocal-space sampling in 'mrad' or 'pixels'
        self._angular_units: Tuple[str,str]
            Reciprocal-space angular units
        self._reciprocal_sampling: Tuple[float,float]
            Reciprocal-space sampling in 'A^-1' or 'pixels'
        self._reciprocal_units: Tuple[str,str]
            Reciprocal-space units

        Returns
        -------
        intensities: (Rx,Ry,Qx,Qy) xp.ndarray
            Raw intensities array stored on device, with dtype xp.float32

        Raises
        ------
        ValueError
            If require_calibrations is True and calibrations are not set

        Warns
        ------
        UserWarning
            If require_calibrations is False and calibrations are not set
        """

        # Copies intensities to device casting to float32
        xp = self._xp

        intensities = xp.asarray(datacube.data, dtype=xp.float32)
        self._grid_scan_shape = intensities.shape[:2]

        # Extracts calibrations
        calibration = datacube.calibration
        real_space_units = calibration.get_R_pixel_units()
        reciprocal_space_units = calibration.get_Q_pixel_units()

        # Real-space
        if force_scan_sampling is not None:
            self._scan_sampling = (force_scan_sampling, force_scan_sampling)
            self._scan_units = "A"
        else:
            if real_space_units == "pixels":
                if require_calibrations:
                    raise ValueError("Real-space calibrations must be given in 'A'")

                if self._verbose:
                    warnings.warn(
                        (
                            "Iterative reconstruction will not be quantitative unless you specify "
                            "real-space calibrations in 'A'"
                        ),
                        UserWarning,
                    )

                self._scan_sampling = (1.0, 1.0)
                self._scan_units = ("pixels",) * 2

            elif real_space_units == "A":
                self._scan_sampling = (calibration.get_R_pixel_size(),) * 2
                self._scan_units = ("A",) * 2
            elif real_space_units == "nm":
                self._scan_sampling = (calibration.get_R_pixel_size() * 10,) * 2
                self._scan_units = ("A",) * 2
            else:
                raise ValueError(
                    f"Real-space calibrations must be given in 'A', not {real_space_units}"
                )

        # Reciprocal-space
        if force_angular_sampling is not None or force_reciprocal_sampling is not None:
            # there is no xor keyword in Python!
            angular = force_angular_sampling is not None
            reciprocal = force_reciprocal_sampling is not None
            assert (angular and not reciprocal) or (
                not angular and reciprocal
            ), "Only one of angular or reciprocal calibration can be forced!"

            # angular calibration specified
            if angular:
                self._angular_sampling = (force_angular_sampling,) * 2
                self._angular_units = ("mrad",) * 2

                if self._energy is not None:
                    self._reciprocal_sampling = (
                        force_angular_sampling
                        / electron_wavelength_angstrom(self._energy)
                        / 1e3,
                    ) * 2
                    self._reciprocal_units = ("A^-1",) * 2

            # reciprocal calibration specified
            if reciprocal:
                self._reciprocal_sampling = (force_reciprocal_sampling,) * 2
                self._reciprocal_units = ("A^-1",) * 2

                if self._energy is not None:
                    self._angular_sampling = (
                        force_reciprocal_sampling
                        * electron_wavelength_angstrom(self._energy)
                        * 1e3,
                    ) * 2
                    self._angular_units = ("mrad",) * 2

        else:
            if reciprocal_space_units == "pixels":
                if require_calibrations:
                    raise ValueError(
                        "Reciprocal-space calibrations must be given in in 'A^-1' or 'mrad'"
                    )

                if self._verbose:
                    warnings.warn(
                        (
                            "Iterative reconstruction will not be quantitative unless you specify "
                            "appropriate reciprocal-space calibrations"
                        ),
                        UserWarning,
                    )

                self._angular_sampling = (1.0, 1.0)
                self._angular_units = ("pixels",) * 2
                self._reciprocal_sampling = (1.0, 1.0)
                self._reciprocal_units = ("pixels",) * 2

            elif reciprocal_space_units == "A^-1":
                reciprocal_size = calibration.get_Q_pixel_size()
                self._reciprocal_sampling = (reciprocal_size,) * 2
                self._reciprocal_units = ("A^-1",) * 2

                if self._energy is not None:
                    self._angular_sampling = (
                        reciprocal_size
                        * electron_wavelength_angstrom(self._energy)
                        * 1e3,
                    ) * 2
                    self._angular_units = ("mrad",) * 2

            elif reciprocal_space_units == "mrad":
                angular_size = calibration.get_Q_pixel_size()
                self._angular_sampling = (angular_size,) * 2
                self._angular_units = ("mrad",) * 2

                if self._energy is not None:
                    self._reciprocal_sampling = (
                        angular_size / electron_wavelength_angstrom(self._energy) / 1e3,
                    ) * 2
                    self._reciprocal_units = ("A^-1",) * 2
            else:
                raise ValueError(
                    (
                        "Reciprocal-space calibrations must be given in 'A^-1' or 'mrad', "
                        f"not {reciprocal_space_units}"
                    )
                )

        return intensities

    def _calculate_intensities_center_of_mass(
        self,
        intensities: np.ndarray,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        com_shifts: np.ndarray = None,
        com_measured: np.ndarray = None,
    ):
        """
        Common preprocessing function to compute and fit diffraction intensities CoM

        Parameters
        ----------
        intensities: (Rx,Ry,Qx,Qy) xp.ndarray
            Raw intensities array stored on device, with dtype xp.float32
        dp_mask: ndarray
            If not None, apply mask to datacube amplitude
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
        com_shifts, tuple of ndarrays (CoMx measured, CoMy measured)
            If not None, com_shifts are fitted on the measured CoM values.
        com_measured: tuple of ndarrays (CoMx measured, CoMy measured)
            If not None, com_measured are passed as com_measured_x, com_measured_y
        Returns
        -------

        com_measured_x: (Rx,Ry) xp.ndarray
            Measured horizontal center of mass gradient
        com_measured_y: (Rx,Ry) xp.ndarray
            Measured vertical center of mass gradient
        com_fitted_x: (Rx,Ry) xp.ndarray
            Best fit horizontal center of mass gradient
        com_fitted_y: (Rx,Ry) xp.ndarray
            Best fit vertical center of mass gradient
        com_normalized_x: (Rx,Ry) xp.ndarray
            Normalized horizontal center of mass gradient
        com_normalized_y: (Rx,Ry) xp.ndarray
            Normalized vertical center of mass gradient
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # for ptycho
        if com_measured:
            com_measured_x, com_measured_y = com_measured

        else:
            # Coordinates
            kx = xp.arange(intensities.shape[-2], dtype=xp.float32)
            ky = xp.arange(intensities.shape[-1], dtype=xp.float32)
            kya, kxa = xp.meshgrid(ky, kx)

            # calculate CoM
            if dp_mask is not None:
                if dp_mask.shape != intensities.shape[-2:]:
                    raise ValueError(
                        (
                            f"Mask shape should be (Qx,Qy):{intensities.shape[-2:]}, "
                            f"not {dp_mask.shape}"
                        )
                    )
                intensities_mask = intensities * xp.asarray(dp_mask, dtype=xp.float32)
            else:
                intensities_mask = intensities

            intensities_sum = xp.sum(intensities_mask, axis=(-2, -1))
            com_measured_x = (
                xp.sum(intensities_mask * kxa[None, None], axis=(-2, -1))
                / intensities_sum
            )
            com_measured_y = (
                xp.sum(intensities_mask * kya[None, None], axis=(-2, -1))
                / intensities_sum
            )

        if com_shifts is None:
            com_measured_x_np = asnumpy(com_measured_x)
            com_measured_y_np = asnumpy(com_measured_y)
            finite_mask = np.isfinite(com_measured_x_np)

            com_shifts = fit_origin(
                (com_measured_x_np, com_measured_y_np),
                fitfunction=fit_function,
                mask=finite_mask,
            )

        # Fit function to center of mass
        com_fitted_x = xp.asarray(com_shifts[0], dtype=xp.float32)
        com_fitted_y = xp.asarray(com_shifts[1], dtype=xp.float32)

        # fix CoM units
        com_normalized_x = (
            xp.nan_to_num(com_measured_x - com_fitted_x) * self._reciprocal_sampling[0]
        )
        com_normalized_y = (
            xp.nan_to_num(com_measured_y - com_fitted_y) * self._reciprocal_sampling[1]
        )

        return (
            com_measured_x,
            com_measured_y,
            com_fitted_x,
            com_fitted_y,
            com_normalized_x,
            com_normalized_y,
        )

    def _solve_for_center_of_mass_relative_rotation(
        self,
        _com_measured_x: np.ndarray,
        _com_measured_y: np.ndarray,
        _com_normalized_x: np.ndarray,
        _com_normalized_y: np.ndarray,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        plot_rotation: bool = True,
        plot_center_of_mass: str = "default",
        maximize_divergence: bool = False,
        force_com_rotation: float = None,
        force_com_transpose: bool = None,
        **kwargs,
    ):
        """
        Common method to solve for the relative rotation between scan directions
        and the reciprocal coordinate system. We do this by minimizing the curl of the
        CoM gradient vector field or, alternatively, maximizing the divergence.

        Parameters
        ----------
        _com_measured_x: (Rx,Ry) xp.ndarray
            Measured horizontal center of mass gradient
        _com_measured_y: (Rx,Ry) xp.ndarray
            Measured vertical center of mass gradient
        _com_normalized_x: (Rx,Ry) xp.ndarray
            Normalized horizontal center of mass gradient
        _com_normalized_y: (Rx,Ry) xp.ndarray
            Normalized vertical center of mass gradient
        rotation_angles_deg: ndarray, optional
            Array of angles in degrees to perform curl minimization over
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        plot_center_of_mass: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        maximize_divergence: bool, optional
            If True, the divergence of the CoM gradient vector field is maximized
        force_com_rotation: float (degrees), optional
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool, optional
            Force whether diffraction intensities need to be transposed.

        Returns
        --------
        _rotation_best_rad: float
            Rotation angle which minimizes CoM curl, in radians
        _rotation_best_transpose: bool
            Whether diffraction intensities need to be transposed to minimize CoM curl
        _com_x: xp.ndarray
            Corrected horizontal center of mass gradient, on calculation device
        _com_y: xp.ndarray
            Corrected vertical center of mass gradient, on calculation device
        com_x: np.ndarray
            Corrected horizontal center of mass gradient, as a numpy array
        com_y: np.ndarray
            Corrected vertical center of mass gradient, as a numpy array

        Displays
        --------
        rotation_curl/div vs rotation_angles_deg, optional
            Vector calculus quantity being minimized/maximized
        com_measured_x/y, com_normalized_x/y and com_x/y, optional
            Measured and normalized CoM gradients
        rotation_best_deg, optional
            Summary statistics
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if force_com_rotation is not None:
            # Rotation known

            _rotation_best_rad = np.deg2rad(force_com_rotation)

            if self._verbose:
                warnings.warn(
                    (
                        "Best fit rotation forced to "
                        f"{force_com_rotation:.0f} degrees."
                    ),
                    UserWarning,
                )

            if force_com_transpose is not None:
                # Transpose known

                _rotation_best_transpose = force_com_transpose

                if self._verbose:
                    warnings.warn(
                        f"Transpose of intensities forced to {force_com_transpose}.",
                        UserWarning,
                    )

            else:
                # Rotation known, transpose unknown
                com_measured_x = (
                    xp.cos(_rotation_best_rad) * _com_normalized_x
                    - xp.sin(_rotation_best_rad) * _com_normalized_y
                )
                com_measured_y = (
                    xp.sin(_rotation_best_rad) * _com_normalized_x
                    + xp.cos(_rotation_best_rad) * _com_normalized_y
                )
                if maximize_divergence:
                    com_grad_x_x = com_measured_x[2:, 1:-1] - com_measured_x[:-2, 1:-1]
                    com_grad_y_y = com_measured_y[1:-1, 2:] - com_measured_y[1:-1, :-2]
                    rotation_div = xp.mean(xp.abs(com_grad_x_x + com_grad_y_y))
                else:
                    com_grad_x_y = com_measured_x[1:-1, 2:] - com_measured_x[1:-1, :-2]
                    com_grad_y_x = com_measured_y[2:, 1:-1] - com_measured_y[:-2, 1:-1]
                    rotation_curl = xp.mean(xp.abs(com_grad_y_x - com_grad_x_y))

                com_measured_x = (
                    xp.cos(_rotation_best_rad) * _com_normalized_y
                    - xp.sin(_rotation_best_rad) * _com_normalized_x
                )
                com_measured_y = (
                    xp.sin(_rotation_best_rad) * _com_normalized_y
                    + xp.cos(_rotation_best_rad) * _com_normalized_x
                )
                if maximize_divergence:
                    com_grad_x_x = com_measured_x[2:, 1:-1] - com_measured_x[:-2, 1:-1]
                    com_grad_y_y = com_measured_y[1:-1, 2:] - com_measured_y[1:-1, :-2]
                    rotation_div_transpose = xp.mean(
                        xp.abs(com_grad_x_x + com_grad_y_y)
                    )
                else:
                    com_grad_x_y = com_measured_x[1:-1, 2:] - com_measured_x[1:-1, :-2]
                    com_grad_y_x = com_measured_y[2:, 1:-1] - com_measured_y[:-2, 1:-1]
                    rotation_curl_transpose = xp.mean(
                        xp.abs(com_grad_y_x - com_grad_x_y)
                    )

                if maximize_divergence:
                    _rotation_best_transpose = rotation_div_transpose > rotation_div
                else:
                    _rotation_best_transpose = rotation_curl_transpose < rotation_curl

                if self._verbose:
                    if _rotation_best_transpose:
                        print("Diffraction intensities should be transposed.")
                    else:
                        print("No need to transpose diffraction intensities.")

        else:
            # Rotation unknown
            if force_com_transpose is not None:
                # Transpose known, rotation unknown

                _rotation_best_transpose = force_com_transpose

                if self._verbose:
                    warnings.warn(
                        f"Transpose of intensities forced to {force_com_transpose}.",
                        UserWarning,
                    )

                rotation_angles_deg = xp.asarray(rotation_angles_deg, dtype=xp.float32)
                rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

                if _rotation_best_transpose:
                    com_measured_x = (
                        xp.cos(rotation_angles_rad) * _com_normalized_y[None]
                        - xp.sin(rotation_angles_rad) * _com_normalized_x[None]
                    )
                    com_measured_y = (
                        xp.sin(rotation_angles_rad) * _com_normalized_y[None]
                        + xp.cos(rotation_angles_rad) * _com_normalized_x[None]
                    )

                    rotation_angles_rad = asnumpy(xp.squeeze(rotation_angles_rad))
                    rotation_angles_deg = asnumpy(rotation_angles_deg)

                    if maximize_divergence:
                        com_grad_x_x = (
                            com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                        )
                        com_grad_y_y = (
                            com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                        )
                        rotation_div_transpose = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )

                        ind_trans_max = xp.argmax(rotation_div_transpose).item()
                        rotation_best_deg = rotation_angles_deg[ind_trans_max]
                        _rotation_best_rad = rotation_angles_rad[ind_trans_max]

                    else:
                        com_grad_x_y = (
                            com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        )
                        com_grad_y_x = (
                            com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                        )
                        rotation_curl_transpose = xp.mean(
                            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                        )

                        ind_trans_min = xp.argmin(rotation_curl_transpose).item()
                        rotation_best_deg = rotation_angles_deg[ind_trans_min]
                        _rotation_best_rad = rotation_angles_rad[ind_trans_min]

                else:
                    com_measured_x = (
                        xp.cos(rotation_angles_rad) * _com_normalized_x[None]
                        - xp.sin(rotation_angles_rad) * _com_normalized_y[None]
                    )
                    com_measured_y = (
                        xp.sin(rotation_angles_rad) * _com_normalized_x[None]
                        + xp.cos(rotation_angles_rad) * _com_normalized_y[None]
                    )

                    rotation_angles_rad = asnumpy(xp.squeeze(rotation_angles_rad))
                    rotation_angles_deg = asnumpy(rotation_angles_deg)

                    if maximize_divergence:
                        com_grad_x_x = (
                            com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                        )
                        com_grad_y_y = (
                            com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                        )
                        rotation_div = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )

                        ind_max = xp.argmax(rotation_div).item()
                        rotation_best_deg = rotation_angles_deg[ind_max]
                        _rotation_best_rad = rotation_angles_rad[ind_max]

                    else:
                        com_grad_x_y = (
                            com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        )
                        com_grad_y_x = (
                            com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                        )
                        rotation_curl = xp.mean(
                            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                        )

                        ind_min = xp.argmin(rotation_curl).item()
                        rotation_best_deg = rotation_angles_deg[ind_min]
                        _rotation_best_rad = rotation_angles_rad[ind_min]

                if self._verbose:
                    print(("Best fit rotation = " f"{rotation_best_deg:.0f} degrees."))

                if plot_rotation:
                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    if _rotation_best_transpose:
                        ax.plot(
                            rotation_angles_deg,
                            asnumpy(rotation_div_transpose)
                            if maximize_divergence
                            else asnumpy(rotation_curl_transpose),
                            label="CoM after transpose",
                        )
                    else:
                        ax.plot(
                            rotation_angles_deg,
                            asnumpy(rotation_div)
                            if maximize_divergence
                            else asnumpy(rotation_curl),
                            label="CoM",
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
                        aspect_ratio = (
                            np.ptp(rotation_div_transpose)
                            if _rotation_best_transpose
                            else np.ptp(rotation_div)
                        )
                        ax.set_ylabel("Mean Absolute Divergence")
                        ax.set_aspect(np.ptp(rotation_angles_deg) / aspect_ratio / 4)
                    else:
                        aspect_ratio = (
                            np.ptp(rotation_curl_transpose)
                            if _rotation_best_transpose
                            else np.ptp(rotation_curl)
                        )
                        ax.set_ylabel("Mean Absolute Curl")
                        ax.set_aspect(np.ptp(rotation_angles_deg) / aspect_ratio / 4)
                    fig.tight_layout()

            else:
                # Transpose unknown, rotation unknown
                rotation_angles_deg = xp.asarray(rotation_angles_deg, dtype=xp.float32)
                rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

                # Untransposed
                com_measured_x = (
                    xp.cos(rotation_angles_rad) * _com_normalized_x[None]
                    - xp.sin(rotation_angles_rad) * _com_normalized_y[None]
                )
                com_measured_y = (
                    xp.sin(rotation_angles_rad) * _com_normalized_x[None]
                    + xp.cos(rotation_angles_rad) * _com_normalized_y[None]
                )

                if maximize_divergence:
                    com_grad_x_x = (
                        com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                    )
                    com_grad_y_y = (
                        com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                    )
                    rotation_div = xp.mean(
                        xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                    )
                else:
                    com_grad_x_y = (
                        com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                    )
                    com_grad_y_x = (
                        com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                    )
                    rotation_curl = xp.mean(
                        xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                    )

                # Transposed
                com_measured_x = (
                    xp.cos(rotation_angles_rad) * _com_normalized_y[None]
                    - xp.sin(rotation_angles_rad) * _com_normalized_x[None]
                )
                com_measured_y = (
                    xp.sin(rotation_angles_rad) * _com_normalized_y[None]
                    + xp.cos(rotation_angles_rad) * _com_normalized_x[None]
                )

                if maximize_divergence:
                    com_grad_x_x = (
                        com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                    )
                    com_grad_y_y = (
                        com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                    )
                    rotation_div_transpose = xp.mean(
                        xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                    )
                else:
                    com_grad_x_y = (
                        com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                    )
                    com_grad_y_x = (
                        com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                    )
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
                        _rotation_best_rad = rotation_angles_rad[ind_max]
                        _rotation_best_transpose = False
                    else:
                        rotation_best_deg = rotation_angles_deg[ind_trans_max]
                        _rotation_best_rad = rotation_angles_rad[ind_trans_max]
                        _rotation_best_transpose = True

                    self._rotation_div = rotation_div
                    self._rotation_div_transpose = rotation_div_transpose
                else:
                    # Minimize Curl
                    ind_min = xp.argmin(rotation_curl).item()
                    ind_trans_min = xp.argmin(rotation_curl_transpose).item()
                    self._rotation_curl = rotation_curl
                    self._rotation_curl_transpose = rotation_curl_transpose
                    if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
                        rotation_best_deg = rotation_angles_deg[ind_min]
                        _rotation_best_rad = rotation_angles_rad[ind_min]
                        _rotation_best_transpose = False
                    else:
                        rotation_best_deg = rotation_angles_deg[ind_trans_min]
                        _rotation_best_rad = rotation_angles_rad[ind_trans_min]
                        _rotation_best_transpose = True

                self._rotation_angles_deg = rotation_angles_deg
                # Print summary
                if self._verbose:
                    print(("Best fit rotation = " f"{rotation_best_deg:.0f} degrees."))
                    if _rotation_best_transpose:
                        print("Diffraction intensities should be transposed.")
                    else:
                        print("No need to transpose diffraction intensities.")

                # Plot Curl/Div rotation
                if plot_rotation:
                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    ax.plot(
                        rotation_angles_deg,
                        asnumpy(rotation_div)
                        if maximize_divergence
                        else asnumpy(rotation_curl),
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
        if _rotation_best_transpose:
            _com_x = (
                xp.cos(_rotation_best_rad) * _com_normalized_y
                - xp.sin(_rotation_best_rad) * _com_normalized_x
            )
            _com_y = (
                xp.sin(_rotation_best_rad) * _com_normalized_y
                + xp.cos(_rotation_best_rad) * _com_normalized_x
            )
        else:
            _com_x = (
                xp.cos(_rotation_best_rad) * _com_normalized_x
                - xp.sin(_rotation_best_rad) * _com_normalized_y
            )
            _com_y = (
                xp.sin(_rotation_best_rad) * _com_normalized_x
                + xp.cos(_rotation_best_rad) * _com_normalized_y
            )

        # 'Public'-facing attributes as numpy arrays
        com_x = asnumpy(_com_x)
        com_y = asnumpy(_com_y)

        # Optionally, plot CoM
        if plot_center_of_mass == "all":
            figsize = kwargs.pop("figsize", (8, 12))
            cmap = kwargs.pop("cmap", "RdBu_r")
            extent = [
                0,
                self._scan_sampling[1] * _com_measured_x.shape[1],
                self._scan_sampling[0] * _com_measured_x.shape[0],
                0,
            ]

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(3, 2), axes_pad=(0.25, 0.5))

            for ax, arr, title in zip(
                grid,
                [
                    _com_measured_x,
                    _com_measured_y,
                    _com_normalized_x,
                    _com_normalized_y,
                    com_x,
                    com_y,
                ],
                [
                    "CoM_x",
                    "CoM_y",
                    "Normalized CoM_x",
                    "Normalized CoM_y",
                    "Corrected CoM_x",
                    "Corrected CoM_y",
                ],
            ):
                ax.imshow(asnumpy(arr), extent=extent, cmap=cmap, **kwargs)
                ax.set_ylabel(f"x [{self._scan_units[0]}]")
                ax.set_xlabel(f"y [{self._scan_units[1]}]")
                ax.set_title(title)

        elif plot_center_of_mass == "default":
            figsize = kwargs.pop("figsize", (8, 4))
            cmap = kwargs.pop("cmap", "RdBu_r")

            extent = [
                0,
                self._scan_sampling[1] * com_x.shape[1],
                self._scan_sampling[0] * com_x.shape[0],
                0,
            ]

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=(0.25, 0.5))

            for ax, arr, title in zip(
                grid,
                [
                    com_x,
                    com_y,
                ],
                [
                    "Corrected CoM_x",
                    "Corrected CoM_y",
                ],
            ):
                ax.imshow(arr, extent=extent, cmap=cmap, **kwargs)
                ax.set_ylabel(f"x [{self._scan_units[0]}]")
                ax.set_xlabel(f"y [{self._scan_units[1]}]")
                ax.set_title(title)

        return (
            _rotation_best_rad,
            _rotation_best_transpose,
            _com_x,
            _com_y,
            com_x,
            com_y,
        )

    def _normalize_diffraction_intensities(
        self,
        diffraction_intensities,
        com_fitted_x,
        com_fitted_y,
        crop_patterns,
    ):
        """
        Fix diffraction intensities CoM, shift to origin, and take square root

        Parameters
        ----------
        diffraction_intensities: (Rx,Ry,Sx,Sy) np.ndarray
            Zero-padded diffraction intensities
        com_fitted_x: (Rx,Ry) xp.ndarray
            Best fit horizontal center of mass gradient
        com_fitted_y: (Rx,Ry) xp.ndarray
            Best fit vertical center of mass gradient
        crop_patterns: bool
            if True, crop patterns to avoid wrap around of patterns
            when centering

        Returns
        -------
        amplitudes: (Rx * Ry, Sx, Sy) np.ndarray
            Flat array of normalized diffraction amplitudes
        mean_intensity: float
            Mean intensity value
        """

        xp = self._xp
        mean_intensity = 0

        diffraction_intensities = self._asnumpy(diffraction_intensities)
        if crop_patterns:
            crop_x = int(
                np.minimum(
                    diffraction_intensities.shape[2] - com_fitted_x.max(),
                    com_fitted_x.min(),
                )
            )
            crop_y = int(
                np.minimum(
                    diffraction_intensities.shape[3] - com_fitted_y.max(),
                    com_fitted_y.min(),
                )
            )

            crop_w = np.minimum(crop_y, crop_x)
            region_of_interest_shape = (crop_w * 2, crop_w * 2)
            amplitudes = np.zeros(
                (
                    diffraction_intensities.shape[0],
                    diffraction_intensities.shape[1],
                    crop_w * 2,
                    crop_w * 2,
                ),
                dtype=np.float32,
            )

            crop_mask = np.zeros(diffraction_intensities.shape[-2:], dtype=np.bool_)
            crop_mask[:crop_w, :crop_w] = True
            crop_mask[-crop_w:, :crop_w] = True
            crop_mask[:crop_w:, -crop_w:] = True
            crop_mask[-crop_w:, -crop_w:] = True
            self._crop_mask = crop_mask

        else:
            region_of_interest_shape = diffraction_intensities.shape[-2:]
            amplitudes = np.zeros(diffraction_intensities.shape, dtype=np.float32)

        com_fitted_x = self._asnumpy(com_fitted_x)
        com_fitted_y = self._asnumpy(com_fitted_y)

        for rx in range(diffraction_intensities.shape[0]):
            for ry in range(diffraction_intensities.shape[1]):
                intensities = get_shifted_ar(
                    diffraction_intensities[rx, ry],
                    -com_fitted_x[rx, ry],
                    -com_fitted_y[rx, ry],
                    bilinear=True,
                    device="cpu",
                )

                if crop_patterns:
                    intensities = intensities[crop_mask].reshape(
                        region_of_interest_shape
                    )

                mean_intensity += np.sum(intensities)
                amplitudes[rx, ry] = np.sqrt(np.maximum(intensities, 0))

        amplitudes = xp.reshape(amplitudes, (-1,) + region_of_interest_shape)
        amplitudes = xp.asarray(amplitudes)
        mean_intensity /= amplitudes.shape[0]

        return amplitudes, mean_intensity

    def show_complex_CoM(
        self,
        com=None,
        cbar=True,
        scalebar=True,
        pixelsize=None,
        pixelunits=None,
        **kwargs,
    ):
        """
        Plot complex-valued CoM image

        Parameters
        ----------

        com = (CoM_x, CoM_y) tuple
              If None is specified, uses (self.com_x, self.com_y) instead
        cbar: bool, optional
            if True, adds colorbar
        scalebar: bool, optional
            if True, adds scalebar to probe
        pixelunits: str, optional
            units for scalebar, default is A
        pixelsize: float, optional
            default is scan sampling
        """

        if com is None:
            com = (self.com_x, self.com_y)

        if pixelsize is None:
            pixelsize = self._scan_sampling[0]
        if pixelunits is None:
            pixelunits = r"$\AA$"

        figsize = kwargs.pop("figsize", (6, 6))
        fig, ax = plt.subplots(figsize=figsize)

        complex_com = com[0] + 1j * com[1]

        show_complex(
            complex_com,
            cbar=cbar,
            figax=(fig, ax),
            scalebar=scalebar,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            ticks=False,
            **kwargs,
        )


class PtychographicReconstruction(PhaseReconstruction, PtychographicConstraints):
    """
    Base ptychographic reconstruction class.
    Inherits from PhaseReconstruction and PtychographicConstraints.
    Defines various common functions and properties for subclasses to inherit.
    """

    def to_h5(self, group):
        """
        Wraps datasets and metadata to write in emdfile classes,
        notably: the object and probe arrays.
        """

        asnumpy = self._asnumpy

        # instantiation metadata
        tf = AffineTransform(angle=-self._rotation_best_rad)
        pos = self.positions

        if pos.ndim == 2:
            origin = np.mean(pos, axis=0)
        else:
            origin = np.mean(pos, axis=(0, 1))
        scan_positions = tf(pos, origin)

        vacuum_probe_intensity = (
            asnumpy(self._vacuum_probe_intensity)
            if self._vacuum_probe_intensity is not None
            else None
        )
        metadata = {
            "energy": self._energy,
            "semiangle_cutoff": self._semiangle_cutoff,
            "rolloff": self._rolloff,
            "object_padding_px": self._object_padding_px,
            "object_type": self._object_type,
            "verbose": self._verbose,
            "device": self._device,
            "name": self.name,
            "vacuum_probe_intensity": vacuum_probe_intensity,
            "positions": scan_positions,
        }

        cls = self.__class__
        class_specific_metadata = {}
        for key in cls._class_specific_metadata:
            class_specific_metadata[key[1:]] = getattr(self, key, None)

        metadata |= class_specific_metadata

        self.metadata = Metadata(
            name="instantiation_metadata",
            data=metadata,
        )

        # preprocessing metadata
        self.metadata = Metadata(
            name="preprocess_metadata",
            data={
                "rotation_angle_rad": self._rotation_best_rad,
                "data_transpose": self._rotation_best_transpose,
                "positions_px": asnumpy(self._positions_px),
                "region_of_interest_shape": self._region_of_interest_shape,
                "num_diffraction_patterns": self._num_diffraction_patterns,
                "sampling": self.sampling,
                "angular_sampling": self.angular_sampling,
            },
        )

        # reconstruction metadata
        is_stack = self._save_iterations and hasattr(self, "object_iterations")
        if is_stack:
            num_iterations = len(self.object_iterations)
            iterations = list(range(0, num_iterations, self._save_iterations_frequency))
            if num_iterations - 1 not in iterations:
                iterations.append(num_iterations - 1)

            error = [self.error_iterations[i] for i in iterations]
        else:
            error = getattr(self, "error", 0.0)

        self.metadata = Metadata(
            name="reconstruction_metadata",
            data={
                "reconstruction_error": error,
            },
        )

        # aberrations metadata
        self.metadata = Metadata(
            name="aberrations_metadata",
            data=self._polar_parameters,
        )

        # object
        self._object_emd = Array(
            name="reconstruction_object",
            data=asnumpy(self._xp.asarray(self._object)),
        )

        # probe
        self._probe_emd = Array(name="reconstruction_probe", data=asnumpy(self._probe))

        if is_stack:
            iterations_labels = [f"iteration_{i:03}" for i in iterations]

            # object
            object_iterations = [
                np.asarray(self.object_iterations[i]) for i in iterations
            ]
            self._object_iterations_emd = Array(
                name="reconstruction_object_iterations",
                data=np.stack(object_iterations, axis=0),
                slicelabels=iterations_labels,
            )

            # probe
            probe_iterations = [self.probe_iterations[i] for i in iterations]
            self._probe_iterations_emd = Array(
                name="reconstruction_probe_iterations",
                data=np.stack(probe_iterations, axis=0),
                slicelabels=iterations_labels,
            )

        # exit_waves
        if self._save_exit_waves:
            self._exit_waves_emd = Array(
                name="reconstruction_exit_waves",
                data=asnumpy(self._xp.asarray(self._exit_waves)),
            )

        # datacube
        if self._save_datacube:
            self.metadata = self._datacube.calibration
            Custom.to_h5(self, group)
        else:
            dc = self._datacube
            self._datacube = None
            Custom.to_h5(self, group)
            self._datacube = dc

    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of arguments/values to pass
        to the class' __init__ function
        """
        # Get data
        dict_data = cls._get_emd_attr_data(cls, group)

        # Get metadata dictionaries
        instance_md = _read_metadata(group, "instantiation_metadata")
        polar_params = _read_metadata(group, "aberrations_metadata")._params

        # Fix calibrations bug
        if "_datacube" in dict_data:
            calibrations_dict = _read_metadata(group, "calibration")._params
            cal = Calibration()
            cal._params.update(calibrations_dict)
            dc = dict_data["_datacube"]
            dc.calibration = cal
        else:
            dc = None

        obj = dict_data["_object_emd"].data
        probe = dict_data["_probe_emd"].data

        # Populate args and return
        kwargs = {
            "datacube": dc,
            "initial_object_guess": np.asarray(obj),
            "initial_probe_guess": np.asarray(probe),
            "vacuum_probe_intensity": instance_md["vacuum_probe_intensity"],
            "initial_scan_positions": instance_md["positions"],
            "energy": instance_md["energy"],
            "object_padding_px": instance_md["object_padding_px"],
            "object_type": instance_md["object_type"],
            "semiangle_cutoff": instance_md["semiangle_cutoff"],
            "rolloff": instance_md["rolloff"],
            "name": instance_md["name"],
            "polar_parameters": polar_params,
            "verbose": True,  # for compatibility
            "device": "cpu",  # for compatibility
        }

        class_specific_kwargs = {}
        for key in cls._class_specific_metadata:
            class_specific_kwargs[key[1:]] = instance_md[key[1:]]

        kwargs |= class_specific_kwargs

        return kwargs

    def _populate_instance(self, group):
        """
        Sets post-initialization properties, notably some preprocessing meta
        """
        xp = self._xp
        asnumpy = self._asnumpy

        # Preprocess metadata
        preprocess_md = _read_metadata(group, "preprocess_metadata")
        self._rotation_best_rad = preprocess_md["rotation_angle_rad"]
        self._rotation_best_transpose = preprocess_md["data_transpose"]
        self._positions_px = xp.asarray(preprocess_md["positions_px"])
        self._angular_sampling = preprocess_md["angular_sampling"]
        self._region_of_interest_shape = preprocess_md["region_of_interest_shape"]
        self._num_diffraction_patterns = preprocess_md["num_diffraction_patterns"]

        # Reconstruction metadata
        reconstruction_md = _read_metadata(group, "reconstruction_metadata")
        error = reconstruction_md["reconstruction_error"]

        # Data
        dict_data = Custom._get_emd_attr_data(Custom, group)
        if "_exit_waves_emd" in dict_data:
            self._exit_waves = dict_data["_exit_waves_emd"].data
            self._exit_waves = xp.asarray(self._exit_waves, dtype=xp.complex64)
        else:
            self._exit_waves = None

        # Check if stack
        if hasattr(error, "__len__"):
            self.object_iterations = list(dict_data["_object_iterations_emd"].data)
            self.probe_iterations = list(dict_data["_probe_iterations_emd"].data)
            self.error_iterations = error
            self.error = error[-1]
        else:
            self.error = error

        # Slim preprocessing to enable visualize
        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self.object = asnumpy(self._object)
        self.probe = self.probe_centered
        self._preprocessed = True

    def _set_polar_parameters(self, parameters: dict):
        """
        Set the probe aberrations dictionary.

        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._polar_parameters.keys():
                self._polar_parameters[symbol] = value

            elif symbol == "defocus":
                self._polar_parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._polar_parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

    def _calculate_scan_positions_in_pixels(self, positions: np.ndarray):
        """
        Method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input probe positions in Å.
            If None, a raster scan using experimental parameters is constructed.

        Returns
        -------
        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        """

        grid_scan_shape = self._grid_scan_shape
        rotation_angle = self._rotation_best_rad
        step_sizes = self._scan_sampling

        if positions is None:
            if grid_scan_shape is not None:
                nx, ny = grid_scan_shape

                if step_sizes is not None:
                    sx, sy = step_sizes
                    x = np.arange(nx) * sx
                    y = np.arange(ny) * sy
                else:
                    raise ValueError()
            else:
                raise ValueError()

            if self._rotation_best_transpose:
                x = (x - np.ptp(x) / 2) / self.sampling[1]
                y = (y - np.ptp(y) / 2) / self.sampling[0]
            else:
                x = (x - np.ptp(x) / 2) / self.sampling[0]
                y = (y - np.ptp(y) / 2) / self.sampling[1]
            x, y = np.meshgrid(x, y, indexing="ij")

        else:
            positions -= np.mean(positions, axis=0)
            x = positions[:, 0] / self.sampling[1]
            y = positions[:, 1] / self.sampling[0]

        if rotation_angle is not None:
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

        if self._rotation_best_transpose:
            positions = np.array([y.ravel(), x.ravel()]).T
        else:
            positions = np.array([x.ravel(), y.ravel()]).T
        positions -= np.min(positions, axis=0)

        if self._object_padding_px is None:
            float_padding = self._region_of_interest_shape / 2
            self._object_padding_px = (float_padding, float_padding)
        elif np.isscalar(self._object_padding_px[0]):
            self._object_padding_px = (
                (self._object_padding_px[0],) * 2,
                (self._object_padding_px[1],) * 2,
            )

        positions[:, 0] += self._object_padding_px[0][0]
        positions[:, 1] += self._object_padding_px[1][0]

        return positions

    def _sum_overlapping_patches_bincounts_base(self, patches: np.ndarray):
        """
        Base bincouts overlapping patches sum function, operating on real-valued arrays.
        Note this assumes the probe is corner-centered.

        Parameters
        ----------
        patches: (Rx*Ry,Sx,Sy) np.ndarray
            Patches to sum

        Returns
        -------
        out_array: (Px,Py) np.ndarray
            Summed array
        """
        xp = self._xp
        x0 = xp.round(self._positions_px[:, 0]).astype("int")
        y0 = xp.round(self._positions_px[:, 1]).astype("int")

        roi_shape = self._region_of_interest_shape
        x_ind = xp.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
        y_ind = xp.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

        flat_weights = patches.ravel()
        indices = (
            (y0[:, None, None] + y_ind[None, None, :]) % self._object_shape[1]
        ) + (
            (x0[:, None, None] + x_ind[None, :, None]) % self._object_shape[0]
        ) * self._object_shape[
            1
        ]
        counts = xp.bincount(
            indices.ravel(), weights=flat_weights, minlength=np.prod(self._object_shape)
        )
        return xp.reshape(counts, self._object_shape)

    def _sum_overlapping_patches_bincounts(self, patches: np.ndarray):
        """
        Sum overlapping patches defined into object shaped array using bincounts.
        Calls _sum_overlapping_patches_bincounts_base on Real and Imaginary parts.

        Parameters
        ----------
        patches: (Rx*Ry,Sx,Sy) np.ndarray
            Patches to sum

        Returns
        -------
        out_array: (Px,Py) np.ndarray
            Summed array
        """

        xp = self._xp
        if xp.iscomplexobj(patches):
            real = self._sum_overlapping_patches_bincounts_base(xp.real(patches))
            imag = self._sum_overlapping_patches_bincounts_base(xp.imag(patches))
            return real + 1.0j * imag
        else:
            return self._sum_overlapping_patches_bincounts_base(patches)

    def _extract_vectorized_patch_indices(self):
        """
        Sets the vectorized row/col indices used for the overlap projection
        Note this assumes the probe is corner-centered.

        Returns
        -------
        self._vectorized_patch_indices_row: np.ndarray
            Row indices for probe patches inside object array
        self._vectorized_patch_indices_col: np.ndarray
            Column indices for probe patches inside object array
        """
        xp = self._xp
        x0 = xp.round(self._positions_px[:, 0]).astype("int")
        y0 = xp.round(self._positions_px[:, 1]).astype("int")

        roi_shape = self._region_of_interest_shape
        x_ind = xp.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
        y_ind = xp.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

        obj_shape = self._object_shape
        vectorized_patch_indices_row = (
            x0[:, None, None] + x_ind[None, :, None]
        ) % obj_shape[0]
        vectorized_patch_indices_col = (
            y0[:, None, None] + y_ind[None, None, :]
        ) % obj_shape[1]

        return vectorized_patch_indices_row, vectorized_patch_indices_col

    def _crop_rotate_object_fov(
        self,
        array,
        padding=0,
    ):
        """
        Crops and rotated object to FOV bounded by current pixel positions.

        Parameters
        ----------
        array: np.ndarray
            Object array to crop and rotate. Only operates on numpy arrays for comptatibility.
        padding: int, optional
            Optional padding outside pixel positions

        Returns
        cropped_rotated_array: np.ndarray
            Cropped and rotated object array
        """

        asnumpy = self._asnumpy
        angle = (
            self._rotation_best_rad
            if self._rotation_best_transpose
            else -self._rotation_best_rad
        )

        tf = AffineTransform(angle=angle)
        rotated_points = tf(
            asnumpy(self._positions_px), origin=asnumpy(self._positions_px_com), xp=np
        )

        min_x, min_y = np.floor(np.amin(rotated_points, axis=0) - padding).astype("int")
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x, max_y = np.ceil(np.amax(rotated_points, axis=0) + padding).astype("int")

        rotated_array = rotate(
            asnumpy(array), np.rad2deg(-angle), reshape=False, axes=(-2, -1)
        )[..., min_x:max_x, min_y:max_y]

        if self._rotation_best_transpose:
            rotated_array = rotated_array.swapaxes(-2, -1)

        return rotated_array

    def tune_angle_and_defocus(
        self,
        angle_guess=None,
        defocus_guess=None,
        transpose=None,
        angle_step_size=1,
        defocus_step_size=20,
        num_angle_values=5,
        num_defocus_values=5,
        max_iter=5,
        plot_reconstructions=True,
        plot_convergence=True,
        return_values=False,
        **kwargs,
    ):
        """
        Run reconstructions over a parameters space of angles and
        defocus values. Should be run after preprocess step.

        Parameters
        ----------
        angle_guess: float (degrees), optional
            initial starting guess for rotation angle between real and reciprocal space
            if None, uses current initialized values
        defocus_guess: float (A), optional
            initial starting guess for defocus
            if None, uses current initialized values
        angle_step_size: float (degrees), optional
            size of change of rotation angle between real and reciprocal space for
            each step in parameter space
        defocus_step_size: float (A), optional
            size of change of defocus for each step in parameter space
        num_angle_values: int, optional
            number of values of angle to test, must be >= 1.
        num_defocus_values: int,optional
            number of values of defocus to test, must be >= 1
        max_iter: int, optional
            number of iterations to run in ptychographic reconstruction
        plot_reconstructions: bool, optional
            if True, plot phase of reconstructed objects
        plot_convergence: bool, optional
            if True, plots error for each iteration for each reconstruction.
        return_values: bool, optional
            if True, returns objects, convergence

        Returns
        -------
        objects: list
            reconstructed objects
        convergence: np.ndarray
            array of convergence values from reconstructions
        """
        # calculate angles and defocus values to test
        if angle_guess is None:
            angle_guess = self._rotation_best_rad * 180 / np.pi
        if defocus_guess is None:
            defocus_guess = -self._polar_parameters["C10"]
        if transpose is None:
            transpose = self._rotation_best_transpose

        if num_angle_values == 1:
            angle_step_size = 0

        if num_defocus_values == 1:
            defocus_step_size = 0

        angles = np.linspace(
            angle_guess - angle_step_size * (num_angle_values - 1) / 2,
            angle_guess + angle_step_size * (num_angle_values - 1) / 2,
            num_angle_values,
        )

        defocus_values = np.linspace(
            defocus_guess - defocus_step_size * (num_defocus_values - 1) / 2,
            defocus_guess + defocus_step_size * (num_defocus_values - 1) / 2,
            num_defocus_values,
        )

        if return_values:
            convergence = []
            objects = []

        # current initialized values
        current_verbose = self._verbose
        current_defocus = -self._polar_parameters["C10"]
        current_rotation_deg = self._rotation_best_rad * 180 / np.pi
        current_transpose = self._rotation_best_transpose

        # Gridspec to plot on
        if plot_reconstructions:
            if plot_convergence:
                spec = GridSpec(
                    ncols=num_defocus_values,
                    nrows=num_angle_values * 2,
                    height_ratios=[1, 1 / 4] * num_angle_values,
                    hspace=0.15,
                    wspace=0.35,
                )
                figsize = kwargs.get(
                    "figsize", (4 * num_defocus_values, 5 * num_angle_values)
                )
            else:
                spec = GridSpec(
                    ncols=num_defocus_values,
                    nrows=num_angle_values,
                    hspace=0.15,
                    wspace=0.35,
                )
                figsize = kwargs.get(
                    "figsize", (4 * num_defocus_values, 4 * num_angle_values)
                )

            fig = plt.figure(figsize=figsize)

        progress_bar = kwargs.pop("progress_bar", False)
        # run loop and plot along the way
        self._verbose = False
        for flat_index, (angle, defocus) in enumerate(
            tqdmnd(angles, defocus_values, desc="Tuning angle and defocus")
        ):
            self._polar_parameters["C10"] = -defocus
            self._probe = None
            self._object = None
            self.preprocess(
                force_com_rotation=angle,
                force_com_transpose=transpose,
                plot_center_of_mass=False,
                plot_rotation=False,
                plot_probe_overlaps=False,
            )

            self.reconstruct(
                reset=True,
                store_iterations=True,
                max_iter=max_iter,
                progress_bar=progress_bar,
                **kwargs,
            )

            if plot_reconstructions:
                row_index, col_index = np.unravel_index(
                    flat_index, (num_angle_values, num_defocus_values)
                )

                if plot_convergence:
                    object_ax = fig.add_subplot(spec[row_index * 2, col_index])
                    convergence_ax = fig.add_subplot(spec[row_index * 2 + 1, col_index])
                    self._visualize_last_iteration_figax(
                        fig,
                        object_ax=object_ax,
                        convergence_ax=convergence_ax,
                        cbar=True,
                    )
                    convergence_ax.yaxis.tick_right()
                else:
                    object_ax = fig.add_subplot(spec[row_index, col_index])
                    self._visualize_last_iteration_figax(
                        fig,
                        object_ax=object_ax,
                        convergence_ax=None,
                        cbar=True,
                    )

                object_ax.set_title(
                    f" angle = {angle:.1f} °, defocus = {defocus:.1f} A \n error = {self.error:.3e}"
                )
                object_ax.set_xticks([])
                object_ax.set_yticks([])

            if return_values:
                objects.append(self.object)
                convergence.append(self.error_iterations.copy())

        # initialize back to pre-tuning values
        self._polar_parameters["C10"] = -current_defocus
        self._probe = None
        self._object = None
        self.preprocess(
            force_com_rotation=current_rotation_deg,
            force_com_transpose=current_transpose,
            plot_center_of_mass=False,
            plot_rotation=False,
            plot_probe_overlaps=False,
        )
        self._verbose = current_verbose

        if plot_reconstructions:
            spec.tight_layout(fig)

        if return_values:
            return objects, convergence

    def _position_correction(
        self,
        relevant_object,
        relevant_probes,
        relevant_overlap,
        relevant_amplitudes,
        current_positions,
        positions_step_size,
        constrain_position_distance,
    ):
        """
        Position correction using estimated intensity gradient.

        Parameters
        --------
        relevant_object: np.ndarray
            Current object estimate
        relevant_probes:np.ndarray
            fractionally-shifted probes
        relevant_overlap: np.ndarray
            object * probe overlap
        relevant_amplitudes: np.ndarray
            Measured amplitudes
        current_positions: np.ndarray
            Current positions estimate
        positions_step_size: float
            Positions step size
        constrain_position_distance: float
            Distance to constrain position correction within original
            field of view in A

        Returns
        --------
        updated_positions: np.ndarray
            Updated positions estimate
        """

        xp = self._xp

        if self._object_type == "potential":
            complex_object = xp.exp(1j * relevant_object)
        else:
            complex_object = relevant_object

        obj_rolled_x_patches = complex_object[
            (self._vectorized_patch_indices_row + 1) % self._object_shape[0],
            self._vectorized_patch_indices_col,
        ]
        obj_rolled_y_patches = complex_object[
            self._vectorized_patch_indices_row,
            (self._vectorized_patch_indices_col + 1) % self._object_shape[1],
        ]

        overlap_fft = xp.fft.fft2(relevant_overlap)

        exit_waves_dx_fft = overlap_fft - xp.fft.fft2(
            obj_rolled_x_patches * relevant_probes
        )
        exit_waves_dy_fft = overlap_fft - xp.fft.fft2(
            obj_rolled_y_patches * relevant_probes
        )

        overlap_fft_conj = xp.conj(overlap_fft)
        estimated_intensity = xp.abs(overlap_fft) ** 2
        measured_intensity = relevant_amplitudes**2

        flat_shape = (relevant_overlap.shape[0], -1)
        difference_intensity = (measured_intensity - estimated_intensity).reshape(
            flat_shape
        )

        partial_intensity_dx = 2 * xp.real(
            exit_waves_dx_fft * overlap_fft_conj
        ).reshape(flat_shape)
        partial_intensity_dy = 2 * xp.real(
            exit_waves_dy_fft * overlap_fft_conj
        ).reshape(flat_shape)

        coefficients_matrix = xp.dstack((partial_intensity_dx, partial_intensity_dy))

        # positions_update = xp.einsum(
        #    "idk,ik->id", xp.linalg.pinv(coefficients_matrix), difference_intensity
        # )

        coefficients_matrix_T = coefficients_matrix.conj().swapaxes(-1, -2)
        positions_update = (
            xp.linalg.inv(coefficients_matrix_T @ coefficients_matrix)
            @ coefficients_matrix_T
            @ difference_intensity[..., None]
        )

        if constrain_position_distance is not None:
            constrain_position_distance /= xp.sqrt(
                self.sampling[0] ** 2 + self.sampling[1] ** 2
            )
            x1 = (current_positions - positions_step_size * positions_update[..., 0])[
                :, 0
            ]
            y1 = (current_positions - positions_step_size * positions_update[..., 0])[
                :, 1
            ]
            x0 = self._positions_px_initial[:, 0]
            y0 = self._positions_px_initial[:, 1]
            if self._rotation_best_transpose:
                x0, y0 = xp.array([y0, x0])
                x1, y1 = xp.array([y1, x1])

            if self._rotation_best_rad is not None:
                rotation_angle = self._rotation_best_rad
                x0, y0 = x0 * xp.cos(-rotation_angle) + y0 * xp.sin(
                    -rotation_angle
                ), -x0 * xp.sin(-rotation_angle) + y0 * xp.cos(-rotation_angle)
                x1, y1 = x1 * xp.cos(-rotation_angle) + y1 * xp.sin(
                    -rotation_angle
                ), -x1 * xp.sin(-rotation_angle) + y1 * xp.cos(-rotation_angle)

            outlier_ind = (x1 > (xp.max(x0) + constrain_position_distance)) + (
                x1 < (xp.min(x0) - constrain_position_distance)
            ) + (y1 > (xp.max(y0) + constrain_position_distance)) + (
                y1 < (xp.min(y0) - constrain_position_distance)
            ) > 0

            positions_update[..., 0][outlier_ind] = 0

        current_positions -= positions_step_size * positions_update[..., 0]
        return current_positions

    def plot_position_correction(
        self,
        scale_arrows=1,
        plot_arrow_freq=1,
        verbose=True,
        **kwargs,
    ):
        """
        Function to plot changes to probe positions during ptychography reconstruciton

        Parameters
        ----------
        scale_arrows: float, optional
            scaling factor to be applied on vectors prior to plt.quiver call
        verbose: bool, optional
            if True, prints AffineTransformation if positions have been updated
        """
        if verbose:
            if hasattr(self, "_tf"):
                print(self._tf)

        asnumpy = self._asnumpy

        extent = [
            0,
            self.sampling[1] * self._object_shape[1],
            self.sampling[0] * self._object_shape[0],
            0,
        ]

        initial_pos = asnumpy(self._positions_initial)
        pos = self.positions

        figsize = kwargs.pop("figsize", (6, 6))
        color = kwargs.pop("color", (1, 0, 0, 1))

        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(
            initial_pos[::plot_arrow_freq, 1],
            initial_pos[::plot_arrow_freq, 0],
            (pos[::plot_arrow_freq, 1] - initial_pos[::plot_arrow_freq, 1])
            * scale_arrows,
            (pos[::plot_arrow_freq, 0] - initial_pos[::plot_arrow_freq, 0])
            * scale_arrows,
            scale_units="xy",
            scale=1,
            color=color,
            **kwargs,
        )

        ax.set_ylabel("x [A]")
        ax.set_xlabel("y [A]")
        ax.set_xlim((extent[0], extent[1]))
        ax.set_ylim((extent[2], extent[3]))
        ax.set_aspect("equal")
        ax.set_title("Probe positions correction")

    def _return_fourier_probe(
        self,
        probe=None,
    ):
        """
        Returns complex fourier probe shifted to center of array from
        corner-centered complex real space probe

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe

        Returns
        -------
        fourier_probe: np.ndarray
            Fourier-transformed and center-shifted probe.
        """
        xp = self._xp

        if probe is None:
            probe = self._probe
        else:
            probe = xp.asarray(probe, dtype=xp.complex64)

        return xp.fft.fftshift(xp.fft.fft2(probe), axes=(-2, -1))

    def _return_fourier_probe_from_centered_probe(
        self,
        probe=None,
    ):
        """
        Returns complex fourier probe shifted to center of array from
        centered complex real space probe

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe

        Returns
        -------
        fourier_probe: np.ndarray
            Fourier-transformed and center-shifted probe.
        """
        xp = self._xp
        return self._return_fourier_probe(xp.fft.ifftshift(probe, axes=(-2, -1)))

    def _return_centered_probe(
        self,
        probe=None,
    ):
        """
        Returns complex probe centered in middle of the array.

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe

        Returns
        -------
        centered_probe: np.ndarray
            Center-shifted probe.
        """
        xp = self._xp

        if probe is None:
            probe = self._probe
        else:
            probe = xp.asarray(probe, dtype=xp.complex64)

        return xp.fft.fftshift(probe, axes=(-2, -1))

    def _return_object_fft(
        self,
        obj=None,
    ):
        """
        Returns absolute value of obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object

        Returns
        -------
        object_fft_amplitude: np.ndarray
            Amplitude of Fourier-transformed and center-shifted obj.
        """
        asnumpy = self._asnumpy

        if obj is None:
            obj = self._object

        obj = self._crop_rotate_object_fov(asnumpy(obj))
        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj))))

    def show_fourier_probe(
        self,
        probe=None,
        cbar=True,
        scalebar=True,
        pixelsize=None,
        pixelunits=None,
        **kwargs,
    ):
        """
        Plot probe in fourier space

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses the `probe_fourier` property
        cbar: bool, optional
            if True, adds colorbar
        scalebar: bool, optional
            if True, adds scalebar to probe
        pixelunits: str, optional
            units for scalebar, default is A^-1
        pixelsize: float, optional
            default is probe reciprocal sampling
        """
        asnumpy = self._asnumpy

        if probe is None:
            probe = self.probe_fourier
        else:
            probe = asnumpy(self._return_fourier_probe(probe))

        if pixelsize is None:
            pixelsize = self._reciprocal_sampling[1]
        if pixelunits is None:
            pixelunits = r"$\AA^{-1}$"

        figsize = kwargs.pop("figsize", (6, 6))
        chroma_boost = kwargs.pop("chroma_boost", 2)

        fig, ax = plt.subplots(figsize=figsize)
        show_complex(
            probe,
            cbar=cbar,
            figax=(fig, ax),
            scalebar=scalebar,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            ticks=False,
            chroma_boost=chroma_boost,
            **kwargs,
        )

    def show_object_fft(self, obj=None, **kwargs):
        """
        Plot FFT of reconstructed object

        Parameters
        ----------
        obj: complex array, optional
            if None is specified, uses the `object_fft` property
        """
        if obj is None:
            object_fft = self.object_fft
        else:
            object_fft = self._return_object_fft(obj)

        figsize = kwargs.pop("figsize", (6, 6))
        cmap = kwargs.pop("cmap", "magma")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 1)
        power = kwargs.pop("power", 0.2)

        pixelsize = 1 / (object_fft.shape[1] * self.sampling[1])
        show(
            object_fft,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            scalebar=True,
            pixelsize=pixelsize,
            ticks=False,
            pixelunits=r"$\AA^{-1}$",
            power=power,
            **kwargs,
        )

    @property
    def probe_fourier(self):
        """Current probe estimate in Fourier space"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(self._return_fourier_probe(self._probe))

    @property
    def probe_centered(self):
        """Current probe estimate shifted to the center"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(self._return_centered_probe(self._probe))

    @property
    def object_fft(self):
        """Fourier transform of current object estimate"""

        if not hasattr(self, "_object"):
            return None

        return self._return_object_fft(self._object)

    @property
    def angular_sampling(self):
        """Angular sampling [mrad]"""
        return getattr(self, "_angular_sampling", None)

    @property
    def sampling(self):
        """Sampling [Å]"""

        if self.angular_sampling is None:
            return None

        return tuple(
            electron_wavelength_angstrom(self._energy) * 1e3 / dk / n
            for dk, n in zip(self.angular_sampling, self._region_of_interest_shape)
        )

    @property
    def positions(self):
        """Probe positions [A]"""

        if self.angular_sampling is None:
            return None

        asnumpy = self._asnumpy

        positions = self._positions_px.copy()
        positions[:, 0] *= self.sampling[0]
        positions[:, 1] *= self.sampling[1]

        return asnumpy(positions)

    @property
    def object_cropped(self):
        """cropped and rotated object"""

        return self._crop_rotate_object_fov(self._object)
