"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods.
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from py4DSTEM.visualize import show_complex
from scipy.ndimage import zoom

try:
    import cupy as cp
    from cupy.fft.config import get_plan_cache
except (ModuleNotFoundError, ImportError):
    cp = np

from emdfile import Array, Custom, Metadata, _read_metadata, tqdmnd
from py4DSTEM.data import Calibration
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.phase.utils import (
    AffineTransform,
    copy_to_device,
    get_array_module,
    polar_aliases,
)
from py4DSTEM.process.utils import (
    electron_wavelength_angstrom,
    fourier_resample,
    get_shifted_ar,
)

warnings.showwarning = lambda msg, *args, **kwargs: print(msg, file=sys.stderr)
warnings.simplefilter("always", UserWarning)


class PhaseReconstruction(Custom):
    """
    Base phase reconstruction class.
    Defines various common functions and properties for subclasses to inherit.
    """

    def set_device(self, device, clear_fft_cache):
        """
        Sets calculation device.

        Parameters
        ----------
        device: str
            Calculation device will be perfomed on. Must be 'cpu' or 'gpu'

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if clear_fft_cache is not None:
            self._clear_fft_cache = clear_fft_cache

        if device is None:
            return self

        if device == "cpu":
            import scipy

            self._xp = np
            self._scipy = scipy

        elif device == "gpu":
            from cupyx import scipy

            self._xp = cp
            self._scipy = scipy

        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        self._device = device

        return self

    def set_storage(self, storage):
        """
        Sets storage device.

        Parameters
        ----------
        storage: str
            Device arrays will be stored on. Must be 'cpu' or 'gpu'

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if storage == "cpu":
            self._xp_storage = np

        elif storage == "gpu":
            if self._xp is np:
                raise ValueError("storage='gpu' and device='cpu' is not supported")
            self._xp_storage = cp

        else:
            raise ValueError(f"storage must be either 'cpu' or 'gpu', not {storage}")

        self._asnumpy = copy_to_device
        self._storage = storage

        return self

    def clear_device_mem(self, device, clear_fft_cache):
        """ """
        if device == "gpu":
            if clear_fft_cache:
                cache = get_plan_cache()
                cache.clear()

            xp = self._xp
            xp._default_memory_pool.free_all_blocks()
            xp._default_pinned_memory_pool.free_all_blocks()

    def copy_attributes_to_device(self, attrs, device):
        """Utility function to copy a set of attrs to device"""
        for attr in attrs:
            array = copy_to_device(getattr(self, attr), device)
            setattr(self, attr, array)

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

    def reinitialize_parameters(
        self,
        device: str = None,
        storage: str = None,
        clear_fft_cache: bool = None,
        verbose: bool = None,
    ):
        """
        Reinitializes common parameters. This is useful when loading a previously-saved
        reconstruction (which set device='cpu' and verbose=True for compatibility) ,
        using different initialization parameters.

        Parameters
        ----------
        device: str, optional
            If not None, assigns appropriate device modules
        storage: str, optional
            If not None, assigns appropriate storage modules
        clear_fft_cache: bool, optional
            If not None, sets the FFT caching parameter
        verbose: bool, optional
            If not None, sets the verbosity to verbose

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """

        if device is not None:
            self.set_device(device, clear_fft_cache)

        if storage is not None:
            self.set_storage(storage)

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
        padded_diffraction_intensities_shape=None,
        vacuum_probe_intensity=None,
        dp_mask=None,
        com_shifts=None,
        com_measured=None,
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

        Additionally, one may wish to zero-pad the diffraction intensity data. Note this does not increase
        the information or resolution, but might be beneficial in a limited number of cases, e.g. when the
        scan step sizes are much smaller than the real-space sampling (dx,dy). This can be achieved by specifying
        a padded_diffraction_intensities_shape which is larger than diffraction_intensities_shape.

        Parameters
        ----------
        datacube: Datacube
            Input 4D diffraction pattern intensities
        diffraction_intensities_shape: (int,int), optional
            Resampled diffraction intensities shape.
            If None, no resamping is performed
        reshaping method: str, optional
            Reshaping method to use, one of 'bin', 'bilinear' or 'fourier' (default)
        padded_diffraction_intensities_shape, (int,int), optional
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
                    np.ones(datacube.Rshape) * com_shifts[0],
                    np.ones(datacube.Rshape) * com_shifts[1],
                )

        if com_measured is not None:
            if np.isscalar(com_measured[0]):
                com_measured = (
                    np.ones(datacube.Rshape) * com_measured[0],
                    np.ones(datacube.Rshape) * com_measured[1],
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

            if com_measured is not None:
                com_measured = (
                    com_measured[0] * resampling_factor_x,
                    com_measured[1] * resampling_factor_x,
                )

            if reshaping_method == "bin":
                bin_factor = int(1 / resampling_factor_x)
                if bin_factor < 1:
                    raise ValueError(
                        f"Calculated binning factor {bin_factor} is less than 1."
                    )

                datacube = datacube.bin_Q(N=bin_factor)
                if vacuum_probe_intensity is not None:
                    # crop edges if necessary
                    if Qx % bin_factor == 0:
                        vacuum_probe_intensity = vacuum_probe_intensity[
                            : -(Qx % bin_factor), :
                        ]
                    if Qy % bin_factor == 0:
                        vacuum_probe_intensity = vacuum_probe_intensity[
                            :, : -(Qy % bin_factor)
                        ]

                    vacuum_probe_intensity = vacuum_probe_intensity.reshape(
                        Qx // bin_factor, bin_factor, Qy // bin_factor, bin_factor
                    ).sum(axis=(1, 3))
                if dp_mask is not None:
                    # crop edges if necessary
                    if Qx % bin_factor == 0:
                        dp_mask = dp_mask[: -(Qx % bin_factor), :]
                    if Qy % bin_factor == 0:
                        dp_mask = dp_mask[:, : -(Qy % bin_factor)]

                    dp_mask = dp_mask.reshape(
                        Qx // bin_factor, bin_factor, Qy // bin_factor, bin_factor
                    ).sum(axis=(1, 3))

            elif reshaping_method == "fourier":
                datacube = datacube.resample_Q(
                    N=resampling_factor_x,
                    method=reshaping_method,
                    conserve_array_sums=True,
                )
                if vacuum_probe_intensity is not None:
                    vacuum_probe_intensity = fourier_resample(
                        vacuum_probe_intensity,
                        output_size=diffraction_intensities_shape,
                        force_nonnegative=True,
                        conserve_array_sums=True,
                    )
                if dp_mask is not None:
                    dp_mask = fourier_resample(
                        dp_mask,
                        output_size=diffraction_intensities_shape,
                        force_nonnegative=True,
                        conserve_array_sums=False,
                    )

            elif reshaping_method == "bilinear":
                datacube = datacube.resample_Q(
                    N=resampling_factor_x,
                    method=reshaping_method,
                    conserve_array_sums=True,
                )
                if vacuum_probe_intensity is not None:
                    vacuum_probe_intensity = zoom(
                        vacuum_probe_intensity,
                        (resampling_factor_x, resampling_factor_x),
                        order=1,
                        mode="grid-wrap",
                        grid_mode=True,
                    )
                    vacuum_probe_intensity = (
                        vacuum_probe_intensity / resampling_factor_x**2
                    )
                if dp_mask is not None:
                    dp_mask = zoom(
                        dp_mask,
                        (resampling_factor_x, resampling_factor_x),
                        order=1,
                        mode="grid-wrap",
                        grid_mode=True,
                    )

            else:
                raise ValueError(
                    (
                        "reshaping_method needs to be one of 'bilinear', 'fourier', or 'bin', "
                        f"not {reshaping_method}."
                    )
                )

        if padded_diffraction_intensities_shape is not None:
            Qx, Qy = datacube.shape[-2:]
            Sx, Sy = padded_diffraction_intensities_shape
            datacube = datacube.pad_Q(output_size=padded_diffraction_intensities_shape)

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

        return datacube, vacuum_probe_intensity, dp_mask, com_shifts, com_measured

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

        # explicit read-only self attributes up-front
        verbose = self._verbose
        energy = self._energy

        intensities = np.asarray(datacube.data, dtype=np.float32)
        self._grid_scan_shape = intensities.shape[:2]

        # Extracts calibrations
        calibration = datacube.calibration
        real_space_units = calibration.get_R_pixel_units()
        reciprocal_space_units = calibration.get_Q_pixel_units()

        # Real-space
        if force_scan_sampling is not None:
            self._scan_sampling = (force_scan_sampling, force_scan_sampling)
            self._scan_units = ("A",) * 2
        else:
            if real_space_units == "pixels":
                if require_calibrations:
                    raise ValueError("Real-space calibrations must be given in 'A'")

                if verbose:
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
            if (
                force_angular_sampling is not None
                and force_reciprocal_sampling is not None
            ):
                raise ValueError(
                    "Only one of angular or reciprocal calibration can be forced."
                )

            # angular calibration specified
            if force_angular_sampling is not None:
                self._angular_sampling = (force_angular_sampling,) * 2
                self._angular_units = ("mrad",) * 2

                if energy is not None:
                    self._reciprocal_sampling = (
                        force_angular_sampling
                        / electron_wavelength_angstrom(energy)
                        / 1e3,
                    ) * 2
                    self._reciprocal_units = ("A^-1",) * 2

            # reciprocal calibration specified
            if force_reciprocal_sampling is not None:
                self._reciprocal_sampling = (force_reciprocal_sampling,) * 2
                self._reciprocal_units = ("A^-1",) * 2

                if energy is not None:
                    self._angular_sampling = (
                        force_reciprocal_sampling
                        * electron_wavelength_angstrom(energy)
                        * 1e3,
                    ) * 2
                    self._angular_units = ("mrad",) * 2

        else:
            if reciprocal_space_units == "pixels":
                if require_calibrations:
                    raise ValueError(
                        "Reciprocal-space calibrations must be given in in 'A^-1' or 'mrad'"
                    )

                if verbose:
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

                if energy is not None:
                    self._angular_sampling = (
                        reciprocal_size * electron_wavelength_angstrom(energy) * 1e3,
                    ) * 2
                    self._angular_units = ("mrad",) * 2

            elif reciprocal_space_units == "mrad":
                angular_size = calibration.get_Q_pixel_size()
                self._angular_sampling = (angular_size,) * 2
                self._angular_units = ("mrad",) * 2

                if energy is not None:
                    self._reciprocal_sampling = (
                        angular_size / electron_wavelength_angstrom(energy) / 1e3,
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
        vectorized_calculation=True,
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
        vectorized_calculation: bool, optional
            If True (default), the calculation is vectorized
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

        # explicit read-only self attributes up-front
        xp = self._xp
        device = self._device
        asnumpy = self._asnumpy

        reciprocal_sampling = self._reciprocal_sampling

        if com_measured:
            com_measured_x = xp.asarray(com_measured[0], dtype=xp.float32)
            com_measured_y = xp.asarray(com_measured[1], dtype=xp.float32)

        else:
            if dp_mask is not None:
                if dp_mask.shape != intensities.shape[-2:]:
                    raise ValueError(
                        (
                            f"Mask shape should be (Qx,Qy):{intensities.shape[-2:]}, "
                            f"not {dp_mask.shape}"
                        )
                    )
                    dp_mask = xp.asarray(dp_mask, dtype=xp.float32)

            # Coordinates
            kx = xp.arange(intensities.shape[-2], dtype=xp.float32)
            ky = xp.arange(intensities.shape[-1], dtype=xp.float32)
            kya, kxa = xp.meshgrid(ky, kx)

            if vectorized_calculation:
                # copy to device
                intensities = copy_to_device(intensities, device)

                # calculate CoM
                if dp_mask is not None:
                    intensities_mask = intensities * dp_mask
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

            else:
                sx, sy = intensities.shape[:2]
                com_measured_x = xp.zeros((sx, sy), dtype=xp.float32)
                com_measured_y = xp.zeros((sx, sy), dtype=xp.float32)

                # loop of dps
                for rx, ry in tqdmnd(
                    sx,
                    sy,
                    desc="Calculating center of mass",
                    unit="probe position",
                    disable=not self._verbose,
                ):
                    masked_intensity = copy_to_device(intensities[rx, ry], device)
                    if dp_mask is not None:
                        masked_intensity *= dp_mask
                    summed_intensity = masked_intensity.sum()
                    com_measured_x[rx, ry] = (
                        xp.sum(masked_intensity * kxa) / summed_intensity
                    )
                    com_measured_y[rx, ry] = (
                        xp.sum(masked_intensity * kya) / summed_intensity
                    )

        if com_shifts is None:
            if fit_function is not None:
                com_measured_x_np = asnumpy(com_measured_x)
                com_measured_y_np = asnumpy(com_measured_y)
                finite_mask = np.isfinite(com_measured_x_np)

                com_shifts = fit_origin(
                    (com_measured_x_np, com_measured_y_np),
                    fitfunction=fit_function,
                    mask=finite_mask,
                )

                com_fitted_x = xp.asarray(com_shifts[0], dtype=xp.float32)
                com_fitted_y = xp.asarray(com_shifts[1], dtype=xp.float32)
            else:
                com_fitted_x = xp.asarray(com_measured_x, dtype=xp.float32)
                com_fitted_y = xp.asarray(com_measured_y, dtype=xp.float32)
        else:
            com_fitted_x = xp.asarray(com_shifts[0], dtype=xp.float32)
            com_fitted_y = xp.asarray(com_shifts[1], dtype=xp.float32)

        # Fit function to center of mass

        # fix CoM units
        com_normalized_x = (
            xp.nan_to_num(com_measured_x - com_fitted_x) * reciprocal_sampling[0]
        )
        com_normalized_y = (
            xp.nan_to_num(com_measured_y - com_fitted_y) * reciprocal_sampling[1]
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
        rotation_angles_deg: np.ndarray = None,
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

        # explicit read-only self attributes up-front
        xp = self._xp
        asnumpy = self._asnumpy
        verbose = self._verbose
        scan_sampling = self._scan_sampling
        scan_units = self._scan_units

        if rotation_angles_deg is None:
            rotation_angles_deg = np.arange(-89.0, 90.0, 1.0)

        if force_com_rotation is not None:
            # Rotation known

            _rotation_best_rad = np.deg2rad(force_com_rotation)

            if verbose:
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

                if verbose:
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

                if verbose:
                    if _rotation_best_transpose:
                        warnings.warn(
                            "Diffraction intensities should be transposed.",
                            UserWarning,
                        )

        else:
            # Rotation unknown
            if force_com_transpose is not None:
                # Transpose known, rotation unknown

                _rotation_best_transpose = force_com_transpose

                if verbose:
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

                if verbose:
                    warnings.warn(
                        f"Best fit rotation = {rotation_best_deg:.0f} degrees.",
                        UserWarning,
                    )

                if plot_rotation:
                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    if _rotation_best_transpose:
                        ax.plot(
                            rotation_angles_deg,
                            (
                                asnumpy(rotation_div_transpose)
                                if maximize_divergence
                                else asnumpy(rotation_curl_transpose)
                            ),
                            label="CoM after transpose",
                        )
                    else:
                        ax.plot(
                            rotation_angles_deg,
                            (
                                asnumpy(rotation_div)
                                if maximize_divergence
                                else asnumpy(rotation_curl)
                            ),
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
                if verbose:
                    warnings.warn(
                        f"Best fit rotation = {rotation_best_deg:.0f} degrees.",
                        UserWarning,
                    )
                    if _rotation_best_transpose:
                        warnings.warn(
                            "Diffraction intensities should be transposed.",
                            UserWarning,
                        )

                # Plot Curl/Div rotation
                if plot_rotation:
                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    ax.plot(
                        rotation_angles_deg,
                        (
                            asnumpy(rotation_div)
                            if maximize_divergence
                            else asnumpy(rotation_curl)
                        ),
                        label="CoM",
                    )
                    ax.plot(
                        rotation_angles_deg,
                        (
                            asnumpy(rotation_div_transpose)
                            if maximize_divergence
                            else asnumpy(rotation_curl_transpose)
                        ),
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

        # Optionally, plot CoM
        if plot_center_of_mass == "all":
            figsize = kwargs.pop("figsize", (8, 12))
            cmap = kwargs.pop("cmap", "RdBu_r")
            extent = [
                0,
                scan_sampling[1] * _com_measured_x.shape[1],
                scan_sampling[0] * _com_measured_x.shape[0],
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
                    _com_x,
                    _com_y,
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
                ax.set_ylabel(f"x [{scan_units[0]}]")
                ax.set_xlabel(f"y [{scan_units[1]}]")
                ax.set_title(title)

        elif plot_center_of_mass == "default" or plot_center_of_mass is True:
            figsize = kwargs.pop("figsize", (8, 4))
            cmap = kwargs.pop("cmap", "RdBu_r")

            extent = [
                0,
                scan_sampling[1] * _com_x.shape[1],
                scan_sampling[0] * _com_x.shape[0],
                0,
            ]

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=(0.25, 0.5))

            for ax, arr, title in zip(
                grid,
                [
                    _com_x,
                    _com_y,
                ],
                [
                    "Corrected CoM_x",
                    "Corrected CoM_y",
                ],
            ):
                ax.imshow(asnumpy(arr), extent=extent, cmap=cmap, **kwargs)
                ax.set_ylabel(f"x [{scan_units[0]}]")
                ax.set_xlabel(f"y [{scan_units[1]}]")
                ax.set_title(title)

        return (
            _rotation_best_rad,
            _rotation_best_transpose,
            _com_x,
            _com_y,
        )

    def _normalize_diffraction_intensities(
        self,
        diffraction_intensities,
        com_fitted_x,
        com_fitted_y,
        positions_mask,
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
        positions_mask: np.ndarray, optional
            Boolean real space mask to select positions in datacube to skip for reconstruction
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

        # explicit read-only self attributes up-front
        asnumpy = self._asnumpy

        mean_intensity = 0

        diffraction_intensities = asnumpy(diffraction_intensities)
        com_fitted_x = asnumpy(com_fitted_x)
        com_fitted_y = asnumpy(com_fitted_y)

        if positions_mask is not None:
            number_of_patterns = np.count_nonzero(positions_mask.ravel())
        else:
            number_of_patterns = np.prod(diffraction_intensities.shape[:2])

        # Aggressive cropping for when off-centered high scattering angle data was recorded
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
            diffraction_intensities_shape_crop = (crop_w * 2, crop_w * 2)
            amplitudes = np.zeros(
                (
                    number_of_patterns,
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

        else:
            crop_mask = None
            diffraction_intensities_shape_crop = diffraction_intensities.shape[-2:]
            amplitudes = np.zeros(
                (number_of_patterns,) + diffraction_intensities_shape_crop,
                dtype=np.float32,
            )

        counter = 0
        for rx, ry in tqdmnd(
            diffraction_intensities.shape[0],
            diffraction_intensities.shape[1],
            desc="Normalizing amplitudes",
            unit="probe position",
            disable=not self._verbose,
        ):
            if positions_mask is not None:
                if not positions_mask[rx, ry]:
                    continue
            intensities = get_shifted_ar(
                diffraction_intensities[rx, ry],
                -com_fitted_x[rx, ry],
                -com_fitted_y[rx, ry],
                bilinear=True,
                device="cpu",
            )

            if crop_patterns:
                intensities = intensities[crop_mask].reshape(
                    diffraction_intensities_shape_crop
                )

            mean_intensity += np.sum(intensities)
            amplitudes[counter] = np.sqrt(np.maximum(intensities, 0))
            counter += 1

        mean_intensity /= amplitudes.shape[0]

        self._diffraction_intensities_shape_crop = diffraction_intensities_shape_crop

        return amplitudes, mean_intensity, crop_mask

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

        # explicit read-only self attributes up-front
        asnumpy = self._asnumpy
        scan_sampling = self._scan_sampling
        scan_units = self._scan_units

        if com is None:
            com = (self._com_x, self._com_y)

        if pixelsize is None:
            pixelsize = scan_sampling[0]
        if pixelunits is None:
            pixelunits = scan_units[0]

        figsize = kwargs.pop("figsize", (6, 6))
        fig, ax = plt.subplots(figsize=figsize)

        complex_com = com[0] + 1j * com[1]

        show_complex(
            asnumpy(complex_com),
            cbar=cbar,
            figax=(fig, ax),
            scalebar=scalebar,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            ticks=False,
            **kwargs,
        )


class PtychographicReconstruction(PhaseReconstruction):
    """
    Base ptychographic reconstruction class.
    Inherits from PhaseReconstruction.
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
            "storage": self._storage,
            "clear_fft_cache": self._clear_fft_cache,
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

        # saving multiple None positions_mask fix
        if self._positions_mask is None:
            positions_mask = None
        elif self._positions_mask[0] is None:
            positions_mask = None
        else:
            positions_mask = self._positions_mask

        # preprocessing metadata
        self.metadata = Metadata(
            name="preprocess_metadata",
            data={
                "rotation_angle_rad": self._rotation_best_rad,
                "data_transpose": self._rotation_best_transpose,
                "positions_px": asnumpy(self._positions_px),
                "region_of_interest_shape": self._region_of_interest_shape,
                "amplitudes_shape": self._amplitudes_shape,
                "num_diffraction_patterns": self._num_diffraction_patterns,
                "sampling": self.sampling,
                "angular_sampling": self.angular_sampling,
                "positions_mask": positions_mask,
            },
        )

        # reconstruction metadata
        is_stack = self._save_iterations and hasattr(self, "object_iterations")
        error = self.error_iterations

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
            num_iterations = len(self.object_iterations)
            iterations = list(range(0, num_iterations, self._save_iterations_frequency))
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
            "storage": "cpu",  # for compatibility
            "clear_fft_cache": True,  # for compatibility
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
        self._amplitudes_shape = preprocess_md["amplitudes_shape"]
        self._num_diffraction_patterns = preprocess_md["num_diffraction_patterns"]
        self._positions_mask = preprocess_md["positions_mask"]

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
        if "_object_iterations_emd" in dict_data.keys():
            self.object_iterations = list(dict_data["_object_iterations_emd"].data)
            self.probe_iterations = list(dict_data["_probe_iterations_emd"].data)

        self.error_iterations = error
        self.error = error[-1]

        # Slim preprocessing to enable visualize
        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self.object = asnumpy(self._object)
        self.probe = self.probe_centered
        self._preprocessed = True

    def _switch_object_type(self, object_type):
        """
        Switches object type to/from "potential"/"complex"

        Returns
        --------
        self: PhaseReconstruction
            Self to enable chaining
        """
        xp = self._xp

        match (self._object_type, object_type):
            case ("potential", "complex"):
                self._object_type = "complex"
                self._object = xp.exp(1j * self._object, dtype=xp.complex64)
            case ("complex", "potential"):
                self._object_type = "potential"
                self._object = xp.angle(self._object)
            case _:
                self._object_type = self._object_type

        return self

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

    def _calculate_scan_positions_in_pixels(
        self,
        positions: np.ndarray,
        positions_mask,
        object_padding_px,
        positions_offset_ang,
    ):
        """
        Method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input probe positions in Å.
            If None, a raster scan using experimental parameters is constructed.
        positions_mask: np.ndarray, optional
            Boolean real space mask to select positions in datacube to skip for reconstruction
        object_padding_px: Tuple[int,int], optional
            Pixel dimensions to pad object with
            If None, the padding is set to half the probe ROI dimensions
        positions_offset_ang, np.ndarray, optional
            Offset of positions in A

        Returns
        -------
        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        object_padding_px: Tupe[int,int]
            Updated object_padding_px
        """

        # explicit read-only self attributes up-front
        grid_scan_shape = self._grid_scan_shape
        rotation_angle = self._rotation_best_rad
        transpose = self._rotation_best_transpose
        step_sizes = self._scan_sampling
        region_of_interest_shape = self._region_of_interest_shape
        sampling = self.sampling

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

            if transpose:
                x = (x - np.ptp(x) / 2) / sampling[1]
                y = (y - np.ptp(y) / 2) / sampling[0]
            else:
                x = (x - np.ptp(x) / 2) / sampling[0]
                y = (y - np.ptp(y) / 2) / sampling[1]
            x, y = np.meshgrid(x, y, indexing="ij")

            if positions_offset_ang is not None:
                if transpose:
                    x += positions_offset_ang[0] / sampling[1]
                    y += positions_offset_ang[1] / sampling[0]
                else:
                    x += positions_offset_ang[0] / sampling[0]
                    y += positions_offset_ang[1] / sampling[1]

            if positions_mask is not None:
                x = x[positions_mask]
                y = y[positions_mask]
        else:
            positions -= np.mean(positions, axis=0)
            x = positions[:, 0] / sampling[1]
            y = positions[:, 1] / sampling[0]

        if rotation_angle is not None:
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

        if transpose:
            positions = np.array([y.ravel(), x.ravel()]).T
        else:
            positions = np.array([x.ravel(), y.ravel()]).T

        positions -= np.min(positions, axis=0)

        if object_padding_px is None:
            float_padding = region_of_interest_shape / 2
            object_padding_px = (float_padding, float_padding)
        elif np.isscalar(object_padding_px[0]):
            object_padding_px = (
                (object_padding_px[0],) * 2,
                (object_padding_px[1],) * 2,
            )

        positions[:, 0] += object_padding_px[0][0]
        positions[:, 1] += object_padding_px[1][0]

        return positions, object_padding_px

    def _sum_overlapping_patches_bincounts_base(
        self, patches: np.ndarray, positions_px
    ):
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
        # explicit read-only self attributes up-front
        xp = get_array_module(patches)
        roi_shape = self._region_of_interest_shape
        object_shape = self._object_shape

        x0 = xp.round(positions_px[:, 0]).astype("int")
        y0 = xp.round(positions_px[:, 1]).astype("int")

        x_ind = xp.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
        y_ind = xp.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

        flat_weights = patches.ravel()
        indices = ((y0[:, None, None] + y_ind[None, None, :]) % object_shape[1]) + (
            (x0[:, None, None] + x_ind[None, :, None]) % object_shape[0]
        ) * object_shape[1]
        counts = xp.bincount(
            indices.ravel(), weights=flat_weights, minlength=np.prod(object_shape)
        )
        counts = xp.reshape(counts, object_shape).astype(xp.float32)
        return counts

    def _sum_overlapping_patches_bincounts(self, patches: np.ndarray, positions_px):
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

        if np.iscomplexobj(patches):
            real = self._sum_overlapping_patches_bincounts_base(
                patches.real, positions_px
            )
            imag = self._sum_overlapping_patches_bincounts_base(
                patches.imag, positions_px
            )
            return real + 1.0j * imag
        else:
            return self._sum_overlapping_patches_bincounts_base(patches, positions_px)

    def _extract_vectorized_patch_indices(
        self,
        positions_px,
    ):
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
        # explicit read-only self attributes up-front
        xp_storage = self._xp_storage
        roi_shape = self._region_of_interest_shape
        obj_shape = self._object_shape

        x0 = xp_storage.round(positions_px[:, 0]).astype("int")
        y0 = xp_storage.round(positions_px[:, 1]).astype("int")

        x_ind = xp_storage.fft.fftfreq(roi_shape[0], d=1 / roi_shape[0]).astype("int")
        y_ind = xp_storage.fft.fftfreq(roi_shape[1], d=1 / roi_shape[1]).astype("int")

        vectorized_patch_indices_row = (
            x0[:, None, None] + x_ind[None, :, None]
        ) % obj_shape[0]
        vectorized_patch_indices_col = (
            y0[:, None, None] + y_ind[None, None, :]
        ) % obj_shape[1]

        return vectorized_patch_indices_row, vectorized_patch_indices_col

    def _set_reconstruction_method_parameters(
        self,
        reconstruction_method,
        reconstruction_parameter,
        reconstruction_parameter_a,
        reconstruction_parameter_b,
        reconstruction_parameter_c,
        step_size,
    ):
        """"""

        if reconstruction_method == "generalized-projections":
            if (
                reconstruction_parameter_a is None
                or reconstruction_parameter_b is None
                or reconstruction_parameter_c is None
            ):
                raise ValueError(
                    (
                        "reconstruction_parameter_a/b/c must all be specified "
                        "when using reconstruction_method='generalized-projections'."
                    )
                )

            use_projection_scheme = True
            projection_a = reconstruction_parameter_a
            projection_b = reconstruction_parameter_b
            projection_c = reconstruction_parameter_c
            reconstruction_parameter = None
            step_size = None
        elif (
            reconstruction_method == "DM_AP"
            or reconstruction_method == "difference-map_alternating-projections"
        ):
            if reconstruction_parameter < 0.0 or reconstruction_parameter > 1.0:
                raise ValueError("reconstruction_parameter must be between 0-1.")

            use_projection_scheme = True
            projection_a = -reconstruction_parameter
            projection_b = 1
            projection_c = 1 + reconstruction_parameter
            step_size = None
        elif (
            reconstruction_method == "RAAR"
            or reconstruction_method == "relaxed-averaged-alternating-reflections"
        ):
            if reconstruction_parameter < 0.0 or reconstruction_parameter > 1.0:
                raise ValueError("reconstruction_parameter must be between 0-1.")

            use_projection_scheme = True
            projection_a = 1 - 2 * reconstruction_parameter
            projection_b = reconstruction_parameter
            projection_c = 2
            step_size = None
        elif (
            reconstruction_method == "RRR"
            or reconstruction_method == "relax-reflect-reflect"
        ):
            if reconstruction_parameter < 0.0 or reconstruction_parameter > 2.0:
                raise ValueError("reconstruction_parameter must be between 0-2.")

            use_projection_scheme = True
            projection_a = -reconstruction_parameter
            projection_b = reconstruction_parameter
            projection_c = 2
            step_size = None
        elif (
            reconstruction_method == "SUPERFLIP"
            or reconstruction_method == "charge-flipping"
        ):
            use_projection_scheme = True
            projection_a = 0
            projection_b = 1
            projection_c = 2
            reconstruction_parameter = None
            step_size = None
        elif (
            reconstruction_method == "GD" or reconstruction_method == "gradient-descent"
        ):
            use_projection_scheme = False
            projection_a = None
            projection_b = None
            projection_c = None
            reconstruction_parameter = None
        else:
            raise ValueError(
                (
                    "reconstruction_method must be one of 'generalized-projections', "
                    "'DM_AP' (or 'difference-map_alternating-projections'), "
                    "'RAAR' (or 'relaxed-averaged-alternating-reflections'), "
                    "'RRR' (or 'relax-reflect-reflect'), "
                    "'SUPERFLIP' (or 'charge-flipping'), "
                    f"or 'GD' (or 'gradient-descent'), not  {reconstruction_method}."
                )
            )

        return (
            use_projection_scheme,
            projection_a,
            projection_b,
            projection_c,
            reconstruction_parameter,
            step_size,
        )

    def _report_reconstruction_summary(
        self,
        max_iter,
        use_projection_scheme,
        reconstruction_method,
        reconstruction_parameter,
        projection_a,
        projection_b,
        projection_c,
        normalization_min,
        max_batch_size,
        step_size,
    ):
        """ """

        # object type
        first_line = f"Performing {max_iter} iterations using a {self._object_type} object type, "

        # stochastic gradient descent
        if max_batch_size is not None:
            if use_projection_scheme:
                raise ValueError(
                    (
                        "Stochastic object/probe updating is inconsistent with 'DM_AP', 'RAAR', 'RRR', and 'SUPERFLIP'. "
                        "Use reconstruction_method='GD' or set max_batch_size=None."
                    )
                )
            else:
                warnings.warn(
                    (
                        first_line + f"with the {reconstruction_method} algorithm, "
                        f"with normalization_min: {normalization_min} and step _size: {step_size}, "
                        f"in batches of max {max_batch_size} measurements."
                    ),
                    UserWarning,
                )

        else:
            # named projection set method
            if reconstruction_parameter is not None:
                warnings.warn(
                    (
                        first_line + f"with the {reconstruction_method} algorithm, "
                        f"with normalization_min: {normalization_min} and α: {reconstruction_parameter}."
                    ),
                    UserWarning,
                )

            # generalized projections (or the even more rare charge-flipping)
            elif projection_a is not None:
                warnings.warn(
                    (
                        first_line + f"with the {reconstruction_method} algorithm, "
                        f"with normalization_min: {normalization_min} and (a,b,c): "
                        f"{projection_a, projection_b, projection_c}."
                    ),
                    UserWarning,
                )

            # gradient descent
            else:
                warnings.warn(
                    (
                        first_line + f"with the {reconstruction_method} algorithm, "
                        f"with normalization_min: {normalization_min} and step _size: {step_size}."
                    ),
                    UserWarning,
                )

    def _constraints(
        self,
        current_object,
        current_probe,
        current_positions,
        initial_positions,
        **kwargs,
    ):
        """Wrapper function for all classes to inherit"""

        current_object = self._object_constraints(current_object, **kwargs)
        current_probe = self._probe_constraints(current_probe, **kwargs)
        current_positions = self._positions_constraints(
            current_positions, initial_positions, **kwargs
        )

        return current_object, current_probe, current_positions

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
            for dk, n in zip(self.angular_sampling, self._amplitudes_shape)
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
