"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC and ptychography.
"""

import warnings
from abc import ABCMeta, abstractmethod
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import rotate

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.io import DataCube
from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.phase.utils import AffineTransform, polar_aliases
from py4DSTEM.process.utils import (
    electron_wavelength_angstrom,
    fourier_resample,
    get_shifted_ar,
)

warnings.simplefilter(action="always", category=UserWarning)


class PhaseReconstruction(metaclass=ABCMeta):
    """
    Base phase reconstruction class.
    Defines various common functions and properties for all subclasses to inherit,
    as well as sets up various abstract methods each subclass must define.
    """

    @abstractmethod
    def preprocess(self):
        """
        Abstract method subclasses must define which prepares measured intensities.

        For DPC, this includes:
        - Fitting diffraction intensitie's CoM and rotation
        - Preparing Fourier-coordinates and operators

        For Ptychography, this includes:
        - Centering diffraction intensities using fitted CoM
        - Padding diffraction intensities to region of interest dimensions
        - Preparing initial guess for scanning positions
        - Preparing initial guesses for the objects and probes arrays
        """
        pass

    @abstractmethod
    def _forward(self):
        """
        Abstract method subclasses must define to perform forward projection.

        For DPC, this consists of differentiating the current object estimate.
        For Ptychography, this consists of overlap and Fourier projections.
        """
        pass

    @abstractmethod
    def _adjoint(self):
        """
        Abstract method subclasses must define to perform forward projection.

        For DPC, this consists of Fourier-integrating the CoM gradient.
        For Ptychography, this consists of inverting the modified Fourier amplitude,
        followed by 'stitching' the probe/object update patches.
        """
        pass

    @abstractmethod
    def _update(self):
        """
        Abstract method subclasses must define to update current object estimates.
        """
        pass

    @abstractmethod
    def reconstruct(self):
        """
        Abstract method subclasses must define which performs the reconstruction
        by calling the subclass _forward(), _adjoint(), and _update() methods.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Abstract method subclasses must define to postprocess and display results.
        """
        pass

    def _preprocess_datacube_and_vacuum_probe(
        self,
        datacube,
        diffraction_intensities_shape=None,
        reshaping_method="fourier",
        probe_roi_shape=None,
        vacuum_probe_intensity=None,
        dp_mask=None,
    ):
        """
        Datacube preprocssing step, to set the reciprocal- and real-space sampling.
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

        Returns
        --------
        datacube: Datacube
            Resampled and Padded datacube
        """

        if diffraction_intensities_shape is not None:

            Qx, Qy = datacube.shape[-2:]
            Sx, Sy = diffraction_intensities_shape

            resampling_factor_x = Sx / Qx
            resampling_factor_y = Sy / Qy

            if resampling_factor_x != resampling_factor_y:
                raise ValueError(
                    "Datacube calibration can only handle uniform Q-sampling."
                )

            Q_pixel_size = datacube.calibration.get_Q_pixel_size()

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
            datacube.calibration.set_Q_pixel_size(Q_pixel_size / resampling_factor_x)

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

        return datacube, vacuum_probe_intensity, dp_mask

    def _extract_intensities_and_calibrations_from_datacube(
        self,
        datacube: DataCube,
        require_calibrations: bool = False,
        dp_mask: np.ndarray = None,
    ):
        """
        Common method to extract intensities and calibrations from datacube.

        Parameters
        ----------
        datacube: Datacube
            Input 4D diffraction pattern intensities
        require_calibrations: bool
            If False, warning is issued instead of raising an error
        dp_mask: ndarray
            If not None, apply mask to datacube amplitude

        Assigns
        --------
        self._intensities: (Rx,Ry,Qx,Qy) xp.ndarray
            Raw intensities array stored on device, with dtype xp.float32
        self._intensities_shape: (Rx,Ry,Qx,Qy) xp.ndarray
            Shape of self._intensities
        self._intensities_sum: (Rx,Ry) xp.ndarray
            Total pixel intensity
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
        self._intensities = xp.asarray(datacube.data, dtype=xp.float32)

        if dp_mask is not None:
            if dp_mask.shape != self._intensities.shape[-2:]:
                raise ValueError(
                    (
                        f"Mask shape should be (Qx,Qy):{self._intensities.shape[-2:]}, "
                        f"not {dp_mask.shape}"
                    )
                )
            self._intensities *= xp.asarray(dp_mask, dtype=xp.float32)

        self._intensities_shape = np.array(self._intensities.shape)
        self._intensities_sum = xp.sum(self._intensities, axis=(-2, -1))

        # Extracts calibrations
        calibration = datacube.calibration
        real_space_units = calibration.get_R_pixel_units()
        reciprocal_space_units = calibration.get_Q_pixel_units()

        # Real-space
        if real_space_units == "pixels":
            if require_calibrations:
                raise ValueError("Real-space calibrations must be given in 'A'")

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
            self._scan_sampling = (calibration.get_R_pixel_size()*10,) * 2
            self._scan_units = ("A",) * 2
        else:
            raise ValueError(
                f"Real-space calibrations must be given in 'A', not {real_space_units}"
            )

        # Reciprocal-space
        if reciprocal_space_units == "pixels":
            if require_calibrations:
                raise ValueError(
                    "Reciprocal-space calibrations must be given in in 'A^-1' or 'mrad'"
                )

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
                    reciprocal_size * electron_wavelength_angstrom(self._energy) * 1e3,
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

    def _calculate_intensities_center_of_mass(
        self,
        intensities: np.ndarray,
        fit_function: str = "plane",
    ):
        """
        Common preprocessing function to compute and fit diffraction intensities CoM

        Parameters
        ----------
        intensities: (Rx,Ry,Qx,Qy) xp.ndarray
            Raw intensities array stored on device, with dtype xp.float32
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'

        Assigns
        --------
        self._region_of_interest_shape: Tuple[int,int]
            Pixel dimensions (Sx,Sy) of the region of interest
            Commonly set to diffraction intensity dimensions (Qx,Qy)
        self._com_measured_x: (Rx,Ry) xp.ndarray
            Measured horizontal center of mass gradient
        self._com_measured_y: (Rx,Ry) xp.ndarray
            Measured vertical center of mass gradient
        self._com_fitted_x: (Rx,Ry) xp.ndarray
            Best fit horizontal center of mass gradient
        self._com_fitted_y: (Rx,Ry) xp.ndarray
            Best fit vertical center of mass gradient
        self._com_normalized_x: (Rx,Ry) xp.ndarray
            Normalized horizontal center of mass gradient
        self._com_normalized_y: (Rx,Ry) xp.ndarray
            Normalized vertical center of mass gradient
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # Intensities shape
        self._region_of_interest_shape = self._intensities_shape[-2:]

        # Coordinates
        kx = xp.arange(self._intensities_shape[-2], dtype=xp.float32)
        ky = xp.arange(self._intensities_shape[-1], dtype=xp.float32)
        kya, kxa = xp.meshgrid(ky, kx)

        # calculate CoM
        self._com_measured_x = (
            xp.sum(intensities * kxa[None, None], axis=(-2, -1)) / self._intensities_sum
        )
        self._com_measured_y = (
            xp.sum(intensities * kya[None, None], axis=(-2, -1)) / self._intensities_sum
        )

        # Fit function to center of mass
        # TO-DO: allow py4DSTEM.process.calibration.fit_origin to accept xp.ndarrays
        or_fits = fit_origin(
            (asnumpy(self._com_measured_x), asnumpy(self._com_measured_y)),
            fitfunction=fit_function,
        )
        self._com_fitted_x = xp.asarray(or_fits[0])
        self._com_fitted_y = xp.asarray(or_fits[1])

        # fix CoM units
        self._com_normalized_x = (
            self._com_measured_x - self._com_fitted_x
        ) * self._reciprocal_sampling[0]
        self._com_normalized_y = (
            self._com_measured_y - self._com_fitted_y
        ) * self._reciprocal_sampling[1]

    def _solve_for_center_of_mass_relative_rotation(
        self,
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

        Assigns
        --------
        self._rotation_curl: np.ndarray
            Array of CoM curl for each angle
        self._rotation_curl_transpose: np.ndarray
            Array of transposed CoM curl for each angle
        self._rotation_best_rad: float
            Rotation angle which minimizes CoM curl, in radians
        self._rotation_best_transpose: bool
            Whether diffraction intensities need to be transposed to minimize CoM curl
        self._com_x: xp.ndarray
            Corrected horizontal center of mass gradient, on calculation device
        self._com_y: xp.ndarray
            Corrected vertical center of mass gradient, on calculation device
        self.com_x: np.ndarray
            Corrected horizontal center of mass gradient, as a numpy array
        self.com_y: np.ndarray
            Corrected vertical center of mass gradient, as a numpy array

        Displays
        --------
        self._rotation_curl/div vs rotation_angles_deg, optional
            Vector calculus quantity being minimized/maximized
        self._com_measured_x/y, self._com_normalized_x/y and self.com_x/y, optional
            Measured and normalized CoM gradients
        rotation_best_deg, optional
            Summary statistics
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if force_com_rotation is not None:
            # Rotation known

            self._rotation_best_rad = np.deg2rad(force_com_rotation)

            if self._verbose:
                warnings.warn(
                    (
                        "Best fit rotation forced to "
                        f"{str(np.round(force_com_rotation))} degrees."
                    ),
                    UserWarning,
                )

            if force_com_transpose is not None:
                # Transpose known

                self._rotation_best_transpose = force_com_transpose

                if self._verbose:
                    warnings.warn(
                        f"Transpose of intensities forced to {force_com_transpose}.",
                        UserWarning,
                    )

            else:
                # Rotation known, transpose unknown
                com_measured_x = (
                    xp.cos(self._rotation_best_rad) * self._com_normalized_x
                    - xp.sin(self._rotation_best_rad) * self._com_normalized_y
                )
                com_measured_y = (
                    xp.sin(self._rotation_best_rad) * self._com_normalized_x
                    + xp.cos(self._rotation_best_rad) * self._com_normalized_y
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
                    xp.cos(self._rotation_best_rad) * self._com_normalized_y
                    - xp.sin(self._rotation_best_rad) * self._com_normalized_x
                )
                com_measured_y = (
                    xp.sin(self._rotation_best_rad) * self._com_normalized_y
                    + xp.cos(self._rotation_best_rad) * self._com_normalized_x
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
                    self._rotation_best_transpose = (
                        rotation_div_transpose > rotation_div
                    )
                else:
                    self._rotation_best_transpose = (
                        rotation_curl_transpose < rotation_curl
                    )

                if self._verbose:
                    if self._rotation_best_transpose:
                        print("Diffraction intensities should be transposed.")
                    else:
                        print("No need to transpose diffraction intensities.")

        else:
            # Rotation unknown
            if force_com_transpose is not None:
                # Transpose known, rotation unknown

                self._rotation_best_transpose = force_com_transpose

                if self._verbose:
                    warnings.warn(
                        f"Transpose of intensities forced to {force_com_transpose}.",
                        UserWarning,
                    )

                rotation_angles_deg = xp.asarray(rotation_angles_deg)
                rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

                if self._rotation_best_transpose:
                    com_measured_x = (
                        xp.cos(rotation_angles_rad) * self._com_normalized_y[None]
                        - xp.sin(rotation_angles_rad) * self._com_normalized_x[None]
                    )
                    com_measured_y = (
                        xp.sin(rotation_angles_rad) * self._com_normalized_y[None]
                        + xp.cos(rotation_angles_rad) * self._com_normalized_x[None]
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
                        self._rotation_div_transpose = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )

                        ind_trans_max = xp.argmax(self._rotation_div_transpose).item()
                        rotation_best_deg = rotation_angles_deg[ind_trans_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_trans_max]

                    else:
                        com_grad_x_y = (
                            com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        )
                        com_grad_y_x = (
                            com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                        )
                        self._rotation_curl_transpose = xp.mean(
                            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                        )

                        ind_trans_min = xp.argmin(self._rotation_curl_transpose).item()
                        rotation_best_deg = rotation_angles_deg[ind_trans_min]
                        self._rotation_best_rad = rotation_angles_rad[ind_trans_min]

                else:
                    com_measured_x = (
                        xp.cos(rotation_angles_rad) * self._com_normalized_x[None]
                        - xp.sin(rotation_angles_rad) * self._com_normalized_y[None]
                    )
                    com_measured_y = (
                        xp.sin(rotation_angles_rad) * self._com_normalized_x[None]
                        + xp.cos(rotation_angles_rad) * self._com_normalized_y[None]
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
                        self._rotation_div = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )

                        ind_max = xp.argmax(self._rotation_div).item()
                        rotation_best_deg = rotation_angles_deg[ind_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_max]

                    else:
                        com_grad_x_y = (
                            com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        )
                        com_grad_y_x = (
                            com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                        )
                        self._rotation_curl = xp.mean(
                            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                        )

                        ind_min = xp.argmin(self._rotation_curl).item()
                        rotation_best_deg = rotation_angles_deg[ind_min]
                        self._rotation_best_rad = rotation_angles_rad[ind_min]

                if self._verbose:
                    print(
                        (
                            "Best fit rotation = "
                            f"{str(np.round(rotation_best_deg))} degrees."
                        )
                    )

                if plot_rotation:

                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    if self._rotation_best_transpose:
                        ax.plot(
                            rotation_angles_deg,
                            asnumpy(self._rotation_div_transpose)
                            if maximize_divergence
                            else asnumpy(self._rotation_curl_transpose),
                            label="CoM after transpose",
                        )
                    else:
                        ax.plot(
                            rotation_angles_deg,
                            asnumpy(self._rotation_div)
                            if maximize_divergence
                            else asnumpy(self._rotation_curl),
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
                            np.ptp(self._rotation_div_transpose)
                            if self._rotation_best_transpose
                            else np.ptp(self._rotation_div)
                        )
                        ax.set_ylabel("Mean Absolute Divergence")
                        ax.set_aspect(np.ptp(rotation_angles_deg) / aspect_ratio / 4)
                    else:
                        aspect_ratio = (
                            np.ptp(self._rotation_curl_transpose)
                            if self._rotation_best_transpose
                            else np.ptp(self._rotation_curl)
                        )
                        ax.set_ylabel("Mean Absolute Curl")
                        ax.set_aspect(np.ptp(rotation_angles_deg) / aspect_ratio / 4)
                    fig.tight_layout()

            else:

                # Transpose unknown, rotation unknown
                rotation_angles_deg = xp.asarray(rotation_angles_deg)
                rotation_angles_rad = xp.deg2rad(rotation_angles_deg)[:, None, None]

                # Untransposed
                com_measured_x = (
                    xp.cos(rotation_angles_rad) * self._com_normalized_x[None]
                    - xp.sin(rotation_angles_rad) * self._com_normalized_y[None]
                )
                com_measured_y = (
                    xp.sin(rotation_angles_rad) * self._com_normalized_x[None]
                    + xp.cos(rotation_angles_rad) * self._com_normalized_y[None]
                )

                if maximize_divergence:
                    com_grad_x_x = (
                        com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                    )
                    com_grad_y_y = (
                        com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                    )
                    self._rotation_div = xp.mean(
                        xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                    )
                else:
                    com_grad_x_y = (
                        com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                    )
                    com_grad_y_x = (
                        com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                    )
                    self._rotation_curl = xp.mean(
                        xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                    )

                # Transposed
                com_measured_x = (
                    xp.cos(rotation_angles_rad) * self._com_normalized_y[None]
                    - xp.sin(rotation_angles_rad) * self._com_normalized_x[None]
                )
                com_measured_y = (
                    xp.sin(rotation_angles_rad) * self._com_normalized_y[None]
                    + xp.cos(rotation_angles_rad) * self._com_normalized_x[None]
                )

                if maximize_divergence:
                    com_grad_x_x = (
                        com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                    )
                    com_grad_y_y = (
                        com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                    )
                    self._rotation_div_transpose = xp.mean(
                        xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                    )
                else:
                    com_grad_x_y = (
                        com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                    )
                    com_grad_y_x = (
                        com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                    )
                    self._rotation_curl_transpose = xp.mean(
                        xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                    )

                rotation_angles_rad = asnumpy(xp.squeeze(rotation_angles_rad))
                rotation_angles_deg = asnumpy(rotation_angles_deg)

                # Find lowest curl/ maximum div value
                if maximize_divergence:
                    # Maximize Divergence
                    ind_max = xp.argmax(self._rotation_div).item()
                    ind_trans_max = xp.argmax(self._rotation_div_transpose).item()

                    if (
                        self._rotation_div[ind_max]
                        >= self._rotation_div_transpose[ind_trans_max]
                    ):
                        rotation_best_deg = rotation_angles_deg[ind_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_max]
                        self._rotation_best_transpose = False
                    else:
                        rotation_best_deg = rotation_angles_deg[ind_trans_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_trans_max]
                        self._rotation_best_transpose = True
                else:
                    # Minimize Curl
                    ind_min = xp.argmin(self._rotation_curl).item()
                    ind_trans_min = xp.argmin(self._rotation_curl_transpose).item()

                    if (
                        self._rotation_curl[ind_min]
                        <= self._rotation_curl_transpose[ind_trans_min]
                    ):
                        rotation_best_deg = rotation_angles_deg[ind_min]
                        self._rotation_best_rad = rotation_angles_rad[ind_min]
                        self._rotation_best_transpose = False
                    else:
                        rotation_best_deg = rotation_angles_deg[ind_trans_min]
                        self._rotation_best_rad = rotation_angles_rad[ind_trans_min]
                        self._rotation_best_transpose = True

                # Print summary
                if self._verbose:
                    print(
                        (
                            "Best fit rotation = "
                            f"{str(np.round(rotation_best_deg))} degrees."
                        )
                    )
                    if self._rotation_best_transpose:
                        print("Diffraction intensities should be transposed.")
                    else:
                        print("No need to transpose diffraction intensities.")

                # Plot Curl/Div rotation
                if plot_rotation:

                    figsize = kwargs.get("figsize", (8, 2))
                    fig, ax = plt.subplots(figsize=figsize)

                    ax.plot(
                        rotation_angles_deg,
                        asnumpy(self._rotation_div)
                        if maximize_divergence
                        else asnumpy(self._rotation_curl),
                        label="CoM",
                    )
                    ax.plot(
                        rotation_angles_deg,
                        asnumpy(self._rotation_div_transpose)
                        if maximize_divergence
                        else asnumpy(self._rotation_curl_transpose),
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
                                np.ptp(self._rotation_div),
                                np.ptp(self._rotation_div_transpose),
                            )
                            / 4
                        )
                    else:
                        ax.set_ylabel("Mean Absolute Curl")
                        ax.set_aspect(
                            np.ptp(rotation_angles_deg)
                            / np.maximum(
                                np.ptp(self._rotation_curl),
                                np.ptp(self._rotation_curl_transpose),
                            )
                            / 4
                        )
                    fig.tight_layout()

        # Calculate corrected CoM
        if self._rotation_best_transpose:
            self._com_x = (
                xp.cos(self._rotation_best_rad) * self._com_normalized_y
                - xp.sin(self._rotation_best_rad) * self._com_normalized_x
            )
            self._com_y = (
                xp.sin(self._rotation_best_rad) * self._com_normalized_y
                + xp.cos(self._rotation_best_rad) * self._com_normalized_x
            )
        else:
            self._com_x = (
                xp.cos(self._rotation_best_rad) * self._com_normalized_x
                - xp.sin(self._rotation_best_rad) * self._com_normalized_y
            )
            self._com_y = (
                xp.sin(self._rotation_best_rad) * self._com_normalized_x
                + xp.cos(self._rotation_best_rad) * self._com_normalized_y
            )

        # 'Public'-facing attributes as numpy arrays
        self.com_x = asnumpy(self._com_x)
        self.com_y = asnumpy(self._com_y)

        # Optionally, plot CoM
        if plot_center_of_mass == "all":

            figsize = kwargs.get("figsize", (8, 12))
            cmap = kwargs.get("cmap", "RdBu_r")
            kwargs.pop("cmap", None)
            kwargs.pop("figsize", None)

            extent = [
                0,
                self._scan_sampling[1] * self._intensities_shape[1],
                self._scan_sampling[0] * self._intensities_shape[0],
                0,
            ]

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(3, 2), axes_pad=(0.25, 0.5))

            for ax, arr, title in zip(
                grid,
                [
                    self._com_measured_x,
                    self._com_measured_y,
                    self._com_normalized_x,
                    self._com_normalized_y,
                    self.com_x,
                    self.com_y,
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
                ax.set_xlabel(f"x [{self._scan_units[0]}]")
                ax.set_ylabel(f"y [{self._scan_units[1]}]")
                ax.set_title(title)

        elif plot_center_of_mass == "default":

            figsize = kwargs.get("figsize", (8, 4))
            cmap = kwargs.get("cmap", "RdBu_r")
            kwargs.pop("cmap", None)
            kwargs.pop("figsize", None)

            extent = [
                0,
                self._scan_sampling[1] * self._intensities_shape[1],
                self._scan_sampling[0] * self._intensities_shape[0],
                0,
            ]

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=(0.25, 0.5))

            for ax, arr, title in zip(
                grid,
                [
                    self.com_x,
                    self.com_y,
                ],
                [
                    "Corrected CoM_x",
                    "Corrected CoM_y",
                ],
            ):
                ax.imshow(asnumpy(arr), extent=extent, cmap=cmap, **kwargs)
                ax.set_xlabel(f"x [{self._scan_units[0]}]")
                ax.set_ylabel(f"y [{self._scan_units[1]}]")
                ax.set_title(title)

    def _set_polar_parameters(self, parameters: dict):
        """
        Set the phase of the phase aberration.

        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.

        Mutates
        -------
        self._polar_parameters: dict
            Updated polar aberrations dictionary
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

    def _normalize_diffraction_intensities(
        self,
        diffraction_intensities,
        com_fitted_x,
        com_fitted_y,
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

        Returns
        -------
        amplitudes: (Rx * Ry, Sx, Sy) np.ndarray
            Flat array of normalized diffraction amplitudes
        mean_intensity: float
            Mean intensity value
        """

        xp = self._xp
        asnumpy = self._asnumpy

        dps = asnumpy(diffraction_intensities)
        com_x = asnumpy(com_fitted_x)
        com_y = asnumpy(com_fitted_y)

        mean_intensity = 0

        amplitudes = xp.zeros_like(dps)

        for rx in range(self._intensities_shape[0]):
            for ry in range(self._intensities_shape[1]):
                intensities = get_shifted_ar(
                    dps[rx, ry], -com_x[rx, ry], -com_y[rx, ry], bilinear=True
                )

                intensities = xp.asarray(intensities)
                mean_intensity += xp.sum(intensities)

                amplitudes[rx, ry] = xp.sqrt(xp.maximum(intensities, 0))

        amplitudes = xp.reshape(
            amplitudes, (-1,) + tuple(self._region_of_interest_shape)
        )
        mean_intensity /= amplitudes.shape[0]

        return amplitudes, mean_intensity

    def _calculate_scan_positions_in_pixels(self, positions: np.ndarray):
        """
        Common static method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input experimental positions [Å].
            If None, a raster scan using experimental parameters is constructed.

        Mutates
        -------
        self._object_padding_px: np.ndarray
            Object array padding in pixels

        Returns
        -------
        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        """

        grid_scan_shape = self._intensities_shape[:2]
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

        else:
            x = positions[:, 0]
            y = positions[:, 1]

        if self._rotation_best_transpose:
            x = (x - np.ptp(x) / 2) / self.sampling[1]
            y = (y - np.ptp(y) / 2) / self.sampling[0]
        else:
            x = (x - np.ptp(x) / 2) / self.sampling[0]
            y = (y - np.ptp(y) / 2) / self.sampling[1]
        x, y = np.meshgrid(x, y, indexing="ij")

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
            self._object_padding_px = self._region_of_interest_shape / 2
        positions += self._object_padding_px

        return positions

    def _wrapped_indices_2D_window(
        self,
        center_position: np.ndarray,
        window_shape: Sequence[int],
        array_shape: Sequence[int],
    ):
        """
        Computes periodic indices for a window_shape probe centered at center_position,
        in object of size array_shape.

        Parameters
        ----------
        center_position: (2,) np.ndarray
            The window center positions in pixels
        window_shape: (2,) Sequence[int]
            The pixel dimensions of the window
        array_shape: (2,) Sequence[int]
            The pixel dimensions of the array the window will be embedded in
        Returns
        -------
        window_indices: length-2 tuple of
            The 2D indices of the window
        """

        asnumpy = self._asnumpy
        sx, sy = array_shape
        nx, ny = window_shape

        cx, cy = np.round(asnumpy(center_position)).astype(int)
        ox, oy = (cx - nx // 2, cy - ny // 2)

        return np.ix_(np.arange(ox, ox + nx) % sx, np.arange(oy, oy + ny) % sy)

    def _sum_overlapping_patches(self, patches: np.ndarray):
        """
        Sum overlapping patches defined into object shaped array

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
        positions = self._positions_px
        patch_shape = self._region_of_interest_shape
        array_shape = self._object_shape

        out_array = xp.zeros(array_shape, patches.dtype)
        for ind, pos in enumerate(positions):
            indices = self._wrapped_indices_2D_window(pos, patch_shape, array_shape)
            out_array[indices] += patches[ind]

        return out_array

    def _sum_overlapping_patches_bincounts_base(self, patches: np.ndarray):
        """
        Base bincouts overlapping patches sum function, operating on real-valued arrays.

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
        x_ind = xp.round(xp.arange(roi_shape[0]) - roi_shape[0] / 2).astype("int")
        y_ind = xp.round(xp.arange(roi_shape[1]) - roi_shape[1] / 2).astype("int")

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

    def _set_vectorized_patch_indices(self):
        """
        Sets the vectorized row/col indices used for the overlap projection

        Assigns
        -------
        self._vectorized_patch_indices_row: np.ndarray
            Row indices for probe patches inside object array
        self._vectorized_patch_indices_col
            Column indices for probe patches inside object array
        """
        xp = self._xp
        x0 = xp.round(self._positions_px[:, 0]).astype("int")
        y0 = xp.round(self._positions_px[:, 1]).astype("int")

        roi_shape = self._region_of_interest_shape
        x_ind = xp.round(xp.arange(roi_shape[0]) - roi_shape[0] / 2).astype("int")
        y_ind = xp.round(xp.arange(roi_shape[1]) - roi_shape[1] / 2).astype("int")

        obj_shape = self._object_shape
        self._vectorized_patch_indices_row = (
            x0[:, None, None] + x_ind[None, :, None]
        ) % obj_shape[0]
        self._vectorized_patch_indices_col = (
            y0[:, None, None] + y_ind[None, None, :]
        ) % obj_shape[1]

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

        if self._rotation_best_transpose == True: 
            angle = self._rotation_best_rad
        else: 
            angle = -self._rotation_best_rad

        tf = AffineTransform(angle=angle)
        rotated_points = tf(
            asnumpy(self._positions_px), origin=asnumpy(self._positions_px_com), xp=np
        )
        min_x, min_y = np.floor(np.amin(rotated_points, axis=0) - padding).astype("int")
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x, max_y = np.ceil(np.amax(rotated_points, axis=0) + padding).astype("int")

        rotated_array = rotate(
            asnumpy(array), np.rad2deg(-angle), reshape=False
        )[min_x:max_x, min_y:max_y]
        
        if self._rotation_best_transpose == True: 
            rotated_array = rotated_array.T

        return rotated_array

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
