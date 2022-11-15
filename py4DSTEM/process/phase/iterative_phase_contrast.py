"""Module for reconstructing phase objects from 4DSTEM experiments using iterative methods, namely DPC and ptychography."""

from typing import Union, Sequence, Mapping, Callable, Iterable, Tuple
from abc import ABCMeta, abstractmethod
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.io import DataCube
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from py4DSTEM.process.utils import get_shifted_ar


class PhaseReconstruction(metaclass=ABCMeta):
    """
    Base phase reconstruction class.
    Defines various common functions and properties for all subclasses to inherit,
    as well as sets up various abstract methods each subclass must define.
    """

    @abstractmethod
    def preprocess(self):
        """
        Abstract method all subclasses must define which prepares measured intensities.

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
        Abstract method all subclasses must define to perform forward projection.

        For DPC, this consists of differentiating the current object estimate.
        For Ptychography, this consists of an overlap projection followed by a Fourier projection.
        """
        pass

    @abstractmethod
    def _adjoint(self):
        """
        Abstract method all subclasses must define to perform forward projection.

        For DPC, this consists of Fourier-integrating the CoM gradient.
        For Ptychography, this consists of inverting the modified Fourier amplitude followed by 'stitching' the probe/object update patches.
        """
        pass

    @abstractmethod
    def _update(self):
        """Abstract method all subclasses must define to update current probe/object estimates."""
        pass

    @abstractmethod
    def reconstruct(self):
        """
        Abstract method all subclasses must define which performs the reconstruction
        by calling the subclass _forward(), _adjoint(), and _update() methods.
        """
        pass

    @abstractmethod
    def show(self):
        """Abstract method all subclasses must define to postprocess and display reconstruction outputs."""
        pass

    def _extract_intensities_and_calibrations_from_datacube(
        self, datacube: DataCube, require_calibrations: bool = False
    ):
        """
        Common preprocessing method to extract intensities and calibrations from datacube

        Parameters
        ----------
        datacube: Datacube
            Input 4D diffraction pattern intensities
        require_calibrations: bool
            If False, warning is issued instead of raising an error

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
                "Iterative reconstruction will not be quantitative unless you specify real-space calibrations in 'A'",
                UserWarning,
            )

            self._scan_sampling = (1.0, 1.0)
            self._scan_units = ("pixels",) * 2

        elif real_space_units == "A":
            self._scan_sampling = (calibration.get_R_pixel_size(),) * 2
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
                "Iterative reconstruction will not be quantitative unless you specify appropriate reciprocal-space calibrations",
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
                f"Reciprocal-space calibrations must be given in 'A^-1' or 'mrad', not {reciprocal_space_units}"
            )

    def _calculate_intensities_center_of_mass(
        self,
        intensities: np.ndarray,
        fit_function: str = "plane",
        plot_center_of_mass: bool = True,
        **kwargs,
    ):
        """
        Common preprocessing function to compute and fit diffraction intensities CoM

        Parameters
        ----------
        intensities: (Rx,Ry,Qx,Qy) xp.ndarray
            Raw intensities array stored on device, with dtype xp.float32
        fit_function: str, optional
            2D fitting function for CoM fitting. Must be 'plane' or 'parabola' or 'bezier_two'
        plot_center_of_mass: bool, optional
            If True, the computed and normalized CoM arrays will be displayed

        Assigns
        --------
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

        Mutates
        -------
        self._region_of_interest_shape: Tuple[int,int]
            Pixel dimensions (Sx,Sy) of the region of interest
            Commonly set to diffraction intensity dimensions (Qx,Qy)

        Displays
        --------
        self._com_measured_x/y and self._com_normalized_x/y, optional
            Measured and normalized CoM gradients
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # Intensities shape

        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._intensities_shape[-2:]
        else:
            self._region_of_interest_shape = np.array(self._region_of_interest_shape)

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
        ) * self._scan_sampling[0]
        self._com_normalized_y = (
            self._com_measured_y - self._com_fitted_y
        ) * self._scan_sampling[1]

        # Optionally, plot
        if plot_center_of_mass:

            figsize = kwargs.get("figsize", (12, 12))
            cmap = kwargs.get("cmap", "RdBu_r")
            kwargs.pop("cmap", None)
            kwargs.pop("figsize", None)

            extent = [
                0,
                self._scan_sampling[0] * self._intensities_shape[0],
                0,
                self._scan_sampling[1] * self._intensities_shape[1],
            ]
            fig, ax = plt.subplots(2, 2, figsize=figsize)

            ax[0, 0].imshow(
                asnumpy(self._com_measured_x), extent=extent, cmap=cmap, **kwargs
            )
            ax[0, 1].imshow(
                asnumpy(self._com_measured_y), extent=extent, cmap=cmap, **kwargs
            )
            ax[1, 0].imshow(
                asnumpy(self._com_normalized_x), extent=extent, cmap=cmap, **kwargs
            )
            ax[1, 1].imshow(
                asnumpy(self._com_normalized_y), extent=extent, cmap=cmap, **kwargs
            )

            for ax_flat in ax.flatten():
                ax_flat.set_xlabel(f"x [{self._scan_units[0]}]")
                ax_flat.set_ylabel(f"y [{self._scan_units[1]}]")

            fig.tight_layout()
            plt.show()

    def _solve_for_center_of_mass_relative_rotation(
        self,
        rotation_angles_deg: np.ndarray = np.arange(-90.0, 90.0, 1.0),
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        **kwargs,
    ):
        """
        Common preprocessing method to solve for the relative rotation between real and reciprocal space.
        We do this by minimizing the curl of the vector field.
        Alternatively, we could also maximing the divergence of the vector field.

        Parameters
        ----------
        rotation_angles_deg: ndarray, optional
            Array of angles in degrees to perform curl minimization over
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        maximize_divergence: bool, optional
            If True, the divergence of the CoM gradient vector field is maximized instead

        Assigns
        --------
        self._rotation_curl: np.ndarray
            Array of CoM curl for each angle
        self._rotation_curl_transpose: np.ndarray
            Array of transposed CoM curl for each angle
        self._rotation_best_rad: float
            Rotation angle which minimizes CoM curl, in radians
        self._rotation_best_tranpose: bool
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
        rotation_best_deg, optional
            Summary statistics
        """

        xp = self._xp
        asnumpy = self._asnumpy

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
            com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
            com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
            self._rotation_div = xp.mean(
                xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
            )
        else:
            com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
            com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
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
            com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
            com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
            self._rotation_div_transpose = xp.mean(
                xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
            )
        else:
            com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
            com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
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
                self._rotation_best_tranpose = False
            else:
                rotation_best_deg = rotation_angles_deg[ind_trans_max]
                self._rotation_best_rad = rotation_angles_rad[ind_trans_max]
                self._rotation_best_tranpose = True
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
                self._rotation_best_tranpose = False
            else:
                rotation_best_deg = rotation_angles_deg[ind_trans_min]
                self._rotation_best_rad = rotation_angles_rad[ind_trans_min]
                self._rotation_best_tranpose = True

        # Calculate corrected CoM
        if self._rotation_best_tranpose is False:
            self._com_x = (
                xp.cos(self._rotation_best_rad) * self._com_normalized_x
                - xp.sin(self._rotation_best_rad) * self._com_normalized_y
            )
            self._com_y = (
                xp.sin(self._rotation_best_rad) * self._com_normalized_x
                + xp.cos(self._rotation_best_rad) * self._com_normalized_y
            )
        else:
            self._com_x = (
                xp.cos(self._rotation_best_rad) * self._com_normalized_y
                - xp.sin(self._rotation_best_rad) * self._com_normalized_x
            )
            self._com_y = (
                xp.sin(self._rotation_best_rad) * self._com_normalized_y
                + xp.cos(self._rotation_best_rad) * self._com_normalized_x
            )

        # 'Public'-facing attributes as numpy arrays
        self.com_x = asnumpy(self._com_x)
        self.com_y = asnumpy(self._com_y)

        # Print summary
        if self._verbose:
            print(f"Best fit rotation = {str(np.round(rotation_best_deg))} degrees")
            if self._rotation_best_tranpose:
                print("Diffraction intensities should be transposed")
            else:
                print("No need to transpose diffraction intensities")

        # Plot Curl/Div rotation
        if plot_rotation:

            figsize = kwargs.get("figsize", (12, 12))
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
                    np.ptp(rotation_angles_deg) / np.ptp(self._rotation_div) / 4
                )
            else:
                ax.set_ylabel("Mean Absolute Curl")
                ax.set_aspect(
                    np.ptp(rotation_angles_deg) / np.ptp(self._rotation_curl) / 4
                )
            fig.tight_layout()
            plt.show()

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


class DPCReconstruction(PhaseReconstruction):
    """
    Iterative Differential Phase Constrast Reconstruction Class.

    Diffraction intensities dimensions         : (Rx,Ry,Qx,Qy)
    Reconstructed phase object dimensions      : (Rx,Ry)

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    energy: float, optional
        The electron energy of the wave functions this contrast transfer function will be applied to in eV
    verbose: bool, optional
        If True, various class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'

    Assigns
    --------
    self._xp: Callable
        Array computing module
    self._asnumpy: Callable
        Array conversion module to numpy
    self._region_of_interest_shape
       None, i.e. same as diffraction intensities (Qx,Qy)
    self._preprocessed: bool
        Flag to signal object has not yet been preprocessed
    """

    def __init__(
        self,
        datacube: DataCube,
        energy: float = None,
        verbose: bool = True,
        device: str = "cpu",
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

        self._energy = energy
        self._datacube = datacube
        self._verbose = verbose
        self._region_of_interest_shape = None
        self._preprocessed = False

    def preprocess(
        self,
        fit_function: str = "plane",
        plot_center_of_mass: bool = True,
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = np.arange(-90.0, 90.0, 1.0),
        **kwargs,
    ):
        """
        DPC preprocessing step.
        Calls the base class methods:

        _extract_intensities_and_calibrations_from_datacube(),
        _calculate_intensities_center_of_mass(), and
        _solve_for_center_of_mass_relative_rotation()

        Parameters
        ----------
        fit_function: str, optional
            2D fitting function for CoM fitting. Must be 'plane' or 'parabola' or 'bezier_two'
        plot_center_of_mass: bool, optional
            If True, the computed and fitted CoM arrays will be displayed
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        maximize_divergence: bool, optional
            If True, the divergence of the CoM gradient vector field is maximized instead
        rotation_angles_deg: np.darray, optional
            Array of angles in degrees to perform curl minimization over

        Mutates
        --------
        self._preprocessed: bool
            Flag to signal object has been preprocessed

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        self._extract_intensities_and_calibrations_from_datacube(
            self._datacube, require_calibrations=False
        )

        self._calculate_intensities_center_of_mass(
            self._intensities,
            fit_function=fit_function,
            plot_center_of_mass=plot_center_of_mass,
            **kwargs,
        )

        self._solve_for_center_of_mass_relative_rotation(
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            maximize_divergence=maximize_divergence,
            **kwargs,
        )

        self._preprocessed = True

        return self

    def _forward(
        self, padded_phase_object: np.ndarray, mask: np.ndarray, mask_inv: np.ndarray
    ):
        """
        DPC forward projection:
        Computes a centered finite-difference approximation to the phase gradient
        and projects to the measured CoM gradient

        Parameters
        ----------
        padded_phase_object: np.ndarray
            Current padded phase object estimate
        mask: np.ndarray
            Mask of object inside padded array
        mask_inv: np.ndarray
            Inverse mask of object inside padded array

        Returns
        --------
        obj_dx: np.ndarray
            Forward-projected horizontal CoM gradient
        obj_dy: np.ndarray
            Forward-projected vertical CoM gradient
        error: float
            Updated estimate error
        """

        xp = self._xp

        # centered finite-differences
        obj_dx = 0.5 * (
            xp.roll(padded_phase_object, 1, axis=0)
            - xp.roll(padded_phase_object, -1, axis=0)
        )
        obj_dy = 0.5 * (
            xp.roll(padded_phase_object, 1, axis=1)
            - xp.roll(padded_phase_object, -1, axis=1)
        )

        # difference from measurement
        obj_dx[mask] += self._com_x.ravel()
        obj_dy[mask] += self._com_y.ravel()
        obj_dx[mask_inv] = 0
        obj_dy[mask_inv] = 0

        error = xp.sqrt(
            xp.mean(
                (obj_dx[mask] - xp.mean(obj_dx[mask])) ** 2
                + (obj_dy[mask] - xp.mean(obj_dy[mask])) ** 2
            )
        )

        return obj_dx, obj_dy, error

    def _adjoint(
        self,
        obj_dx: np.ndarray,
        obj_dy: np.ndarray,
        kx_op: np.ndarray,
        ky_op: np.ndarray,
    ):
        """
        DPC adjoint projection:
        Fourier-integrates the current estimate of the CoM gradient

        Parameters
        ----------
        obj_dx: np.ndarray
            Forward-projected horizontal phase gradient
        obj_dy: np.ndarray
            Forward-projected vertical phase gradient
        kx_op: np.ndarray
            Scaled k_x operator
        ky_op: np.ndarray
            Scaled k_y operator

        Returns
        --------
        phase_update: np.ndarray
            Adjoint-projected phase object
        """

        xp = self._xp

        phase_update = xp.real(
            xp.fft.ifft2(xp.fft.fft2(obj_dx) * kx_op + xp.fft.fft2(obj_dy) * ky_op)
        )

        return phase_update

    def _update(
        self,
        padded_phase_object: np.ndarray,
        phase_update: np.ndarray,
        step_size: float,
    ):
        """
        DPC update step:

        Parameters
        ----------
        padded_phase_object: np.ndarray
            Current padded phase object estimate
        phase_update: np.ndarray
            Adjoint-projected phase object
        step_size: float
            Update step size

        Returns
        --------
        updated_padded_phase_object: np.ndarray
            Updated padded phase object estimate
        """

        padded_phase_object += step_size * phase_update
        return padded_phase_object

    def reconstruct(
        self,
        padding_factor: float = 2,
        max_iter: int = 64,
        step_size: float = 0.9,
        progress_bar: bool = True,
        store_iterations: bool = False,
    ):
        """
        Performs Iterative DPC Reconstruction:

        Parameters
        ----------
        padding_factor: float, optional
            Factor to pad object by to reduce periodic artifacts
        max_iter: int, optional
            Maximum number of iterations
        step_size: float, optional
            Reconstruction update step size
        progress_bar: bool, optional
            If True, reconstruction progress bar will be printed
        store_iterations: bool, optional
            If True, all reconstruction iterations will be stored

        Assigns
        --------
        self._object_phase: xp.ndarray
            Reconstructed phase object, on calculation device
        self.object_phase: np.ndarray
            Reconstructed phase object, as a numpy array
        self.error: float
            RMS error
        self._object_phase_iterations, optional
            max_iter length list storing reconstructed phase objects at each iteration
        self._error_iterations, optional
            max_iter length list storing RMS errors at each iteration

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # initialization
        padded_object_shape = np.round(
            self._intensities_shape[:2] * padding_factor
        ).astype("int")
        padded_phase_object = xp.zeros(padded_object_shape)
        mask = xp.zeros(padded_object_shape, dtype="bool")
        mask[: self._intensities_shape[0], : self._intensities_shape[1]] = True
        mask_inv = xp.logical_not(mask)

        if store_iterations:
            self._object_phase_iterations = []
            self._error_iterations = []

        # Fourier coordinates and operators
        kx = xp.fft.fftfreq(padded_object_shape[0], d=self._reciprocal_sampling[0])
        ky = xp.fft.fftfreq(padded_object_shape[1], d=self._reciprocal_sampling[1])
        kya, kxa = xp.meshgrid(ky, kx)
        k_den = kxa**2 + kya**2
        k_den[0, 0] = np.inf
        k_den = 1 / k_den
        kx_op = -1j * 0.25 * kxa * k_den
        ky_op = -1j * 0.25 * kya * k_den

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing phase",
            unit=" iter",
            disable=not progress_bar,
        ):

            # forward operator
            com_dx, com_dy, error = self._forward(padded_phase_object, mask, mask_inv)

            # adjoint operator
            phase_update = self._adjoint(com_dx, com_dy, kx_op, ky_op)

            # update
            padded_phase_object = self._update(
                padded_phase_object, phase_update, step_size
            )

            if store_iterations:
                self._object_phase_iterations.append(
                    padded_phase_object[
                        : self._intensities_shape[0], : self._intensities_shape[1]
                    ].copy()
                )
                self._error_iterations.append(error.item())

        # crop result
        self._object_phase = padded_phase_object[
            : self._intensities_shape[0], : self._intensities_shape[1]
        ]
        self.object_phase = asnumpy(self._object_phase)
        self.error = error.item()

        return self

    def _show_last_iteration(
        self, cbar: bool = False, plot_convergence: bool = False, **kwargs
    ):
        """
        Displays last iteration of reconstructed phase object.

        Parameters
        --------
        cbar: bool, optional
            If true, displays a colorbar
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        """

        figsize = kwargs.get("figsize", (8, 8))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        if plot_convergence:
            figsize = (figsize[0], figsize[1] + figsize[0] / 4)
            spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.1)
        else:
            spec = GridSpec(ncols=1, nrows=1)

        extent = [
            0,
            self._scan_sampling[0] * self._intensities_shape[0],
            0,
            self._scan_sampling[1] * self._intensities_shape[1],
        ]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(spec[0])
        im = ax.imshow(self.object_phase, extent=extent, cmap=cmap, **kwargs)
        ax.set_xlabel(f"x [{self._scan_units[0]}]")
        ax.set_ylabel(f"y [{self._scan_units[1]}]")
        ax.set_title(f"DPC Phase Reconstruction\nRMS error: {self.error:.3f}")

        if cbar:

            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "_error_iterations"):

            errors = self._error_iterations
            ax = fig.add_subplot(spec[1])

            ax.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Log RMS error")
            ax.yaxis.tick_right()

        plt.show()

    def _show_all_iterations(
        self,
        cbar: bool,
        plot_convergence: bool,
        iterations_grid: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays last iteration of reconstructed phase object.

        Parameters
        --------
        cbar: bool, optional
            If true, displays a colorbar
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        """

        asnumpy = self._asnumpy

        # TODO: 'auto' is likely not a good name
        if iterations_grid == "auto":
            iterations_grid = (2, 4)

        figsize = kwargs.get("figsize", (12, 6))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        total_grids = np.prod(iterations_grid)
        errors = self._error_iterations
        phases = self._object_phase_iterations
        max_iter = len(phases)

        if plot_convergence:
            grid_range = range(0, max_iter, max_iter // (total_grids - 2))
        else:
            grid_range = range(0, max_iter, max_iter // (total_grids - 1))
        extent = [
            0,
            self._scan_sampling[0] * self._intensities_shape[0],
            0,
            self._scan_sampling[1] * self._intensities_shape[1],
        ]

        gridspec = GridSpec(
            nrows=iterations_grid[0], ncols=iterations_grid[1], hspace=0
        )
        fig = plt.figure(figsize=figsize)

        for n, spec in enumerate(gridspec):

            if plot_convergence and n == len(grid_range):

                ax = fig.add_subplot(spec)
                ax.semilogy(np.arange(len(errors)), errors, **kwargs)
                ax.set_xlabel("Iteration Number")
                ax.set_ylabel("Log RMS error")
                ax.yaxis.tick_right()
                ax.set_aspect(max_iter / np.ptp(np.log10(np.array(errors))))

            else:
                ax = fig.add_subplot(spec)
                im = ax.imshow(
                    asnumpy(phases[grid_range[n]]), extent=extent, cmap=cmap, **kwargs
                )
                ax.set_xlabel(f"x [{self._scan_units[0]}]")
                ax.set_ylabel(f"y [{self._scan_units[1]}]")
                ax.set_title(
                    f"Iteration: {grid_range[n]}\nRMS error: {errors[grid_range[n]]:.3f}"
                )

                if cbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad="2.5%")
                    fig.colorbar(im, cax=cax, orientation="vertical")

        fig.tight_layout()

    def show(
        self,
        plot_convergence: bool = False,
        iterations_grid: Tuple[int, int] = None,
        cbar: bool = False,
        **kwargs,
    ):
        """
        Displays reconstructed phase object.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        if iterations_grid is None:
            self._show_last_iteration(
                plot_convergence=plot_convergence, cbar=cbar, **kwargs
            )
        else:
            self._show_all_iterations(
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                cbar=cbar,
                **kwargs,
            )

        return self
