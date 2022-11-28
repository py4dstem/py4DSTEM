"""Module for reconstructing phase objects from 4DSTEM experiments using iterative methods, namely DPC and ptychography."""

from typing import Union, Sequence, Mapping, Callable, Iterable, Tuple
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

import numpy as np
import warnings

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.io import DataCube
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.process.calibration import fit_origin
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from py4DSTEM.process.utils import get_shifted_ar, get_CoM
from py4DSTEM.process.phase.utils import (
    fft_shift,
    ComplexProbe,
    polar_symbols,
    polar_aliases,
)


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
        ) * self._reciprocal_sampling[0]
        self._com_normalized_y = (
            self._com_measured_y - self._com_fitted_y
        ) * self._reciprocal_sampling[1]

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

            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(fig,111,nrows_ncols=(2,2),axes_pad=0.1)

            for ax, arr in zip(grid,[self._com_measured_x,self._com_measured_y,self._com_normalized_x,self._com_normalized_y]):
                ax.imshow(
                    asnumpy(arr.T), extent=extent, origin='lower',cmap=cmap, **kwargs
                    )
                ax.set_xlabel(f"x [{self._scan_units[0]}]")
                ax.set_ylabel(f"y [{self._scan_units[1]}]")

            plt.show()

    def _solve_for_center_of_mass_relative_rotation(
        self,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        force_com_rotation: float = None,
        force_com_transpose: bool = None,
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
                    f"Best fit rotation forced to {str(np.round(force_com_rotation))} degrees.",
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
                    rotation_div_transpose = xp.mean(xp.abs(com_grad_x_x + com_grad_y_y))
                else:
                    com_grad_x_y = com_measured_x[1:-1, 2:] - com_measured_x[1:-1, :-2]
                    com_grad_y_x = com_measured_y[2:, 1:-1] - com_measured_y[:-2, 1:-1]
                    rotation_curl_transpose = xp.mean(xp.abs(com_grad_y_x - com_grad_x_y))

                if maximize_divergence:
                    self._rotation_best_transpose = rotation_div_transpose > rotation_div
                else:
                    self._rotation_best_transpose = rotation_curl_transpose < rotation_curl

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
                        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                        self._rotation_div_transpose = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )
                        
                        ind_trans_max = xp.argmax(self._rotation_div_transpose).item()
                        rotation_best_deg = rotation_angles_deg[ind_trans_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_trans_max]

                    else:
                        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
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
                        com_grad_x_x = com_measured_x[:, 2:, 1:-1] - com_measured_x[:, :-2, 1:-1]
                        com_grad_y_y = com_measured_y[:, 1:-1, 2:] - com_measured_y[:, 1:-1, :-2]
                        self._rotation_div = xp.mean(
                            xp.abs(com_grad_x_x + com_grad_y_y), axis=(-2, -1)
                        )
                        
                        ind_max = xp.argmax(self._rotation_div).item()
                        rotation_best_deg = rotation_angles_deg[ind_max]
                        self._rotation_best_rad = rotation_angles_rad[ind_max]

                    else:
                        com_grad_x_y = com_measured_x[:, 1:-1, 2:] - com_measured_x[:, 1:-1, :-2]
                        com_grad_y_x = com_measured_y[:, 2:, 1:-1] - com_measured_y[:, :-2, 1:-1]
                        self._rotation_curl = xp.mean(
                            xp.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
                        )
                        
                        ind_min = xp.argmin(self._rotation_curl).item()
                        rotation_best_deg = rotation_angles_deg[ind_min]
                        self._rotation_best_rad = rotation_angles_rad[ind_min]

                if self._verbose:
                    print(f"Best fit rotation = {str(np.round(rotation_best_deg))} degrees.")

                if plot_rotation:

                    figsize = kwargs.get("figsize", (12, 12))
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
                        aspect_ratio = np.ptp(self._rotation_div_transpose) if self._rotation_best_transpose else np.ptp(self._rotation_div)
                        ax.set_ylabel("Mean Absolute Divergence")
                        ax.set_aspect(
                            np.ptp(rotation_angles_deg) / aspect_ratio / 4
                        )
                    else:
                        aspect_ratio = np.ptp(self._rotation_curl_transpose) if self._rotation_best_transpose else np.ptp(self._rotation_curl)
                        ax.set_ylabel("Mean Absolute Curl")
                        ax.set_aspect(
                            np.ptp(rotation_angles_deg) / aspect_ratio / 4
                        )
                    fig.tight_layout()
                    plt.show()

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
                    print(f"Best fit rotation = {str(np.round(rotation_best_deg))} degrees.")
                    if self._rotation_best_transpose:
                        print("Diffraction intensities should be transposed.")
                    else:
                        print("No need to transpose diffraction intensities.")

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
                            np.ptp(rotation_angles_deg) / np.maximum(np.ptp(self._rotation_div),np.ptp(self._rotation_div_transpose)) / 4
                        )
                    else:
                        ax.set_ylabel("Mean Absolute Curl")
                        ax.set_aspect(
                            np.ptp(rotation_angles_deg) / np.maximum(np.ptp(self._rotation_curl),np.ptp(self._rotation_curl_transpose)) / 4
                        )
                    fig.tight_layout()
                    plt.show()
                
        # Calculate corrected CoM
        if self._rotation_best_transpose is False:
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

    def _pad_diffraction_intensities(
        self,
        diffraction_intensities: np.ndarray,
        region_of_interest_shape: Tuple[int, int],
    ):
        """
        Common static method to zero-pad diffraction intensities to a certain region of interest shape.

        Parameters
        ----------
        diffraction_intensities: (Rx,Ry,Qx,Qy) np.ndarray
            Array of diffraction intensities to be zero-padded
        region_of_interest_shape: (2,) Tuple[int,int]
            Pixel dimensions (Sx,Sy) the CBED patterns will be padded to

        Returns
        -------
        padded_diffraction_intensities: (Rx,Ry,Sx,Sy) np.ndarray
            Zero-padded diffraction intensities
        """

        xp = self._xp
        diffraction_intensities_shape = np.array(diffraction_intensities.shape[-2:])

        if np.all(diffraction_intensities_shape != self._intensities_shape[-2:]):
            raise ValueError()

        if any(
            dp_shape > roi_shape
            for dp_shape, roi_shape in zip(
                diffraction_intensities_shape, region_of_interest_shape
            )
        ):
            raise ValueError()

        if np.all(diffraction_intensities_shape != region_of_interest_shape):
            padding_list = [(0, 0), (0, 0)]  # No padding along first two dimensions
            for current_dim, target_dim in zip(
                diffraction_intensities_shape, region_of_interest_shape
            ):
                pad_value = target_dim - current_dim
                pad_tuple = (pad_value // 2, pad_value // 2 + pad_value % 2)
                padding_list.append(pad_tuple)

            diffraction_intensities = xp.pad(
                diffraction_intensities, tuple(padding_list), mode="constant"
            )

        return diffraction_intensities

    def _pad_vacuum_probe_intensity(
        self,
        vacuum_probe_intensity: np.ndarray,
        region_of_interest_shape: Tuple[int, int],
    ):
        """
        Common static method to vacuum probe intensity to a certain region of interest shape.

        Parameters
        ----------
        vacuum_probe_intensity: (Qx,Qy) np.ndarray
            Vacuum probe to be zero-padded
        region_of_interest_shape: (2,) Tuple[int,int]
            Pixel dimensions (Sx,Sy) the CBED patterns will be padded to

        Returns
        -------
        padded_vacuum_probe: (Sx,Sy) np.ndarray
            Zero-padded vacuum probe
        """
        xp = self._xp
        vacuum_probe_intensity_shape = np.array(vacuum_probe_intensity.shape)
        vacuum_probe_intensity = xp.asarray(vacuum_probe_intensity, dtype=xp.float32)

        if any(
            vp_shape > roi_shape
            for vp_shape, roi_shape in zip(
                vacuum_probe_intensity_shape, region_of_interest_shape
            )
        ):
            raise ValueError()

        if np.all(vacuum_probe_intensity_shape != region_of_interest_shape):
            padding_list = []
            for current_dim, target_dim in zip(
                vacuum_probe_intensity_shape, region_of_interest_shape
            ):
                pad_value = target_dim - current_dim
                pad_tuple = (pad_value // 2, pad_value // 2 + pad_value % 2)
                padding_list.append(pad_tuple)

            vacuum_probe_intensity = xp.pad(
                vacuum_probe_intensity, tuple(padding_list), mode="constant"
            )

        return vacuum_probe_intensity

    def _normalize_diffraction_intensities(
        self, diffraction_intensities, com_fitted_x, com_fitted_y
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
        diffraction_intensities: (Rx * Ry, Sx, Sy) np.ndarray
            Flat array of normalized diffraction intensities
        """

        xp = self._xp
        asnumpy = self._asnumpy

        dps = asnumpy(diffraction_intensities)
        com_x = asnumpy(com_fitted_x)
        com_y = asnumpy(com_fitted_y)

        mean_intensity = 0

        for rx in range(self._intensities_shape[0]):
            for ry in range(self._intensities_shape[1]):
                amplitudes = xp.asarray(
                    get_shifted_ar(
                        dps[rx, ry], 
                        -com_x[rx, ry], -com_y[rx, ry], bilinear=True
                    )
                )
                # amplitudes /= xp.sum(amplitudes)
                mean_intensity += xp.sum(amplitudes)


                diffraction_intensities[rx, ry] = xp.sqrt(xp.maximum(amplitudes, 0))
        diffraction_intensities = xp.reshape(
            diffraction_intensities, (-1,) + tuple(self._region_of_interest_shape)
        )
        mean_intensity /= diffraction_intensities.shape[0]
        return diffraction_intensities, mean_intensity

    def _calculate_scan_positions_in_pixels(self, positions: np.ndarray):
        """
        Common static method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input experimental positions [Å].
            If None, a raster scan using experimental parameters is constructed.

        Assigns
        -------
        self._object_px_padding: np.ndarray
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

        self._object_px_padding = self._region_of_interest_shape / 2
        positions += self._object_px_padding

        return positions

    def _wrapped_indices_2D_window(
        self,
        center_position: np.ndarray,
        window_shape: Sequence[int],
        array_shape: Sequence[int],
    ):
        """
        Computes periodic indices for a window_shape probe centered at center_position, in object of size array_shape.
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
            self._positions_px length array of self._region_of_interest shape patches to sum

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
        """ """
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
        Sum overlapping patches defined into object shaped array using bincounts

        Parameters
        ----------
        patches: (Rx*Ry,Sx,Sy) np.ndarray
            self._positions_px length array of self._region_of_interest shape patches to sum

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
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        force_com_rotation: float = None,
        force_com_transpose: bool = None,

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
        force_ com_rotation: float (degrees), optional 
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool (optional)
            Force whether diffraction intensities need to be transposed.
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
            calculate_rotation = True, 
            force_com_rotation = force_com_rotation,
            force_com_transpose = force_com_transpose,
            **kwargs,
        )

        self._preprocessed = True

        return self

    def _forward(
            self, padded_phase_object: np.ndarray, mask: np.ndarray, mask_inv: np.ndarray, error: float, step_size: float
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
        dx, dy = self._scan_sampling

        # centered finite-differences
        obj_dx = (
            xp.roll(padded_phase_object, 1, axis=0)
            - xp.roll(padded_phase_object, -1, axis=0)
        )/(2*dx)
        obj_dy = (
            xp.roll(padded_phase_object, 1, axis=1)
            - xp.roll(padded_phase_object, -1, axis=1)
        )/(2*dy)

        # difference from measurement
        obj_dx[mask] += self._com_x.ravel()
        obj_dy[mask] += self._com_y.ravel()
        obj_dx[mask_inv] = 0
        obj_dy[mask_inv] = 0

        new_error = xp.sqrt(
            xp.mean(
                (obj_dx[mask] - xp.mean(obj_dx[mask])) ** 2
                + (obj_dy[mask] - xp.mean(obj_dy[mask])) ** 2
            )
        )

        if new_error > error:
            step_size /= 2

        return obj_dx, obj_dy, new_error, step_size

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
        error = 0.

        if store_iterations:
            self._object_phase_iterations = []
            self._error_iterations = []

        # Fourier coordinates and operators
        kx = xp.fft.fftfreq(padded_object_shape[0], d=self._scan_sampling[0])
        ky = xp.fft.fftfreq(padded_object_shape[1], d=self._scan_sampling[1])
        kya, kxa = xp.meshgrid(ky, kx)
        k_den = kxa**2 + kya**2
        k_den[0, 0] = np.inf
        k_den = 1 / k_den
        kx_op = -1j * 0.25 * step_size * kxa * k_den
        ky_op = -1j * 0.25 * step_size * kya * k_den

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing phase",
            unit=" iter",
            disable=not progress_bar,
        ):

            # forward operator
            com_dx, com_dy, error, step_size = self._forward(padded_phase_object, mask, mask_inv, error, step_size)

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
        im = ax.imshow(self.object_phase.T, extent=extent, cmap=cmap, origin='lower',**kwargs)
        ax.set_xlabel(f"x [{self._scan_units[0]}]")
        ax.set_ylabel(f"y [{self._scan_units[1]}]")
        ax.set_title(f"DPC Phase Reconstruction - RMS error: {self.error:.3e}")

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
        max_iter = len(phases) - 1

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
                    asnumpy(phases[grid_range[n]].T), extent=extent, origin='lower',cmap=cmap, **kwargs
                )
                ax.set_xlabel(f"x [{self._scan_units[0]}]")
                ax.set_ylabel(f"y [{self._scan_units[1]}]")
                ax.set_title(
                    f"Iteration: {grid_range[n]}\nRMS error: {errors[grid_range[n]]:.3e}"
                )

                if cbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad="2.5%")
                    fig.colorbar(im, cax=cax, orientation="vertical")

        #fig.tight_layout()

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


class PtychographicReconstruction(PhaseReconstruction):
    """
    Iterative Ptychographic Reconstruction Class.

    Diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions      : (Sx,Sy)
    Reconstructed object dimensions     : (Px,Py)

    such that (Sx,Sy) >= (Qx,Qy) is the region-of-interest (ROI) size of our probe
    and (Px,Py) >= (Sx,Sy) is the padded-object size we position our ROI around in.

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions this contrast transfer function will be applied to in eV
    region_of_interest_shape: Tuple[int,int]
        Pixel dimensions (Sx,Sy) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the diffraction intensities dimensions (Qx,Qy)
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (Px,Py)
        If None, initialized to 1.0j
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy)
        If None, initialized to Probe object with specified semiangle_cutoff, energy, and aberrations
    scan_positions: np.ndarray, optional
        Probe positions in Å for each diffraction intensity
        If None, initialized to a grid scan
    verbose: bool, optional
        If True, various class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial probe guess
    vacuum_probe_intensity: np.ndarray, optional
        Vacuum probe to use as intensity aperture for initial probe guess
    polar_parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration magnitudes should be given in Å
        and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.

    Assigns
    --------
    self._xp: Callable
        Array computing module
    self._intensities: (Rx,Ry,Qx,Qy) xp.ndarray
        Raw intensities array stored on device, with dtype xp.float32
    self._preprocessed: bool
        Flag to signal object has not yet been preprocessed
    """

    def __init__(
        self,
        datacube: DataCube,
        energy: float,
        region_of_interest_shape: Tuple[int, int] = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        scan_positions: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        semiangle_cutoff: float = None,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        # Should probably be abstracted in a device.py similar to:
        # https://github.com/abTEM/abTEM/blob/95da2f5ba900f2530f2689af845be85e96b1129a/abtem/device.py
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

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))

        if polar_parameters is None:
            polar_parameters = {}

        polar_parameters.update(kwargs)
        self._set_polar_parameters(polar_parameters)

        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._region_of_interest_shape = region_of_interest_shape
        self._object = initial_object_guess
        self._probe = initial_probe_guess
        self._scan_positions = scan_positions
        self._datacube = datacube
        self._verbose = verbose
        self._preprocessed = False

    def preprocess(
        self,
        fit_function: str = "plane",
        plot_center_of_mass: bool = True,
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        plot_probe_overlaps: bool = True,
        force_com_rotation: float = None,
        force_com_transpose: float = None,
        **kwargs,
    ):
        """
        Ptychographic preprocessing step.
        Calls the base class methods:

        _extract_intensities_and_calibrations_from_datacube,
        _compute_center_of_mass(),
        _solve_CoM_rotation(),
        _pad_diffraction_intensities()
        _normalize_diffraction_intensities()
        _calculate_scan_positions_in_px()

        Additionally, it initializes an (Px,Py) array of 1.0j
        and an ideal complex probe using the specified polar parameters.

        Parameters
        ----------
        fit_function: str, optional
            2D fitting function for CoM fitting. Must be 'plane' or 'parabola' or 'bezier_two'
        plot_center_of_mass: bool, optional
            If True, the computed and fitted CoM arrays will be displayed
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        rotation_angles_deg: np.darray, optional
            Array of angles in degrees to perform curl minimization over
        plot_probe_overlaps: bool, optional
            If True, the initial probe overlaps scanned over the object will be displayed
        force_com_rotation: float (degrees), optional 
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool, optional
            Force whether diffraction intensities need to be transposed.
            
        Assigns
        --------
        self._preprocessed: bool
            Flag to signal object has been preprocessed

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        xp = self._xp
        asnumpy = self._asnumpy

        self._extract_intensities_and_calibrations_from_datacube(
            self._datacube, require_calibrations=True
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
            force_com_rotation = force_com_rotation,
            force_com_transpose = force_com_transpose,
            **kwargs,
        )

        self._intensities = self._pad_diffraction_intensities(
            self._intensities, self._region_of_interest_shape
        )

        (
            self._intensities,
            self._mean_diffraction_intensity,
        ) = self._normalize_diffraction_intensities(
            self._intensities,
            self._com_fitted_x,
            self._com_fitted_y,
        )

        self._positions_px = self._calculate_scan_positions_in_pixels(
            self._scan_positions
        )

        # Object Initialization
        if self._object is None:
            pad_x, pad_y = self._object_px_padding
            p, q = np.max(self._positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._object = xp.ones((p, q), dtype=xp.complex64)
        else:
            self._object = xp.asarray(self._object, dtype=xp.complex64)

        self._positions_px = xp.asarray(self._positions_px, dtype=xp.float32)
        self._positions = self._positions_px.copy()
        self._positions[:, 0] *= self.sampling[0]
        self._positions[:, 1] *= self.sampling[1]

        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self._positions_px_fractional = self._positions_px - xp.round(
            self._positions_px
        )

        # Vectorized Patches
        self._object_shape = self._object.shape
        self._set_vectorized_patch_indices()

        # Probe Initialization
        if self._probe is None:
            if self._vacuum_probe_intensity is not None:
                self._semiangle_cutoff = np.inf
                self._vacuum_probe_intensity = self._pad_vacuum_probe_intensity(
                    self._vacuum_probe_intensity, self._region_of_interest_shape
                )
                vacuum_probe_intensity = asnumpy(self._vacuum_probe_intensity)
                probe_x0, probe_y0 = get_CoM(vacuum_probe_intensity)
                shift_x = self._region_of_interest_shape[0] / 2 - probe_x0
                shift_y = self._region_of_interest_shape[1] / 2 - probe_y0
                self._vacuum_probe_intensity = xp.asarray(
                    get_shifted_ar(
                        vacuum_probe_intensity, shift_x, shift_y, bilinear=True
                    )
                )
            self._probe = (
                ComplexProbe(
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    energy=self._energy,
                    semiangle_cutoff=self._semiangle_cutoff,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    parameters=self._polar_parameters,
                    device="cpu" if xp is np else "gpu",
                )
                .build()
                ._array
            )

        else:
            if isinstance(self._probe, ComplexProbe):
                if self._probe._gpts != self._region_of_interest_shape:
                    raise ValueError()
                if hasattr(self._probe, "_array"):
                    self._probe = self._probe._array
                else:
                    self._probe._xp = xp
                    self._probe = self._probe.build()._array
            else:
                self._probe = xp.asarray(self._probe, dtype=xp.complex64)

        # Normalize probe to match mean diffraction intensity
        probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probe)) ** 2)
        self._probe *= np.sqrt(self._mean_diffraction_intensity / probe_intensity)

        if plot_probe_overlaps:

            shifted_probes = fft_shift(self._probe, self._positions_px_fractional, xp)
            probe_intensities = xp.abs(shifted_probes) ** 2
            probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)

            figsize = kwargs.get("figsize", (8, 8))
            cmap = kwargs.get("cmap", "gray")
            kwargs.pop("figsize", None)
            kwargs.pop("cmap", None)

            extent = [
                0,
                self.sampling[0] * self._object_shape[0],
                0,
                self.sampling[1] * self._object_shape[1],
            ]
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(asnumpy(probe_overlap.T), extent=extent, cmap=cmap, origin='lower', **kwargs)
            ax.scatter(
                asnumpy(self._positions[:, 0]),
                asnumpy(self._positions[:, 1]),
                s=2.5,
                color=(1, 0, 0, 1),
            )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            plt.show()

        self._preprocessed = True

        return self

    def _overlap_projection(self, current_object, current_probe):
        """ """

        xp = self._xp

        self._shifted_probes = fft_shift(
            current_probe, self._positions_px_fractional, xp
        )
        exit_waves = (
            current_object[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        return xp.fft.fft2(exit_waves)

    def _fourier_projection(self, amplitudes, fourier_exit_waves):
        """ """

        xp = self._xp
        abs_fourier_exit_waves = xp.abs(fourier_exit_waves)
        error = (
            xp.mean(xp.abs(amplitudes - abs_fourier_exit_waves) ** 2)
            / self._mean_diffraction_intensity
        )

        difference_gradient_fourier = (amplitudes - abs_fourier_exit_waves) * xp.exp(
            1j * xp.angle(fourier_exit_waves)
        )

        return difference_gradient_fourier, error

    def _forward(self, current_object, current_probe, amplitudes):
        """ """

        fourier_exit_waves = self._overlap_projection(current_object, current_probe)
        difference_gradient_fourier, error = self._fourier_projection(
            amplitudes, fourier_exit_waves
        )

        return difference_gradient_fourier, error

    def _adjoint(
        self,
        current_object,
        current_probe,
        difference_gradient_fourier,
        fix_probe: bool,
        normalization_min: float,
    ):
        """ """

        xp = self._xp

        difference_gradient = xp.fft.ifft2(difference_gradient_fourier)

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(self._shifted_probes) ** 2
        )
        probe_normalization = 1 / xp.sqrt(
            probe_normalization**2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        object_update = (
            self._sum_overlapping_patches_bincounts(
                xp.conj(self._shifted_probes) * difference_gradient
            )
            * probe_normalization
        )

        if fix_probe:
            probe_update = None
        else:
            object_normalization = xp.sum(
                (xp.abs(current_object) ** 2)[
                    self._vectorized_patch_indices_row,
                    self._vectorized_patch_indices_col,
                ],
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                object_normalization**2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            probe_update = (
                xp.sum(
                    xp.conj(current_object)[
                        self._vectorized_patch_indices_row,
                        self._vectorized_patch_indices_col,
                    ]
                    * difference_gradient,
                    axis=0,
                )
                * object_normalization
            )

        return object_update, probe_update

    def _update(
        self,
        current_object,
        object_update,
        current_probe,
        probe_update,
        step_size: float = 0.9,
    ):
        """ """
        current_object += step_size * object_update
        if probe_update is not None:
            current_probe += step_size * probe_update
        return current_object, current_probe

    def _threshold_object_constraint(self, current_object, pure_phase_object: bool):
        """ """
        xp = self._xp
        phase = xp.exp(1.0j * xp.angle(current_object))
        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(current_object), 1.0)
        return amplitude * phase

    def _constraints(self, current_object, current_probe, pure_phase_object: bool):
        """ """
        current_object = self._threshold_object_constraint(
            current_object, pure_phase_object
        )
        return current_object, current_probe

    def reconstruct(
        self,
        max_iter: int = 64,
        step_size: float = 0.9,
        normalization_min: float = 1e-4,
        warmup_iter: int = 12,
        pure_phase_object_iter: int = 64,
        progress_bar: bool = True,
        store_iterations: bool = False,
    ):
        """ """
        xp = self._xp
        asnumpy = self._asnumpy

        # initialization
        if store_iterations:
            self._object_iterations = []
            self._probe_iterations = []
            self._error_iterations = []

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing object and probe",
            unit=" iter",
            disable=not progress_bar,
        ):

            # forward operator
            difference_gradient_fourier, error = self._forward(
                self._object, self._probe, self._intensities
            )

            # adjoint operator
            object_update, probe_update = self._adjoint(
                self._object,
                self._probe,
                difference_gradient_fourier,
                fix_probe=a0 < warmup_iter,
                normalization_min=normalization_min,
            )

            # update
            self._object, self._probe = self._update(
                self._object,
                object_update,
                self._probe,
                probe_update,
                step_size=step_size,
            )

            # constraints
            self._object, self._probe = self._constraints(
                self._object, self._probe, pure_phase_object=a0 < pure_phase_object_iter
            )

            if store_iterations:
                self._object_iterations.append(self._object.copy())
                self._probe_iterations.append(self._probe.copy())
                self._error_iterations.append(error.item())

        # store result
        self.object = asnumpy(self._object)
        self.probe = asnumpy(self._probe)
        self.error = error.item()

        return self

    def _show_last_iteration(
        self, cbar: bool, plot_convergence: bool, plot_probe, object_mode: str, **kwargs
    ):
        """ """
        figsize = kwargs.get("figsize", (5, 5))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        if plot_convergence:
            figsize = (figsize[0], figsize[1] + figsize[0] / 4)
            if plot_probe:
                figsize = (figsize[0] * 2, figsize[1])
                spec = GridSpec(ncols=2, nrows=2, height_ratios=[4, 1], hspace=0.1)
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.1)
        else:
            if plot_probe:
                figsize = (figsize[0] * 2, figsize[1])
                spec = GridSpec(ncols=2, nrows=1)
            else:
                spec = GridSpec(ncols=1, nrows=1)

        extent = [
            0,
            self.sampling[0] * self._object_shape[0],
            self.sampling[1] * self._object_shape[1],
            0,
        ]

        fig = plt.figure(figsize=figsize)

        if plot_probe:

            # Object
            ax = fig.add_subplot(spec[0, 0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(self.object.T), extent=extent, cmap=cmap, **kwargs
                )
                ax.set_title(f"Reconstructed Object Phase")
            elif object_mode == "amplitude":
                im = ax.imshow(np.abs(self.object.T), extent=extent, cmap=cmap, **kwargs)
                ax.set_title(f"Reconstructed Object Amplitude")
            else:
                im = ax.imshow(
                    np.abs(self.object.T) ** 2, extent=extent, cmap=cmap, **kwargs
                )
                ax.set_title(f"Reconstructed Object Intensity")
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Probe
            probe_extent = [
                0,
                self.sampling[0] * self._region_of_interest_shape[0],
                self.sampling[1] * self._region_of_interest_shape[1],
                0,
            ]

            ax = fig.add_subplot(spec[0, 1])
            im = ax.imshow(
                np.abs(self.probe) ** 2, extent=probe_extent, cmap="Greys_r", **kwargs
            )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Reconstructed Probe Intensity")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        else:
            ax = fig.add_subplot(spec[0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(self.object.T), extent=extent, cmap=cmap, **kwargs
                )
                ax.set_title(f"Reconstructed Object Phase")
            elif object_mode == "amplitude":
                im = ax.imshow(np.abs(self.object.T), extent=extent, cmap=cmap, **kwargs)
                ax.set_title(f"Reconstructed Object Amplitude")
            else:
                im = ax.imshow(
                    np.abs(self.object.T) ** 2, extent=extent, cmap=cmap, **kwargs
                )
                ax.set_title(f"Reconstructed Object Intensity")
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "_error_iterations"):
            errors = self._error_iterations
            if plot_probe:
                ax = fig.add_subplot(spec[1, :])
            else:
                ax = fig.add_subplot(spec[1])

            ax.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Log RMS error")
            ax.yaxis.tick_right()

        fig.suptitle(f"RMS error: {self.error:.3e}")
        plt.show()

    def _show_all_iterations(
        self,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        iterations_grid: Tuple[int, int],
        object_mode: str,
        **kwargs,
    ):
        """ """
        asnumpy = self._asnumpy

        if iterations_grid == "auto":
            iterations_grid = (2, 4)
        figsize = kwargs.get("figsize", (13, 6.5))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        if plot_probe:
            total_grids = (np.prod(iterations_grid) / 2).astype("int")
        else:
            total_grids = np.prod(iterations_grid)
        errors = self._error_iterations
        objects = self._object_iterations
        if plot_probe:
            probes = self._probe_iterations
        max_iter = len(objects) - 1

        if plot_convergence and not plot_probe:
            grid_range = range(0, max_iter, max_iter // (total_grids - 2))
        else:
            grid_range = range(0, max_iter, max_iter // (total_grids - 1))

        if plot_probe:
            grid_range = np.tile(grid_range, 2)
            if plot_convergence:
                grid_range = grid_range[:-1]

        extent = [
            0,
            self._scan_sampling[0] * self._intensities_shape[0],
            self._scan_sampling[1] * self._intensities_shape[1],
            0,
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
                if n // (total_grids) and plot_probe:
                    im = ax.imshow(
                        np.abs(asnumpy(probes[grid_range[n]].T)) ** 2,
                        extent=extent,
                        cmap="Greys_r",
                        **kwargs,
                    )
                    ax.set_title(f"Iter: {grid_range[n]} Probe")
                else:
                    if object_mode == "phase":
                        im = ax.imshow(
                            np.angle(asnumpy(objects[grid_range[n]].T)),
                            extent=extent,
                            cmap=cmap,
                            **kwargs,
                        )
                        ax.set_title(f"Iter: {grid_range[n]} Phase")
                    elif object_mode == "amplitude":
                        im = ax.imshow(
                            np.abs(asnumpy(objects[grid_range[n]].T)),
                            extent=extent,
                            cmap=cmap,
                            **kwargs,
                        )
                        ax.set_title(f"Iter: {grid_range[n]} Amplitude")
                    else:
                        im = ax.imshow(
                            np.abs(asnumpy(objects[grid_range[n]].T)) ** 2,
                            extent=extent,
                            cmap=cmap,
                            **kwargs,
                        )
                        ax.set_title(f"Iter: {grid_range[n]} Intensity")

                    if cbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad="2.5%")
                        fig.colorbar(im, cax=cax, orientation="vertical")

                ax.set_xlabel("x [A]")
                ax.set_ylabel("y [A]")

        fig.tight_layout()

    def show(
        self,
        plot_convergence: bool = False,
        iterations_grid: Tuple[int, int] = None,
        cbar: bool = False,
        plot_probe: bool = True,
        object_mode: str = "phase",
        **kwargs,
    ):
        """
        Displays reconstructed phase object.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        object_mode: str
            Specifies the attribute of the object to plot, one of 'phase', 'amplitude', 'intensity'

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """

        if (
            object_mode != "phase"
            and object_mode != "amplitude"
            and object_mode != "intensity"
        ):
            raise ValueError(
                f"object_mode needs to be one of 'phase', 'amplitude', or 'intensity', not {object_mode}"
            )

        if iterations_grid is None:
            self._show_last_iteration(
                plot_convergence=plot_convergence,
                plot_probe=plot_probe,
                object_mode=object_mode,
                cbar=cbar,
                **kwargs,
            )
        else:
            self._show_all_iterations(
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                plot_probe=plot_probe,
                object_mode=object_mode,
                cbar=cbar,
                **kwargs,
            )

        return self
