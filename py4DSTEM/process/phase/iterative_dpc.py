"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC.
"""

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

try:
    import cupy as cp
except ImportError:
    cp = None

from emdfile import tqdmnd
from py4DSTEM.classes import DataCube
from py4DSTEM.process.phase.iterative_base_class import PhaseReconstruction

warnings.simplefilter(action="always", category=UserWarning)


class DPCReconstruction(PhaseReconstruction):
    """
    Iterative Differential Phase Constrast Reconstruction Class.

    Diffraction intensities dimensions         : (Rx,Ry,Qx,Qy)
    Reconstructed phase object dimensions      : (Rx,Ry)

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    dp_mask: ndarray, optional
        Mask for datacube intensities (Qx,Qy)
    energy: float, optional
        The electron energy of the wave functions in eV
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
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
        dp_mask: np.ndarray = None,
        energy: float = None,
        verbose: bool = True,
        device: str = "cpu",
    ):
        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
            from scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
            from cupyx.scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
        else:
            raise ValueError(f"device must be either 'cpu' or 'gpu', not {device}")

        self._energy = energy
        self._datacube = datacube
        self._verbose = verbose
        self._preprocessed = False
        self._dp_mask = dp_mask

    def preprocess(
        self,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        maximize_divergence: bool = False,
        fit_function: str = "plane",
        force_com_rotation: float = None,
        force_com_transpose: bool = None,
        force_com_shifts: float = None,
        plot_center_of_mass: str = "default",
        plot_rotation: bool = True,
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
        rotation_angles_deg: np.darray, optional
            Array of angles in degrees to perform curl minimization over
        maximize_divergence: bool, optional
            If True, the divergence of the CoM gradient vector field is maximized
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
        force_ com_rotation: float (degrees), optional
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool (optional)
            Force whether diffraction intensities need to be transposed.
        force_com_shifts: tuple of ndarrays (CoMx, CoMy)
            Force CoM fitted shifts
        plot_center_of_mass: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed

        Mutates
        --------
        self._preprocessed: bool
            Flag to signal object has been preprocessed

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        self._intensities = self._extract_intensities_and_calibrations_from_datacube(
            self._datacube,
            require_calibrations=False,
        )

        (
            self._com_measured_x,
            self._com_measured_y,
            self._com_fitted_x,
            self._com_fitted_y,
            self._com_normalized_x,
            self._com_normalized_y,
        ) = self._calculate_intensities_center_of_mass(
            self._intensities,
            dp_mask=self._dp_mask,
            fit_function=fit_function,
            com_shifts=force_com_shifts,
        )

        (
            self._rotation_best_rad,
            self._rotation_best_transpose,
            self._com_x,
            self._com_y,
            self.com_x,
            self.com_y,
        ) = self._solve_for_center_of_mass_relative_rotation(
            self._com_measured_x,
            self._com_measured_y,
            self._com_normalized_x,
            self._com_normalized_y,
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            plot_center_of_mass=plot_center_of_mass,
            maximize_divergence=maximize_divergence,
            force_com_rotation=force_com_rotation,
            force_com_transpose=force_com_transpose,
            **kwargs,
        )
        self._preprocessed = True

        return self

    def _forward(
        self,
        padded_phase_object: np.ndarray,
        mask: np.ndarray,
        mask_inv: np.ndarray,
        error: float,
        step_size: float,
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
        error: float
            Current error estimate
        step_size: float
            Current reconstruction step-size

        Returns
        --------
        obj_dx: np.ndarray
            Forward-projected horizontal CoM gradient
        obj_dy: np.ndarray
            Forward-projected vertical CoM gradient
        error: float
            Updated estimate error
        step_size: float
            Updated reconstruction step-size. Halved if error increased.
        """

        xp = self._xp
        dx, dy = self._scan_sampling

        # centered finite-differences
        obj_dx = (
            xp.roll(padded_phase_object, 1, axis=0)
            - xp.roll(padded_phase_object, -1, axis=0)
        ) / (2 * dx)
        obj_dy = (
            xp.roll(padded_phase_object, 1, axis=1)
            - xp.roll(padded_phase_object, -1, axis=1)
        ) / (2 * dy)

        # difference from measurement
        obj_dx[mask] += self._com_x.ravel()
        obj_dy[mask] += self._com_y.ravel()
        obj_dx[mask_inv] = 0
        obj_dy[mask_inv] = 0

        new_error = xp.mean(obj_dx[mask] ** 2 + obj_dy[mask] ** 2) / (
            xp.mean(self._com_x.ravel() ** 2 + self._com_y.ravel() ** 2)
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

    def _constraints(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma,
        butterworth_filter,
        q_lowpass,
        q_highpass,
    ):
        """
        DPC constraints operator.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_filter: bool
            If True, applies real-space gaussian filter
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel
        butterworth_filter: bool
            If True, applies high-pass butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter


        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma, False
            )

        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
            )
            current_object = xp.real(current_object)

        return current_object

    def reconstruct(
        self,
        reset: bool = None,
        padding_factor: float = 2,
        max_iter: int = 64,
        step_size: float = 1.0,
        stopping_criterion: float = 1e-6,
        progress_bar: bool = True,
        gaussian_filter_sigma: float = None,
        gaussian_filter_iter: int = np.inf,
        butterworth_filter_iter: int = np.inf,
        q_lowpass: float = None,
        q_highpass: float = None,
        store_iterations: bool = False,
    ):
        """
        Performs Iterative DPC Reconstruction:

        Parameters
        ----------
        reset: bool, optional
            If True, previous reconstructions are ignored
        padding_factor: float, optional
            Factor to pad object by to reduce periodic artifacts
        max_iter: int, optional
            Maximum number of iterations
        step_size: float, optional
            Reconstruction update step size
        stopping_criterion: float, optional
            step_size below which reconstruction exits
        progress_bar: bool, optional
            If True, reconstruction progress bar will be printed
        gaussian_filter_sigma: float, optional
            Standard deviation of gaussian kernel
        gaussian_filter_iter: int, optional
            Number of iterations to run using object smoothness constraint
        butterworth_filter_iter: int, optional
            Number of iterations to run using high-pass butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        store_iterations: bool, optional
            If True, all reconstruction iterations will be stored

        Assigns
        --------
        self._object_phase: xp.ndarray
            Reconstructed phase object, on calculation device
        self.object_phase: np.ndarray
            Reconstructed phase object, as a numpy array
        self.error: float
            NMSE error
        self.object_phase_iterations, optional
            Reconstructed phase objects at each iteration as numpy arrays
        self.error_iterations, optional
            NMSE errors at each iteration

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # Initialization
        padded_object_shape = np.round(
            np.array(self._grid_scan_shape) * padding_factor
        ).astype("int")
        mask = xp.zeros(padded_object_shape, dtype="bool")
        mask[: self._grid_scan_shape[0], : self._grid_scan_shape[1]] = True
        mask_inv = xp.logical_not(mask)

        # Fourier coordinates and operators
        kx = xp.fft.fftfreq(padded_object_shape[0], d=self._scan_sampling[0])
        ky = xp.fft.fftfreq(padded_object_shape[1], d=self._scan_sampling[1])
        kya, kxa = xp.meshgrid(ky, kx)
        k_den = kxa**2 + kya**2
        k_den[0, 0] = np.inf
        k_den = 1 / k_den
        kx_op = -1j * 0.25 * kxa * k_den
        ky_op = -1j * 0.25 * kya * k_den

        if reset is None and hasattr(self, "error"):
            warnings.warn(
                (
                    "Continuing reconstruction from previous result. "
                    "Use reset=True for a fresh start."
                ),
                UserWarning,
            )

        # Restart
        if not hasattr(self, "_padded_phase_object") or reset:
            self.error = np.inf
            self._step_size = step_size
            self._padded_phase_object = xp.zeros(padded_object_shape)

        if store_iterations and (not hasattr(self, "object_phase_iterations") or reset):
            self.object_phase_iterations = []
            self.error_iterations = []

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing phase",
            unit=" iter",
            disable=not progress_bar,
        ):
            if self._step_size < stopping_criterion:
                break

            # forward operator
            com_dx, com_dy, self.error, self._step_size = self._forward(
                self._padded_phase_object, mask, mask_inv, self.error, self._step_size
            )

            # adjoint operator
            phase_update = self._adjoint(com_dx, com_dy, kx_op, ky_op)

            # update
            self._padded_phase_object = self._update(
                self._padded_phase_object, phase_update, self._step_size
            )

            # constraints
            (self._padded_phase_object) = self._constraints(
                self._padded_phase_object,
                gaussian_filter=a0 < gaussian_filter_iter
                and gaussian_filter_sigma is not None,
                gaussian_filter_sigma=gaussian_filter_sigma,
                butterworth_filter=a0 < butterworth_filter_iter
                and (q_lowpass is not None or q_highpass is not None),
                q_lowpass=q_lowpass,
                q_highpass=q_highpass,
            )

            if store_iterations:
                self.object_phase_iterations.append(
                    asnumpy(
                        self._padded_phase_object[
                            : self._grid_scan_shape[0], : self._grid_scan_shape[1]
                        ].copy()
                    )
                )
                self.error_iterations.append(self.error.item())

        if self._step_size < stopping_criterion:
            warnings.warn(
                f"Step-size has decreased below stopping criterion {stopping_criterion}.",
                UserWarning,
            )

        # crop result
        self._object_phase = self._padded_phase_object[
            : self._grid_scan_shape[0], : self._grid_scan_shape[1]
        ]
        self.object_phase = asnumpy(self._object_phase)

        return self

    def _visualize_last_iteration(self, cbar: bool, plot_convergence: bool, **kwargs):
        """
        Displays last iteration of reconstructed phase object.

        Parameters
        --------
        cbar: bool, optional
            If true, displays a colorbar
        plot_convergence: bool, optional
            If true, the NMSE error plot is displayed
        """

        figsize = kwargs.get("figsize", (8, 8))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        if plot_convergence:
            spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            spec = GridSpec(ncols=1, nrows=1)
        fig = plt.figure(figsize=figsize)

        extent = [
            0,
            self._scan_sampling[1] * self._grid_scan_shape[1],
            self._scan_sampling[0] * self._grid_scan_shape[0],
            0,
        ]

        ax1 = fig.add_subplot(spec[0])
        im = ax1.imshow(self.object_phase, extent=extent, cmap=cmap, **kwargs)
        ax1.set_ylabel(f"x [{self._scan_units[0]}]")
        ax1.set_xlabel(f"y [{self._scan_units[1]}]")
        ax1.set_title(f"DPC Phase Reconstruction - NMSE error: {self.error:.3e}")

        if cbar:
            divider = make_axes_locatable(ax1)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "_error_iterations"):
            errors = self._error_iterations
            ax2 = fig.add_subplot(spec[1])
            ax2.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax2.set_xlabel("Iteration Number")
            ax2.set_ylabel("Log NMSE error")
            ax2.yaxis.tick_right()

        spec.tight_layout(fig)

    def _visualize_all_iterations(
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
            If true, the NMSE error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        """

        if iterations_grid == "auto":
            iterations_grid = (2, 4)

        figsize = kwargs.get("figsize", (12, 7))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        total_grids = np.prod(iterations_grid)
        errors = self.error_iterations
        phases = self.object_phase_iterations
        max_iter = len(phases) - 1
        grid_range = range(0, max_iter + 1, max_iter // (total_grids - 1))

        extent = [
            0,
            self._scan_sampling[1] * self._grid_scan_shape[1],
            self._scan_sampling[0] * self._grid_scan_shape[0],
            0,
        ]

        if plot_convergence:
            spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            spec = GridSpec(ncols=1, nrows=1)
        fig = plt.figure(figsize=figsize)

        grid = ImageGrid(
            fig,
            spec[0],
            nrows_ncols=iterations_grid,
            axes_pad=(0.75, 0.5) if cbar else 0.5,
            cbar_mode="each" if cbar else None,
            cbar_pad="2.5%" if cbar else None,
        )

        for n, ax in enumerate(grid):
            im = ax.imshow(
                phases[grid_range[n]],
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_ylabel(f"x [{self._scan_units[0]}]")
            ax.set_xlabel(f"y [{self._scan_units[1]}]")
            if cbar:
                grid.cbar_axes[n].colorbar(im)
            ax.set_title(
                f"Iteration: {grid_range[n]}\nNMSE error: {errors[grid_range[n]]:.3e}"
            )

        if plot_convergence:
            ax2 = fig.add_subplot(spec[1])
            ax2.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax2.set_xlabel("Iteration Number")
            ax2.set_ylabel("Log NMSE error")
            ax2.yaxis.tick_right()

            spec.tight_layout(fig)

    def visualize(
        self,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        cbar: bool = False,
        **kwargs,
    ):
        """
        Displays reconstructed phase object.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the NMSE error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        if iterations_grid is None:
            self._visualize_last_iteration(
                plot_convergence=plot_convergence, cbar=cbar, **kwargs
            )
        else:
            self._visualize_all_iterations(
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                cbar=cbar,
                **kwargs,
            )

        return self

    @property
    def sampling(self):
        """Sampling [Å]"""

        return self._scan_sampling
