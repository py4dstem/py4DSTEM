"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC.
"""

import warnings
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

try:
    import cupy as cp
except ImportError:
    cp = None

from emdfile import Array, Custom, Metadata, _read_metadata, tqdmnd

from py4DSTEM.data import Calibration
from py4DSTEM.datacube import DataCube
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
    initial_object_guess: np.ndarray, optional
        Cropped initial guess of dpc phase
    energy: float, optional
        The electron energy of the wave functions in eV
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    name: str, optional
        Class name
    """

    def __init__(
        self,
        datacube: DataCube = None,
        initial_object_guess: np.ndarray = None,
        energy: float = None,
        verbose: bool = True,
        device: str = "cpu",
        name: str = "dpc_reconstruction",
    ):
        Custom.__init__(self, name=name)

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

        self.set_save_defaults()

        # Data
        self._datacube = datacube
        self._object_phase = initial_object_guess

        # Metadata
        self._energy = energy
        self._verbose = verbose
        self._device = device
        self._preprocessed = False

    def to_h5(self, group):
        """
        Wraps datasets and metadata to write in emdfile classes,
        notably: the object phase array.
        """

        # instantiation metadata
        self.metadata = Metadata(
            name="instantiation_metadata",
            data={
                "energy": self._energy,
                "verbose": self._verbose,
                "device": self._device,
                "name": self.name,
            },
        )

        # preprocessing metadata
        self.metadata = Metadata(
            name="preprocess_metadata",
            data={
                "rotation_angle_rad": self._rotation_best_rad,
                "data_transpose": self._rotation_best_transpose,
                "sampling": self.sampling,
            },
        )

        # reconstruction metadata
        is_stack = self._save_iterations and hasattr(self, "object_phase_iterations")
        if is_stack:
            num_iterations = len(self.object_phase_iterations)
            iterations = list(range(0, num_iterations, self._save_iterations_frequency))
            if num_iterations - 1 not in iterations:
                iterations.append(num_iterations - 1)

            error = [self.error_iterations[i] for i in iterations]
        else:
            error = self.error

        self.metadata = Metadata(
            name="reconstruction_metadata",
            data={
                "reconstruction_error": error,
                "final_step_size": self._step_size,
            },
        )

        if is_stack:
            iterations_labels = [f"iteration_{i:03}" for i in iterations]

            # object
            object_iterations = [
                np.asarray(self.object_phase_iterations[i]) for i in iterations
            ]
            self._object_emd = Array(
                name="reconstruction_object",
                data=np.stack(object_iterations, axis=0),
                slicelabels=iterations_labels,
            )

        else:
            # object
            self._object_emd = Array(
                name="reconstruction_object",
                data=self._asnumpy(self._object_phase),
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

        # Fix calibrations bug
        if "_datacube" in dict_data:
            calibrations_dict = _read_metadata(group, "calibration")._params
            cal = Calibration()
            cal._params.update(calibrations_dict)
            dc = dict_data["_datacube"]
            dc.calibration = cal
        else:
            dc = None

        # Check if stack
        if dict_data["_object_emd"].is_stack:
            obj = dict_data["_object_emd"][-1].data
        else:
            obj = dict_data["_object_emd"].data

        # Populate args and return
        kwargs = {
            "datacube": dc,
            "initial_object_guess": np.asarray(obj),
            "energy": instance_md["energy"],
            "verbose": instance_md["verbose"],
            "name": instance_md["name"],
            "device": instance_md["device"],
        }

        return kwargs

    def _populate_instance(self, group):
        """
        Sets post-initialization properties, notably some preprocessing meta
        optional; during read, this method is run after object instantiation.
        """
        # Preprocess metadata
        preprocess_md = _read_metadata(group, "preprocess_metadata")
        self._rotation_best_rad = preprocess_md["rotation_angle_rad"]
        self._rotation_best_transpose = preprocess_md["data_transpose"]
        self._preprocessed = False

        # Reconstruction metadata
        reconstruction_md = _read_metadata(group, "reconstruction_metadata")
        error = reconstruction_md["reconstruction_error"]

        # Data
        dict_data = Custom._get_emd_attr_data(Custom, group)

        # Check if stack
        if hasattr(error, "__len__"):
            self.object_phase_iterations = list(dict_data["_object_emd"].data)
            self.error_iterations = error
            self.error = error[-1]
        else:
            self.error = error

        self._step_size = reconstruction_md["final_step_size"]

    def preprocess(
        self,
        dp_mask: np.ndarray = None,
        padding_factor: float = 2,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        maximize_divergence: bool = False,
        fit_function: str = "plane",
        force_com_rotation: float = None,
        force_com_transpose: bool = None,
        force_com_shifts: Union[Sequence[np.ndarray], Sequence[float]] = None,
        force_com_measured: Sequence[np.ndarray] = None,
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
        dp_mask: ndarray, optional
            Mask for datacube intensities (Qx,Qy)
        padding_factor: float, optional
            Factor to pad object by to reduce periodic artifacts
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
        force_com_measured: tuple of ndarrays (CoMx measured, CoMy measured)
            Force CoM measured shifts
        plot_center_of_mass: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """
        xp = self._xp

        # set additional metadata
        self._dp_mask = dp_mask

        if self._datacube is None:
            if force_com_measured is None:
                raise ValueError(
                    (
                        "The preprocess() method requires either a DataCube "
                        "or `force_com_measured`. "
                        "Please run dpc.attach_datacube(DataCube) to attach DataCube."
                    )
                )
            else:
                self._datacube = DataCube(
                    data=np.empty(force_com_measured[0].shape + (1, 1))
                )

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
            com_measured=force_com_measured,
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

        # Object Initialization
        padded_object_shape = np.round(
            np.array(self._grid_scan_shape) * padding_factor
        ).astype("int")
        self._padded_object_phase = xp.zeros(padded_object_shape, dtype=xp.float32)
        if self._object_phase is not None:
            self._padded_object_phase[
                : self._grid_scan_shape[0], : self._grid_scan_shape[1]
            ] = xp.asarray(self._object_phase, dtype=xp.float32)

        self._padded_object_phase_initial = self._padded_object_phase.copy()

        # Fourier coordinates and operators
        kx = xp.fft.fftfreq(padded_object_shape[0], d=self._scan_sampling[0])
        ky = xp.fft.fftfreq(padded_object_shape[1], d=self._scan_sampling[1])
        kya, kxa = xp.meshgrid(ky, kx)
        k_den = kxa**2 + kya**2
        k_den[0, 0] = np.inf
        k_den = 1 / k_den
        self._kx_op = -1j * 0.25 * kxa * k_den
        self._ky_op = -1j * 0.25 * kya * k_den

        self._preprocessed = True

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

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
        updated_padded_object_phase: np.ndarray
            Updated padded phase object estimate
        """

        padded_phase_object += step_size * phase_update
        return padded_phase_object

    def _object_gaussian_constraint(self, current_object, gaussian_filter_sigma):
        """
        Smoothness constrain used for blurring object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        gaussian_filter = self._gaussian_filter

        gaussian_filter_sigma /= self.sampling[0]
        current_object = gaussian_filter(current_object, gaussian_filter_sigma)

        return current_object

    def _object_butterworth_constraint(
        self, current_object, q_lowpass, q_highpass, butterworth_order
    ):
        """
        Butterworth filter used for low/high-pass filtering.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        qx = xp.fft.fftfreq(current_object.shape[0], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[1], self.sampling[1])

        qya, qxa = xp.meshgrid(qy, qx)
        qra = xp.sqrt(qxa**2 + qya**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object)
        current_object -= current_object_mean
        current_object = xp.fft.ifft2(xp.fft.fft2(current_object) * env)
        current_object += current_object_mean

        return xp.real(current_object)

    def _object_anti_gridding_contraint(self, current_object):
        """
        Zero outer pixels of object fft to remove gridding artifacts

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        # find indices to zero
        width_x = current_object.shape[0]
        width_y = current_object.shape[1]
        ind_min_x = int(xp.floor(width_x / 2) - 2)
        ind_max_x = int(xp.ceil(width_x / 2) + 2)
        ind_min_y = int(xp.floor(width_y / 2) - 2)
        ind_max_y = int(xp.ceil(width_y / 2) + 2)

        # zero pixels
        object_fft = xp.fft.fft2(current_object)
        object_fft[ind_min_x:ind_max_x] = 0
        object_fft[:, ind_min_y:ind_max_y] = 0

        return xp.real(xp.fft.ifft2(object_fft))

    def _constraints(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma,
        butterworth_filter,
        q_lowpass,
        q_highpass,
        butterworth_order,
        anti_gridding,
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
            Standard deviation of gaussian kernel in A
        butterworth_filter: bool
            If True, applies high-pass butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        anti_gridding: bool
            If true, zero outer pixels of object fft to remove
            gridding artifacts

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma
            )

        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )

        if anti_gridding:
            current_object = self._object_anti_gridding_contraint(
                current_object,
            )

        return current_object

    def reconstruct(
        self,
        reset: bool = None,
        max_iter: int = 64,
        step_size: float = None,
        stopping_criterion: float = 1e-6,
        backtrack: bool = True,
        progress_bar: bool = True,
        gaussian_filter_sigma: float = None,
        gaussian_filter_iter: int = np.inf,
        butterworth_filter_iter: int = np.inf,
        q_lowpass: float = None,
        q_highpass: float = None,
        butterworth_order: float = 2,
        anti_gridding: float = True,
        store_iterations: bool = False,
    ):
        """
        Performs Iterative DPC Reconstruction:

        Parameters
        ----------
        reset: bool, optional
            If True, previous reconstructions are ignored
        max_iter: int, optional
            Maximum number of iterations
        step_size: float, optional
            Reconstruction update step size
        stopping_criterion: float, optional
            step_size below which reconstruction exits
        backtrack: bool, optional
            If True, steps that increase the error metric are rejected
            and iteration continues with a reduced step size from the
            previous iteration
        progress_bar: bool, optional
            If True, reconstruction progress bar will be printed
        gaussian_filter_sigma: float, optional
            Standard deviation of gaussian kernel in A
        gaussian_filter_iter: int, optional
            Number of iterations to run using object smoothness constraint
        butterworth_filter_iter: int, optional
            Number of iterations to run using high-pass butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        anti_gridding: bool
            If true, zero outer pixels of object fft to remove
            gridding artifacts
        store_iterations: bool, optional
            If True, all reconstruction iterations will be stored

        Returns
        --------
        self: DPCReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if reset is None and hasattr(self, "error"):
            warnings.warn(
                (
                    "Continuing reconstruction from previous result. "
                    "Use reset=True for a fresh start."
                ),
                UserWarning,
            )

        # Restart
        if store_iterations and (not hasattr(self, "object_phase_iterations") or reset):
            self.object_phase_iterations = []
            self.error_iterations = []

        if reset:
            self.error = np.inf
            self._step_size = step_size if step_size is not None else 0.5
            self._padded_object_phase = self._padded_object_phase_initial.copy()

        self.error = getattr(self, "error", np.inf)

        if step_size is None:
            self._step_size = getattr(self, "_step_size", 0.5)
        else:
            self._step_size = step_size

        mask = xp.zeros(self._padded_object_phase.shape, dtype="bool")
        mask[: self._grid_scan_shape[0], : self._grid_scan_shape[1]] = True
        mask_inv = xp.logical_not(mask)

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing phase",
            unit=" iter",
            disable=not progress_bar,
        ):
            if self._step_size < stopping_criterion:
                break

            previous_iteration = self._padded_object_phase.copy()

            # forward operator
            com_dx, com_dy, new_error, self._step_size = self._forward(
                self._padded_object_phase, mask, mask_inv, self.error, self._step_size
            )

            # if the error went up after the previous step, go back to the step
            # before the error rose and continue with the halved step size
            if (new_error > self.error) and backtrack:
                self._padded_object_phase = previous_iteration
                self._step_size /= 2
                print(f"Iteration {a0}, step reduced to {self._step_size}")
                continue
            self.error = new_error

            # adjoint operator
            phase_update = self._adjoint(com_dx, com_dy, self._kx_op, self._ky_op)

            # update
            self._padded_object_phase = self._update(
                self._padded_object_phase, phase_update, self._step_size
            )

            # constraints
            self._padded_object_phase = self._constraints(
                self._padded_object_phase,
                gaussian_filter=a0 < gaussian_filter_iter
                and gaussian_filter_sigma is not None,
                gaussian_filter_sigma=gaussian_filter_sigma,
                butterworth_filter=a0 < butterworth_filter_iter
                and (q_lowpass is not None or q_highpass is not None),
                q_lowpass=q_lowpass,
                q_highpass=q_highpass,
                butterworth_order=butterworth_order,
                anti_gridding=anti_gridding,
            )

            if store_iterations:
                self.object_phase_iterations.append(
                    asnumpy(
                        self._padded_object_phase[
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
        self._object_phase = self._padded_object_phase[
            : self._grid_scan_shape[0], : self._grid_scan_shape[1]
        ]
        self.object_phase = asnumpy(self._object_phase)

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def _visualize_last_iteration(
        self, fig, cbar: bool, plot_convergence: bool, **kwargs
    ):
        """
        Displays last iteration of reconstructed phase object.

        Parameters
        --------
        fig, optional
            Matplotlib figure to draw Gridspec on
        cbar: bool, optional
            If true, displays a colorbar
        plot_convergence: bool, optional
            If true, the NMSE error plot is displayed
        """

        figsize = kwargs.pop("figsize", (8, 8))
        cmap = kwargs.pop("cmap", "magma")

        if plot_convergence:
            spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            spec = GridSpec(ncols=1, nrows=1)

        if fig is None:
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
        fig,
        cbar: bool,
        plot_convergence: bool,
        iterations_grid: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays last iteration of reconstructed phase object.

        Parameters
        --------
        fig, optional
            Matplotlib figure to draw Gridspec on
        cbar: bool, optional
            If true, displays a colorbar
        plot_convergence: bool, optional
            If true, the NMSE error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        """

        if not hasattr(self, "object_phase_iterations"):
            raise ValueError(
                (
                    "Object iterations were not saved during reconstruction. "
                    "Please re-run using store_iterations=True."
                )
            )

        if iterations_grid == "auto":
            num_iter = len(self.error_iterations)

            if num_iter == 1:
                return self._visualize_last_iteration(
                    plot_convergence=plot_convergence,
                    cbar=cbar,
                    **kwargs,
                )
            else:
                iterations_grid = (2, 4) if num_iter > 8 else (2, num_iter // 2)

        auto_figsize = (
            (3 * iterations_grid[1], 3 * iterations_grid[0] + 1)
            if plot_convergence
            else (3 * iterations_grid[1], 3 * iterations_grid[0])
        )
        figsize = kwargs.pop("figsize", auto_figsize)
        cmap = kwargs.pop("cmap", "magma")

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

        if fig is None:
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
        fig=None,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        cbar: bool = False,
        **kwargs,
    ):
        """
        Displays reconstructed phase object.

        Parameters
        ----------
        fig, optional
            Matplotlib figure to draw Gridspec on
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
                fig=fig, plot_convergence=plot_convergence, cbar=cbar, **kwargs
            )
        else:
            self._visualize_all_iterations(
                fig=fig,
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
