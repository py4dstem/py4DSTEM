from typing import Sequence, Tuple, Union

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from emdfile import tqdmnd

from py4DSTEM import show
from py4DSTEM.datacube import DataCube
from py4DSTEM.preprocess.utils import bin2D
from py4DSTEM.process.calibration import fit_origin, get_origin
from py4DSTEM.process.diffraction import Crystal
from py4DSTEM.process.phase.utils import copy_to_device
from py4DSTEM.utils import fourier_resample
from py4DSTEM.visualize import return_scaled_histogram_ordering

from scipy.ndimage import rotate, zoom
from scipy.spatial.transform import Rotation as R

from mpire import WorkerPool, cpu_count
from threadpoolctl import threadpool_limits


try:
    import cupy as cp

    get_array_module = cp.get_array_module
except (ImportError, ModuleNotFoundError):
    cp = None

    def get_array_module(*args):
        return np


class Tomography:
    """ """

    def __init__(
        self,
        datacubes: Union[Sequence[DataCube], Sequence[str]] = None,
        import_kwargs: dict = {},
        object_shape_x_y_z: Tuple = None,
        voxel_size_A: float = None,
        datacube_R_pixel_size_A: float = None,
        datacube_Q_pixel_size_inv_A: float = None,  # do we even need this?
        tilt_deg: Union[Sequence, np.ndarray] = None,
        shift_px: Union[Sequence, np.ndarray] = None,
        tilt_rotation_axis_angle_deg: float = None,
        tilt_rotation_axis_shift_px: float = None,
        transpose_xy: bool = False,
        initial_object_guess: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = "cpu",
        clear_fft_cache: bool = True,
        name: str = "tomography",
    ):
        """
        Nanobeam  tomography!

        Parameters
        ----------
        datacubes:List of `DataCube`s of list of strings
            list of py4DSTEM `DataCube`s or strings for where to load the data from
        import_kwargs: dict
            arguments to pass to `read` or `import_file` if passes a string for datacubes
        object_shape_x_y_z: 3-tuple
            shape (in voxels) of the 3D object to be reconstructed
        voxel_size_A: float
            size of each voxel in object. Voxel size is uniform in x, y, and z
        datacube_R_pixel_size_A: float
            step size (in A) for datacube
        datacube_Q_pixel_size_inv_A: float
            step size (in A^-1) for datacube
        tilt_deg: np.ndarray or list
            list of tilt angles for datacubes
        shift_px:
            list of x, y shifts for each scan. If transpose_xy is True, these are also transposed
        tilt_rotation_axis_angle_deg: float
            rotation angle of scan direction to tilt axis
        tilt_rotation_axis_shift_px: float
            shift of centering of datacubes so that the axis is centered in the object
            (in number of datacube pixels)
        transpose_xy: bool
            if True, swaps x and y
        initial_object_guess: np.ndarray
            initial guess for object
        verbose: bool
            if False, supresses output messages
        device: str, optional
            device calculation will be perfomed on. Must be 'cpu' or 'gpu'
        storage: str, optional
            device non-frequent arrays will be stored on. Must be 'cpu' or 'gpu'
        clear_fft_cache: bool, optional (TODO, not yet implemented)
            if True, and device = 'gpu', clears the cached fft plan at the end of function calls

        Other notes for users:
            - the tilt axis is along y
            - for a thin foil geometry, pass a smaller value for z
        """
        self._datacubes = datacubes
        self._import_kwargs = import_kwargs
        self._object_shape_x_y_z = object_shape_x_y_z
        self._voxel_size_A = voxel_size_A
        self._datacube_R_pixel_size_A_init = datacube_R_pixel_size_A
        self._datacube_Q_pixel_size_inv_A = datacube_Q_pixel_size_inv_A
        self._tilt_deg = tilt_deg
        self._shift_px = shift_px
        self._tilt_rotation_axis_angle_deg = tilt_rotation_axis_angle_deg
        self._tilt_rotation_axis_shift_px = tilt_rotation_axis_shift_px
        self._transpose_xy = transpose_xy
        self._verbose = verbose
        self._initial_object_guess = initial_object_guess

        self.set_device(device, clear_fft_cache)
        self.set_storage(storage)

    def preprocess(
        self,
        diffraction_intensities_shape: int = None,
        resizing_method: str = "bin",
        bin_real_space: int = None,
        crop_reciprocal_space: float = None,
        crop_real_space: float = None,
        q_max_inv_A: int = None,
        diffraction_space_mask_com=None,
        force_centering_shifts: Sequence[Tuple] = None,
        masks_real_space: Union[np.ndarray, Sequence[np.ndarray]] = None,
        r: float = None,
        rscale: float = 1.2,
        fast_center: bool = False,
        fitfunction: str = "plane",
        robust: bool = False,
        robust_steps: int = 3,
        robust_thresh: int = 2,
        force_q_to_r_rotation_deg=None,
        force_q_to_r_transpose=False,
        dp_shift_method="pixel",
        num_points: int = None,
        device: str = None,
        clear_fft_cache: bool = True,
        progress_bar: bool = True,
    ):
        """
        Preprocessing for nanobeam tomography

        Parameters
        ----------
        diffraction_intensites_shape: int
            shape of diffraction patterns to reshape data into
        resizing_method: float
            method to reshape diffraction space ("bin", "fourier", "bilinear")
        bin_real_space: int
            factor for binnning in real space
        crop_reciprocal_space: float
            if not None, crops reciprocal space on all sides by integer.
            Can pass an single integer of a 4-tuple
        crop_real_space: float
            if not None, crops real space on all sides by integer.
            Can pass an single integer of a 4-tuple
        q_max_inv_A: int
            maximum q in inverse angstroms
        diffraction_space_mask_com: np.ndarray
            applies mask to datacube while solving for CoM rotation
        force_centering_shifts: list of 2-tuples of np.ndarrays of Rshape
            forces the qx and qy shifts of diffraction patterns
        masks_real_space: list of np.ndarray or np.ndarray
            mask for real space. can be the same for each datacube of individually specified.
        r: (float or None)
            the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (float)
             expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        fast_center: (bool)
            skip the center of mass refinement step.
            arrays are returned for qx0,qy0
        force_q_to_r_rotation_deg:float
            force q to r rotation in degrees
            if `tilt_rotation_axis_angle_deg` is not None, this angle is added
        force_q_to_r_transpose: bool
            force q to r transpose
        fitfunction: "str"
            fit function for origin ('plane' or 'parabola' or 'bezier_two' or 'constant').
        robust: bool
            if set to True, origin fit will be repeated with outliers
            removed.
        robust_steps: int
            number of robust iterations performed after initial fit.
        robust_thresh: int
            threshold for including points, in units of root-mean-square (standard deviations) error
            of the predicted values after fitting.
        dp_shift_method: float
            method to shift diffraction patterns "subpixel" or "pixel"
        num_points: int
            number of points for bilinear interpolation in real space
        """
        self.set_device(device, clear_fft_cache)

        xp_storage = self._xp_storage
        storage = self._storage
        xp = self._xp

        self._num_datacubes = len(self._datacubes)

        self._diffraction_patterns_projected = []
        self._positions_vox = []
        self._positions_vox_F = []
        self._positions_vox_dF = []

        self._force_q_to_r_transpose = force_q_to_r_transpose
        self._force_q_to_r_rotation_deg = force_q_to_r_rotation_deg

        self._datacube_R_pixel_size_A = self._datacube_R_pixel_size_A_init

        if num_points is None:
            num_points = self._object_shape_x_y_z[2]
        self._num_points = num_points

        # preprocessing of diffraction data
        for a0 in tqdmnd(
            self._num_datacubes,
            desc="importing datacubes",
            unit=" iter",
            disable=not progress_bar,
        ):
            # load and preprocess datacube
            (datacube, mask_real_space, diffraction_space_mask_com, q_max_inv_A) = (
                self._prepare_datacube(
                    datacube_number=a0,
                    diffraction_intensities_shape=diffraction_intensities_shape,
                    diffraction_space_mask_com=diffraction_space_mask_com,
                    resizing_method=resizing_method,
                    bin_real_space=bin_real_space,
                    masks_real_space=masks_real_space,
                    crop_reciprocal_space=crop_reciprocal_space,
                    crop_real_space=crop_real_space,
                    q_max_inv_A=q_max_inv_A,
                )
            )

            # initialize object
            if a0 == 0:
                if self._initial_object_guess:
                    self._object_initial = copy_to_device(
                        self._initial_object_guess, storage
                    )
                    del self._initial_object_guess
                else:
                    diffraction_shape = self._initial_datacube_shape[-1]
                    self._object_initial = xp_storage.zeros(
                        (
                            self._object_shape_x_y_z[0],
                            self._object_shape_x_y_z[1] * self._object_shape_x_y_z[2],
                            diffraction_shape * diffraction_shape * diffraction_shape,
                        ),
                        dtype="float32",
                    )
                self._object_shape_6D = self._object_shape_x_y_z + (
                    diffraction_shape,
                    diffraction_shape,
                    diffraction_shape,
                )

            # ellpitical fitting?!

            # hmmm how to handle this? we might need to rotate diffraction patterns
            # solve for QR rotation if necessary
            # if a0 is 0 only
            # if force_transpose is not None and force_com_rotation is not None:
            #     dc = self._datacubes[datacube_to_solve_rotation]
            #     _solve_for_center_of_mass_relative_rotation():

            # initialize positions
            mask_real_space = self._calculate_scan_positions(
                datacube_number=a0,
                mask_real_space=mask_real_space,
            )

            # align and reshape
            if force_centering_shifts is not None:
                if np.isscalar(force_centering_shifts[a0][0]):
                    qx0_fit = (
                        np.ones((datacube.data.shape[0:2]))
                        * force_centering_shifts[a0][0]
                    )
                    qy0_fit = (
                        np.ones((datacube.data.shape[0:2]))
                        * force_centering_shifts[a0][1]
                    )
                else:
                    qx0_fit = force_centering_shifts[a0][0]
                    qy0_fit = force_centering_shifts[a0][1]

            else:
                (qx0_fit, qy0_fit) = self._solve_for_diffraction_pattern_centering(
                    datacube=datacube,
                    r=r,
                    rscale=rscale,
                    fast_center=fast_center,
                    fitfunction=fitfunction,
                    robust=robust,
                    robust_steps=robust_steps,
                    robust_thresh=robust_thresh,
                )

            self._reshape_diffraction_patterns(
                datacube_number=a0,
                datacube=datacube,
                mask_real_space=mask_real_space,
                qx0_fit=qx0_fit,
                qy0_fit=qy0_fit,
                q_max_inv_A=q_max_inv_A,
                dp_shift_method=dp_shift_method,
            )

            self._solve_for_indicies(
                datacube_number=a0,
                num_points=num_points,
            )

        s = self._object_shape_6D
        cylinder_mask = np.zeros((s[0:3]))
        x = np.arange(s[1])
        y = np.arange(s[2])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        center = (np.mean(x), np.mean(y))
        cylinder_mask[
            :,
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            <= (np.max((center[0], center[1]))) ** 2,
        ] = 1

        self._cylinder_mask = cylinder_mask

        weights_diff_all = xp.array(self._weights_diff).flatten()
        ind_diff_all = xp.array(self._ind_diff).flatten()
        weights_diff_all_counted = xp.bincount(
            ind_diff_all,
            weights=weights_diff_all,
            minlength=s[3] * s[4] * s[5],
        )
        self._weights_diff_all_counted = weights_diff_all_counted

        center = (
            (self._object_shape_6D[3] - 1) / 2,
            (self._object_shape_6D[4] - 1) / 2,
            (self._object_shape_6D[5] - 1) / 2,
        )

        qx = np.arange(self._object_shape_6D[3])
        qy = np.arange(self._object_shape_6D[4])
        qz = np.arange(self._object_shape_6D[5])

        qxx, qyy, qzz = np.meshgrid(qx, qy, qz)

        diffraction_edge_mask = (qxx - center[0]) ** 2 + (qyy - center[1]) ** 2 + (
            qzz - center[2]
        ) ** 2 <= center[0] ** 2
        diffraction_edge_mask = diffraction_edge_mask.ravel()
        diffraction_edge_mask = np.array(diffraction_edge_mask, dtype="int")

        self._diffraction_edge_mask = diffraction_edge_mask

        return self

    def reconstruct(
        self,
        num_iter: int = 1,
        store_iterations: bool = False,
        store_initial_object: bool = True,
        store_error_per_step: bool = False,
        reset: bool = True,
        step_size: float = 0.5,
        zero_edges_real: bool = True,
        zero_edges_diffraction: bool = True,
        cylinder_mask: bool = True,
        diffraction_gaussian_filter: float = 0,
        real_space_gaussian_filter: float = 0,
        baseline_thresh: float = None,
        diffraction_shrinkage: float = False,
        diffraction_shrinkage_threshold: float = None,
        position_refinement: bool = False,
        position_refinement_frequency: int = 1,
        position_refinement_step_size: float = 1,
        position_refinement_max_num_iter: int = 10,
        position_refinement_stop_criteria_shift_size: float = 0.2,
        position_refinement_y_frequency: int = 4,
        position_refinement_max_total_displacement: int = None,
        position_refinement_max_step_displacement: int = None,
        distributed=False,
        num_jobs=None,
        threads_per_job=1,
        device: str = None,
        clear_fft_cache: bool = True,
        progress_bar: bool = True,
    ):
        """
        Main loop for reconstruct

        Parameters
        ----------
        num_iter: int
            number of iterations
        store_iterations: bool
            if True, stores number of iterations
        store_initial_object: bool
            if True, keeps a copy of an initial object to reset without preprocessing
        store_error_per_step: bool
            if True, stores error for each tilt for each iteration and order of tilts
        reset: bool
            if True, resets object
        step_size: float
            from 0 to 1, step size for update
        progres_bar: bool
            if True, shows progress bar
        zero_edges_real: bool
            if True, zeros edges along y and z
        zero_edges_diffraction: bool
            if True, zeros diffraction edges with spherical mask
        cylinderical_mask: bool
            if True, applies cylindrical mask to reconstruction
        diffraction_gaussian_filter: float
            Gaussian filter sigma for diffraction space (in pixels)
        real_space_gaussian_filter: flooat
            Gaussian filter for real space (in pixels)
        baseline_thresh: float
            if not None, data is cropped below threshold. Value is percentile of object.
        diffraction_shrinkage: bool
            if True, subtracts baseline from each kernel in real space and zeros any residual negative values. If no
            theshold is provided, uses mean of object
        diffraction_shrinkage_threshold: None
            threshold for shrinkage
        position_refinement: bool
            if True, refines positions
        position_refinement_frequency: int
            frequency of iterations to run position refinement
        position_refinement_step_size: float
            scaling factor for step size
        position_refinement_max_num_iter: int
            maximum number of iterations to run for position refinement
        position_refinement_stop_criteria_shift_size: float
            below a certain threshold stop searching for updated positions
        position_refinement_y_frequency: int
            if y_frequency is greater than 1, only uses y rows of that frequency for
            position update
        position_refinement_max_total_displacement: float
            maximum total displacement per datacube
        position_refinement_max_step_displacement: float
            maximum displacement per step
        """
        self.set_device(device, clear_fft_cache)
        if device is not None:
            self._cylinder_mask = copy_to_device(self._cylinder_mask, device)
            self._weights_diff_all_counted = copy_to_device(
                self._weights_diff_all_counted, device
            )
            self._diffraction_edge_mask = copy_to_device(
                self._weights_diff_all_counted, device
            )
            self._ind_real = copy_to_device(self._ind_real, device)
            self._ind_diff = copy_to_device(self._ind_diff, device)
            self._weights_real = copy_to_device(self._weights_real, device)
            self._weights_diff = copy_to_device(self._weights_diff, device)

        device = self._device

        if reset is True:
            self.error_iterations = []

            if store_iterations:
                self.object_iterations = []

            if store_initial_object:
                self._object = self._object_initial.copy()
            else:
                self._object = self._object_initial

            if store_error_per_step:
                self._tilt_order = []
                self._error_per_step = []

        for a0 in tqdmnd(
            num_iter,
            desc="Reconstructing object",
            unit=" iter",
            disable=not progress_bar,
        ):
            error_iteration = 0
            random_tilt_order = np.arange(self._num_datacubes)
            np.random.shuffle(random_tilt_order)

            num_points = self._num_points

            if distributed is True and self._device == "cpu":
                num_jobs = num_jobs or cpu_count() // threads_per_job

                def f(args):
                    with threadpool_limits(limits=threads_per_job):
                        return self._reconstruct(**args)

            for a1 in range(self._num_datacubes):
                error_iteration_datacube = 0
                a1_shuffle = random_tilt_order[a1]
                diffraction_patterns_projected = copy_to_device(
                    self._diffraction_patterns_projected[a1_shuffle], device
                )

                if distributed is False:
                    for a2 in range(self._object_shape_6D[0]):
                        x_index, yy, zz, update_r_summed, error = self._reconstruct(
                            a2=a2,
                            a1_shuffle=a1_shuffle,
                            num_points=num_points,
                            diffraction_patterns_projected=diffraction_patterns_projected,
                            step_size=step_size,
                        )
                        error_iteration_datacube += error

                        self._object[x_index, yy, zz] += update_r_summed

                elif distributed is True and self._device == "cpu":
                    inputs = [
                        (
                            {
                                "a2": a2,
                                "a1_shuffle": a1_shuffle,
                                "num_points": num_points,
                                "diffraction_patterns_projected": diffraction_patterns_projected,
                                "step_size": step_size,
                            },
                        )
                        for a2 in range(self._object_shape_6D[0])
                    ]

                    with WorkerPool(
                        n_jobs=num_jobs,
                    ) as pool:
                        results = pool.map(
                            f,
                            inputs,
                            progress_bar=False,
                        )

                    for a2 in range(self._object_shape_6D[0]):
                        self._object[
                            results[a2][0], results[a2][1], results[a2][2]
                        ] += results[a2][3]
                        error_iteration_datacube += results[a2][4]

                else:
                    raise ValueError(("distributed not implemented for gpu"))

                if store_error_per_step:
                    self._tilt_order.append(a1_shuffle)
                    self._error_per_step.append(error_iteration_datacube)

                error_iteration += error_iteration_datacube

            self._constraints(
                zero_edges_real=zero_edges_real,
                zero_edges_diffraction=zero_edges_diffraction,
                cylinder_mask=cylinder_mask,
                diffraction_gaussian_filter=diffraction_gaussian_filter,
                real_space_gaussian_filter=real_space_gaussian_filter,
                baseline_thresh=baseline_thresh,
                diffraction_shrinkage=diffraction_shrinkage,
                diffraction_shrinkage_threshold=diffraction_shrinkage_threshold,
            )

            if position_refinement and a0 % position_refinement_frequency < 1e-6:
                self.position_refinement(
                    max_num_iter=position_refinement_max_num_iter,
                    stop_criteria_shift_size=position_refinement_stop_criteria_shift_size,
                    y_frequency=position_refinement_y_frequency,
                    step_size=position_refinement_step_size,
                    max_total_displacement=position_refinement_max_total_displacement,
                    max_step_displacement=position_refinement_max_step_displacement,
                )

            self.error_iterations.append(error_iteration)
            self.error = error_iteration
            if store_iterations:
                self.object_iterations.append(self._object.copy())

        return self

    def _reconstruct(
        self,
        a2,
        a1_shuffle,
        num_points,
        diffraction_patterns_projected,
        step_size,
    ):
        object_sliced = self._forward(
            datacube_number=a1_shuffle,
            x_index=a2,
            num_points=num_points,
        )

        update, error = self._calculate_update(
            object_sliced=object_sliced,
            diffraction_patterns_projected=diffraction_patterns_projected,
            datacube_number=a1_shuffle,
            x_index=a2,
        )

        update *= step_size
        (x_index, i_real, i_diff, update_r_summed) = self._back(
            num_points=num_points,
            datacube_number=a1_shuffle,
            x_index=a2,
            update=update,
        )

        return x_index, i_real, i_diff, update_r_summed, error

    def position_refinement(
        self,
        max_num_iter: int = 10,
        stop_criteria_shift_size: float = 0.2,
        y_frequency: int = 4,
        datacube_numbers: Union[int, np.ndarray] = None,
        step_size: float = 1,
        max_total_displacement: int = None,
        max_step_displacement: int = None,
    ):
        """
        Refining positions by searching a 2x2 grid space and updating based
        on forward projection error

        Parameters
        ----------
        max_num_iter: int
            maximum number of iterations to run per datacube
        stop_criteria_shift_size: float
            below a certain threshold stop searching for updated positions
        y_frequency: int
            if y_frequency is greater than 1, only uses y rows of that frequency for
            position update
        datacube_numbers: int or np.ndarray of integers
            datacubes to update positions for with default to update all datacubes
        step_size: float
            scaling factor for position_update
        max_total_displacement: float
            maximum total displacement per datacube
        max_step_displacement: float
            maximum displacement per step
        """
        xp = self._xp
        device = self._device
        s = self._object_shape_6D
        num_points = self._num_points

        if not hasattr(self, "_position_refinements"):
            self._position_refinements = np.zeros((self._num_datacubes, 2))

        if datacube_numbers is None:
            datacube_numbers = np.arange(self._num_datacubes)
        elif np.isscalar(datacube_numbers):
            datacube_numbers = np.atleast_1d(np.asarray((datacube_numbers)))

        y_values = np.arange(s[1] // y_frequency) * y_frequency

        random_tilt_order = datacube_numbers.copy()

        for a0 in range(datacube_numbers.shape[0]):
            for a1 in range(max_num_iter):
                positions_save = (
                    self._positions_vox[a0][0].copy(),
                    self._positions_vox[a0][1].copy(),
                )
                positions_F_save = (
                    self._positions_vox_F[a0][0].copy(),
                    self._positions_vox_F[a0][1].copy(),
                )

                diffraction_patterns_projected = copy_to_device(
                    self._diffraction_patterns_projected[a0], device
                )
                error_shifts = np.zeros((y_values.shape[0], 4))
                position_deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

                for a2 in range(y_values.shape[0]):
                    object_sliced = self._forward(
                        datacube_number=a0,
                        x_index=a2,
                        num_points=num_points,
                    )

                    for a3 in range(4):
                        self._positions_vox[a0] = (
                            positions_save[0].copy() + position_deltas[a3][0],
                            positions_save[1].copy() + position_deltas[a3][1],
                        )
                        self._positions_vox_F[a0] = (
                            positions_F_save[0].copy() + position_deltas[a3][0],
                            positions_F_save[1].copy() + position_deltas[a3][1],
                        )

                        _, error = self._calculate_update(
                            object_sliced=object_sliced,
                            diffraction_patterns_projected=diffraction_patterns_projected,
                            datacube_number=a0,
                            x_index=a2,
                        )

                        error_shifts[a2, a3] = error

                error_shifts_mean = np.ma.array(
                    data=error_shifts, mask=error_shifts == 0
                )

                weights = error_shifts_mean.mean(0).data
                weights -= weights.mean()
                weights = -1 * weights
                weights /= np.abs(weights).sum()

                position_delta = (
                    np.asarray(position_deltas, dtype="float") * weights[:, None]
                ).sum(0)

                position_delta = position_delta * step_size
                if max_step_displacement is not None:
                    position_delta = np.clip(
                        position_delta, -max_step_displacement, max_step_displacement
                    )

                if max_total_displacement is not None:
                    position_delta = np.clip(
                        position_delta,
                        -max_total_displacement - self._position_refinements[a0],
                        max_total_displacement - self._position_refinements[a0],
                    )

                x_vox = positions_save[0].copy() + position_delta[0]
                y_vox = positions_save[1].copy() + position_delta[1]
                x_vox_F = np.floor(x_vox).astype("int")
                y_vox_F = np.floor(y_vox).astype("int")
                dx = x_vox - x_vox_F
                dy = y_vox - y_vox_F

                self._positions_vox[a0] = (
                    copy_to_device(x_vox, device),
                    copy_to_device(y_vox, device),
                )
                self._positions_vox_F[a0] = (
                    copy_to_device(x_vox_F, device),
                    copy_to_device(y_vox_F, device),
                )
                self._positions_vox_dF[a0] = (
                    copy_to_device(dx, device),
                    copy_to_device(dy, device),
                )
                self._position_refinements[a0] += position_delta

                if np.max(position_delta) < stop_criteria_shift_size:
                    break

    def _prepare_datacube(
        self,
        datacube_number,
        diffraction_intensities_shape,
        diffraction_space_mask_com,
        resizing_method,
        bin_real_space,
        masks_real_space,
        crop_reciprocal_space,
        crop_real_space,
        q_max_inv_A,
    ):
        """
        datacube_number: int
            index of datacube
        diffraction_intensites_shape: int
            shape of diffraction patterns to reshape data into
        diffraction_space_mask_com: np.ndarray
            applies mask to datacube while solving for CoM rotation
        resizing_method: float
            method to reshape diffraction space ("bin", "fourier", "bilinear")
        bin_real_space: int
            factor for binnning in real space
        masks_real_space: list of np.ndarray or np.ndarray
            mask for real space. can be the same for each datacube of individually specified.
        crop_reciprocal_space: float
            if not None, crops reciprocal space on all sides by integer.
            Can pass an single integer of a 4-tuple
        crop_real_space: float
            if not None, crops real space on all sides by integer.
            Can pass an single integer of a 4-tuple
        q_max_inv_A: int
            maximum q in inverse angtroms
        """
        if type(self._datacubes[datacube_number]) is str:
            try:
                from py4DSTEM import import_file

                datacube = import_file(
                    self._datacubes[datacube_number], **self._import_kwargs
                )

            except:
                from py4DSTEM import read

                datacube = read(self._datacubes[datacube_number], **self._import_kwargs)
        else:
            datacube = self._datacubes[datacube_number]

        if crop_real_space:
            if np.isscalar(crop_real_space):
                datacube.crop_R(
                    (
                        crop_real_space,
                        -crop_real_space,
                        crop_real_space,
                        -crop_real_space,
                    )
                )
            else:
                datacube.crop_R(
                    (
                        crop_real_space[0],
                        crop_real_space[1],
                        crop_real_space[2],
                        crop_real_space[3],
                    )
                )

        if masks_real_space is not None:
            if type(masks_real_space) is np.ndarray:
                mask_real_space = masks_real_space
            else:
                mask_real_space = masks_real_space[datacube_number]
            mask_real_space = np.ndarray(masks_real_space, dtype="bool")
        else:
            mask_real_space = None

        if crop_reciprocal_space is not None:
            if np.isscalar(crop_reciprocal_space):
                datacube.crop_Q(
                    (
                        crop_reciprocal_space,
                        -crop_reciprocal_space,
                        crop_reciprocal_space,
                        -crop_reciprocal_space,
                    )
                )
            else:
                datacube.crop_Q(
                    (
                        crop_reciprocal_space[0],
                        crop_reciprocal_space[1],
                        crop_reciprocal_space[2],
                        crop_reciprocal_space[3],
                    )
                )

        if self._force_q_to_r_rotation_deg is not None:
            for a0 in range(datacube.shape[0]):
                for a1 in range(datacube.shape[1]):
                    datacube.data[a0, a1] = np.clip(
                        rotate(
                            datacube.data[a0, a1],
                            self._force_q_to_r_rotation_deg,
                            reshape=False,
                            order=0,
                        ),
                        0,
                        np.inf,
                    )
        if self._transpose_xy:
            datacube.data = datacube.data.swapaxes(-1, -2)

        # resize diffraction space
        if diffraction_intensities_shape is not None:
            Q = datacube.shape[-1]
            S = diffraction_intensities_shape
            resampling_factor = S / Q

            if resizing_method == "bin":
                datacube = datacube.bin_Q(N=int(1 / resampling_factor))
                if diffraction_space_mask_com is not None:
                    diffraction_space_mask_com = bin2D(
                        diffraction_space_mask_com, int(1 / resampling_factor)
                    )

            elif resizing_method == "fourier":
                datacube = datacube.resample_Q(
                    N=resampling_factor, method=resizing_method
                )
                if diffraction_space_mask_com is not None:
                    diffraction_space_mask_com = fourier_resample(
                        diffraction_space_mask_com,
                        output_size=(S, S),
                        force_nonnegative=True,
                    )

            elif resizing_method == "bilinear":
                datacube = datacube.resample_Q(
                    N=resampling_factor, method=resizing_method
                )
                if diffraction_space_mask_com is not None:
                    diffraction_space_mask_com = zoom(
                        diffraction_space_mask_com,
                        (resampling_factor, resampling_factor),
                        order=1,
                    )

            else:
                raise ValueError(
                    (
                        "reshaping_method needs to be one of 'bilinear', 'fourier', or 'bin', "
                        f"not {resizing_method}."
                    )
                )

            if datacube_number == 0:
                self._datacube_Q_pixel_size_inv_A /= resampling_factor
                if q_max_inv_A is not None:
                    q_max_inv_A *= resampling_factor
                else:
                    q_max_inv_A = (
                        self._datacube_Q_pixel_size_inv_A * datacube.Qshape[0] / 2
                    )
        else:
            if datacube_number == 0 and q_max_inv_A is None:
                q_max_inv_A = self._datacube_Q_pixel_size_inv_A * datacube.Qshape[0] / 2

        # bin real space
        if bin_real_space is not None:
            datacube.bin_R(bin_real_space)
            if mask_real_space is not None:
                mask_real_space = bin2D(mask_real_space, bin_real_space)
                mask_real_space = np.floor(
                    mask_real_space / bin_real_space / bin_real_space
                )
                mask_real_space = np.ndarray(masks_real_space, dtype="bool")
            if datacube_number == 0:
                self._datacube_R_pixel_size_A *= bin_real_space

        self._initial_datacube_shape = datacube.data.shape

        return datacube, mask_real_space, diffraction_space_mask_com, q_max_inv_A

    def _calculate_scan_positions(
        self,
        datacube_number,
        mask_real_space,
    ):
        """
        Calculate scan positions in angstroms and voxels

        Parameters
        ----------
        datacube_number: int
            index of datacube
        mask_real_space: np.ndarray
            mask for real space

        Returns
        --------
        mask_real_space: np.ndarray
            mask for real space

        """
        device = self._device

        # calculate shape
        field_of_view_px = self._object_shape_6D[0:2]
        self._field_of_view_A = (
            self._voxel_size_A * field_of_view_px[0],
            self._voxel_size_A * field_of_view_px[1],
        )

        # calculate positions
        s = self._initial_datacube_shape

        step_size = self._datacube_R_pixel_size_A

        x = np.arange(s[0], dtype="float")
        y = np.arange(s[1], dtype="float")

        if self._shift_px is not None:
            if self._transpose_xy:
                x += self._shift_px[datacube_number][1]
                y += self._shift_px[datacube_number][0]
            else:
                x += self._shift_px[datacube_number][0]
                y += self._shift_px[datacube_number][1]

        x *= step_size
        y *= step_size

        x, y = np.meshgrid(x, y, indexing="ij")

        if self._tilt_rotation_axis_angle_deg is not None:
            rotation_angle = np.deg2rad(self._tilt_rotation_axis_angle_deg)
            x_mean = x.mean()
            y_mean = y.mean()
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

            x -= x.mean()
            y -= y.mean()

            x += x_mean
            y += y_mean

        if self._transpose_xy:
            x_temp = x.copy()
            y_temp = y.copy()
            x = y_temp.copy()
            y = x_temp.copy()

        if self._tilt_rotation_axis_shift_px:
            y -= self._tilt_rotation_axis_shift_px

        # remove data outside FOV
        if mask_real_space is None:
            mask_real_space = np.ones(x.shape, dtype="bool")
        else:
            if self._transpose_xy:
                mask_real_space = mask_real_space.swapaxes(-1, -2)

        mask_real_space[x >= self._field_of_view_A[0]] = False
        mask_real_space[x < 0] = False
        mask_real_space[y >= self._field_of_view_A[1]] = False
        mask_real_space[y < 0] = False

        # calculate positions in voxels
        x = x[mask_real_space].ravel()
        y = y[mask_real_space].ravel()

        x_vox = x / self._voxel_size_A
        y_vox = y / self._voxel_size_A

        x_vox_F = np.floor(x_vox).astype("int")
        y_vox_F = np.floor(y_vox).astype("int")
        dx = x_vox - x_vox_F
        dy = y_vox - y_vox_F

        # store pixels
        self._positions_vox.append(
            (copy_to_device(x_vox, device), copy_to_device(y_vox, device))
        )
        self._positions_vox_F.append(
            (copy_to_device(x_vox_F, device), copy_to_device(y_vox_F, device))
        )
        self._positions_vox_dF.append(
            (copy_to_device(dx, device), copy_to_device(dy, device))
        )

        return mask_real_space

    def _solve_for_diffraction_pattern_centering(
        self,
        datacube,
        r,
        rscale,
        fast_center,
        fitfunction,
        robust,
        robust_steps,
        robust_thresh,
    ):
        """
        Solve for qx and qy shifts

        Parameters
        ----------
        r: (float or None)
            the approximate radius of the center disk. If None (default),
            tries to compute r using the get_probe_size method.  The data used for this
            is controlled by dp_max.
        rscale (float)
             expand 'r' by this amount to form a mask about the center disk
            when taking its center of mass
        fast_center: (bool)
            skip the center of mass refinement step.
            arrays are returned for qx0,qy0
        fitfunction: "str"
            fit function for origin ('plane' or 'parabola' or 'bezier_two' or 'constant').
        robust: bool
            if set to True, origin fit will be repeated with outliers
            removed.
        robust_steps: int
            number of robust iterations performed after initial fit.
        robust_thresh: int
            threshold for including points, in units of root-mean-square (standard deviations) error
            of the predicted values after fitting.

        Returns
        --------
        qx0_fit, qy0_fit: (np.ndarray, np.ndarray)
            qx and qy shifts

        """

        (qx0, qy0, _) = get_origin(
            datacube,
            r=r,
            rscale=rscale,
            fast_center=fast_center,
            verbose=False,
        )

        (qx0_fit, qy0_fit, qx0_res, qy0_res) = fit_origin(
            (qx0, qy0),
            fitfunction=fitfunction,
            returnfitp=False,
            robust=robust,
            robust_steps=robust_steps,
            robust_thresh=robust_thresh,
        )

        return qx0_fit, qy0_fit

    def _reshape_diffraction_patterns(
        self,
        datacube_number,
        datacube,
        mask_real_space,
        qx0_fit,
        qy0_fit,
        q_max_inv_A,
        dp_shift_method,
    ):
        """
        Reshapes diffraction data into a 2 column array

        Parameters
        ----------
        datacube_number: int
            index of datacube
        datacube: DataCube
            datacube to be reshapped
        mask_real_space: np.ndarray
            mask for real space
        qx0_fit: np.ndarray
            qx shifts
        qy0_fit: int
            qy shifts
        q_max_inv_A: int
            maximum q in inverse angstroms
        dp_shift_method: float
            method to shift diffraction patterns "subpixel" or "pixel"
        """
        # calculate bincount array
        if datacube_number == 0:
            self._make_diffraction_masks(q_max_inv_A=q_max_inv_A)

        diffraction_patterns_reshaped = self._reshape_4D_array_to_2D(
            data=datacube.data,
            qx0_fit=qx0_fit,
            qy0_fit=qy0_fit,
            ind_diffraction_ravel=self._ind_diffraction_rotate_transpose_ravel,
            dp_shift_method=dp_shift_method,
        )

        del datacube

        self._diffraction_patterns_projected.append(
            diffraction_patterns_reshaped[mask_real_space.ravel()]
        )

    def _make_diffraction_masks(self, q_max_inv_A):
        """
        make masks to convert 2D diffraction patterns to 1D arrays

        Parameters
        ----------
        q_max_inv_A: int
            maximum q in inverse angstroms

        """

        s = self._initial_datacube_shape

        mask = np.ones((s[-1], s[-1]), dtype="bool")
        mask[:, int(np.ceil(s[-1] / 2)) :] = 0
        mask[: int(np.ceil(s[-1] / 2)), int(np.floor(s[-1] / 2))] = 0

        ind_diffraction = np.roll(
            np.arange(s[-1] * s[-1]).reshape(s[-1], s[-1]),
            (int(np.floor(s[-1] / 2)), int(np.floor(s[-1] / 2))),
            axis=(0, 1),
        )

        ind_diffraction[mask] = 1e10

        a = np.argsort(ind_diffraction.ravel())
        i = np.empty_like(a)
        i[a] = np.arange(a.size)
        i = i.reshape((s[-1], s[-1]))

        ind_diffraction = i
        ind_diffraction_rot = np.rot90(ind_diffraction, 2)

        ind_diffraction[mask] = ind_diffraction_rot[mask]

        ind_diffraction_rotate_transpose = ind_diffraction.copy()

        if self._force_q_to_r_transpose:
            ind_diffraction_rotate_transpose = (
                ind_diffraction_rotate_transpose.swapaxes(-1, -2)
            )

        self._ind_diffraction = ind_diffraction
        self._ind_diffraction_ravel = ind_diffraction.ravel()
        self._ind_diffraction_rotate_transpose = ind_diffraction_rotate_transpose
        self._ind_diffraction_rotate_transpose_ravel = (
            ind_diffraction_rotate_transpose.ravel()
        )
        self._q_length = np.unique(self._ind_diffraction).shape[0]

        # pixels to remove
        q_max_px = q_max_inv_A / self._datacube_Q_pixel_size_inv_A

        x = np.arange(s[-1]) - ((s[-1] - 1) / 2)
        y = np.arange(s[-1]) - ((s[-1] - 1) / 2)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        circular_mask = ((xx) ** 2 + (yy) ** 2) ** 0.5 < q_max_px

        self._circular_mask = circular_mask
        self._circular_mask_ravel = circular_mask.ravel()
        self._circular_mask_bincount = np.asarray(
            np.bincount(
                self._ind_diffraction_ravel,
                circular_mask.ravel(),
                minlength=self._q_length,
            ),
            dtype="bool",
        )

    def _solve_for_indicies(
        self,
        datacube_number,
        num_points,
    ):
        """ """
        xp = self._xp
        s = self._object_shape_6D
        device = self._device

        tilt_deg = self._tilt_deg[datacube_number]
        tilt = -np.deg2rad(tilt_deg)

        # solve for real space coordinates
        s_max = np.max((s[1], s[2]))
        y = np.arange(s_max)
        z = np.arange(s_max)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        sin = np.sin(tilt)
        cos = np.cos(tilt)
        r = [[cos, sin], [-sin, cos]]
        points = np.array((yy.ravel(), zz.ravel())).T
        points = points @ r
        line_y = points[:, 0]
        line_z = points[:, 1]
        line_y -= np.mean(line_y) - (s_max - 1) / 2 + ((s_max - s[1]) / 2)
        line_z -= np.mean(line_z) - (s_max - 1) / 2 + ((s_max - s[2]) / 2)

        yF = np.floor(line_y).astype("int")
        zF = np.floor(line_z).astype("int")
        dy = line_y - yF
        dz = line_z - zF

        ind0 = np.hstack((yF, yF + 1, yF, yF + 1))

        ind1 = np.hstack((zF, zF, zF + 1, zF + 1))

        weights_real = np.hstack(
            (
                (1 - dy) * (1 - dz),
                (dy) * (1 - dz),
                (1 - dy) * (dz),
                (dy) * (dz),
            )
        )

        ind_real = np.ravel_multi_index((ind0, ind1), (s[1], s[2]), mode="clip")

        # solve for diffraction space coordinates
        line_y_diff = np.arange(s[-1]) * np.cos(tilt)
        line_z_diff = np.arange(s[-1]) * np.sin(tilt)
        line_y_diff -= np.mean(line_y_diff) - (s[-1] - 1) / 2
        line_z_diff -= np.mean(line_z_diff) - (s[-1] - 1) / 2

        yF_diff = np.floor(line_y_diff).astype("int")
        zF_diff = np.floor(line_z_diff).astype("int")
        dy_diff = line_y_diff - yF_diff
        dz_diff = line_z_diff - zF_diff

        qx = np.arange(s[-1])
        qy = np.arange(s[-1])
        qxx, qyy = np.meshgrid(qx, qy, indexing="ij")

        ind0_diff = np.hstack(
            (
                np.tile(yF_diff, s[-1]),
                np.tile(yF_diff + 1, s[-1]),
                np.tile(yF_diff, s[-1]),
                np.tile(yF_diff + 1, s[-1]),
            )
        )

        ind1_diff = np.hstack(
            (
                np.tile(zF_diff, s[-1]),
                np.tile(zF_diff, s[-1]),
                np.tile(zF_diff + 1, s[-1]),
                np.tile(zF_diff + 1, s[-1]),
            )
        )

        weights_diff = np.hstack(
            (
                np.tile(((1 - dy_diff) * (1 - dz_diff)), s[-1]),
                np.tile(((dy_diff) * (1 - dz_diff)), s[-1]),
                np.tile(((1 - dy_diff) * (dz_diff)), s[-1]),
                np.tile(((dy_diff) * (dz_diff)), s[-1]),
            )
        )

        ind_diff = np.ravel_multi_index(
            (
                np.tile(qxx.ravel(), 4),
                ind0_diff.ravel(),
                ind1_diff.ravel(),
            ),
            (s[-1], s[-1], s[-1]),
            "clip",
        )

        # normalization real space
        bincount_real_max = s[0] * s[1] * s[2]

        ind_real_bincount_weight = np.bincount(
            ind_real.ravel(), weights_real.ravel(), minlength=bincount_real_max
        )
        ind_real_bincount = np.bincount(ind_real.ravel(), minlength=bincount_real_max)

        ind_real_bincount_weight = ind_real_bincount_weight[ind_real_bincount > 0]
        ind_real_bincount = ind_real_bincount[ind_real_bincount > 0]

        ind_real_bincount_weight[ind_real_bincount_weight == 0] = 1

        correction_factor_real = 1 / ind_real_bincount_weight

        correction_factor_real = np.repeat(correction_factor_real, ind_real_bincount)
        sorted_indicies = np.argsort(np.argsort(ind_real.ravel()))
        correction_factor_real = correction_factor_real[sorted_indicies].reshape(
            ind_real.shape
        )
        weights_real = weights_real * correction_factor_real

        if datacube_number == 0:
            self._ind_real = []
            self._weights_real = []
            self._ind_diff = []
            self._weights_diff = []

        self._ind_real.append(xp.asarray(ind_real))
        self._ind_diff.append(xp.asarray(ind_diff))
        self._weights_real.append(xp.asarray(weights_real))
        self._weights_diff.append(xp.asarray(weights_diff))

    def _reshape_4D_array_to_2D(
        self,
        data,
        qx0_fit=None,
        qy0_fit=None,
        ind_diffraction_ravel=None,
        dp_shift_method="subpixel",
    ):
        """
        reshape diffraction 4D-data to 2D ravelled patterns

        Parameters
        ----------
        data: np.ndarrray
            4D datacube data to be reshapped
        qx0_fit: np.ndarray
            qx shifts
        qy0_fit: int
            qy shifts
        ind_diffraction: np.ndarray
            1D array (length of number of pixels in diffraciton space to project 4D array into)
        dp_shift_method: float
            method to shift diffraction patterns "subpixel" or "pixel"


        Returns
        --------
        diffraction_patterns_reshaped: np.ndarray
            diffraction patterns ravelled
        """

        s = data.shape

        center = ((s[-1] - 1) / 2, (s[-1] - 1) / 2)
        diffraction_patterns_reshaped = np.zeros((s[0] * s[1], self._q_length))

        if ind_diffraction_ravel is None:
            ind_diffraction_ravel = self._ind_diffraction_ravel

        for a0 in range(s[0]):
            for a1 in range(s[0]):
                dp = data[a0, a1]
                index = np.ravel_multi_index((a0, a1), (s[0], s[1]))
                if qx0_fit is not None:
                    qx0 = center[0] - qx0_fit[a0, a1]
                    qy0 = center[1] - qy0_fit[a0, a1]

                    if dp_shift_method == "subpixel":
                        xF = int(np.floor(qx0))
                        yF = int(np.floor(qy0))

                        wx = qx0 - xF
                        wy = qy0 - yF

                        dp_projected = (
                            (
                                (
                                    (1 - wx)
                                    * (1 - wy)
                                    * np.bincount(
                                        ind_diffraction_ravel,
                                        np.roll(dp, (xF, yF), axis=(0, 1)).ravel(),
                                        minlength=self._q_length,
                                    )
                                )
                            )
                            + (
                                (wx)
                                * (1 - wy)
                                * np.bincount(
                                    ind_diffraction_ravel,
                                    np.roll(dp, (xF + 1, yF), axis=(0, 1)).ravel(),
                                    minlength=self._q_length,
                                )
                            )
                            + (
                                (1 - wx)
                                * (wy)
                                * np.bincount(
                                    ind_diffraction_ravel,
                                    np.roll(dp, (xF, yF + 1), axis=(0, 1)).ravel(),
                                    minlength=self._q_length,
                                )
                            )
                            + (
                                (wx)
                                * (wy)
                                * np.bincount(
                                    ind_diffraction_ravel,
                                    np.roll(dp, (xF + 1, yF + 1), axis=(0, 1)).ravel(),
                                    minlength=self._q_length,
                                )
                            )
                        )

                        diffraction_patterns_reshaped[index] = dp_projected

                    elif dp_shift_method == "pixel":
                        xF = int(qx0)
                        yF = int(qy0)

                        dp_projected = np.bincount(
                            ind_diffraction_ravel,
                            np.roll(dp, (xF, yF), axis=(0, 1)).ravel(),
                            minlength=self._q_length,
                        )
                        diffraction_patterns_reshaped[index] = dp_projected
                else:
                    diffraction_patterns_reshaped[index] = np.bincount(
                        ind_diffraction_ravel,
                        dp.ravel(),
                        minlength=self._q_length,
                    )

        diffraction_patterns_reshaped = diffraction_patterns_reshaped[
            :, self._circular_mask_bincount
        ]
        return diffraction_patterns_reshaped

    def _reshape_2D_array_to_4D(self, data, xy_shape=None, positions=None):
        """
        reshape ravelled diffraction 2D-data to 4D-data

        Parameters
        ----------
        data: np.ndarrray
            2D datacube data to be reshapped
        xy_shape: 2-tuple
            if None, takes 6D object shape
        real_space_mask: np.ndarray
            Must be xy_shape. Fills 4D datacube with zeros wherever is masked
        positions: np.ndarray
            2-tuple of np.ndarrays specifyign positions

        Returns
        --------
        data_reshaped: np.ndarray
            data reshapped in 4D-array

        """
        xp = self._xp

        if xy_shape is None:
            s = (
                self._object_shape_6D[0],
                self._object_shape_6D[1],
                self._object_shape_6D[-1],
                self._object_shape_6D[-1],
            )
        else:
            s = (
                xy_shape[0],
                xy_shape[1],
                self._object_shape_6D[-1],
                self._object_shape_6D[-1],
            )
        a = xp.argsort(self._ind_diffraction_ravel[self._circular_mask_ravel])
        i = xp.empty_like(a)
        i[a] = xp.arange(a.size)

        data_reshaped = xp.zeros((s[0] * s[1], s[2] * s[3]))

        if positions is not None:
            data_masked = data.copy()
            data = xp.zeros((s[0] * s[1], data_masked.shape[1]))

            positions = (
                np.asarray(copy_to_device(positions[0], "cpu"), dtype="int"),
                np.asarray(copy_to_device(positions[1], "cpu"), dtype="int"),
            )

            indicies = np.ravel_multi_index(
                (
                    positions[0],
                    positions[1],
                ),
                (s[0], s[1]),
            )
            data[indicies] = data_masked

        if s[3] % 2 > 0:
            data_reshaped[:, self._circular_mask_ravel] = xp.repeat(data, 2, axis=1)[
                :, 1:
            ]
            data_reshaped[1:] /= 2
        else:
            data_reshaped[:, self._circular_mask_ravel] = xp.repeat(data, 2, axis=1) / 2
        data_reshaped[:, self._circular_mask_ravel] = data_reshaped[
            :, self._circular_mask_ravel
        ][:, i]
        data_reshaped = data_reshaped.reshape((s[0], s[1], s[2], s[3]))

        return data_reshaped

    def _forward(
        self,
        datacube_number: float,
        x_index: int,
        num_points: int,
    ):
        """
        Forward projection of object for simulation of diffraction data

        Parameters
        ----------
        datacube_number: float
            index of datacube
        x_index: int
            x slice for forward projection
        num_points: int
            number of points for bilinear interpolation

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        diffraction_patterns_reshaped: np.ndarray
            datacube with diffraction data reshapped in 2D arrays
        """
        xp = self._xp
        s = self._object_shape_6D
        device = self._device
        obj = copy_to_device(self._object[x_index], device)

        s_max = np.max((s[1], s[2]))
        ind_real = self._ind_real[datacube_number].reshape((4, s_max, s_max))
        ind_diff = self._ind_diff[datacube_number].reshape((4, s[-1], s[-1]))
        weights_real = self._weights_real[datacube_number].reshape((4, s_max, s_max))
        weights_diff = self._weights_diff[datacube_number].reshape((4, s[-1], s[-1]))

        obj_q_summed = (obj[:, ind_diff] * weights_diff).sum((1))
        bincount_diff = (
            xp.tile(
                self._ind_diffraction_ravel,
                (s[1] * s[2]),
            )
            + xp.repeat(
                xp.arange(s[1] * s[2]), obj_q_summed.shape[1] * obj_q_summed.shape[2]
            )
            * self._q_length
        )

        obj_q_summed = xp.bincount(
            bincount_diff,
            obj_q_summed.ravel(),
            minlength=s[1] * s[2] * self._q_length,
        ).reshape((-1, self._q_length))[:, self._circular_mask_bincount]

        obj_projected = (obj_q_summed[ind_real] * weights_real[:, :, :, None]).sum(
            (0, 2)
        )

        return obj_projected

    def _calculate_update(
        self,
        object_sliced,
        diffraction_patterns_projected,
        datacube_number,
        x_index,
    ):
        """
        Calculate update for back projection

        Parameters
        ----------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        diffraction_patterns_projected: np.ndarray
            projected diffraction patterns for the relevant tilt
        datacube_number: int
            index of datacube
        x_index: int
            x slice of object to be sliced


        Returns
        --------
        update: np.ndarray
            difference between current object sliced in diffraciton space and
            experimental diffraction patterns
        """
        xp = self._xp
        device = self._device

        s = self._object_shape_6D

        ind0 = self._positions_vox_F[datacube_number][0] == x_index - 1
        ind1 = self._positions_vox_F[datacube_number][0] == x_index

        dp_length = diffraction_patterns_projected.shape[1]

        dp_patterns = np.hstack(
            [
                diffraction_patterns_projected[ind0].ravel(),
                diffraction_patterns_projected[ind0].ravel(),
                diffraction_patterns_projected[ind1].ravel(),
                diffraction_patterns_projected[ind1].ravel(),
            ]
        )

        if dp_patterns.shape[0] == 0:
            update = xp.zeros(object_sliced.shape)
            error = 0
            error = copy_to_device(error, "cpu")

        else:
            weights = np.hstack(
                [
                    np.repeat(
                        (self._positions_vox_dF[datacube_number][0][ind0])
                        * (1 - self._positions_vox_dF[datacube_number][1][ind0]),
                        dp_length,
                    ),
                    np.repeat(
                        (self._positions_vox_dF[datacube_number][0][ind0])
                        * (self._positions_vox_dF[datacube_number][1][ind0]),
                        dp_length,
                    ),
                    np.repeat(
                        (1 - self._positions_vox_dF[datacube_number][0][ind1])
                        * (1 - self._positions_vox_dF[datacube_number][1][ind1]),
                        dp_length,
                    ),
                    np.repeat(
                        (1 - self._positions_vox_dF[datacube_number][0][ind1])
                        * (self._positions_vox_dF[datacube_number][1][ind1]),
                        dp_length,
                    ),
                ]
            )

            weights = copy_to_device(weights, device)

            positions_y = xp.clip(
                xp.hstack(
                    [
                        self._positions_vox_F[datacube_number][1][ind0],
                        self._positions_vox_F[datacube_number][1][ind0] + 1,
                        self._positions_vox_F[datacube_number][1][ind1],
                        self._positions_vox_F[datacube_number][1][ind1] + 1,
                    ],
                ),
                0,
                s[1] - 1,
            )

            bincount_x = (
                xp.tile(xp.arange(dp_length), dp_patterns.shape[0] // dp_length)
                + xp.repeat(positions_y, dp_length) * dp_length
            )

            bincount_x = xp.asarray(bincount_x, dtype="int")

            dp_patterns_counted = xp.bincount(
                bincount_x, weights=dp_patterns * weights, minlength=s[1] * dp_length
            ).reshape((s[1], dp_length))

            update = dp_patterns_counted - object_sliced

            error = (
                xp.mean(update.ravel() ** 2) ** 0.5 / dp_patterns_counted.mean(0).sum()
            )

            error = copy_to_device(error, "cpu")

        return update, error

    def _back(
        self,
        num_points: int,
        datacube_number: int,
        x_index: int,
        update,
    ):
        """
        back propagate

        Parameters
        ----------
        num_points: int
            number of points for bilinear interpolation
        datacube_number: int
            index of datacube
        x_index: int
            x slice for back projection
        update: np.ndarray
            difference between current object sliced in diffraciton space and
            experimental diffraction patterns
        """
        xp = self._xp
        storage = self._storage

        s = self._object_shape_6D
        s_max = np.max((s[1], s[2]))

        ind_update = xp.tile(self._circular_mask_ravel, 4)

        a = xp.argsort(self._ind_diffraction_ravel[self._circular_mask_ravel])
        i = xp.empty_like(a)
        i[a] = xp.arange(a.size)
        i = xp.tile(i, 4) + xp.repeat(xp.arange(4), i.shape[0]) * (i.shape[0])

        if s[-1] % 2 > 0:
            normalize = xp.ones((xp.repeat(update, 2, axis=1)[:, 1:]).shape) * 2
            normalize[:, 0] = 1

            update_reshaped = (
                (xp.tile(xp.repeat(update, 2, axis=1)[:, 1:] / normalize, 4))[:, i]
            ) * (self._weights_diff[datacube_number][ind_update])
        else:
            normalize = xp.ones((xp.repeat(update, 2, axis=1)).shape) * 2

            update_reshaped = (
                (xp.tile(xp.repeat(update, 2, axis=1) / normalize, (4)))[:, i]
            ) * (self._weights_diff[datacube_number][ind_update])

        ind_real = self._ind_real[datacube_number].ravel()
        ind_diff = self._ind_diff[datacube_number][ind_update]

        ind_diff_bincount = xp.bincount(ind_diff)
        diff_max = ind_diff_bincount.shape[0]

        real_shape = ind_real.shape[0]
        diff_shape = ind_diff.shape[0]

        bincount_diff = (
            xp.tile(ind_diff, s_max)
            + (xp.repeat(xp.arange(s_max), diff_shape)) * diff_max
        )

        update_q_summed = xp.bincount(
            bincount_diff,
            update_reshaped.ravel(),
            minlength=((diff_max) * s_max),
        ).reshape((s_max, -1))[:, ind_diff_bincount > 0]

        # update_q_summed = xp.tile(update_q_summed, (s[2] * 4, 1)) / (s[2])
        update_q_summed = xp.tile(xp.repeat(update_q_summed, s_max, axis=0), (4, 1)) / (
            s_max
        )

        diff_shape_bin = update_q_summed.shape[-1]

        ind_real_bincount = xp.bincount(ind_real)
        real_max = ind_real_bincount.shape[0]

        bincount_real = (
            xp.tile(xp.arange(diff_shape_bin), real_shape)
            + xp.repeat(ind_real, diff_shape_bin) * diff_shape_bin
        )

        update_r_summed = (
            xp.bincount(
                bincount_real,
                (
                    update_q_summed
                    * self._weights_real[datacube_number].ravel()[:, None]
                ).ravel(),
                minlength=((real_max) * diff_shape_bin),
            )
        ).reshape((-1, diff_shape_bin))[ind_real_bincount > 0]

        weights_diff_all_counted = self._weights_diff_all_counted
        weights_diff_all_counted = weights_diff_all_counted[xp.unique(ind_diff)]
        weights_diff_all_counted[weights_diff_all_counted < 1] = 1
        update_r_summed[None, :] = update_r_summed[None, :] / weights_diff_all_counted

        i_real, i_diff = xp.meshgrid(
            xp.unique(ind_real), xp.unique(ind_diff), indexing="ij"
        )

        i_real = copy_to_device(i_real, storage)
        i_diff = copy_to_device(i_diff, storage)

        return x_index, i_real, i_diff, copy_to_device(update_r_summed, storage)

    def _constraints(
        self,
        zero_edges_real: bool,
        zero_edges_diffraction: bool,
        cylinder_mask: bool,
        baseline_thresh: float,
        diffraction_gaussian_filter: float,
        real_space_gaussian_filter: float,
        diffraction_shrinkage: bool,
        diffraction_shrinkage_threshold: float,
    ):
        """
        Constrains for object
        TODO: add constrains and break into multiple functions possibly

        Parameters
        ----------
        zero_edges_real: bool
            If True, zeros edges along y and z
        zero_edges_diffraction: bool
            If True, zeros diffraction edges with spherical mask
        cylinderical_mask: bool
            If True, applies cylinderical mask
        diffraction_gaussian_filter: float
            Gaussian filter sigma for diffraction space (in pixels)
        real_space_gaussian_filter: flooat
            Gaussian filter for real space (in pixels)
        diffraction_shrinkage: bool
            if True, subtracts baseline from each kernel in real space and zeros any residual negative values. If no
            theshold is provided, uses mean of object
        diffraction_shrinkage_threshold: None
            threshold for shrinkage
        """
        if cylinder_mask:
            storage = self._storage
            s = self._object_shape_6D

            int_zero = (
                copy_to_device(
                    self._cylinder_mask.reshape((s[0], s[1] * s[2])), storage
                )
                == 0
            )

            self._object[int_zero] = 0

        elif zero_edges_real:
            xp = self._xp_storage
            s = self._object_shape_6D
            y = xp.arange(s[1])
            z = xp.arange(s[2])
            yy, zz = xp.meshgrid(y, z, indexing="ij")
            ind_zero = xp.where(
                (yy.ravel() == 0)
                | (zz.ravel() == 0)
                | (yy.ravel() == y.max())
                | (zz.ravel() == z.max())
            )[0]
            self._object[:, ind_zero] = 0

        if zero_edges_diffraction:
            storage = self._storage
            diffraction_edge_mask = copy_to_device(self._diffraction_edge_mask, storage)
            self._object = self._object * diffraction_edge_mask[None, None, :]

        if diffraction_gaussian_filter > 0:
            from scipy.ndimage import gaussian_filter

            storage = self._storage
            s = self._object.shape

            obj_6D = copy_to_device(self.object_6D, device="cpu")

            obj_6D = gaussian_filter(
                obj_6D, diffraction_gaussian_filter, axes=(-1, -2, -3)
            )  # axes only supported in cpu

            self._object = copy_to_device(obj_6D.reshape(s), device=storage)

        if real_space_gaussian_filter > 0:
            from scipy.ndimage import gaussian_filter

            storage = self._storage
            s = self._object.shape

            obj_6D = copy_to_device(self.object_6D, device="cpu")

            obj_6D = gaussian_filter(
                obj_6D, real_space_gaussian_filter, axes=(0, 1, 2)
            )  # axes only supported in cpu

            self._object = copy_to_device(obj_6D.reshape(s), device=storage)

        if baseline_thresh is not None:
            _, vmin, _ = return_scaled_histogram_ordering(
                self._object, vmin=baseline_thresh
            )
            xp = self._xp_storage
            self._object = xp.clip(self._object - vmin, 0, np.inf)

        if diffraction_shrinkage is True:
            if diffraction_shrinkage_threshold is None:
                diffraction_shrinkage_threshold = self._object.mean()
            self._object -= diffraction_shrinkage_threshold
            self._object[self._object < 0] = 0

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

    def set_device(self, device, clear_fft_cache):
        """
        Sets calculation device.

        Parameters
        ----------
        device: str
            Calculation device will be perfomed on. Must be 'cpu' or 'gpu'

        Returns
        --------
        self: TomoReconstruction
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

    @property
    def object_6D(self):
        """6D object"""
        return copy_to_device(self._object.reshape(self._object_shape_6D), "cpu")

    def recovered_4D_scan(self, index):
        """recovered 4D-STEM scan from projected patterns"""

        scan = self._reshape_2D_array_to_4D(
            self._diffraction_patterns_projected[index],
            positions=self._positions_vox_F[index],
        )

        return copy_to_device(scan, "cpu")

    def visualize(self, plot_convergence=True, figsize=(10, 10)):
        """
        vis
        """

        if plot_convergence:
            spec = GridSpec(
                ncols=2,
                nrows=2,
                height_ratios=[4, 1],
                hspace=0.15,
                # width_ratios=[
                #     (extent[1] / extent[2]) / (probe_extent[1] / probe_extent[2]),
                #     1,
                # ],
                wspace=0.15,
            )

        else:
            spec = GridSpec(ncols=2, nrows=1)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(spec[0, 0])
        show(
            self.object_6D.mean((2, 3, 4, 5)),
            figax=(fig, ax),
            cmap="magma",
            title="real space object",
        )

        ax = fig.add_subplot(spec[0, 1])
        ind_diff = self._object_shape_6D[-1] // 2
        show(
            self.object_6D.mean((0, 1, 2, 5)),
            figax=(fig, ax),
            cmap="magma",
            title="diffraction space object",
        )

        if plot_convergence:
            ax = fig.add_subplot(spec[1, :])
            ax.plot(self.error_iterations, color="b")
            ax.set_xlabel("iterations")
            ax.set_ylabel("error")

        return self

    def show_error_per_iteration(self, **kwargs):
        num_iter = len(self._tilt_order) // self._num_datacubes
        iterations = np.repeat(np.arange(num_iter), self._num_datacubes)
        order = np.asarray(self._tilt_order)
        ind = np.argsort(
            np.ravel_multi_index((iterations, order), (num_iter, self._num_datacubes))
        )
        error = (np.asarray(self._error_per_step)[ind]).reshape(
            (num_iter, self._num_datacubes)
        )

        tilts_order = self._tilt_deg
        tilts_order = np.argsort(tilts_order)

        error = error[:, tilts_order]

        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        fig, ax = show(
            error, cmap="magma", returnfig=True, vmax=vmax, vmin=vmin, **kwargs
        )

        ax.set_title("error")
        ax.set_ylabel("iteration")
        ax.set_xlabel("tilts (negative -> positive)")
        ax.set_xticks([])

        return self

    def widget(
        self,
        cyliner_mask=False,
        mode="dark-field",
        virtual_image_mask_radius=4,
        **kwargs,
    ):
        """ """
        from ipywidgets import HBox, VBox, widgets, interact, Dropdown, Label, Layout
        from skimage.feature import peak_local_max
        from scipy.ndimage import gaussian_filter
        from py4DSTEM.visualize import return_scaled_histogram_ordering

        obj_6D = self.object_6D.copy()
        obj_6D /= obj_6D.mean()

        if cyliner_mask:
            cylinder_mask = self._cylinder_mask
            obj_6D *= cyliner_mask[:, :, :, None, None, None]

        diffraction_kernel = np.ones((obj_6D.shape[3:]))
        if mode == "dark-field":
            center = obj_6D.shape[3] // 2
            diffraction_kernel[
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
            ] = 0
        elif mode == "bright-field":
            center = obj_6D.shape[3] // 2
            diffraction_kernel[
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
                center - virtual_image_mask_radius : center + virtual_image_mask_radius,
            ] = 0
            diffraction_kernel = -1 * diffraction_kernel + 1

        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        _, vmin, vmax = return_scaled_histogram_ordering(
            ((obj_6D) * diffraction_kernel[None, None, None, :, :, :]).mean((3, 4, 5)),
            vmin=vmin,
            vmax=vmax,
        )

        # %matplotlib ipympl

        with plt.ioff():
            fig = plt.figure(figsize=(6.5, 3))
            ax0 = fig.add_subplot(1, 3, 1)
            ax1 = fig.add_subplot(1, 3, 2)
            ax2 = fig.add_subplot(1, 3, 3, projection="3d")

        x = obj_6D.shape[0] // 2
        y = obj_6D.shape[1] // 2
        z = obj_6D.shape[2] // 2
        gaussian_filter_sigma = 1.5
        min_distance = 2

        ax0.imshow(
            (obj_6D[:, :, z] * diffraction_kernel[None, None, :, :, :]).mean((2, 3, 4)),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax0.scatter(y, x, color="red")

        ax1.imshow(
            (
                obj_6D[
                    :,
                    y,
                    :,
                ]
                * diffraction_kernel[None, None, :, :, :]
            )
            .mean((2, 3, 4))
            .T,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        ax1.scatter(x, z, color="red")

        plot_data = gaussian_filter(
            obj_6D[
                x,
                y,
                z,
            ],
            gaussian_filter_sigma,
        )
        ind = peak_local_max(plot_data, min_distance=min_distance)
        ax2.scatter(
            ind[:, 0],
            ind[:, 1],
            ind[:, 2],
            s=plot_data[ind[:, 0], ind[:, 1], ind[:, 2]] / 2,
            color="red",
        )

        ax0.set_title("xy")
        ax1.set_title("xz")
        ax2.set_title("Diffraction")

        ax0.set_xticks([])
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax0.set_yticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])

        ax0.set_xlabel("y")
        ax0.set_ylabel("x")

        ax1.set_xlabel("x")
        ax1.set_ylabel("z")

        ax2.set_xlim([0, obj_6D.shape[3]])
        ax2.set_ylim([0, obj_6D.shape[4]])
        ax2.set_zlim([0, obj_6D.shape[5]])

        plt.tight_layout()

        im0 = ax0.get_images()[0]
        im1 = ax1.get_images()[0]

        # interact
        def update_images(
            x,
            y,
            z,
            gaussian_filter_diffraction,
            minimum_threshold,
            intensities_power,
            scale_intensities,
            block_center,
        ):
            ax0.clear()
            ax0.imshow(
                (
                    obj_6D[
                        :,
                        :,
                        z,
                    ]
                    * diffraction_kernel[None, None, :, :, :]
                ).mean((2, 3, 4)),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            ax0.scatter(y, x, color="red")

            ax1.clear()
            ax1.imshow(
                (
                    obj_6D[
                        :,
                        y,
                        :,
                    ]
                    * diffraction_kernel[None, None, :, :, :]
                )
                .mean((2, 3, 4))
                .T,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            ax1.scatter(x, z, color="red")

            plot_data = gaussian_filter(obj_6D[x, y, z], gaussian_filter_diffraction)
            ind = peak_local_max(plot_data, min_distance=min_distance)

            max_intensity = np.max(plot_data[ind[:, 0], ind[:, 1], ind[:, 2]])
            min_intensity = minimum_threshold * max_intensity

            ind_keep = (
                plot_data[ind[:, 0], ind[:, 1], ind[:, 2]] ** scale_intensities
                > min_intensity**intensities_power
            )
            ind = ind[ind_keep]

            if block_center:
                center = (obj_6D.shape[3] / 2, obj_6D.shape[4] / 2, obj_6D.shape[5] / 2)
                dist = (
                    (ind[:, 0] - center[0]) ** 2
                    + (ind[:, 1] - center[1]) ** 2
                    + (ind[:, 2] - center[2]) ** 2
                ) ** 0.5
                ind = ind[dist > min_distance]

            ax2.clear()
            ax2.scatter(
                ind[:, 0],
                ind[:, 1],
                ind[:, 2],
                s=scale_intensities
                * plot_data[ind[:, 0], ind[:, 1], ind[:, 2]] ** intensities_power,
                color="red",
            )

            ax0.set_xticks([])
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax0.set_yticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])

            ax0.set_xlabel("y")
            ax0.set_ylabel("x")

            ax1.set_xlabel("x")
            ax1.set_ylabel("z")

            ax2.set_xlim([0, obj_6D.shape[3]])
            ax2.set_ylim([0, obj_6D.shape[4]])
            ax2.set_zlim([0, obj_6D.shape[5]])

            ax0.set_title("xy")
            ax1.set_title("xz")
            ax2.set_title("Diffraction")

            plt.tight_layout()

            fig.canvas.draw_idle()

        style = {
            "description_width": "initial",
        }

        layout = Layout(width="250px", height="30px")

        x = widgets.IntSlider(
            value=obj_6D.shape[0] // 2,
            min=0,
            max=obj_6D.shape[0] - 1,
            step=1,
            description="x",
            style=style,
            layout=layout,
        )

        y = widgets.IntSlider(
            value=obj_6D.shape[1] // 2,
            min=0,
            max=obj_6D.shape[1] - 1,
            step=1,
            description="y",
            style=style,
            layout=layout,
        )

        z = widgets.IntSlider(
            value=obj_6D.shape[2] // 2,
            min=0,
            max=obj_6D.shape[2] - 1,
            step=1,
            description="z",
            style=style,
            layout=layout,
        )

        gaussian_filter_diffraction = widgets.FloatSlider(
            value=1.2,
            min=0,
            max=3,
            step=0.1,
            description="filter",
            style=style,
            layout=layout,
        )

        minimum_threshold = widgets.FloatSlider(
            value=0.25,
            min=0,
            max=1,
            step=0.01,
            description="min threshold",
            style=style,
            layout=layout,
        )

        intensities_power = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=2,
            step=0.05,
            description="intensities power",
            style=style,
            layout=layout,
        )

        scale_intensities = widgets.FloatSlider(
            value=2,
            min=0,
            max=20,
            step=0.2,
            description="scale intensities",
            style=style,
            layout=layout,
        )

        block_center = widgets.Checkbox(
            value=True, description="block center", disabled=False
        )

        widgets.interactive_output(
            update_images,
            {
                "x": x,
                "y": y,
                "z": z,
                "gaussian_filter_diffraction": gaussian_filter_diffraction,
                "minimum_threshold": minimum_threshold,
                "intensities_power": intensities_power,
                "scale_intensities": scale_intensities,
                "block_center": block_center,
            },
        )

        fig.canvas.resizable = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_visible = True
        fig.canvas.layout.width = "675px"
        fig.canvas.layout.height = "400px"
        fig.canvas.toolbar_position = "bottom"

        widget = widgets.VBox(
            [
                fig.canvas,
                HBox([x, y]),
                HBox([z, gaussian_filter_diffraction]),
                HBox([minimum_threshold, scale_intensities]),
                HBox([intensities_power, block_center]),
            ],
        )

        display(widget)

        return self
