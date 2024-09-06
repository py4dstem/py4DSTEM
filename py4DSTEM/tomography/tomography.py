from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from emdfile import tqdmnd

from py4DSTEM import show
import numpy as np
from py4DSTEM.datacube import DataCube
from py4DSTEM.preprocess.utils import bin2D
from py4DSTEM.process.calibration import fit_origin, get_origin
from py4DSTEM.process.diffraction import Crystal
from py4DSTEM.process.phase.utils import copy_to_device
from py4DSTEM.utils import fourier_resample
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation as R

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
        tilt_deg: Sequence[np.ndarray] = None,
        translation_px: Sequence[np.ndarray] = None,
        scanning_to_tilt_rotation: float = None,
        initial_object_guess: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = "cpu",
        clear_fft_cache: bool = True,
        name: str = "tomography",
    ):
        """
        Nanobeam  tomography!

        """

        self._datacubes = datacubes
        self._import_kwargs = import_kwargs
        self._object_shape_x_y_z = object_shape_x_y_z
        self._voxel_size_A = voxel_size_A
        self._datacube_R_pixel_size_A = datacube_R_pixel_size_A
        self._datacube_Q_pixel_size_inv_A = datacube_Q_pixel_size_inv_A
        self._tilt_deg = tilt_deg
        self._translation_px = translation_px
        self._scanning_to_tilt_rotation = scanning_to_tilt_rotation
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
        q_max_inv_A: int = None,
        force_q_to_r_rotation_deg: float = None,
        force_q_to_r_transpose: bool = None,
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
    ):
        """
        diffraction_intensites_shape: int
            shape of diffraction patterns to reshape data into
        resizing_method: float
            method to reshape diffraction space ("bin", "fourier", "bilinear")
        bin_real_space: int
            factor for binnning in real space
        crop_reciprocal_space: float
            if not None, crops reciprocal space on all sides by integer
        q_max_inv_A: int
            maximum q in inverse angstroms
        force_q_to_r_rotation_deg:float
            force q to r rotation in degrees. If False solves for rotation
            with datacube specified with `datacube_to_solve_rotation` using
            center of mass method.
        force_q_to_r_transpose: bool
            force q to r transpose. If False, solves for transpose
            with datacube specified with `datacube_to_solve_rotation` using
            center of mass method.
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
        fitfunction: "str"
            fit function for origin ('plane' or 'parabola' or 'bezier_two' or 'constant').
        robust: bool
            If set to True, origin fit will be repeated with outliers
            removed.
        robust_steps: int
            number of robust iterations performed after initial fit.
        robust_thresh: int
            threshold for including points, in units of root-mean-square (standard deviations) error
            of the predicted values after fitting.
        """
        xp_storage = self._xp_storage
        storage = self._storage

        self._num_datacubes = len(self._datacubes)

        self._diffraction_patterns_projected = []
        self._positions_ang = []
        self._positions_vox = []
        self._positions_vox_F = []
        self._positions_vox_dF = []

        # preprocessing of diffraction data
        for a0 in range(self._num_datacubes):
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
            )

        return self

    def reconstruct(
        self,
        num_iter: int = 1,
        store_iterations: bool = False,
        reset: bool = True,
        step_size: float = 0.5,
        num_points: int = 60,
        progress_bar: bool = True,
        zero_edges: bool = True,
    ):
        """
        Main loop for reconstruct

        Parameters
        ----------
        num_iter: int
            Number of iterations
        store_iterations: bool
            if True, stores number of iterations
        reset: bool
            if True, resets object
        step_size: float
            from 0 to 1, step size for update
        num_points: int
            number of points for bilinear interpolation in real space
        progres_bar: bool
            if True, shows progress bar
        zero_edges: bool
            If True, zero edges along y and z
        """
        device = self._device

        if reset is True:
            self.error_iterations = []

            if store_iterations:
                self.object_iterations = []

            self._object = self._object_initial.copy()

        for a0 in tqdmnd(
            num_iter,
            desc="Reconstructing object",
            unit=" iter",
            disable=not progress_bar,
        ):
            error_iteration = 0
            random_tilt_order = np.arange(self._num_datacubes)
            np.random.shuffle(random_tilt_order)

            for a1 in range(self._num_datacubes):
                a1_shuffle = random_tilt_order[a1]
                diffraction_patterns_projected = copy_to_device(
                    self._diffraction_patterns_projected[a1_shuffle], device
                )

                for a2 in range(self._object_shape_6D[0]):
                    object_sliced = self._forward(
                        x_index=a2,
                        tilt_deg=self._tilt_deg[a1_shuffle],
                        num_points=num_points,
                    )

                    update, error = self._calculate_update(
                        object_sliced=object_sliced,
                        diffraction_patterns_projected=diffraction_patterns_projected,
                        x_index=a2,
                        datacube_number=a1_shuffle,
                    )

                    error_iteration += error

                    update *= step_size
                    self._back(
                        num_points=num_points,
                        x_index=a2,
                        update=update,
                    )

            self._constraints(
                zero_edges=zero_edges,
            )

            self.error_iterations.append(error_iteration)
            self.error = error_iteration
            if store_iterations:
                self.object_iterations.append(self._object.copy())

        return self

    def _prepare_datacube(
        self,
        datacube_number,
        diffraction_intensities_shape,
        diffraction_space_mask_com,
        resizing_method,
        bin_real_space,
        masks_real_space,
        crop_reciprocal_space,
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
            if not None, crops reciprocal space on all sides by integer
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

        if masks_real_space is not None:
            if type(masks_real_space) is np.ndarray:
                mask_real_space = masks_real_space
            else:
                mask_real_space = masks_real_space[datacube_number]
            mask_real_space = np.ndarray(masks_real_space, dtype="bool")
        else:
            mask_real_space = None

        if crop_reciprocal_space is not None:
            datacube.crop_Q(
                (
                    crop_reciprocal_space,
                    -crop_reciprocal_space,
                    crop_reciprocal_space,
                    -crop_reciprocal_space,
                )
            )

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
        field_of_view_px = self._object_initial.shape[0:2]
        self._field_of_view_A = (
            self._voxel_size_A * field_of_view_px[0],
            self._voxel_size_A * field_of_view_px[1],
        )

        # calculate positions
        s = self._initial_datacube_shape

        step_size = self._datacube_R_pixel_size_A

        x = np.arange(s[0])
        y = np.arange(s[1])

        if self._translation_px is not None:
            x += self._translation_px[datacube_number][0]
            y += self._translation_px[datacube_number][1]

        x *= step_size
        y *= step_size

        x, y = np.meshgrid(x, y, indexing="ij")

        if self._scanning_to_tilt_rotation is not None:
            rotation_angle = np.deg2rad(self._scanning_to_tilt_rotation)
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

        # remove data outside FOV
        if mask_real_space is None:
            mask_real_space = np.ones(x.shape, dtype="bool")
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
        self._positions_ang.append((x, y))
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

    # def _solve_for_center_of_mass_relative_rotation():

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
            If set to True, origin fit will be repeated with outliers
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
        """
        # calculate bincount array
        if datacube_number == 0:
            self._make_diffraction_masks(q_max_inv_A=q_max_inv_A)

            # TODO deal with with diffraction patterns cut off by detector?

        diffraction_patterns_reshaped = self._reshape_4D_array_to_2D(
            data=datacube.data,
            qx0_fit=qx0_fit,
            qy0_fit=qy0_fit,
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

        self._ind_diffraction = ind_diffraction
        self._ind_diffraction_ravel = ind_diffraction.ravel()
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

    def _reshape_4D_array_to_2D(self, data, qx0_fit=None, qy0_fit=None):
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


        Returns
        --------
        diffraction_patterns_reshaped: np.ndarray
            diffraction patterns ravelled
        """

        s = data.shape

        center = ((s[-1] - 1) / 2, (s[-1] - 1) / 2)
        diffraction_patterns_reshaped = np.zeros((s[0] * s[1], self._q_length))

        for a0 in range(s[0]):
            for a1 in range(s[0]):
                dp = data[a0, a1]
                index = np.ravel_multi_index((a0, a1), (s[0], s[1]))

                if qx0_fit is not None:
                    qx0 = qx0_fit[a0, a1] - center[0]
                    qy0 = qy0_fit[a0, a1] - center[1]

                    xF = int(np.floor(qx0))
                    yF = int(np.floor(qy0))

                    wx = qx0 - xF
                    wy = qy0 - yF

                    diffraction_patterns_reshaped[index] = (
                        (
                            (
                                (1 - wx)
                                * (1 - wy)
                                * np.bincount(
                                    self._ind_diffraction_ravel,
                                    np.roll(dp, (xF, yF), axis=(0, 1)).ravel(),
                                    minlength=self._q_length,
                                )
                            )
                        )
                        + (
                            (wx)
                            * (1 - wy)
                            * np.bincount(
                                self._ind_diffraction_ravel,
                                np.roll(dp, (xF + 1, yF), axis=(0, 1)).ravel(),
                                minlength=self._q_length,
                            )
                        )
                        + (
                            (1 - wx)
                            * (wy)
                            * np.bincount(
                                self._ind_diffraction_ravel,
                                np.roll(dp, (xF, yF + 1), axis=(0, 1)).ravel(),
                                minlength=self._q_length,
                            )
                        )
                        + (
                            (wx)
                            * (wy)
                            * np.bincount(
                                self._ind_diffraction_ravel,
                                np.roll(dp, (xF + 1, yF + 1), axis=(0, 1)).ravel(),
                                minlength=self._q_length,
                            )
                        )
                    )
                else:

                    diffraction_patterns_reshaped[index] = np.bincount(
                        self._ind_diffraction_ravel,
                        dp.ravel(),
                        minlength=self._q_length,
                    )

        diffraction_patterns_reshaped = diffraction_patterns_reshaped[
            :, self._circular_mask_bincount
        ]
        return diffraction_patterns_reshaped

    def _reshape_2D_array_to_4D(self, data, xy_shape=None):
        """
        reshape ravelled diffraction 2D-data to 4D-data

        Parameters
        ----------
        data: np.ndarrray
            2D datacube data to be reshapped
        xy_shape: 2-tuple
            if None, takes 6D object shape

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
        data_reshaped[:, self._circular_mask_ravel] = xp.repeat(data, 2, axis=1)[:, 1:]
        data_reshaped[1:] /= 2
        data_reshaped[:, self._circular_mask_ravel] = data_reshaped[
            :, self._circular_mask_ravel
        ][:, i]
        data_reshaped = data_reshaped.reshape((s[0], s[1], s[2], s[3]))

        return data_reshaped

    def _real_space_radon(
        self,
        current_object: np.ndarray,
        tilt_deg: int,
        x_index: int,
        num_points: int,
    ):
        """
        Real space projection of current object

        Parameters
        ----------
        current_object: np.ndarray
            current object estimate
        tilt_deg: float
            tilt of object in degrees
        x_index: int
            x slice of object to be sliced
        num_points: float
            number of points for bilinear interpolation

        Returns
        --------
        current_object_projected: np.ndarray
            projection of current object

        """
        xp = self._xp
        device = self._device

        current_object = copy_to_device(current_object, device)

        s = current_object.shape

        tilt = xp.deg2rad(tilt_deg)

        padding = int(xp.ceil(xp.abs(xp.tan(tilt) * s[2])))

        line_z = xp.arange(0, 1, 1 / num_points) * (s[2] - 1)
        line_y = line_z * xp.tan(tilt) + padding

        offset = xp.arange(s[1], dtype="int")

        current_object_reshape = xp.pad(
            current_object[x_index],
            ((padding, padding), (0, 0), (0, 0), (0, 0), (0, 0)),
        ).reshape(((s[1] + padding * 2) * s[2], s[3], s[4], s[5]))

        current_object_projected = xp.zeros((s[1], s[3], s[4], s[5]))

        yF = xp.floor(line_y).astype("int")
        zF = xp.floor(line_z).astype("int")
        dy = line_y - yF
        dz = line_z - zF

        ind0 = np.hstack(
            (
                xp.tile(yF, (s[1], 1)) + offset[:, None],
                xp.tile(yF + 1, (s[1], 1)) + offset[:, None],
                xp.tile(yF, (s[1], 1)) + offset[:, None],
                xp.tile(yF + 1, (s[1], 1)) + offset[:, None],
            )
        )

        ind1 = np.hstack(
            (
                xp.tile(zF, (s[1], 1)),
                xp.tile(zF, (s[1], 1)),
                xp.tile(zF + 1, (s[1], 1)),
                xp.tile(zF + 1, (s[1], 1)),
            )
        )

        weights = np.hstack(
            (
                xp.tile(((1 - dy) * (1 - dz)), (s[1], 1)),
                xp.tile(((dy) * (1 - dz)), (s[1], 1)),
                xp.tile(((1 - dy) * (dz)), (s[1], 1)),
                xp.tile(((dy) * (dz)), (s[1], 1)),
            )
        )

        current_object_projected += (
            current_object_reshape[
                xp.ravel_multi_index(
                    (ind0, ind1), (s[1] + 2 * padding, s[2]), mode="clip"
                )
            ]
            * weights[:, :, None, None, None]
        ).sum(1)

        return current_object_projected

    def _diffraction_space_slice(
        self,
        current_object_projected: np.ndarray,
        tilt_deg: int,
    ):
        """
        Slicing of diffraction space for rotated object

        Parameters
        ----------
        current_object_rotated: np.ndarray
            current object estimate projected
        tilt_deg: float
            tilt of object in degrees

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space

        """
        xp = self._xp

        s = current_object_projected.shape

        tilt = xp.deg2rad(tilt_deg)

        line_y_diff = xp.fft.fftfreq(s[-1], 1 / s[-1]) * xp.cos(tilt)
        line_z_diff = xp.fft.fftfreq(s[-1], 1 / s[-1]) * xp.sin(tilt)

        yF_diff = xp.floor(line_y_diff).astype("int")
        zF_diff = xp.floor(line_z_diff).astype("int")
        dy_diff = line_y_diff - yF_diff
        dz_diff = line_z_diff - zF_diff

        current_object_sliced = xp.zeros((s[0], s[-1], s[-1]))

        current_object_sliced = (
            current_object_projected[:, :, yF_diff, zF_diff]
            * ((1 - dy_diff) * (1 - dz_diff))[None, None, :]
            + current_object_projected[:, :, yF_diff + 1, zF_diff]
            * ((dy_diff) * (1 - dz_diff))[None, None, :]
            + current_object_projected[:, :, yF_diff, zF_diff + 1]
            * ((1 - dy_diff) * (dz_diff))[None, None, :]
            + current_object_projected[:, :, yF_diff + 1, zF_diff + 1]
            * ((dy_diff) * (dz_diff))[None, None, :]
        )

        return self._asnumpy(current_object_sliced)

    def _forward(
        self,
        x_index: int,
        tilt_deg: float,
        num_points: int,
    ):
        """
        Forward projection of object for simulation of diffraction data

        Parameters
        ----------
        x_index: int
            x slice for forward projection
        tilt_deg: float
            tilt of object in degrees
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

        tilt = xp.deg2rad(tilt_deg)

        # solve for real space coordinates
        line_z = xp.linspace(0, 1, num_points) * (s[2] - 1)
        line_y = line_z * xp.tan(tilt)
        offset = xp.arange(s[1], dtype="int")

        yF = xp.floor(line_y).astype("int")
        zF = xp.floor(line_z).astype("int")
        dy = line_y - yF
        dz = line_z - zF

        ind0 = np.hstack(
            (
                xp.tile(yF, (s[1], 1)) + offset[:, None],
                xp.tile(yF + 1, (s[1], 1)) + offset[:, None],
                xp.tile(yF, (s[1], 1)) + offset[:, None],
                xp.tile(yF + 1, (s[1], 1)) + offset[:, None],
            )
        )

        ind1 = np.hstack(
            (
                xp.tile(zF, (s[1], 1)),
                xp.tile(zF, (s[1], 1)),
                xp.tile(zF + 1, (s[1], 1)),
                xp.tile(zF + 1, (s[1], 1)),
            )
        )

        weights_real = np.hstack(
            (
                xp.tile(((1 - dy) * (1 - dz)), (s[1], 1)),
                xp.tile(((dy) * (1 - dz)), (s[1], 1)),
                xp.tile(((1 - dy) * (dz)), (s[1], 1)),
                xp.tile(((dy) * (dz)), (s[1], 1)),
            )
        )

        # solve for diffraction space coordinates
        length = s[-1] * np.cos(tilt)
        line_y_diff = xp.arange(-(s[-1] - 1) / 2, s[-1] / 2) * length / s[-1]
        line_z_diff = line_y_diff * xp.tan(tilt) + (s[-1] - 1) / 2
        line_y_diff += (s[-1] - 1) / 2

        # line_y_diff = np.fft.fftfreq(s[-1], 1 / s[-1]) * xp.cos(tilt) + (s[-1]-1)/2
        # line_z_diff = np.fft.fftfreq(s[-1], 1 / s[-1]) * xp.sin(tilt) + (s[-1]-1)/2

        yF_diff = xp.floor(line_y_diff).astype("int")
        zF_diff = xp.floor(line_z_diff).astype("int")
        dy_diff = line_y_diff - yF_diff
        dz_diff = line_z_diff - zF_diff

        qx = xp.arange(s[-1])
        qy = xp.arange(s[-1])
        qxx, qyy = xp.meshgrid(qx, qy, indexing="ij")

        ind0_diff = np.hstack(
            (
                xp.tile(yF_diff, s[-1]),
                xp.tile(yF_diff + 1, s[-1]),
                xp.tile(yF_diff, s[-1]),
                xp.tile(yF_diff + 1, s[-1]),
            )
        )

        ind1_diff = np.hstack(
            (
                xp.tile(zF_diff, s[-1]),
                xp.tile(zF_diff, s[-1]),
                xp.tile(zF_diff + 1, s[-1]),
                xp.tile(zF_diff + 1, s[-1]),
            )
        )

        weights_diff = np.hstack(
            (
                xp.repeat(((1 - dy_diff) * (1 - dz_diff)), s[-1]),
                xp.repeat(((dy_diff) * (1 - dz_diff)), s[-1]),
                xp.repeat(((1 - dy_diff) * (dz_diff)), s[-1]),
                xp.repeat(((dy_diff) * (dz_diff)), s[-1]),
            )
        )

        ind_diff = xp.ravel_multi_index(
            (
                xp.tile(qxx.ravel(), 4),
                ind0_diff.ravel(),
                ind1_diff.ravel(),
            ),
            (s[-1], s[-1], s[-1]),
            "clip",
        )

        bincount_x = (
            xp.tile(
                (xp.tile(self._ind_diffraction_ravel, 4)),
                (s[1]),
            )
            + xp.repeat(xp.arange(s[1]), ind_diff.shape[0]) * self._q_length
        )

        ind_real = xp.ravel_multi_index((ind0, ind1), (s[1], s[2]), mode="clip")
        self.ind_real = ind_real

        obj_projected = (
            (
                xp.bincount(
                    bincount_x,
                    (
                        (obj[ind_real] * weights_real[:, :, None]).mean(1)[:, ind_diff]
                    ).ravel()
                    * xp.tile(weights_diff, s[1]).ravel(),
                    minlength=self._q_length * s[1],
                ).reshape(s[1], self._q_length)[:, self._circular_mask_bincount]
            )
            * s[2]
            * 4
        )

        self._ind0 = ind0
        self._ind1 = ind1
        self._weights_real = weights_real
        self._bincount_x = bincount_x
        self._ind_diff = ind_diff
        self._weights_diff = weights_diff

        return obj_projected

    def _calculate_update(
        self, object_sliced, diffraction_patterns_projected, x_index, datacube_number
    ):
        """
        Calculate update for back projection

        Parameters
        ----------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        diffraction_patterns_projected: np.ndarray
            projected diffraction patterns for the relevant tilt
        x_index: int
            x slice of object to be sliced
        datacube_number: int
            index of datacube

        Returns
        --------
        update: np.ndarray
            difference between current object sliced in diffraciton space and
            experimental diffraction patterns
        """
        xp = self._xp

        s = self._object_shape_6D

        ind0 = self._positions_vox_F[datacube_number][0] == x_index
        ind1 = self._positions_vox_F[datacube_number][0] == x_index + 1

        dp_length = diffraction_patterns_projected.shape[1]

        dp_patterns = np.hstack(
            [
                diffraction_patterns_projected[ind0].ravel(),
                diffraction_patterns_projected[ind0].ravel(),
                diffraction_patterns_projected[ind1].ravel(),
                diffraction_patterns_projected[ind1].ravel(),
            ]
        )

        weights = np.hstack(
            [
                np.repeat(
                    (1 - self._positions_vox_dF[datacube_number][0][ind0])
                    * (1 - self._positions_vox_dF[datacube_number][1][ind0]),
                    dp_length,
                ),
                np.repeat(
                    (1 - self._positions_vox_dF[datacube_number][0][ind0])
                    * (self._positions_vox_dF[datacube_number][1][ind0]),
                    dp_length,
                ),
                np.repeat(
                    (self._positions_vox_dF[datacube_number][0][ind1])
                    * (1 - self._positions_vox_dF[datacube_number][1][ind1]),
                    dp_length,
                ),
                np.repeat(
                    (self._positions_vox_dF[datacube_number][0][ind1])
                    * (self._positions_vox_dF[datacube_number][1][ind1]),
                    dp_length,
                ),
            ]
        )

        positions_y = xp.clip(
            xp.hstack(
                [
                    self._positions_vox[datacube_number][1][ind0],
                    self._positions_vox[datacube_number][1][ind0] + 1,
                    self._positions_vox[datacube_number][1][ind1],
                    self._positions_vox[datacube_number][1][ind1] + 1,
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

        error = xp.mean(update.ravel() ** 2) / xp.mean(dp_patterns_counted.ravel() ** 2)

        error = copy_to_device(error, "cpu")

        return update, error

    def _back(
        self,
        num_points: int,
        x_index: int,
        update,
    ):
        """
        back propagate

        Parameters
        ----------
        num_points: int
            number of points for bilinear interpolation
        x_index: int
            x slice for back projection
        update: np.ndarray
            difference between current object sliced in diffraciton space and
            experimental diffraction patterns
        """
        xp = self._xp
        storage = self._storage

        s = self._object_shape_6D

        ind_update = xp.tile(self._circular_mask_ravel, 4)

        a = xp.argsort(self._ind_diffraction_ravel[self._circular_mask_ravel])
        i = xp.empty_like(a)
        i[a] = xp.arange(a.size)
        i = xp.tile(i, 4) + xp.repeat(xp.arange(4), i.shape[0]) * (i.shape[0])

        normalize = xp.ones((xp.repeat(update, 2, axis=1)[:, 1:]).shape) * 2
        normalize[:, 0] = 1

        update_reshaped = xp.repeat(
            (
                (xp.tile(xp.repeat(update, 2, axis=1)[:, 1:] / normalize, (4)))[:, i]
                * (self._weights_diff[ind_update])
            ),
            4 * num_points,
            axis=0,
        ) / (num_points)

        real_index = xp.ravel_multi_index(
            (self._ind0.ravel(), self._ind1.ravel()), (s[1], s[2]), mode="clip"
        )
        diff_index = self._ind_diff[ind_update]

        diff_bincount = xp.bincount(diff_index)
        diff_max = diff_bincount.shape[0]

        real_shape = real_index.shape[0]
        diff_shape = diff_index.shape[0]

        bincount_diff = (
            xp.tile(diff_index, real_shape)
            + (xp.repeat(xp.arange(real_shape), diff_shape)) * diff_max
        )

        update_q_summed = xp.bincount(
            bincount_diff,
            update_reshaped.ravel(),
            minlength=((diff_max) * real_shape),
        ).reshape((real_shape, -1))[:, diff_bincount > 0]

        diff_shape_bin = update_q_summed.shape[-1]

        real_bincount = xp.bincount(real_index)
        real_max = real_bincount.shape[0]

        bincount_real = (
            xp.tile(xp.arange(diff_shape_bin), real_shape)
            + xp.repeat(real_index, diff_shape_bin) * diff_shape_bin
        )

        update_r_summed = (
            xp.bincount(
                bincount_real,
                (update_q_summed * self._weights_real.ravel()[:, None]).ravel(),
                minlength=((real_max) * diff_shape_bin),
            )
        ).reshape((-1, diff_shape_bin))[real_bincount > 0]

        yy, zz = xp.meshgrid(
            xp.unique(real_index), xp.unique(diff_index), indexing="ij"
        )

        yy = copy_to_device(yy, storage)
        zz = copy_to_device(zz, storage)

        self._object[x_index, yy, zz] += copy_to_device(update_r_summed, storage)

    def _constraints(
        self,
        zero_edges: bool,
    ):
        """
        Constrains for object
        TODO: add constrains and break into multiple functions possibly

        Parameters
        ----------
        zero_edges: bool
            If True, zero edges along y and z
        """

        if zero_edges:
            s = self._object_shape_6D
            y = np.arange(s[1])
            z = np.arange(s[2])
            yy, zz = np.meshgrid(y, z, indexing="ij")
            ind_zero = np.where(
                (yy.ravel() == 0)
                | (zz.ravel() == 0)
                | (yy.ravel() == y.max())
                | (zz.ravel() == z.max())
            )[0]
            self._object[:, ind_zero] = 0

    def _make_test_object(
        self,
        sx: int,
        sy: int,
        sz: int,
        sq: int,
        q_max: float,
        r: int,
        num: int,
    ):
        """
        Make test object with 3D gold cubes at random orientations

        Parameters
        ----------
        sx: int
            x size (pixels)
        sy: int
            y size (pixels)
        sz: int
            z size (pixels)
        sq: int
            q size (pixels)
        q_max: float
            maximum scattering angle (A^-1)
        r: int
            length of 3D gold cubes
        num: int
            number of cubes

        Returns
        --------
        test_object: np.ndarray
            6D test object
        """
        xp_storage = self._xp_storage
        storage = self._storage

        test_object = xp_storage.zeros((sx, sy, sz, sq, sq, sq))

        diffraction_cloud = self._make_diffraction_cloud(sq, q_max, [0, 0, 0])

        test_object[:, :, :, 0, 0, 0] = copy_to_device(diffraction_cloud.sum(), storage)

        for a0 in range(num):
            s1 = xp_storage.random.randint(r, sx - r)
            s2 = xp_storage.random.randint(r, sy - r)
            h = xp_storage.random.randint(r, sz - r, size=1)
            t = xp_storage.random.randint(0, 360, size=3)

            cloud = copy_to_device(self._make_diffraction_cloud(sq, q_max, t), storage)

            test_object[s1 - r : s1 + r, s2 - r : s2 + r, h[0] - r : h[0] + r] = cloud

        return test_object

    def _forward_simulation(
        self,
        current_object: np.ndarray,
        tilt_deg: int,
        x_index: int,
        num_points: np.ndarray = 60,
    ):
        """
        Forward projection of object for simulation of diffraction data

        Parameters
        ----------
        current_object: np.ndarray
            current object estimate
        tilt_deg: float
            tilt of object in degrees
        x_index: int
            x slice of object to be sliced
        num_points: float
            number of points for bilinear interpolation

        Returns
        --------
        current_object_sliced: np.ndarray
            projection of current object sliced in diffraciton space
        """
        current_object_projected = self._real_space_radon(
            current_object,
            tilt_deg,
            x_index,
            num_points,
        )

        current_object_sliced = self._diffraction_space_slice(
            current_object_projected,
            tilt_deg,
        )

        return current_object_sliced

    def _make_diffraction_cloud(
        self,
        sq,
        q_max,
        rot,
    ):
        """
        Make 3D diffraction cloud

        Parameters
        ----------
        sq: int
            q size (pixels)
        q_max: float
            maximum scattering angle (A^-1)
        rot: 3-tuple
            rotation of cloud

        Returns
        --------
        diffraction_cloud: np.ndarray
            3D structure factor

        """
        xp = self._xp

        gold = self._make_gold(q_max)

        diffraction_cloud = xp.zeros((sq, sq, sq))

        q_step = q_max * 2 / (sq - 1)

        qz = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)
        qx = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)
        qy = xp.fft.ifftshift(xp.arange(sq) * q_step - q_step * (sq - 1) / 2)

        qxa, qya, qza = xp.meshgrid(qx, qy, qz, indexing="ij")

        g_vecs = gold.g_vec_all.copy()
        r = R.from_euler("zxz", [rot[0], rot[1], rot[2]])
        g_vecs = r.as_matrix() @ g_vecs

        cut_off = 0.1

        for a0 in range(gold.g_vec_all.shape[1]):
            bragg_spot = g_vecs[:, a0]
            distance = xp.sqrt(
                (qxa - bragg_spot[0]) ** 2
                + (qya - bragg_spot[1]) ** 2
                + (qza - bragg_spot[2]) ** 2
            )

            update_index = distance < cut_off
            update = xp.zeros((distance.shape))
            update[update_index] = cut_off - distance[update_index]
            update -= xp.min(update)
            update /= xp.sum(update)
            update *= gold.struct_factors_int[a0]
            diffraction_cloud += update

        return diffraction_cloud

    def _make_gold(
        self,
        q_max,
    ):
        """
        Calculate structure factor for gold up to q_max

        Parameters
        ----------
        q_max: float
            maximum scattering angle (A^-1)

        Returns
        --------
        crystal: Crystal
            gold crystal with structure factor calculated to q_max

        """

        pos = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        atom_num = 79
        a = 4.08
        cell = a

        crystal = Crystal(pos, atom_num, cell)

        crystal.calculate_structure_factors(q_max)

        return crystal

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
            self.object_6D.mean((0, 1, 2))[:, :, ind_diff],
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

    @property
    def object_6D(self):
        """6D object"""

        return self._object.reshape(self._object_shape_6D)
