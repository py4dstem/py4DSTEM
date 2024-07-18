import warnings
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from emdfile import tqdmnd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.process.phase.utils import (
    AffineTransform,
    ComplexProbe,
    bilinear_resample,
    copy_to_device,
    fft_shift,
    generate_batches,
    partition_list,
    rotate_point,
    spatial_frequencies,
)
from py4DSTEM.process.utils import (
    align_and_shift_images,
    electron_wavelength_angstrom,
    get_CoM,
    get_shifted_ar,
)
from py4DSTEM.visualize import return_scaled_histogram_ordering, show, show_complex
from scipy.ndimage import gaussian_filter, rotate

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np


class ObjectNDMethodsMixin:
    """
    Mixin class for object methods applicable to 2D,2.5D, and 3D objects.
    """

    def _initialize_object(
        self,
        initial_object,
        positions_px,
        object_type,
    ):
        """ """
        # explicit read-only self attributes up-front
        xp = self._xp

        object_padding_px = self._object_padding_px
        region_of_interest_shape = self._region_of_interest_shape

        if initial_object is None:
            pad_x = object_padding_px[0][1]
            pad_y = object_padding_px[1][1]
            p, q = np.round(np.max(positions_px, axis=0))
            p = np.max([np.round(p + pad_x), region_of_interest_shape[0]]).astype("int")
            q = np.max([np.round(q + pad_y), region_of_interest_shape[1]]).astype("int")
            if object_type == "potential":
                _object = xp.zeros((p, q), dtype=xp.float32)
            elif object_type == "complex":
                _object = xp.ones((p, q), dtype=xp.complex64)
        else:
            if object_type == "potential":
                _object = xp.asarray(initial_object, dtype=xp.float32)
            elif object_type == "complex":
                _object = xp.asarray(initial_object, dtype=xp.complex64)

        return _object

    def _crop_rotate_object_fov(
        self,
        array,
        positions_px=None,
        padding=0,
    ):
        """
        Crops and rotated object to FOV bounded by current pixel positions.

        Parameters
        ----------
        array: np.ndarray
            Object array to crop and rotate. Only operates on numpy arrays for compatibility.
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

        if positions_px is None:
            positions_px = asnumpy(self._positions_px)
        else:
            positions_px = asnumpy(positions_px)

        tf = AffineTransform(angle=angle)
        rotated_points = tf(positions_px, origin=positions_px.mean(0), xp=np)

        min_x, min_y = np.floor(np.amin(rotated_points, axis=0) - padding).astype("int")
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x, max_y = np.ceil(np.amax(rotated_points, axis=0) + padding).astype("int")

        rotated_array = rotate(
            asnumpy(array), np.rad2deg(-angle), order=1, reshape=False, axes=(-2, -1)
        )[..., min_x:max_x, min_y:max_y]

        if self._rotation_best_transpose:
            rotated_array = rotated_array.swapaxes(-2, -1)

        return rotated_array

    def _return_projected_cropped_potential(
        self,
        obj=None,
        return_kwargs=False,
        **kwargs,
    ):
        """Utility function to accommodate multiple classes"""
        if obj is None:
            obj = self.object_cropped
        else:
            obj = self._crop_rotate_object_fov(obj)

        if np.iscomplexobj(obj):
            obj = np.angle(obj)

        if return_kwargs:
            return obj, kwargs
        else:
            return obj

    def _return_object_fft(
        self,
        obj=None,
        apply_hanning_window=False,
        **kwargs,
    ):
        """
        Returns absolute value of obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        apply_hanning_window: bool, optional
            If True, a 2D Hann window is applied to the object before FFT

        Returns
        -------
        object_fft_amplitude: np.ndarray
            Amplitude of Fourier-transformed and center-shifted obj.
        """
        xp = self._xp
        asnumpy = self._asnumpy

        if obj is None:
            obj = self._object

        if np.iscomplexobj(obj):
            obj = xp.angle(obj)

        obj = self._crop_rotate_object_fov(asnumpy(obj))

        if apply_hanning_window:
            sx, sy = obj.shape
            wx = np.hanning(sx)
            wy = np.hanning(sy)
            obj *= wx[:, None] * wy[None, :]

        return np.abs(np.fft.fftshift(np.fft.fft2(obj)))

    def show_object_fft(
        self,
        obj=None,
        apply_hanning_window=True,
        scalebar=True,
        pixelsize=None,
        pixelunits=None,
        **kwargs,
    ):
        """
        Plot FFT of reconstructed object

        Parameters
        ----------
        obj: complex array, optional
            If None is specified, uses the `object_fft` property
        apply_hanning_window: bool, optional
            If True, a 2D Hann window is applied to the object before FFT
        scalebar: bool, optional
            if True, adds scalebar to probe
        pixelunits: str, optional
            units for scalebar, default is A^-1
        pixelsize: float, optional
            default is object FFT sampling
        """

        object_fft = self._return_object_fft(
            obj, apply_hanning_window=apply_hanning_window, **kwargs
        )

        if pixelsize is None:
            pixelsize = 1 / (object_fft.shape[1] * self.sampling[1])
        if pixelunits is None:
            pixelunits = r"$\AA^{-1}$"

        figsize = kwargs.pop("figsize", (4, 4))
        cmap = kwargs.pop("cmap", "magma")
        ticks = kwargs.pop("ticks", False)
        vmin = kwargs.pop("vmin", 0.001)
        vmax = kwargs.pop("vmax", 0.999)

        # remove additional 3D FFT parameters before passing to show
        kwargs.pop("orientation_matrix", None)
        kwargs.pop("vertical_lims", None)
        kwargs.pop("horizontal_lims", None)

        show(
            object_fft,
            figsize=figsize,
            cmap=cmap,
            scalebar=scalebar,
            pixelsize=pixelsize,
            ticks=ticks,
            pixelunits=pixelunits,
            vmin=vmin,
            vmax=vmax,
            aspect=object_fft.shape[1] / object_fft.shape[0],
            **kwargs,
        )

    def _reset_reconstruction(
        self,
        store_iterations,
        reset,
    ):
        """ """
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []

        # reset can be True, False, or None (default)
        if reset is True:
            self.error_iterations = []
            self._object = self._object_initial.copy()
            self._probe = self._probe_initial.copy()
            self._positions_px = self._positions_px_initial.copy()
            self._object_type = self._object_type_initial
            self._exit_waves = None

            # delete positions affine transform
            if hasattr(self, "_tf"):
                del self._tf

        elif reset is None:
            # continued run
            if hasattr(self, "error"):
                warnings.warn(
                    (
                        "Continuing reconstruction from previous result. "
                        "Use reset=True for a fresh start."
                    ),
                    UserWarning,
                )

            # first start
            else:
                self.error_iterations = []
                self._exit_waves = None
        else:
            # reset=False first start
            if not hasattr(self, "error"):
                self.error_iterations = []
                self._exit_waves = None

    @property
    def object_fft(self):
        """Fourier transform of current object estimate"""

        if not hasattr(self, "_object"):
            return None

        return self._return_object_fft(self._object)

    @property
    def object_cropped(self):
        """Cropped and rotated object"""

        return self._crop_rotate_object_fov(self._object)


class Object2p5DMethodsMixin:
    """
    Mixin class for object methods unique to 2.5D objects.
    Overwrites ObjectNDMethodsMixin.
    """

    def _precompute_propagator_arrays(
        self,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        energy: float,
        slice_thicknesses: Sequence[float],
        theta_x: float = None,
        theta_y: float = None,
    ):
        """
        Precomputes propagator arrays complex wave-function will be convolved by,
        for all slice thicknesses.

        Parameters
        ----------
        gpts: Tuple[int,int]
            Wavefunction pixel dimensions
        sampling: Tuple[float,float]
            Wavefunction sampling in A
        energy: float
            The electron energy of the wave functions in eV
        slice_thicknesses: Sequence[float]
            Array of slice thicknesses in A
        theta_x: float, optional
            x tilt of propagator in mrad
        theta_y: float, optional
            y tilt of propagator in mrad

        Returns
        -------
        propagator_arrays: np.ndarray
            (T,Sx,Sy) shape array storing propagator arrays
        """
        xp = self._xp

        # Frequencies
        kx, ky = spatial_frequencies(gpts, sampling)
        kx = xp.asarray(kx, dtype=xp.float32)
        ky = xp.asarray(ky, dtype=xp.float32)

        # Propagators
        wavelength = electron_wavelength_angstrom(energy)
        num_slices = slice_thicknesses.shape[0]
        propagators = xp.empty(
            (num_slices, kx.shape[0], ky.shape[0]), dtype=xp.complex64
        )

        for i, dz in enumerate(slice_thicknesses):
            propagators[i] = xp.exp(
                1.0j * (-(kx**2)[:, None] * np.pi * wavelength * dz)
            )
            propagators[i] *= xp.exp(1.0j * (-(ky**2)[None] * np.pi * wavelength * dz))

            if theta_x is not None:
                propagators[i] *= xp.exp(
                    1.0j * (-2 * kx[:, None] * np.pi * dz * np.tan(theta_x / 1e3))
                )

            if theta_y is not None:
                propagators[i] *= xp.exp(
                    1.0j * (-2 * ky[None] * np.pi * dz * np.tan(theta_y / 1e3))
                )

        return propagators

    def _propagate_array(self, array: np.ndarray, propagator_array: np.ndarray):
        """
        Propagates array by Fourier convolving array with propagator_array.

        Parameters
        ----------
        array: np.ndarray
            Wavefunction array to be convolved
        propagator_array: np.ndarray
            Propagator array to convolve array with

        Returns
        -------
        propagated_array: np.ndarray
            Fourier-convolved array
        """
        xp = self._xp

        return xp.fft.ifft2(xp.fft.fft2(array) * propagator_array)

    def _initialize_object(
        self,
        initial_object,
        num_slices,
        positions_px,
        object_type,
    ):
        """ """
        # explicit read-only self attributes up-front
        xp = self._xp

        object_padding_px = self._object_padding_px
        region_of_interest_shape = self._region_of_interest_shape

        if initial_object is None:
            pad_x = object_padding_px[0][1]
            pad_y = object_padding_px[1][1]
            p, q = np.round(np.max(positions_px, axis=0))
            p = np.max([np.round(p + pad_x), region_of_interest_shape[0]]).astype("int")
            q = np.max([np.round(q + pad_y), region_of_interest_shape[1]]).astype("int")
            if object_type == "potential":
                _object = xp.zeros((num_slices, p, q), dtype=xp.float32)
            elif object_type == "complex":
                _object = xp.ones((num_slices, p, q), dtype=xp.complex64)
        else:
            if object_type == "potential":
                _object = xp.asarray(initial_object, dtype=xp.float32)
            elif object_type == "complex":
                _object = xp.asarray(initial_object, dtype=xp.complex64)

        return _object

    def _return_projected_cropped_potential(
        self,
        obj=None,
        return_kwargs=False,
        **kwargs,
    ):
        """Utility function to accommodate multiple classes"""

        if obj is None:
            obj = self.object_cropped
        else:
            obj = self._crop_rotate_object_fov(obj)

        if np.iscomplexobj(obj):
            obj = np.angle(obj).sum(0)
        else:
            obj = obj.sum(0)

        if return_kwargs:
            return obj, kwargs
        else:
            return obj

    def _return_object_fft(
        self,
        obj=None,
        apply_hanning_window=False,
        **kwargs,
    ):
        """
        Returns obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        apply_hanning_window: bool, optional
            If True, a 2D Hann window is applied to the object before FFT

        Returns
        -------
        object_fft_amplitude: np.ndarray
            Amplitude of Fourier-transformed and center-shifted obj.
        """
        xp = self._xp

        if obj is None:
            obj = self._object

        if np.iscomplexobj(obj):
            obj = xp.angle(obj)

        obj = self._crop_rotate_object_fov(obj.sum(axis=0))

        if apply_hanning_window:
            sx, sy = obj.shape
            wx = np.hanning(sx)
            wy = np.hanning(sy)
            obj *= wx[:, None] * wy[None, :]

        return np.abs(np.fft.fftshift(np.fft.fft2(obj)))

    def show_depth_section(
        self,
        ptA: Tuple[float, float],
        ptB: Tuple[float, float],
        aspect_ratio: float = "auto",
        plot_line_profile: bool = False,
        ms_object=None,
        specify_calibrated: bool = True,
        gaussian_filter_sigma: float = None,
        cbar: bool = True,
        **kwargs,
    ):
        """
        Displays line profile depth section

        Parameters
        ----------
        ptA: Tuple[float,float]
            Starting point (x1,y1) for line profile depth section
            If either is None, assumed to be array start.
            Specified in Angstroms unless specify_calibrated is False
        ptB: Tuple[float,float]
            End point (x2,y2) for line profile depth section
            If either is None, assumed to be array end.
            Specified in Angstroms unless specify_calibrated is False
        aspect_ratio: float, optional
            aspect ratio for depth profile plot
        plot_line_profile: bool
            If True, also plots line profile showing where depth profile is taken
        ms_object: np.array
            Object to plot slices of. If None, uses current object
        specify_calibrated: bool (optional)
            If False, ptA and ptB points specified in pixels instead of Angstroms
        gaussian_filter_sigma: float (optional)
            Standard deviation of gaussian kernel in A
        cbar: bool, optional
            If True, displays a colorbar
        """
        if ms_object is None:
            ms_object = self.object_cropped

        if np.iscomplexobj(ms_object):
            ms_object = np.angle(ms_object)

        x1, y1 = ptA
        x2, y2 = ptB

        if x1 is None:
            x1 = 0
        if y1 is None:
            y1 = 0
        if x2 is None:
            x2 = self.sampling[0] * ms_object.shape[1]
        if y2 is None:
            y2 = self.sampling[1] * ms_object.shape[2]

        if specify_calibrated:
            x1 /= self.sampling[0]
            x2 /= self.sampling[0]
            y1 /= self.sampling[1]
            y2 /= self.sampling[1]

        x1, x2 = np.array([x1, x2]).clip(0, ms_object.shape[1])
        y1, y2 = np.array([y1, y2]).clip(0, ms_object.shape[2])

        angle = np.arctan2(x2 - x1, y2 - y1)

        x0 = ms_object.shape[1] / 2
        y0 = ms_object.shape[2] / 2

        x1_0, y1_0 = rotate_point((x0, y0), (x1, y1), angle)
        x2_0, y2_0 = rotate_point((x0, y0), (x2, y2), angle)

        rotated_object = np.roll(
            rotate(ms_object, np.rad2deg(angle), reshape=False, axes=(-1, -2)),
            -int(x1_0),
            axis=1,
        )

        if gaussian_filter_sigma is not None:
            gaussian_filter_sigma /= self.sampling[0]
            rotated_object = gaussian_filter(rotated_object, gaussian_filter_sigma)

        y1_0, y2_0 = (
            np.array([y1_0, y2_0]).astype("int").clip(0, rotated_object.shape[2])
        )
        plot_im = rotated_object[:, 0, y1_0:y2_0]

        # Plotting
        if plot_line_profile:
            ncols = 2
        else:
            ncols = 1
        col_index = 0

        spec = GridSpec(ncols=ncols, nrows=1, wspace=0.15)

        figsize = kwargs.pop("figsize", (4 * ncols, 4))
        fig = plt.figure(figsize=figsize)
        cmap = kwargs.pop("cmap", "magma")

        # Line profile
        if plot_line_profile:
            ax = fig.add_subplot(spec[0, col_index])

            extent_line = [
                0,
                self.sampling[1] * ms_object.shape[2],
                self.sampling[0] * ms_object.shape[1],
                0,
            ]

            ax.imshow(ms_object.sum(0), cmap="gray", extent=extent_line)

            ax.plot(
                [y1 * self.sampling[0], y2 * self.sampling[1]],
                [x1 * self.sampling[0], x2 * self.sampling[1]],
                color="red",
            )

            ax.set_xlabel("y [A]")
            ax.set_ylabel("x [A]")
            ax.set_title("Multislice depth profile location")
            col_index += 1

        # Main visualization

        extent = [
            0,
            self.sampling[1] * plot_im.shape[1],
            self._slice_thicknesses[0] * plot_im.shape[0],
            0,
        ]

        ax = fig.add_subplot(spec[0, col_index])
        im = ax.imshow(plot_im, cmap=cmap, extent=extent)

        if aspect_ratio is not None:
            if aspect_ratio == "auto":
                aspect_ratio = extent[1] / extent[2]
                if plot_line_profile:
                    aspect_ratio *= extent_line[2] / extent_line[1]

            ax.set_aspect(aspect_ratio)
            cbar = False

        ax.set_xlabel("r [A]")
        ax.set_ylabel("z [A]")
        ax.set_title("Multislice depth profile")

        if cbar:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            fig.colorbar(im, cax=ax_cb)

        spec.tight_layout(fig)

    def show_slices(
        self,
        ms_object=None,
        cbar: bool = True,
        common_color_scale: bool = True,
        padding: int = 0,
        num_cols: int = 3,
        show_fft: bool = False,
        **kwargs,
    ):
        """
        Displays reconstructed slices of object

        Parameters
        --------
        ms_object: nd.array, optional
            Object to plot slices of. If None, uses current object
        cbar: bool, optional
            If True, displays a colorbar
        padding: int, optional
            Padding to leave uncropped
        num_cols: int, optional
            Number of GridSpec columns
        show_fft: bool, optional
            if True, plots fft of object slices
        """

        if ms_object is None:
            ms_object = self._object

        rotated_object = self._crop_rotate_object_fov(ms_object, padding=padding)

        if show_fft:
            rotated_object = np.abs(
                np.fft.fftshift(
                    np.fft.fft2(rotated_object, axes=(-2, -1)), axes=(-2, -1)
                )
            )

        rotated_shape = rotated_object.shape

        if np.iscomplexobj(rotated_object):
            rotated_object = np.angle(rotated_object)

        extent = [
            0,
            self.sampling[1] * rotated_shape[2],
            self.sampling[0] * rotated_shape[1],
            0,
        ]

        num_rows = np.ceil(self._num_slices / num_cols).astype("int")
        wspace = 0.35 if cbar else 0.15

        axsize = kwargs.pop("axsize", (3, 3))
        cmap = kwargs.pop("cmap", "magma")

        if common_color_scale:
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            rotated_object, vmin, vmax = return_scaled_histogram_ordering(
                rotated_object, vmin=vmin, vmax=vmax
            )
        else:
            vmin = None
            vmax = None

        spec = GridSpec(
            ncols=num_cols,
            nrows=num_rows,
            hspace=0.15,
            wspace=wspace,
        )

        figsize = (axsize[0] * num_cols, axsize[1] * num_rows)
        fig = plt.figure(figsize=figsize)

        for flat_index, obj_slice in enumerate(rotated_object):
            row_index, col_index = np.unravel_index(flat_index, (num_rows, num_cols))
            ax = fig.add_subplot(spec[row_index, col_index])
            im = ax.imshow(
                obj_slice,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                **kwargs,
            )

            ax.set_title(f"Slice index: {flat_index}")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            if row_index < num_rows - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("y [A]")

            if col_index > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel("x [A]")

        spec.tight_layout(fig)


class Object3DMethodsMixin:
    """
    Mixin class for object methods unique to 3D objects.
    Overwrites ObjectNDMethodsMixin and Object2p5DMethodsMixin.
    """

    _swap_zxy_to_xyz = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def _project_sliced_object(self, array: np.ndarray, output_z):
        """
        Projects voxel-sliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to project
        output_z: int
            Output_dimension to project array to.

        Returns
        -------
        projected_array: np.ndarray
            projected array
        """
        xp = self._xp
        input_z = array.shape[0]

        voxels_per_slice = np.ceil(input_z / output_z).astype("int")
        pad_size = voxels_per_slice * output_z - input_z

        padded_array = xp.pad(array, ((0, pad_size), (0, 0), (0, 0)))

        return xp.sum(
            padded_array.reshape(
                (
                    -1,
                    voxels_per_slice,
                )
                + array.shape[1:]
            ),
            axis=1,
        )

    def _expand_sliced_object(self, array: np.ndarray, output_z):
        """
        Expands supersliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to expand
        output_z: int
            Output_dimension to expand array to.

        Returns
        -------
        expanded_array: np.ndarray
            expanded array
        """
        xp = self._xp
        input_z = array.shape[0]

        voxels_per_slice = np.ceil(output_z / input_z).astype("int")
        remainder_size = voxels_per_slice - (voxels_per_slice * input_z - output_z)

        voxels_in_slice = xp.repeat(voxels_per_slice, input_z)
        voxels_in_slice[-1] = remainder_size if remainder_size > 0 else voxels_per_slice

        normalized_array = array / xp.asarray(voxels_in_slice)[:, None, None]
        return xp.repeat(normalized_array, voxels_per_slice, axis=0)[:output_z]

    def _rotate_zxy_volume(
        self,
        volume_array,
        rot_matrix,
        order=3,
    ):
        """ """

        xp = self._xp
        affine_transform = self._scipy.ndimage.affine_transform
        swap_zxy_to_xyz = self._swap_zxy_to_xyz

        volume = volume_array.copy()
        volume_shape = xp.asarray(volume.shape)
        tf = xp.asarray(swap_zxy_to_xyz.T @ rot_matrix.T @ swap_zxy_to_xyz)

        in_center = (volume_shape - 1) / 2
        out_center = tf @ in_center
        offset = in_center - out_center

        volume = affine_transform(volume, tf, offset=offset, order=order)

        return volume

    def _initialize_object(
        self,
        initial_object,
        positions_px,
        object_type,
        main_tilt_axis="vertical",
    ):
        """ """
        # explicit read-only self attributes up-front
        xp = self._xp
        object_padding_px = self._object_padding_px
        region_of_interest_shape = self._region_of_interest_shape

        if initial_object is None:
            pad_x = object_padding_px[0][1]
            pad_y = object_padding_px[1][1]
            p, q = np.round(np.max(positions_px, axis=0))
            p = np.max([np.round(p + pad_x), region_of_interest_shape[0]]).astype("int")
            q = np.max([np.round(q + pad_y), region_of_interest_shape[1]]).astype("int")

            if main_tilt_axis == "vertical":
                _object = xp.zeros((q, p, q), dtype=xp.float32)
            elif main_tilt_axis == "horizontal":
                _object = xp.zeros((p, p, q), dtype=xp.float32)
            else:
                _object = xp.zeros((max(p, q), p, q), dtype=xp.float32)
        else:
            _object = xp.asarray(initial_object, dtype=xp.float32)

        return _object

    def _return_projected_cropped_potential(
        self,
        obj=None,
        return_kwargs=False,
        **kwargs,
    ):
        """Utility function to accommodate multiple classes"""

        asnumpy = self._asnumpy

        rot_matrix = kwargs.pop("orientation_matrix", None)
        v_lims = kwargs.pop("vertical_lims", (None, None))
        h_lims = kwargs.pop("horizontal_lims", (None, None))

        if obj is None:
            obj = self._object

        if rot_matrix is not None:
            obj = self._rotate_zxy_volume(
                obj,
                rot_matrix=rot_matrix,
            )

        start_v, end_v = v_lims
        start_h, end_h = h_lims
        obj = asnumpy(obj.sum(0)[start_v:end_v, start_h:end_h])

        if return_kwargs:
            return obj, kwargs
        else:
            return obj

    def _return_object_fft(
        self,
        obj=None,
        apply_hanning_window=False,
        orientation_matrix=None,
        vertical_lims: Tuple[int, int] = (None, None),
        horizontal_lims: Tuple[int, int] = (None, None),
        **kwargs,
    ):
        """
        Returns obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        apply_hanning_window: bool, optional
            If True, a 2D Hann window is applied to the object before FFT
        orientation_matrix: np.ndarray, optional
            orientation matrix to rotate zone-axis
        vertical_lims: tuple(int,int), optional
            min/max vertical indices
        horizontal_lims: tuple(int,int), optional
            min/max horizontal indices

        Returns
        -------
        object_fft_amplitude: np.ndarray
            Amplitude of Fourier-transformed and center-shifted obj.
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if obj is None:
            obj = self._object
        else:
            obj = xp.asarray(obj, dtype=xp.float32)

        if orientation_matrix is not None:
            obj = self._rotate_zxy_volume(
                obj,
                rot_matrix=orientation_matrix,
            )

        start_v, end_v = vertical_lims
        start_h, end_h = horizontal_lims
        obj = asnumpy(obj.sum(0)[start_v:end_v, start_h:end_h])

        if apply_hanning_window:
            sx, sy = obj.shape
            wx = np.hanning(sx)
            wy = np.hanning(sy)
            obj *= wx[:, None] * wy[None, :]

        return np.abs(np.fft.fftshift(np.fft.fft2(obj)))

    @property
    def object_supersliced(self):
        """Returns super-sliced object"""
        return self._project_sliced_object(self._object, self._num_slices)


class ProbeMethodsMixin:
    """
    Mixin class for probe methods applicable to a single probe.
    """

    def _initialize_probe(
        self,
        initial_probe,
        vacuum_probe_intensity,
        mean_diffraction_intensity,
        semiangle_cutoff,
        crop_patterns,
    ):
        """ """
        # explicit read-only self attributes up-front
        xp = self._xp
        device = self._device

        crop_mask = self._crop_mask
        region_of_interest_shape = self._region_of_interest_shape
        sampling = self.sampling
        energy = self._energy
        rolloff = self._rolloff
        polar_parameters = self._polar_parameters

        if initial_probe is None:
            if vacuum_probe_intensity is not None:
                semiangle_cutoff = np.inf
                vacuum_probe_intensity = xp.asarray(
                    vacuum_probe_intensity, dtype=xp.float32
                )

                probe_x0, probe_y0 = get_CoM(
                    vacuum_probe_intensity,
                    device=device,
                )
                vacuum_probe_intensity = get_shifted_ar(
                    vacuum_probe_intensity,
                    -probe_x0,
                    -probe_y0,
                    bilinear=True,
                    device=device,
                )

                if crop_patterns:
                    vacuum_probe_intensity = vacuum_probe_intensity[crop_mask].reshape(
                        self._diffraction_intensities_shape_crop
                    )

                sx, sy = vacuum_probe_intensity.shape
                tx, ty = region_of_interest_shape
                if sx != tx or sy != ty and self._resample_exit_waves is True:
                    vacuum_probe_intensity = bilinear_resample(
                        vacuum_probe_intensity,
                        output_size=(tx, ty),
                        vectorized=True,
                        conserve_array_sums=True,
                        xp=xp,
                    )

            _probe = (
                ComplexProbe(
                    gpts=region_of_interest_shape,
                    sampling=sampling,
                    energy=energy,
                    semiangle_cutoff=semiangle_cutoff,
                    rolloff=rolloff,
                    vacuum_probe_intensity=vacuum_probe_intensity,
                    parameters=polar_parameters,
                    device=device,
                )
                .build()
                ._array
            )

            # Normalize probe to match mean diffraction intensity
            probe_intensity = xp.sum(xp.abs(xp.fft.fft2(_probe)) ** 2)
            _probe *= xp.sqrt(mean_diffraction_intensity / probe_intensity)

        else:
            if isinstance(initial_probe, ComplexProbe):
                if initial_probe._gpts != region_of_interest_shape:
                    raise ValueError()
                if hasattr(initial_probe, "_array"):
                    _probe = initial_probe._array
                else:
                    initial_probe._xp = xp
                    _probe = initial_probe.build()._array

                # Normalize probe to match mean diffraction intensity
                probe_intensity = xp.sum(xp.abs(xp.fft.fft2(_probe)) ** 2)
                _probe *= xp.sqrt(mean_diffraction_intensity / probe_intensity)
            else:
                _probe = xp.asarray(initial_probe, dtype=xp.complex64)

        return _probe, semiangle_cutoff

    def _return_fourier_probe(
        self,
        probe=None,
        remove_initial_probe_aberrations=False,
    ):
        """
        Returns complex fourier probe shifted to center of array from
        corner-centered complex real space probe

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe
        remove_initial_probe_aberrations: bool, optional
            If True, removes initial probe aberrations from Fourier probe

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

        fourier_probe = xp.fft.fft2(probe)

        if remove_initial_probe_aberrations:
            fourier_probe *= xp.conjugate(self._known_aberrations_array)

        return xp.fft.fftshift(fourier_probe, axes=(-2, -1))

    def _return_fourier_probe_from_centered_probe(
        self,
        probe=None,
        remove_initial_probe_aberrations=False,
    ):
        """
        Returns complex fourier probe shifted to center of array from
        centered complex real space probe

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe
        remove_initial_probe_aberrations: bool, optional
            If True, removes initial probe aberrations from Fourier probe

        Returns
        -------
        fourier_probe: np.ndarray
            Fourier-transformed and center-shifted probe.
        """
        xp = self._xp
        return self._return_fourier_probe(
            xp.fft.ifftshift(probe, axes=(-2, -1)),
            remove_initial_probe_aberrations=remove_initial_probe_aberrations,
        )

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

    def _return_probe_intensities(self, probe):
        """
        Returns probe intensities summing up to 1.
        """
        if probe is None:
            probe = self.probe_centered

        intensity_arrays = np.abs(np.array(probe, ndmin=3)) ** 2
        probe_ratio = list(intensity_arrays.sum((-2, -1)) / intensity_arrays.sum())

        return probe_ratio

    def show_probe(
        self,
        probe=None,
        cbar=True,
        scalebar=True,
        pixelsize=None,
        pixelunits=None,
        W=6,
        **kwargs,
    ):
        """
        Plot probe in real space

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses the `probe_fourier` property
        remove_initial_probe_aberrations: bool, optional
            If True, removes initial probe aberrations from Fourier probe
        cbar: bool, optional
            if True, adds colorbar
        scalebar: bool, optional
            if True, adds scalebar to probe
        pixelsize: float, optional
            default is probe reciprocal sampling
        pixelunits: str, optional
            units for scalebar, default is A^-1
        W: int, optional
            if not None, sets the width of the image grid
        """
        asnumpy = self._asnumpy

        if pixelsize is None:
            pixelsize = self.sampling[1]
        if pixelunits is None:
            pixelunits = r"$\AA$"

        intensities = self._return_probe_intensities(probe)
        title = [
            f"Probe {iter} intensity: {ratio*100:.1f}%"
            for iter, ratio in enumerate(intensities)
        ]

        axsize = kwargs.pop("axsize", (4, 4))
        chroma_boost = kwargs.pop("chroma_boost", 1)
        ticks = kwargs.pop("ticks", False)
        title = kwargs.pop("title", title if len(title) > 1 else title[0])

        if probe is None:
            probe = list(np.array(self.probe_centered, ndmin=3))
        else:
            if isinstance(probe, np.ndarray) and probe.ndim == 2:
                probe = [probe]
            probe = [
                asnumpy(
                    self._return_centered_probe(
                        pr,
                    )
                )
                for pr in probe
            ]

        probe = list(partition_list(probe, W))
        probe = probe if len(probe) > 1 else probe[0]

        show_complex(
            probe,
            cbar=cbar,
            axsize=axsize,
            scalebar=scalebar,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            ticks=ticks,
            chroma_boost=chroma_boost,
            title=title,
            **kwargs,
        )

    def show_fourier_probe(
        self,
        probe=None,
        remove_initial_probe_aberrations=False,
        cbar=True,
        scalebar=True,
        pixelsize=None,
        pixelunits=None,
        W=6,
        **kwargs,
    ):
        """
        Plot probe in fourier space

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses the `probe_fourier` property
        remove_initial_probe_aberrations: bool, optional
            If True, removes initial probe aberrations from Fourier probe
        cbar: bool, optional
            if True, adds colorbar
        scalebar: bool, optional
            if True, adds scalebar to probe
        pixelsize: float, optional
            default is probe reciprocal sampling
        pixelunits: str, optional
            units for scalebar, default is A^-1
        W: int, optional
            if not None, sets the width of the image grid
        """
        asnumpy = self._asnumpy

        if pixelsize is None:
            pixelsize = self._reciprocal_sampling[1]
        if pixelunits is None:
            pixelunits = r"$\AA^{-1}$"

        intensities = self._return_probe_intensities(probe)
        title = [
            f"Probe {iter} intensity: {ratio*100:.1f}%"
            for iter, ratio in enumerate(intensities)
        ]

        axsize = kwargs.pop("axsize", (4, 4))
        chroma_boost = kwargs.pop("chroma_boost", 1)
        ticks = kwargs.pop("ticks", False)
        title = kwargs.pop("title", title if len(title) > 1 else title[0])

        if probe is None:
            if remove_initial_probe_aberrations:
                probe = self.probe_fourier_residual
            else:
                probe = self.probe_fourier
            probe = list(np.array(probe, ndmin=3))
        else:
            if isinstance(probe, np.ndarray) and probe.ndim == 2:
                probe = [probe]
            probe = [
                asnumpy(
                    self._return_fourier_probe(
                        pr,
                        remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                    )
                )
                for pr in probe
            ]

        probe = list(partition_list(probe, W))
        probe = probe if len(probe) > 1 else probe[0]

        show_complex(
            probe,
            cbar=cbar,
            axsize=axsize,
            scalebar=scalebar,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            ticks=ticks,
            chroma_boost=chroma_boost,
            title=title,
            **kwargs,
        )

    def _return_single_probe(self, probe=None):
        """Current probe estimate"""
        xp = self._xp

        if probe is not None:
            return xp.asarray(probe)
        else:
            if not hasattr(self, "_probe"):
                return None

            return self._probe

    @property
    def probe_fourier(self):
        """Current probe estimate in Fourier space"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(self._return_fourier_probe(self._probe))

    @property
    def probe_fourier_residual(self):
        """Current probe estimate in Fourier space"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(
            self._return_fourier_probe(
                self._probe, remove_initial_probe_aberrations=True
            )
        )

    @property
    def probe_centered(self):
        """Current probe estimate shifted to the center"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(self._return_centered_probe(self._probe))


class ProbeMixedMethodsMixin:
    """
    Mixin class for probe methods unique to mixed probes.
    Overwrites ProbeMethodsMixin.
    """

    def _initialize_probe(
        self,
        initial_probe,
        vacuum_probe_intensity,
        mean_diffraction_intensity,
        semiangle_cutoff,
        crop_patterns,
    ):
        """ """

        # explicit read-only self attributes up-front
        xp = self._xp
        num_probes = self._num_probes
        region_of_interest_shape = self._region_of_interest_shape

        if initial_probe is None or isinstance(initial_probe, ComplexProbe):
            # calls ProbeMethodsMixin for first probe
            # annoyingly can't use super() as Mixins are defined right->left
            # but MRO is defined left->right..
            _probe, semiangle_cutoff = ProbeMethodsMixin._initialize_probe(
                self,
                initial_probe,
                vacuum_probe_intensity,
                mean_diffraction_intensity,
                semiangle_cutoff,
                crop_patterns,
            )

            sx, sy = region_of_interest_shape
            _probes = xp.zeros((num_probes, sx, sy), dtype=xp.complex64)
            _probes[0] = _probe

            # Randomly shift phase of other probes
            for i_probe in range(1, num_probes):
                shift_x = xp.exp(
                    -2j * np.pi * (xp.random.rand() - 0.5) * xp.fft.fftfreq(sx)
                )
                shift_y = xp.exp(
                    -2j * np.pi * (xp.random.rand() - 0.5) * xp.fft.fftfreq(sy)
                )
                _probes[i_probe] = (
                    _probes[i_probe - 1] * shift_x[:, None] * shift_y[None]
                )
        else:
            _probes = xp.asarray(initial_probe, dtype=xp.complex64)

        return _probes, semiangle_cutoff

    def _return_single_probe(self, probe=None):
        """Current probe estimate"""
        xp = self._xp

        if probe is not None:
            return xp.asarray(probe[0])
        else:
            if not hasattr(self, "_probe"):
                return None

            return self._probe[0]


class ObjectNDProbeMethodsMixin:
    """
    Mixin class for methods applicable to 2D, 2.5D, and 3D objects using a single probe.
    """

    def _return_shifted_probes(self, current_probe, positions_px_fractional):
        """Simple utility to de-duplicate _overlap_projection"""

        xp = self._xp
        shifted_probes = fft_shift(current_probe, positions_px_fractional, xp)
        return shifted_probes

    def _overlap_projection(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        shifted_probes,
    ):
        """
        Ptychographic overlap projection method.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        shifted_probes:np.ndarray
            fractionally-shifted probes
        object_patches: np.ndarray
            Patched object view
        overlap: np.ndarray
            shifted_probes * object_patches
        """

        xp = self._xp

        object_patches = current_object[
            vectorized_patch_indices_row, vectorized_patch_indices_col
        ]

        if self._object_type == "potential":
            object_patches = xp.exp(1j * object_patches)

        overlap = shifted_probes * object_patches

        return shifted_probes, object_patches, overlap

    def _return_farfield_amplitudes(self, fourier_overlap):
        """Small utility to de-duplicate mixed-state Fourier projection."""

        xp = self._xp
        return xp.abs(fourier_overlap)

    def cross_correlate_amplitudes_to_probe_aperture(
        self, upsample_factor=4, progress_bar=True, probe=None
    ):
        """
        Cross-correlates the measured amplitudes with the current probe aperture.
        Modifies self._amplitudes in-place.

        Parameters
        ----------
        upsample_factor: float
            Upsampling factor used in cross-correlation. Must be larger than 2
        probe: np.ndarray, optional
            Probe to use for centering. Passed to _return_single_probe(probe)

        Returns
        -------
        self to accommodate chaining
        """
        xp = self._xp
        storage = self._storage

        num_dps = self._num_diffraction_patterns

        single_probe = self._return_single_probe(probe)
        probe_aperture = copy_to_device(xp.abs(xp.fft.fft2(single_probe)), storage)

        for idx in tqdmnd(
            num_dps,
            desc="Cross-correlating amplitudes",
            unit="DP",
            disable=not progress_bar,
        ):
            self._amplitudes[idx] = align_and_shift_images(
                probe_aperture,
                self._amplitudes[idx],
                upsample_factor=upsample_factor,
                device=storage,
            )

        return self

    def _gradient_descent_fourier_projection(self, amplitudes, overlap, fourier_mask):
        """
        Ptychographic fourier projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap
        fourier_mask: np.ndarray
            Mask to apply at the detector-plane for zeroing-out unreliable gradients
            Useful when detector has artifacts such as dead-pixels

        Returns
        --------
        exit_waves:np.ndarray
            Difference between modified and estimated exit waves
        error: float
            Reconstruction error
        """

        xp = self._xp

        fourier_overlap = xp.fft.fft2(overlap)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_overlap = bilinear_resample(
                fourier_overlap,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        if fourier_mask is not None:
            fourier_overlap *= fourier_mask

        farfield_amplitudes = self._return_farfield_amplitudes(fourier_overlap)
        error = xp.sum(xp.abs(amplitudes - farfield_amplitudes) ** 2)
        fourier_modified_overlap = amplitudes * xp.exp(1j * xp.angle(fourier_overlap))

        fourier_modified_overlap = fourier_modified_overlap - fourier_overlap
        if fourier_mask is not None:
            fourier_modified_overlap *= fourier_mask

        # resample back to region_of_interest_shape, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_modified_overlap = bilinear_resample(
                fourier_modified_overlap,
                output_size=self._region_of_interest_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        exit_waves = xp.fft.ifft2(fourier_modified_overlap)

        return exit_waves, error

    def _projection_sets_fourier_projection(
        self,
        amplitudes,
        overlap,
        exit_waves,
        fourier_mask,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Ptychographic fourier projection method for DM_AP and RAAR methods.
        Generalized projection using three parameters: a,b,c

            DM_AP(\\alpha)   :   a =  -\\alpha, b = 1, c = 1 + \\alpha
              DM: DM_AP(1.0), AP: DM_AP(0.0)

            RAAR(\\beta)     :   a = 1-2\\beta, b = \\beta, c = 2
              DM : RAAR(1.0)

            RRR(\\gamma)     :   a = -\\gamma, b = \\gamma, c = 2
              DM: RRR(1.0)

            SUPERFLIP       :   a = 0, b = 1, c = 2

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap
        exit_waves: np.ndarray
            previously estimated exit waves
        fourier_mask: np.ndarray
            Mask to apply at the detector-plane for zeroing-out unreliable gradients
            Useful when detector has artifacts such as dead-pixels
            Currently not implemented for projection-sets
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        exit_waves:np.ndarray
            Updated exit_waves
        error: float
            Reconstruction error
        """

        if fourier_mask is not None:
            raise NotImplementedError()

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        if exit_waves is None:
            exit_waves = overlap.copy()

        factor_to_be_projected = projection_c * overlap + projection_y * exit_waves
        fourier_projected_factor = xp.fft.fft2(factor_to_be_projected)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_projected_factor = bilinear_resample(
                fourier_projected_factor,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        farfield_amplitudes = self._return_farfield_amplitudes(fourier_projected_factor)
        error = xp.sum(xp.abs(amplitudes - farfield_amplitudes) ** 2)

        fourier_projected_factor = amplitudes * xp.exp(
            1j * xp.angle(fourier_projected_factor)
        )

        # resample back to region_of_interest_shape, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_projected_factor = bilinear_resample(
                fourier_projected_factor,
                output_size=self._region_of_interest_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        projected_factor = xp.fft.ifft2(fourier_projected_factor)

        exit_waves = (
            projection_x * exit_waves
            + projection_a * overlap
            + projection_b * projected_factor
        )

        return exit_waves, error

    def _forward(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        current_probe,
        positions_px_fractional,
        amplitudes,
        exit_waves,
        fourier_mask,
        use_projection_scheme,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Ptychographic forward operator.
        Calls _overlap_projection() and the appropriate _fourier_projection().

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        amplitudes: np.ndarray
            Normalized measured amplitudes
        exit_waves: np.ndarray
            previously estimated exit waves
        fourier_mask: np.ndarray
            Mask to apply at the detector-plane for zeroing-out unreliable gradients
            Useful when detector has artifacts such as dead-pixels
        use_projection_scheme: bool,
            If True, use generalized projection update
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        shifted_probes:np.ndarray
            fractionally-shifted probes
        object_patches: np.ndarray
            Patched object view
        overlap: np.ndarray
            object * probe overlap
        exit_waves:np.ndarray
            Updated exit_waves
        error: float
            Reconstruction error
        """
        shifted_probes = self._return_shifted_probes(
            current_probe, positions_px_fractional
        )

        shifted_probes, object_patches, overlap = self._overlap_projection(
            current_object,
            vectorized_patch_indices_row,
            vectorized_patch_indices_col,
            shifted_probes,
        )

        if use_projection_scheme:
            exit_waves, error = self._projection_sets_fourier_projection(
                amplitudes,
                overlap,
                exit_waves,
                fourier_mask,
                projection_a,
                projection_b,
                projection_c,
            )

        else:
            exit_waves, error = self._gradient_descent_fourier_projection(
                amplitudes,
                overlap,
                fourier_mask,
            )

        return shifted_probes, object_patches, overlap, exit_waves, error

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        step_size,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for GD method.
        Computes object and probe update steps.

        Parameters
        --------
        object_patches: np.ndarray
            Patched object view
        shifted_probes:np.ndarray
            fractionally-shifted probes
        exit_waves:np.ndarray
            Updated exit_waves
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        object_update: np.ndarray
            Updated object estimate
        probe_update: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(shifted_probes) ** 2,
            positions_px,
        )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        if self._object_type == "potential":
            current_object += step_size * (
                self._sum_overlapping_patches_bincounts(
                    xp.real(
                        -1j
                        * xp.conj(object_patches)
                        * xp.conj(shifted_probes)
                        * exit_waves
                    ),
                    positions_px,
                )
                * probe_normalization
            )
        else:
            current_object += step_size * (
                self._sum_overlapping_patches_bincounts(
                    xp.conj(shifted_probes) * exit_waves, positions_px
                )
                * probe_normalization
            )

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(object_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe += step_size * (
                xp.sum(
                    xp.conj(object_patches) * exit_waves,
                    axis=0,
                )
                * object_normalization
            )

        return current_object, current_probe

    def _projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for DM_AP and RAAR methods.
        Computes object and probe update steps.

        Parameters
        --------
        object_patches: np.ndarray
            Patched object view
        shifted_probes:np.ndarray
            fractionally-shifted probes
        exit_waves:np.ndarray
            Updated exit_waves
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(shifted_probes) ** 2,
            positions_px,
        )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        if self._object_type == "potential":
            current_object = (
                self._sum_overlapping_patches_bincounts(
                    xp.real(
                        -1j
                        * xp.conj(object_patches)
                        * xp.conj(shifted_probes)
                        * exit_waves
                    ),
                    positions_px,
                )
                * probe_normalization
            )
        else:
            current_object = (
                self._sum_overlapping_patches_bincounts(
                    xp.conj(shifted_probes) * exit_waves,
                    positions_px,
                )
                * probe_normalization
            )

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(object_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe = (
                xp.sum(
                    xp.conj(object_patches) * exit_waves,
                    axis=0,
                )
                * object_normalization
            )

        return current_object, current_probe

    def _adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        use_projection_scheme: bool,
        step_size: float,
        normalization_min: float,
        fix_probe: bool,
    ):
        """
        Ptychographic adjoint operator.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        shifted_probes:np.ndarray
            fractionally-shifted probes
        exit_waves:np.ndarray
            Updated exit_waves
        use_projection_scheme: bool,
            If True, use generalized projection update
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        if use_projection_scheme:
            current_object, current_probe = self._projection_sets_adjoint(
                current_object,
                current_probe,
                object_patches,
                shifted_probes,
                positions_px,
                exit_waves,
                normalization_min,
                fix_probe,
            )
        else:
            current_object, current_probe = self._gradient_descent_adjoint(
                current_object,
                current_probe,
                object_patches,
                shifted_probes,
                positions_px,
                exit_waves,
                step_size,
                normalization_min,
                fix_probe,
            )

        return current_object, current_probe

    def _position_correction(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        shifted_probes,
        overlap,
        amplitudes,
        current_positions,
        current_positions_initial,
        positions_step_size,
        max_position_update_distance,
        max_position_total_distance,
    ):
        """
        Position correction using estimated intensity gradient.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        shifted_probes:np.ndarray
            fractionally-shifted probes
        overlap: np.ndarray
            object * probe overlap
        amplitudes: np.ndarray
            Measured amplitudes
        current_positions: np.ndarray
            Current positions estimate
        positions_step_size: float
            Positions step size
        max_position_update_distance: float
            Maximum allowed distance for update in A
        max_position_total_distance: float
            Maximum allowed distance from initial probe positions

        Returns
        --------
        updated_positions: np.ndarray
            Updated positions estimate
        """

        xp = self._xp
        storage = self._storage

        overlap_fft = xp.fft.fft2(overlap)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            overlap_fft = bilinear_resample(
                overlap_fft,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        # unperturbed
        overlap_fft_conj = xp.conj(overlap_fft)

        estimated_intensity = self._return_farfield_amplitudes(overlap_fft) ** 2
        measured_intensity = amplitudes**2

        # book-keeping
        flat_shape = (measured_intensity.shape[0], -1)
        difference_intensity = (measured_intensity - estimated_intensity).reshape(
            flat_shape
        )

        # dx overlap projection perturbation
        _, _, overlap_dx = self._overlap_projection(
            current_object,
            (vectorized_patch_indices_row + 1) % self._object_shape[0],
            vectorized_patch_indices_col,
            shifted_probes,
        )

        # dy overlap projection perturbation
        _, _, overlap_dy = self._overlap_projection(
            current_object,
            vectorized_patch_indices_row,
            (vectorized_patch_indices_col + 1) % self._object_shape[1],
            shifted_probes,
        )

        overlap_dx_fft = xp.fft.fft2(overlap_dx)
        overlap_dy_fft = xp.fft.fft2(overlap_dy)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            overlap_dx_fft = bilinear_resample(
                overlap_dx_fft,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )
            overlap_dy_fft = bilinear_resample(
                overlap_dy_fft,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        # partial intensities
        overlap_dx_fft = overlap_fft - overlap_dx_fft
        overlap_dy_fft = overlap_fft - overlap_dy_fft
        partial_intensity_dx = 2 * xp.real(overlap_dx_fft * overlap_fft_conj)
        partial_intensity_dy = 2 * xp.real(overlap_dy_fft * overlap_fft_conj)

        # handle mixed-state, is this correct?
        if partial_intensity_dx.ndim == 4:
            partial_intensity_dx = partial_intensity_dx.sum(1)
            partial_intensity_dy = partial_intensity_dy.sum(1)

        partial_intensity_dx = partial_intensity_dx.reshape(flat_shape)
        partial_intensity_dy = partial_intensity_dy.reshape(flat_shape)

        # least-squares fit
        coefficients_matrix = xp.dstack((partial_intensity_dx, partial_intensity_dy))
        coefficients_matrix_T = coefficients_matrix.conj().swapaxes(-1, -2)
        positions_update = (
            xp.linalg.inv(coefficients_matrix_T @ coefficients_matrix)
            @ coefficients_matrix_T
            @ difference_intensity[..., None]
        )

        positions_update = positions_update[..., 0] * positions_step_size

        if max_position_update_distance is not None:
            max_position_update_distance /= xp.sqrt(
                self.sampling[0] ** 2 + self.sampling[1] ** 2
            )
            update_norms = xp.linalg.norm(positions_update, axis=1)
            outlier_ind = update_norms > max_position_update_distance
            positions_update[outlier_ind] /= (
                update_norms[outlier_ind, None] / max_position_update_distance
            )

        if max_position_total_distance is not None:
            max_position_total_distance /= xp.sqrt(
                self.sampling[0] ** 2 + self.sampling[1] ** 2
            )
            deltas = (
                xp.asarray(current_positions - current_positions_initial)
                - positions_update
            )
            dsts = xp.linalg.norm(deltas, axis=1)
            outlier_ind = dsts > max_position_total_distance
            positions_update[outlier_ind] = 0

        current_positions -= copy_to_device(positions_update, storage)

        return current_positions

    def _return_self_consistency_errors(
        self,
        max_batch_size=None,
    ):
        """Compute the self-consistency errors for each probe position"""

        xp = self._xp
        xp_storage = self._xp_storage
        device = self._device
        asnumpy = self._asnumpy

        # Batch-size
        if max_batch_size is None:
            max_batch_size = self._num_diffraction_patterns

        errors = np.array([])

        for start, end in generate_batches(
            self._num_diffraction_patterns, max_batch=max_batch_size
        ):
            # batch indices
            positions_px = self._positions_px[start:end]
            positions_px_fractional = positions_px - xp_storage.round(positions_px)

            (
                vectorized_patch_indices_row,
                vectorized_patch_indices_col,
            ) = self._extract_vectorized_patch_indices(positions_px)

            amplitudes_device = copy_to_device(self._amplitudes[start:end], device)

            # Overlaps
            shifted_probes = self._return_shifted_probes(
                self._probe, positions_px_fractional
            )
            _, _, overlap = self._overlap_projection(
                self._object,
                vectorized_patch_indices_row,
                vectorized_patch_indices_col,
                shifted_probes,
            )

            fourier_overlap = xp.fft.fft2(overlap)

            # resample to match data, note: this needs to happen in reciprocal-space
            if self._resample_exit_waves:
                fourier_overlap = bilinear_resample(
                    fourier_overlap,
                    output_size=self._amplitudes_shape,
                    vectorized=True,
                    conserve_array_sums=True,
                    xp=xp,
                )

            farfield_amplitudes = self._return_farfield_amplitudes(fourier_overlap)

            # Normalized mean-squared errors
            batch_errors = xp.sum(
                xp.abs(amplitudes_device - farfield_amplitudes) ** 2, axis=(-2, -1)
            )
            errors = np.hstack((errors, batch_errors))

        errors /= self._mean_diffraction_intensity

        return asnumpy(errors)


class Object2p5DProbeMethodsMixin:
    """
    Mixin class for methods unique to 2.5D objects using a single probe.
    Overwrites ObjectNDProbeMethodsMixin.
    """

    def _overlap_projection(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        shifted_probes_in,
    ):
        """
        Ptychographic overlap projection method.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        shifted_probes: np.ndarray
            Shifted probes at each layer
        object_patches: np.ndarray
            Patched object view
        overlap: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions
        """

        xp = self._xp

        object_patches = current_object[
            :,
            vectorized_patch_indices_row,
            vectorized_patch_indices_col,
        ]

        if self._object_type == "potential":
            object_patches = xp.exp(1j * object_patches)

        shifted_probes = xp.empty_like(object_patches)
        shifted_probes[0] = shifted_probes_in

        for s in range(self._num_slices):
            # transmit
            overlap = object_patches[s] * shifted_probes[s]

            # propagate
            if s + 1 < self._num_slices:
                shifted_probes[s + 1] = self._propagate_array(
                    overlap, self._propagator_arrays[s]
                )

        return shifted_probes, object_patches, overlap

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        step_size,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for GD method.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        propagated_probes: np.ndarray
            Shifted probes at each layer
        exit_waves:np.ndarray
            Updated exit_waves
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        for s in reversed(range(self._num_slices)):
            probe = shifted_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = self._sum_overlapping_patches_bincounts(
                xp.abs(probe) ** 2,
                positions_px,
            )

            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            if self._object_type == "potential":
                current_object[s] += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(-1j * xp.conj(obj) * xp.conj(probe) * exit_waves),
                        positions_px,
                    )
                    * probe_normalization
                )
            else:
                current_object[s] += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        xp.conj(probe) * exit_waves, positions_px
                    )
                    * probe_normalization
                )

            # back-transmit
            exit_waves *= xp.conj(obj)

            if s > 0:
                # back-propagate
                exit_waves = self._propagate_array(
                    exit_waves, xp.conj(self._propagator_arrays[s - 1])
                )
            elif not fix_probe:
                # probe-update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )

                current_probe += (
                    step_size
                    * xp.sum(
                        exit_waves,
                        axis=0,
                    )
                    * object_normalization
                )

        return current_object, current_probe

    def _projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for DM_AP and RAAR methods.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        propagated_probes: np.ndarray
            Shifted probes at each layer
        exit_waves:np.ndarray
            Updated exit_waves
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        # careful not to modify exit_waves in-place for projection set methods
        exit_waves_copy = exit_waves.copy()

        for s in reversed(range(self._num_slices)):
            probe = shifted_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = self._sum_overlapping_patches_bincounts(
                xp.abs(probe) ** 2,
                positions_px,
            )
            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            if self._object_type == "potential":
                current_object[s] = (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(-1j * xp.conj(obj) * xp.conj(probe) * exit_waves_copy),
                        positions_px,
                    )
                    * probe_normalization
                )
            else:
                current_object[s] = (
                    self._sum_overlapping_patches_bincounts(
                        xp.conj(probe) * exit_waves_copy,
                        positions_px,
                    )
                    * probe_normalization
                )

            # back-transmit
            exit_waves_copy *= xp.conj(obj)

            if s > 0:
                # back-propagate
                exit_waves_copy = self._propagate_array(
                    exit_waves_copy, xp.conj(self._propagator_arrays[s - 1])
                )

            elif not fix_probe:
                # probe-update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )

                current_probe = (
                    xp.sum(
                        exit_waves_copy,
                        axis=0,
                    )
                    * object_normalization
                )

        return current_object, current_probe

    def show_transmitted_probe(
        self,
        max_batch_size=None,
        plot_fourier_probe: bool = False,
        remove_initial_probe_aberrations=False,
        **kwargs,
    ):
        """
        Plots the min, max, and mean transmitted probe after propagation and transmission.

        Parameters
        ----------
        max_batch_size: int, optional
            Max number of probes to calculate at once
        plot_fourier_probe: boolean, optional
            If True, the transmitted probes are also plotted in Fourier space
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
        kwargs:
            Passed to show_complex
        """

        xp = self._xp
        xp_storage = self._xp_storage
        asnumpy = self._asnumpy

        if max_batch_size is None:
            max_batch_size = self._num_diffraction_patterns

        mean_transmitted = xp.zeros_like(self._probe)
        intensities_compare = [np.inf, 0]

        for start, end in generate_batches(
            self._num_diffraction_patterns, max_batch=max_batch_size
        ):
            # batch indices
            positions_px = self._positions_px[start:end]
            positions_px_fractional = positions_px - xp_storage.round(positions_px)

            (
                vectorized_patch_indices_row,
                vectorized_patch_indices_col,
            ) = self._extract_vectorized_patch_indices(positions_px)

            # overlaps
            shifted_probes = self._return_shifted_probes(
                self._probe, positions_px_fractional
            )
            _, _, overlap = self._overlap_projection(
                self._object,
                vectorized_patch_indices_row,
                vectorized_patch_indices_col,
                shifted_probes,
            )

            # store relevant arrays
            mean_transmitted += overlap.sum(0)

            intensities = xp.sum(xp.abs(overlap) ** 2, axis=(-2, -1))
            min_intensity = intensities.min()
            max_intensity = intensities.max()

            if min_intensity < intensities_compare[0]:
                min_intensity_transmitted = overlap[xp.argmin(intensities)]
                intensities_compare[0] = min_intensity

            if max_intensity > intensities_compare[1]:
                max_intensity_transmitted = overlap[xp.argmax(intensities)]
                intensities_compare[1] = max_intensity

        mean_transmitted /= self._num_diffraction_patterns

        probes = [
            asnumpy(self._return_centered_probe(probe))
            for probe in [
                mean_transmitted,
                min_intensity_transmitted,
                max_intensity_transmitted,
            ]
        ]
        title = [
            "Mean transmitted probe",
            "Min-intensity transmitted probe",
            "Max-intensity transmitted probe",
        ]

        if plot_fourier_probe:
            bottom_row = [
                asnumpy(
                    self._return_fourier_probe(
                        probe,
                        remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                    )
                )
                for probe in [
                    mean_transmitted,
                    min_intensity_transmitted,
                    max_intensity_transmitted,
                ]
            ]
            probes = [probes, bottom_row]

            title += [
                "Mean transmitted Fourier probe",
                "Min-intensity transmitted Fourier probe",
                "Max-intensity transmitted Fourier probe",
            ]

        title = kwargs.get("title", title)
        ticks = kwargs.get("ticks", False)
        axsize = kwargs.get("axsize", (4, 4))

        show_complex(
            probes,
            title=title,
            ticks=ticks,
            axsize=axsize,
            **kwargs,
        )

        self.clear_device_mem(self._device, self._clear_fft_cache)


class ObjectNDProbeMixedMethodsMixin:
    """
    Mixin class for methods applicable to 2D, 2.5D, and 3D objects using mixed probes.
    Overwrites ObjectNDProbeMethodsMixin.
    """

    def _overlap_projection(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        shifted_probes,
    ):
        """
        Ptychographic overlap projection method.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        shifted_probes:np.ndarray
            fractionally-shifted probes
        object_patches: np.ndarray
            Patched object view
        overlap: np.ndarray
            shifted_probes * object_patches
        """

        xp = self._xp

        object_patches = current_object[
            vectorized_patch_indices_row, vectorized_patch_indices_col
        ]

        if self._object_type == "potential":
            object_patches = xp.exp(1j * object_patches)

        overlap = shifted_probes * xp.expand_dims(object_patches, axis=1)

        return shifted_probes, object_patches, overlap

    def _return_farfield_amplitudes(self, fourier_overlap):
        """Small utility to de-duplicate mixed-state Fourier projection."""

        xp = self._xp
        return xp.sqrt(xp.sum(xp.abs(fourier_overlap) ** 2, axis=1))

    def _gradient_descent_fourier_projection(self, amplitudes, overlap, fourier_mask):
        """
        Ptychographic fourier projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap
        fourier_mask: np.ndarray
            Mask to apply at the detector-plane for zeroing-out unreliable gradients
            Useful when detector has artifacts such as dead-pixels

        Returns
        --------
        exit_waves:np.ndarray
            Difference between modified and estimated exit waves
        error: float
            Reconstruction error
        """

        xp = self._xp

        fourier_overlap = xp.fft.fft2(overlap)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_overlap = bilinear_resample(
                fourier_overlap,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        if fourier_mask is not None:
            fourier_overlap *= fourier_mask
        farfield_amplitudes = self._return_farfield_amplitudes(fourier_overlap)
        error = xp.sum(xp.abs(amplitudes - farfield_amplitudes) ** 2)

        farfield_amplitudes[farfield_amplitudes == 0.0] = np.inf
        amplitude_modification = amplitudes / farfield_amplitudes

        fourier_modified_overlap = amplitude_modification[:, None] * fourier_overlap

        fourier_modified_overlap = fourier_modified_overlap - fourier_overlap
        if fourier_mask is not None:
            fourier_modified_overlap *= fourier_mask

        # resample back to region_of_interest_shape, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_modified_overlap = bilinear_resample(
                fourier_modified_overlap,
                output_size=self._region_of_interest_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        exit_waves = xp.fft.ifft2(fourier_modified_overlap)

        return exit_waves, error

    def _projection_sets_fourier_projection(
        self,
        amplitudes,
        overlap,
        exit_waves,
        fourier_mask,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Ptychographic fourier projection method for DM_AP and RAAR methods.
        Generalized projection using three parameters: a,b,c

            DM_AP(\\alpha)   :   a =  -\\alpha, b = 1, c = 1 + \\alpha
              DM: DM_AP(1.0), AP: DM_AP(0.0)

            RAAR(\\beta)     :   a = 1-2\\beta, b = \\beta, c = 2
              DM : RAAR(1.0)

            RRR(\\gamma)     :   a = -\\gamma, b = \\gamma, c = 2
              DM: RRR(1.0)

            SUPERFLIP       :   a = 0, b = 1, c = 2

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap
        exit_waves: np.ndarray
            previously estimated exit waves
        fourier_mask: np.ndarray
            Mask to apply at the detector-plane for zeroing-out unreliable gradients
            Useful when detector has artifacts such as dead-pixels
            Currently not implemented for projection sets
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        exit_waves:np.ndarray
            Updated exit_waves
        error: float
            Reconstruction error
        """

        if fourier_mask is not None:
            raise NotImplementedError()

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        if exit_waves is None:
            exit_waves = overlap.copy()

        factor_to_be_projected = projection_c * overlap + projection_y * exit_waves
        fourier_projected_factor = xp.fft.fft2(factor_to_be_projected)

        # resample to match data, note: this needs to happen in reciprocal-space
        if self._resample_exit_waves:
            fourier_projected_factor = bilinear_resample(
                fourier_projected_factor,
                output_size=self._amplitudes_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        farfield_amplitudes = self._return_farfield_amplitudes(fourier_projected_factor)
        error = xp.sum(xp.abs(amplitudes - farfield_amplitudes) ** 2)

        farfield_amplitudes[farfield_amplitudes == 0.0] = np.inf
        amplitude_modification = amplitudes / farfield_amplitudes

        fourier_projected_factor *= amplitude_modification[:, None]

        # resample back to region_of_interest_shape, note: this needs to happen in real-space
        if self._resample_exit_waves:
            fourier_projected_factor = bilinear_resample(
                fourier_projected_factor,
                output_size=self._region_of_interest_shape,
                vectorized=True,
                conserve_array_sums=True,
                xp=xp,
            )

        projected_factor = xp.fft.ifft2(fourier_projected_factor)

        exit_waves = (
            projection_x * exit_waves
            + projection_a * overlap
            + projection_b * projected_factor
        )

        return exit_waves, error

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        step_size,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for GD method.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        shifted_probes:np.ndarray
            fractionally-shifted probes
        exit_waves:np.ndarray
            Updated exit_waves
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        probe_normalization = xp.zeros_like(current_object)
        object_update = xp.zeros_like(current_object)

        for i_probe in range(self._num_probes):
            probe_normalization += self._sum_overlapping_patches_bincounts(
                xp.abs(shifted_probes[:, i_probe]) ** 2,
                positions_px,
            )
            if self._object_type == "potential":
                object_update += step_size * self._sum_overlapping_patches_bincounts(
                    xp.real(
                        -1j
                        * xp.conj(object_patches)
                        * xp.conj(shifted_probes[:, i_probe])
                        * exit_waves[:, i_probe]
                    ),
                    positions_px,
                )
            else:
                object_update += step_size * self._sum_overlapping_patches_bincounts(
                    xp.conj(shifted_probes[:, i_probe]) * exit_waves[:, i_probe],
                    positions_px,
                )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        current_object += object_update * probe_normalization

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(object_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe += step_size * (
                xp.sum(
                    xp.expand_dims(xp.conj(object_patches), axis=1) * exit_waves,
                    axis=0,
                )
                * object_normalization[None]
            )

        return current_object, current_probe

    def _projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for DM_AP and RAAR methods.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        shifted_probes:np.ndarray
            fractionally-shifted probes
        exit_waves:np.ndarray
            Updated exit_waves
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        probe_normalization = xp.zeros_like(current_object)
        current_object = xp.zeros_like(current_object)

        for i_probe in range(self._num_probes):
            probe_normalization += self._sum_overlapping_patches_bincounts(
                xp.abs(shifted_probes[:, i_probe]) ** 2,
                positions_px,
            )
            if self._object_type == "potential":
                current_object += self._sum_overlapping_patches_bincounts(
                    xp.real(
                        -1j
                        * xp.conj(object_patches)
                        * xp.conj(shifted_probes[:, i_probe])
                        * exit_waves[:, i_probe]
                    ),
                    positions_px,
                )
            else:
                current_object += self._sum_overlapping_patches_bincounts(
                    xp.conj(shifted_probes[:, i_probe]) * exit_waves[:, i_probe],
                    positions_px,
                )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        current_object *= probe_normalization

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(object_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe = (
                xp.sum(
                    xp.expand_dims(xp.conj(object_patches), axis=1) * exit_waves,
                    axis=0,
                )
                * object_normalization[None]
            )

        return current_object, current_probe


class Object2p5DProbeMixedMethodsMixin:
    """
    Mixin class for methods unique to 2.5D objects using mixed probes.
    Overwrites ObjectNDProbeMethodsMixin and ObjectNDProbeMixedMethodsMixin.
    """

    def _overlap_projection(
        self,
        current_object,
        vectorized_patch_indices_row,
        vectorized_patch_indices_col,
        shifted_probes_in,
    ):
        """
        Ptychographic overlap projection method.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        propagated_probes: np.ndarray
            Shifted probes at each layer
        object_patches: np.ndarray
            Patched object view
        transmitted_probes: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions
        """

        xp = self._xp

        object_patches = current_object[
            :,
            vectorized_patch_indices_row,
            vectorized_patch_indices_col,
        ]

        if self._object_type == "potential":
            object_patches = xp.exp(1j * object_patches)

        num_probe_positions = object_patches.shape[1]

        shifted_shape = (
            self._num_slices,
            num_probe_positions,
            self._num_probes,
            self._region_of_interest_shape[0],
            self._region_of_interest_shape[1],
        )

        shifted_probes = xp.empty(shifted_shape, dtype=object_patches.dtype)
        shifted_probes[0] = shifted_probes_in

        for s in range(self._num_slices):
            # transmit
            overlap = xp.expand_dims(object_patches[s], axis=1) * shifted_probes[s]

            # propagate
            if s + 1 < self._num_slices:
                shifted_probes[s + 1] = self._propagate_array(
                    overlap, self._propagator_arrays[s]
                )

        return shifted_probes, object_patches, overlap

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        step_size,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for GD method.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        propagated_probes: np.ndarray
            Shifted probes at each layer
        exit_waves:np.ndarray
            Updated exit_waves
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        for s in reversed(range(self._num_slices)):
            probe = shifted_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = xp.zeros_like(current_object[s])
            object_update = xp.zeros_like(current_object[s])

            for i_probe in range(self._num_probes):
                probe_normalization += self._sum_overlapping_patches_bincounts(
                    xp.abs(probe[:, i_probe]) ** 2,
                    positions_px,
                )

                if self._object_type == "potential":
                    object_update += (
                        step_size
                        * self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * xp.conj(obj)
                                * xp.conj(probe[:, i_probe])
                                * exit_waves[:, i_probe]
                            ),
                            positions_px,
                        )
                    )
                else:
                    object_update += (
                        step_size
                        * self._sum_overlapping_patches_bincounts(
                            xp.conj(probe[:, i_probe]) * exit_waves[:, i_probe],
                            positions_px,
                        )
                    )

            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            current_object[s] += object_update * probe_normalization

            # back-transmit
            exit_waves *= xp.expand_dims(xp.conj(obj), axis=1)

            if s > 0:
                # back-propagate
                exit_waves = self._propagate_array(
                    exit_waves, xp.conj(self._propagator_arrays[s - 1])
                )
            elif not fix_probe:
                # probe-update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )

                current_probe += (
                    step_size
                    * xp.sum(
                        exit_waves,
                        axis=0,
                    )
                    * object_normalization[None]
                )

        return current_object, current_probe

    def _projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        positions_px,
        exit_waves,
        normalization_min,
        fix_probe,
    ):
        """
        Ptychographic adjoint operator for DM_AP and RAAR methods.
        Computes object and probe update steps.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        object_patches: np.ndarray
            Patched object view
        propagated_probes: np.ndarray
            Shifted probes at each layer
        exit_waves:np.ndarray
            Updated exit_waves
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe: bool, optional
            If True, probe will not be updated

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        """
        xp = self._xp

        # careful not to modify exit_waves in-place for projection set methods
        exit_waves_copy = exit_waves.copy()
        for s in reversed(range(self._num_slices)):
            probe = shifted_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = xp.zeros_like(current_object[s])
            object_update = xp.zeros_like(current_object[s])

            for i_probe in range(self._num_probes):
                probe_normalization += self._sum_overlapping_patches_bincounts(
                    xp.abs(probe[:, i_probe]) ** 2,
                    positions_px,
                )

                if self._object_type == "potential":
                    object_update += self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * xp.conj(obj)
                            * xp.conj(probe[:, i_probe])
                            * exit_waves_copy[:, i_probe]
                        ),
                        positions_px,
                    )
                else:
                    object_update += self._sum_overlapping_patches_bincounts(
                        xp.conj(probe[:, i_probe]) * exit_waves_copy[:, i_probe],
                        positions_px,
                    )

            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            current_object[s] = object_update * probe_normalization

            # back-transmit
            exit_waves_copy *= xp.expand_dims(xp.conj(obj), axis=1)

            if s > 0:
                # back-propagate
                exit_waves_copy = self._propagate_array(
                    exit_waves_copy, xp.conj(self._propagator_arrays[s - 1])
                )

            elif not fix_probe:
                # probe-update
                object_normalization = xp.sum(
                    (xp.abs(obj) ** 2),
                    axis=0,
                )
                object_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * object_normalization) ** 2
                    + (normalization_min * xp.max(object_normalization)) ** 2
                )

                current_probe = (
                    xp.sum(
                        exit_waves_copy,
                        axis=0,
                    )
                    * object_normalization[None]
                )

        return current_object, current_probe

    def show_transmitted_probe(
        self,
        **kwargs,
    ):
        raise NotImplementedError()


class MultipleMeasurementsMethodsMixin:
    """
    Mixin class for methods unique to classes with multiple measurements.
    Overwrites various Mixins.
    """

    def _reset_reconstruction(
        self,
        store_iterations,
        reset,
        use_projection_scheme,
    ):
        """ """
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []

        # reset can be True, False, or None (default)
        if reset is True:
            self.error_iterations = []
            self._object = self._object_initial.copy()
            self._probes_all = [pr.copy() for pr in self._probes_all_initial]
            self._positions_px_all = self._positions_px_initial_all.copy()
            self._object_type = self._object_type_initial

            if use_projection_scheme:
                self._exit_waves = [None] * len(self._probes_all)
            else:
                self._exit_waves = None

            # delete positions affine transform
            if hasattr(self, "_tf"):
                del self._tf

        elif reset is None:
            # continued run
            if hasattr(self, "error"):
                warnings.warn(
                    (
                        "Continuing reconstruction from previous result. "
                        "Use reset=True for a fresh start."
                    ),
                    UserWarning,
                )

            # first start
            else:
                self.error_iterations = []
                if use_projection_scheme:
                    self._exit_waves = [None] * len(self._probes_all)
                else:
                    self._exit_waves = None
        else:
            # reset=False first start
            if not hasattr(self, "error"):
                self.error_iterations = []
                if use_projection_scheme:
                    self._exit_waves = [None] * len(self._probes_all)
                else:
                    self._exit_waves = None

    def _return_single_probe(self, probe=None):
        """Current probe estimate"""
        xp = self._xp

        if probe is not None:
            _probes = [xp.asarray(pr) for pr in probe]
        else:
            if not hasattr(self, "_probes_all"):
                return None
            _probes = self._probes_all

        probe = xp.zeros(self._region_of_interest_shape, dtype=np.complex64)

        for pr in _probes:
            probe += pr

        return probe / len(_probes)

    def _return_average_positions(
        self, positions=None, cum_probes_per_measurement=None
    ):
        """Average positions estimate"""
        xp_storage = self._xp_storage

        if positions is not None:
            _pos = xp_storage.asarray(positions)
        else:
            if not hasattr(self, "_positions_px_all"):
                return None
            _pos = self._positions_px_all

        if cum_probes_per_measurement is None:
            cum_probes_per_measurement = self._cum_probes_per_measurement

        num_probes_per_measurement = np.diff(cum_probes_per_measurement)
        num_measurements = len(num_probes_per_measurement)

        if np.any(num_probes_per_measurement != num_probes_per_measurement[0]):
            return None

        avg_positions = xp_storage.zeros(
            (num_probes_per_measurement[0], 2), dtype=xp_storage.float32
        )

        for index in range(num_measurements):
            start_idx = cum_probes_per_measurement[index]
            end_idx = cum_probes_per_measurement[index + 1]
            avg_positions += _pos[start_idx:end_idx]

        return avg_positions / num_measurements

    def _return_self_consistency_errors(
        self,
        **kwargs,
    ):
        """Compute the self-consistency errors for each probe position"""
        raise NotImplementedError()

    @property
    def probe_fourier(self):
        """Current probe estimate in Fourier space"""
        if not hasattr(self, "_probes_all"):
            return None

        asnumpy = self._asnumpy
        return [asnumpy(self._return_fourier_probe(pr)) for pr in self._probes_all]

    @property
    def probe_fourier_residual(self):
        """Current probe estimate in Fourier space"""
        if not hasattr(self, "_probes_all"):
            return None

        asnumpy = self._asnumpy
        return [
            asnumpy(
                self._return_fourier_probe(pr, remove_initial_probe_aberrations=True)
            )
            for pr in self._probes_all
        ]

    @property
    def probe_centered(self):
        """Current probe estimate shifted to the center"""
        if not hasattr(self, "_probes_all"):
            return None

        asnumpy = self._asnumpy
        return [asnumpy(self._return_centered_probe(pr)) for pr in self._probes_all]

    @property
    def positions(self):
        """Probe positions [A]"""

        if self.angular_sampling is None:
            return None

        asnumpy = self._asnumpy
        positions_all = []

        for index in range(self._num_measurements):
            start_idx = self._cum_probes_per_measurement[index]
            end_idx = self._cum_probes_per_measurement[index + 1]
            positions = self._positions_px_all[start_idx:end_idx].copy()
            positions[:, 0] *= self.sampling[0]
            positions[:, 1] *= self.sampling[1]
            positions_all.append(asnumpy(positions))

        return np.asarray(positions_all)
