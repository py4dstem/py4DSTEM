"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely overlap tomography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg
from scipy.ndimage import rotate as rotate_np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np

from emdfile import Custom, tqdmnd
from py4DSTEM import DataCube
from py4DSTEM.process.phase.iterative_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.iterative_ptychographic_constraints import (
    Object2p5DConstraintsMixin,
    Object3DConstraintsMixin,
    ObjectNDConstraintsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
)
from py4DSTEM.process.phase.iterative_ptychographic_methods import (
    Object2p5DMethodsMixin,
    Object2p5DProbeMethodsMixin,
    Object3DMethodsMixin,
    ObjectNDMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeListMethodsMixin,
    ProbeMethodsMixin,
)
from py4DSTEM.process.phase.iterative_ptychographic_visualizations import (
    VisualizationsMixin,
)
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
)

warnings.simplefilter(action="always", category=UserWarning)


class OverlapTomographicReconstruction(
    VisualizationsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
    Object3DConstraintsMixin,
    Object2p5DConstraintsMixin,
    ObjectNDConstraintsMixin,
    Object2p5DProbeMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeListMethodsMixin,
    ProbeMethodsMixin,
    Object3DMethodsMixin,
    Object2p5DMethodsMixin,
    ObjectNDMethodsMixin,
    PtychographicReconstruction,
):
    """
    Overlap Tomographic Reconstruction Class.

    List of diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions              : (Sx,Sy)
    Reconstructed object dimensions             : (Px,Py,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our probe
    and (Px,Py,Py) is the padded-object electrostatic potential volume,
    where x-axis is the tilt.

    Parameters
    ----------
    datacube: List of DataCubes
        Input list of 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    num_slices: int
        Number of slices to use in the forward model
    tilt_orientation_matrices: Sequence[np.ndarray]
        List of orientation matrices for each tilt
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial probe guess in mrad
    semiangle_cutoff_pixels: float, optional
        Semiangle cutoff for the initial probe guess in pixels
    rolloff: float, optional
        Semiangle rolloff for the initial probe guess
    vacuum_probe_intensity: np.ndarray, optional
        Vacuum probe to use as intensity aperture for initial probe guess
    polar_parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in Å and angles should be given in radians.
    object_padding_px: Tuple[int,int], optional
        Pixel dimensions to pad object with
        If None, the padding is set to half the probe ROI dimensions
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (Px,Py,Py)
        If None, initialized to 1.0
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy). If None,
        initialized to ComplexProbe with semiangle_cutoff, energy, and aberrations
    initial_scan_positions: list of np.ndarray, optional
        Probe positions in Å for each diffraction intensity per tilt
        If None, initialized to a grid scan centered along tilt axis
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    object_type: str, optional
        The object can be reconstructed as a real potential ('potential') or a complex
        object ('complex')
    positions_mask: np.ndarray, optional
        Boolean real space mask to select positions to ignore in reconstruction
    name: str, optional
        Class name
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = ("_num_slices", "_tilt_orientation_matrices")

    def __init__(
        self,
        energy: float,
        num_slices: int,
        tilt_orientation_matrices: Sequence[np.ndarray],
        datacube: Sequence[DataCube] = None,
        semiangle_cutoff: float = None,
        semiangle_cutoff_pixels: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        object_padding_px: Tuple[int, int] = None,
        object_type: str = "potential",
        positions_mask: np.ndarray = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        initial_scan_positions: Sequence[np.ndarray] = None,
        verbose: bool = True,
        device: str = "cpu",
        name: str = "overlap-tomographic_reconstruction",
        **kwargs,
    ):
        Custom.__init__(self, name=name)

        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
            from scipy.ndimage import affine_transform, gaussian_filter, rotate, zoom

            self._gaussian_filter = gaussian_filter
            self._zoom = zoom
            self._rotate = rotate
            self._affine_transform = affine_transform
            from scipy.special import erf

            self._erf = erf
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
            from cupyx.scipy.ndimage import (
                affine_transform,
                gaussian_filter,
                rotate,
                zoom,
            )

            self._gaussian_filter = gaussian_filter
            self._zoom = zoom
            self._rotate = rotate
            self._affine_transform = affine_transform
            from cupyx.scipy.special import erf

            self._erf = erf
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

        num_tilts = len(tilt_orientation_matrices)
        if initial_scan_positions is None:
            initial_scan_positions = [None] * num_tilts

        if object_type != "potential":
            raise NotImplementedError()

        self.set_save_defaults()

        # Data
        self._datacube = datacube
        self._object = initial_object_guess
        self._probe_init = initial_probe_guess

        # Common Metadata
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._scan_positions = initial_scan_positions
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._semiangle_cutoff_pixels = semiangle_cutoff_pixels
        self._rolloff = rolloff
        self._object_type = object_type
        self._object_padding_px = object_padding_px
        self._positions_mask = positions_mask
        self._verbose = verbose
        self._device = device
        self._preprocessed = False

        # Class-specific Metadata
        self._num_slices = num_slices
        self._tilt_orientation_matrices = tuple(tilt_orientation_matrices)
        self._num_tilts = num_tilts

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "fourier",
        probe_roi_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_probe_overlaps: bool = True,
        rotation_real_space_degrees: float = None,
        diffraction_patterns_rotate_degrees: float = None,
        diffraction_patterns_transpose: bool = None,
        force_com_shifts: Sequence[float] = None,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        progress_bar: bool = True,
        object_fov_mask: np.ndarray = None,
        crop_patterns: bool = False,
        main_tilt_axis: str = "vertical",
        **kwargs,
    ):
        """
        Ptychographic preprocessing step.

        Additionally, it initializes an (Px,Py, Py) array of 1.0
        and a complex probe using the specified polar parameters.

        Parameters
        ----------
        diffraction_intensities_shape: Tuple[int,int], optional
            Pixel dimensions (Qx',Qy') of the resampled diffraction intensities
            If None, no resampling of diffraction intenstities is performed
        reshaping_method: str, optional
            Method to use for reshaping, either 'bin, 'bilinear', or 'fourier' (default)
        probe_roi_shape, (int,int), optional
            Padded diffraction intensities shape.
            If None, no padding is performed
        dp_mask: ndarray, optional
            Mask for datacube intensities (Qx,Qy)
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
        plot_probe_overlaps: bool, optional
            If True, initial probe overlaps scanned over the object will be displayed
        rotation_real_space_degrees: float (degrees), optional
            In plane rotation around z axis between x axis and tilt axis in
            real space (forced to be in xy plane)
        diffraction_patterns_rotate_degrees: float, optional
            Relative rotation angle between real and reciprocal space
        diffraction_patterns_transpose: bool, optional
            Whether diffraction intensities need to be transposed.
        force_com_shifts: list of tuple of ndarrays (CoMx, CoMy)
            Amplitudes come from diffraction patterns shifted with
            the CoM in the upper left corner for each probe unless
            shift is overwritten. One tuple per tilt.
        force_scan_sampling: float, optional
            Override DataCube real space scan pixel size calibrations, in Angstrom
        force_angular_sampling: float, optional
            Override DataCube reciprocal pixel size calibration, in mrad
        force_reciprocal_sampling: float, optional
            Override DataCube reciprocal pixel size calibration, in A^-1
        object_fov_mask: np.ndarray (boolean)
            Boolean mask of FOV. Used to calculate additional shrinkage of object
            If None, probe_overlap intensity is thresholded
        crop_patterns: bool
            If True, crop patterns to avoid wrap around of patterns when centering
        main_tilt_axis: str
            The default, 'vertical' (first scan dimension), results in object size (q,p,q),
            'horizontal' (second scan dimension) results in object size (p,p,q),
            any other value (e.g. None) results in object size (max(p,q),p,q).

        Returns
        --------
        self: OverlapTomographicReconstruction
            Self to accommodate chaining
        """
        xp = self._xp
        asnumpy = self._asnumpy

        # set additional metadata
        self._diffraction_intensities_shape = diffraction_intensities_shape
        self._reshaping_method = reshaping_method
        self._probe_roi_shape = probe_roi_shape
        self._dp_mask = dp_mask

        if self._datacube is None:
            raise ValueError(
                (
                    "The preprocess() method requires a DataCube. "
                    "Please run ptycho.attach_datacube(DataCube) first."
                )
            )

        if self._positions_mask is not None:
            self._positions_mask = np.asarray(self._positions_mask, dtype="bool")

            if self._positions_mask.ndim == 2:
                warnings.warn(
                    "2D `positions_mask` assumed the same for all measurements.",
                    UserWarning,
                )
                self._positions_mask = np.tile(
                    self._positions_mask, (self._num_tilts, 1, 1)
                )

            num_probes_per_tilt = np.insert(
                self._positions_mask.sum(axis=(-2, -1)), 0, 0
            )

        else:
            self._positions_mask = [None] * self._num_tilts
            num_probes_per_tilt = [0] + [dc.R_N for dc in self._datacube]
            num_probes_per_tilt = np.array(num_probes_per_tilt)

        # prepopulate relevant arrays
        self._mean_diffraction_intensity = []
        self._num_diffraction_patterns = num_probes_per_tilt.sum()
        self._cum_probes_per_tilt = np.cumsum(num_probes_per_tilt)
        self._positions_px_all = np.empty((self._num_diffraction_patterns, 2))

        # calculate roi_shape
        roi_shape = self._datacube[0].Qshape
        if diffraction_intensities_shape is not None:
            roi_shape = diffraction_intensities_shape
        if probe_roi_shape is not None:
            roi_shape = tuple(max(q, s) for q, s in zip(roi_shape, probe_roi_shape))
        self._amplitudes = xp.empty((self._num_diffraction_patterns,) + roi_shape)
        self._region_of_interest_shape = np.array(roi_shape)

        # TO-DO: generalize this
        if force_com_shifts is None:
            force_com_shifts = [None] * self._num_tilts

        self._rotation_best_rad = np.deg2rad(diffraction_patterns_rotate_degrees)
        self._rotation_best_transpose = diffraction_patterns_transpose

        # loop over DPs for preprocessing
        for tilt_index in tqdmnd(
            self._num_tilts,
            desc="Preprocessing data",
            unit="tilt",
            disable=not progress_bar,
        ):
            # preprocess datacube, vacuum and masks only for first tilt
            if tilt_index == 0:
                (
                    self._datacube[tilt_index],
                    self._vacuum_probe_intensity,
                    self._dp_mask,
                    force_com_shifts[tilt_index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[tilt_index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    probe_roi_shape=self._probe_roi_shape,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    dp_mask=self._dp_mask,
                    com_shifts=force_com_shifts[tilt_index],
                )

            else:
                (
                    self._datacube[tilt_index],
                    _,
                    _,
                    force_com_shifts[tilt_index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[tilt_index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    probe_roi_shape=self._probe_roi_shape,
                    vacuum_probe_intensity=None,
                    dp_mask=None,
                    com_shifts=force_com_shifts[tilt_index],
                )

            # calibrations
            intensities = self._extract_intensities_and_calibrations_from_datacube(
                self._datacube[tilt_index],
                require_calibrations=True,
                force_scan_sampling=force_scan_sampling,
                force_angular_sampling=force_angular_sampling,
                force_reciprocal_sampling=force_reciprocal_sampling,
            )

            # calculate CoM
            (
                com_measured_x,
                com_measured_y,
                com_fitted_x,
                com_fitted_y,
                com_normalized_x,
                com_normalized_y,
            ) = self._calculate_intensities_center_of_mass(
                intensities,
                dp_mask=self._dp_mask,
                fit_function=fit_function,
                com_shifts=force_com_shifts[tilt_index],
            )

            # corner-center amplitudes
            idx_start = self._cum_probes_per_tilt[tilt_index]
            idx_end = self._cum_probes_per_tilt[tilt_index + 1]
            (
                self._amplitudes[idx_start:idx_end],
                mean_diffraction_intensity_temp,
                self._crop_mask,
            ) = self._normalize_diffraction_intensities(
                intensities,
                com_fitted_x,
                com_fitted_y,
                self._positions_mask[tilt_index],
                crop_patterns,
            )

            self._mean_diffraction_intensity.append(mean_diffraction_intensity_temp)

            del (
                intensities,
                com_measured_x,
                com_measured_y,
                com_fitted_x,
                com_fitted_y,
                com_normalized_x,
                com_normalized_y,
            )

            # initialize probe positions
            (
                self._positions_px_all[idx_start:idx_end],
                self._object_padding_px,
            ) = self._calculate_scan_positions_in_pixels(
                self._scan_positions[tilt_index],
                self._positions_mask[tilt_index],
                self._object_padding_px,
            )

        # handle semiangle specified in pixels
        if self._semiangle_cutoff_pixels:
            self._semiangle_cutoff = (
                self._semiangle_cutoff_pixels * self._angular_sampling[0]
            )

        # initialize object
        self._object = self._initialize_object(
            self._object,
            self._positions_px_all,
            self._object_type,
            main_tilt_axis,
        )

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[0]

        # center probe positions
        self._positions_px_all = xp.asarray(self._positions_px_all, dtype=xp.float32)

        for tilt_index in range(self._num_tilts):
            idx_start = self._cum_probes_per_tilt[tilt_index]
            idx_end = self._cum_probes_per_tilt[tilt_index + 1]
            self._positions_px = self._positions_px_all[idx_start:idx_end]
            self._positions_px_com = xp.mean(self._positions_px, axis=0)
            self._positions_px -= (
                self._positions_px_com - xp.array(self._object_shape) / 2
            )

            self._positions_px_all[idx_start:idx_end] = self._positions_px.copy()

        self._positions_px_initial_all = self._positions_px_all.copy()
        self._positions_initial_all = self._positions_px_initial_all.copy()
        self._positions_initial_all[:, 0] *= self.sampling[0]
        self._positions_initial_all[:, 1] *= self.sampling[1]

        # initialize probe
        self._probes_all = []
        self._probes_all_initial = []
        self._probes_all_initial_aperture = []
        list_Q = isinstance(self._probe_init, (list, tuple))

        for tilt_index in range(self._num_tilts):
            _probe, self._semiangle_cutoff = self._initialize_probe(
                self._probe_init[tilt_index] if list_Q else self._probe_init,
                self._vacuum_probe_intensity,
                self._mean_diffraction_intensity[tilt_index],
                self._semiangle_cutoff,
                crop_patterns,
            )

            self._probes_all.append(_probe)
            self._probes_all_initial.append(_probe.copy())
            self._probes_all_initial_aperture.append(xp.abs(xp.fft.fft2(_probe)))

        del self._probe_init

        # initialize aberrations
        self._known_aberrations_array = ComplexProbe(
            energy=self._energy,
            gpts=self._region_of_interest_shape,
            sampling=self.sampling,
            parameters=self._polar_parameters,
            device=self._device,
        )._evaluate_ctf()

        # Precomputed propagator arrays
        self._slice_thicknesses = np.tile(
            self._object_shape[1] * self.sampling[1] / self._num_slices,
            self._num_slices - 1,
        )
        self._propagator_arrays = self._precompute_propagator_arrays(
            self._region_of_interest_shape,
            self.sampling,
            self._energy,
            self._slice_thicknesses,
        )

        # overlaps
        if object_fov_mask is None:
            probe_overlap_3D = xp.zeros_like(self._object)
            old_rot_matrix = np.eye(3)  # identity

            for tilt_index in range(self._num_tilts):
                idx_start = self._cum_probes_per_tilt[tilt_index]
                idx_end = self._cum_probes_per_tilt[tilt_index + 1]
                rot_matrix = self._tilt_orientation_matrices[tilt_index]

                probe_overlap_3D = self._rotate_zxy_volume(
                    probe_overlap_3D,
                    rot_matrix @ old_rot_matrix.T,
                )

                self._positions_px = self._positions_px_all[idx_start:idx_end]
                self._positions_px_fractional = self._positions_px - xp.round(
                    self._positions_px
                )
                shifted_probes = fft_shift(
                    self._probes_all[tilt_index], self._positions_px_fractional, xp
                )
                probe_intensities = xp.abs(shifted_probes) ** 2
                probe_overlap = self._sum_overlapping_patches_bincounts(
                    probe_intensities
                )

                probe_overlap_3D += probe_overlap[None]
                old_rot_matrix = rot_matrix

            probe_overlap_3D = self._rotate_zxy_volume(
                probe_overlap_3D,
                old_rot_matrix.T,
            )

            probe_overlap_3D_blurred = self._gaussian_filter(probe_overlap_3D, 1.0)
            self._object_fov_mask = asnumpy(
                probe_overlap_3D_blurred > 0.25 * probe_overlap_3D_blurred.max()
            )

        else:
            self._object_fov_mask = np.asarray(object_fov_mask)

            self._positions_px = self._positions_px_all[: self._cum_probes_per_tilt[1]]
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            shifted_probes = fft_shift(
                self._probes_all[0], self._positions_px_fractional, xp
            )
            probe_intensities = xp.abs(shifted_probes) ** 2
            probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)

        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        if plot_probe_overlaps:
            figsize = kwargs.pop("figsize", (13, 4))
            chroma_boost = kwargs.pop("chroma_boost", 1)
            power = kwargs.pop("power", 2)

            # initial probe
            complex_probe_rgb = Complex2RGB(
                self.probe_centered,
                power=power,
                chroma_boost=chroma_boost,
            )

            # propagated
            propagated_probe = self._probe.copy()

            for s in range(self._num_slices - 1):
                propagated_probe = self._propagate_array(
                    propagated_probe, self._propagator_arrays[s]
                )
            complex_propagated_rgb = Complex2RGB(
                asnumpy(self._return_centered_probe(propagated_probe)),
                power=power,
                chroma_boost=chroma_boost,
            )

            extent = [
                0,
                self.sampling[1] * self._object_shape[1],
                self.sampling[0] * self._object_shape[0],
                0,
            ]

            probe_extent = [
                0,
                self.sampling[1] * self._region_of_interest_shape[1],
                self.sampling[0] * self._region_of_interest_shape[0],
                0,
            ]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

            ax1.imshow(
                complex_probe_rgb,
                extent=probe_extent,
            )

            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad="2.5%")
            add_colorbar_arg(
                cax1,
                chroma_boost=chroma_boost,
            )
            ax1.set_ylabel("x [A]")
            ax1.set_xlabel("y [A]")
            ax1.set_title("Initial probe intensity")

            ax2.imshow(
                complex_propagated_rgb,
                extent=probe_extent,
            )

            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="5%", pad="2.5%")
            add_colorbar_arg(
                cax2,
                chroma_boost=chroma_boost,
            )
            ax2.set_ylabel("x [A]")
            ax2.set_xlabel("y [A]")
            ax2.set_title("Propagated probe intensity")

            ax3.imshow(
                asnumpy(probe_overlap),
                extent=extent,
                cmap="Greys_r",
            )
            ax3.scatter(
                self.positions[0, :, 1],
                self.positions[0, :, 0],
                s=2.5,
                color=(1, 0, 0, 1),
            )
            ax3.set_ylabel("x [A]")
            ax3.set_xlabel("y [A]")
            ax3.set_xlim((extent[0], extent[1]))
            ax3.set_ylim((extent[2], extent[3]))
            ax3.set_title("Object field of view")

            fig.tight_layout()

        self._preprocessed = True

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def _constraints(
        self,
        current_object,
        current_probe,
        current_positions,
        fix_com,
        fit_probe_aberrations,
        fit_probe_aberrations_max_angular_order,
        fit_probe_aberrations_max_radial_order,
        constrain_probe_amplitude,
        constrain_probe_amplitude_relative_radius,
        constrain_probe_amplitude_relative_width,
        constrain_probe_fourier_amplitude,
        constrain_probe_fourier_amplitude_max_width_pixels,
        constrain_probe_fourier_amplitude_constant_intensity,
        fix_probe_aperture,
        initial_probe_aperture,
        fix_positions,
        global_affine_transformation,
        gaussian_filter,
        gaussian_filter_sigma,
        butterworth_filter,
        q_lowpass,
        q_highpass,
        butterworth_order,
        object_positivity,
        shrinkage_rad,
        object_mask,
        tv_denoise,
        tv_denoise_weights,
        tv_denoise_inner_iter,
    ):
        """
        Ptychographic constraints operator.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        current_positions: np.ndarray
            Current positions estimate
        fix_com: bool
            If True, probe CoM is fixed to the center
        fit_probe_aberrations: bool
            If True, fits the probe aberrations to a low-order expansion
        fit_probe_aberrations_max_angular_order: bool
            Max angular order of probe aberrations basis functions
        fit_probe_aberrations_max_radial_order: bool
            Max radial order of probe aberrations basis functions
        constrain_probe_amplitude: bool
            If True, probe amplitude is constrained by top hat function
        constrain_probe_amplitude_relative_radius: float
            Relative location of top-hat inflection point, between 0 and 0.5
        constrain_probe_amplitude_relative_width: float
            Relative width of top-hat sigmoid, between 0 and 0.5
        constrain_probe_fourier_amplitude: bool
            If True, probe aperture is constrained by fitting a sigmoid for each angular frequency.
        constrain_probe_fourier_amplitude_max_width_pixels: float
            Maximum pixel width of fitted sigmoid functions.
        constrain_probe_fourier_amplitude_constant_intensity: bool
            If True, the probe aperture is additionally constrained to a constant intensity.
        fix_probe_aperture: bool,
            If True, probe Fourier amplitude is replaced by initial probe aperture.
        initial_probe_aperture: np.ndarray,
            Initial probe aperture to use in replacing probe Fourier amplitude.
        fix_positions: bool
            If True, positions are not updated
        gaussian_filter: bool
            If True, applies real-space gaussian filter
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A
        butterworth_filter: bool
            If True, applies fourier-space butterworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        object_positivity: bool
            If True, forces object to be positive
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        object_mask: np.ndarray (boolean)
            If not None, used to calculate additional shrinkage using masked-mean of object
        tv_denoise: bool
            If True, applies TV denoising on object
        tv_denoise_weights: [float,float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        constrained_probe: np.ndarray
            Constrained probe estimate
        constrained_positions: np.ndarray
            Constrained positions estimate
        """

        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma, pure_phase_object=False
            )

        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )
        if tv_denoise:
            current_object = self._object_denoise_tv_pylops(
                current_object,
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object = self._object_shrinkage_constraint(
                current_object,
                shrinkage_rad,
                object_mask,
            )

        if object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        if fix_com:
            current_probe = self._probe_center_of_mass_constraint(current_probe)

        if fix_probe_aperture:
            current_probe = self._probe_aperture_constraint(
                current_probe,
                initial_probe_aperture,
            )
        elif constrain_probe_fourier_amplitude:
            current_probe = self._probe_fourier_amplitude_constraint(
                current_probe,
                constrain_probe_fourier_amplitude_max_width_pixels,
                constrain_probe_fourier_amplitude_constant_intensity,
            )

        if fit_probe_aberrations:
            current_probe = self._probe_aberration_fitting_constraint(
                current_probe,
                fit_probe_aberrations_max_angular_order,
                fit_probe_aberrations_max_radial_order,
            )

        if constrain_probe_amplitude:
            current_probe = self._probe_amplitude_constraint(
                current_probe,
                constrain_probe_amplitude_relative_radius,
                constrain_probe_amplitude_relative_width,
            )

        if not fix_positions:
            current_positions = self._positions_center_of_mass_constraint(
                current_positions
            )

            if global_affine_transformation:
                current_positions = self._positions_affine_transformation_constraint(
                    self._positions_px_initial, current_positions
                )

        return current_object, current_probe, current_positions

    def reconstruct(
        self,
        max_iter: int = 8,
        reconstruction_method: str = "gradient-descent",
        reconstruction_parameter: float = 1.0,
        reconstruction_parameter_a: float = None,
        reconstruction_parameter_b: float = None,
        reconstruction_parameter_c: float = None,
        max_batch_size: int = None,
        seed_random: int = None,
        step_size: float = 0.5,
        normalization_min: float = 1,
        positions_step_size: float = 0.9,
        fix_com: bool = True,
        fix_probe_iter: int = 0,
        fix_probe_aperture_iter: int = 0,
        constrain_probe_amplitude_iter: int = 0,
        constrain_probe_amplitude_relative_radius: float = 0.5,
        constrain_probe_amplitude_relative_width: float = 0.05,
        constrain_probe_fourier_amplitude_iter: int = 0,
        constrain_probe_fourier_amplitude_max_width_pixels: float = 3.0,
        constrain_probe_fourier_amplitude_constant_intensity: bool = False,
        fix_positions_iter: int = np.inf,
        max_position_update_distance: float = None,
        max_position_total_distance: float = None,
        global_affine_transformation: bool = True,
        gaussian_filter_sigma: float = None,
        gaussian_filter_iter: int = np.inf,
        fit_probe_aberrations_iter: int = 0,
        fit_probe_aberrations_max_angular_order: int = 4,
        fit_probe_aberrations_max_radial_order: int = 4,
        butterworth_filter_iter: int = np.inf,
        q_lowpass: float = None,
        q_highpass: float = None,
        butterworth_order: float = 2,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
        tv_denoise_iter=np.inf,
        tv_denoise_weights=None,
        tv_denoise_inner_iter=40,
        collective_tilt_updates: bool = False,
        store_iterations: bool = False,
        progress_bar: bool = True,
        reset: bool = None,
    ):
        """
        Ptychographic reconstruction main method.

        Parameters
        --------
        max_iter: int, optional
            Maximum number of iterations to run
        reconstruction_method: str, optional
            Specifies which reconstruction algorithm to use, one of:
            "generalized-projections",
            "DM_AP" (or "difference-map_alternating-projections"),
            "RAAR" (or "relaxed-averaged-alternating-reflections"),
            "RRR" (or "relax-reflect-reflect"),
            "SUPERFLIP" (or "charge-flipping"), or
            "GD" (or "gradient_descent")
        reconstruction_parameter: float, optional
            Reconstruction parameter for various reconstruction methods above.
        reconstruction_parameter_a: float, optional
            Reconstruction parameter a for reconstruction_method='generalized-projections'.
        reconstruction_parameter_b: float, optional
            Reconstruction parameter b for reconstruction_method='generalized-projections'.
        reconstruction_parameter_c: float, optional
            Reconstruction parameter c for reconstruction_method='generalized-projections'.
        max_batch_size: int, optional
            Max number of probes to update at once
        seed_random: int, optional
            Seeds the random number generator, only applicable when max_batch_size is not None
        step_size: float, optional
            Update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        positions_step_size: float, optional
            Positions update step size
        fix_com: bool, optional
            If True, fixes center of mass of probe
        fix_probe_iter: int, optional
            Number of iterations to run with a fixed probe before updating probe estimate
        fix_probe_aperture_iter: int, optional
            Number of iterations to run with a fixed probe Fourier amplitude before updating probe estimate
        constrain_probe_amplitude_iter: int, optional
            Number of iterations to run while constraining the real-space probe with a top-hat support.
        constrain_probe_amplitude_relative_radius: float
            Relative location of top-hat inflection point, between 0 and 0.5
        constrain_probe_amplitude_relative_width: float
            Relative width of top-hat sigmoid, between 0 and 0.5
        constrain_probe_fourier_amplitude_iter: int, optional
            Number of iterations to run while constraining the Fourier-space probe by fitting a sigmoid for each angular frequency.
        constrain_probe_fourier_amplitude_max_width_pixels: float
            Maximum pixel width of fitted sigmoid functions.
        constrain_probe_fourier_amplitude_constant_intensity: bool
            If True, the probe aperture is additionally constrained to a constant intensity.
        fix_positions_iter: int, optional
            Number of iterations to run with fixed positions before updating positions estimate
        max_position_update_distance: float, optional
            Maximum allowed distance for update in A
        max_position_total_distance: float, optional
            Maximum allowed distance from initial positions
        global_affine_transformation: bool, optional
            If True, positions are assumed to be a global affine transform from initial scan
        gaussian_filter_sigma: float, optional
            Standard deviation of gaussian kernel in A
        gaussian_filter_iter: int, optional
            Number of iterations to run using object smoothness constraint
        fit_probe_aberrations_iter: int, optional
            Number of iterations to run while fitting the probe aberrations to a low-order expansion
        fit_probe_aberrations_max_angular_order: bool
            Max angular order of probe aberrations basis functions
        fit_probe_aberrations_max_radial_order: bool
            Max radial order of probe aberrations basis functions
        butterworth_filter_iter: int, optional
            Number of iterations to run using high-pass butteworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        object_positivity: bool, optional
            If True, forces object to be positive
        tv_denoise: bool
            If True, applies TV denoising on object
        tv_denoise_weights: [float,float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        collective_tilt_updates: bool
            if True perform collective tilt updates
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        reset: bool, optional
            If True, previous reconstructions are ignored

        Returns
        --------
        self: OverlapTomographicReconstruction
            Self to accommodate chaining
        """
        asnumpy = self._asnumpy
        xp = self._xp

        # set and report reconstruction method
        (
            use_projection_scheme,
            projection_a,
            projection_b,
            projection_c,
            reconstruction_parameter,
            step_size,
        ) = self._set_reconstruction_method_parameters(
            reconstruction_method,
            reconstruction_parameter,
            reconstruction_parameter_a,
            reconstruction_parameter_b,
            reconstruction_parameter_c,
            step_size,
        )

        if self._verbose:
            self._report_reconstruction_summary(
                max_iter,
                np.inf,
                use_projection_scheme,
                reconstruction_method,
                reconstruction_parameter,
                projection_a,
                projection_b,
                projection_c,
                normalization_min,
                step_size,
                max_batch_size,
            )

        # batching
        shuffled_indices = np.arange(self._num_diffraction_patterns)
        unshuffled_indices = np.zeros_like(shuffled_indices)

        if max_batch_size is not None:
            xp.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        # initialization
        self._reset_reconstruction(store_iterations, reset, use_projection_scheme)

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing object and probe",
            unit=" iter",
            disable=not progress_bar,
        ):
            error = 0.0

            if collective_tilt_updates:
                collective_object = xp.zeros_like(self._object)

            tilt_indices = np.arange(self._num_tilts)
            np.random.shuffle(tilt_indices)

            old_rot_matrix = np.eye(3)  # identity

            for tilt_index in tilt_indices:
                self._active_tilt_index = tilt_index

                tilt_error = 0.0

                rot_matrix = self._tilt_orientation_matrices[self._active_tilt_index]
                self._object = self._rotate_zxy_volume(
                    self._object,
                    rot_matrix @ old_rot_matrix.T,
                )

                object_sliced = self._project_sliced_object(
                    self._object, self._num_slices
                )

                _probe = self._probes_all[self._active_tilt_index]
                _probe_initial_aperture = self._probes_all_initial_aperture[
                    self._active_tilt_index
                ]

                if not use_projection_scheme:
                    object_sliced_old = object_sliced.copy()

                start_tilt = self._cum_probes_per_tilt[self._active_tilt_index]
                end_tilt = self._cum_probes_per_tilt[self._active_tilt_index + 1]

                num_diffraction_patterns = end_tilt - start_tilt
                shuffled_indices = np.arange(num_diffraction_patterns)
                unshuffled_indices = np.zeros_like(shuffled_indices)

                # randomize
                if not use_projection_scheme:
                    np.random.shuffle(shuffled_indices)

                unshuffled_indices[shuffled_indices] = np.arange(
                    num_diffraction_patterns
                )

                positions_px = self._positions_px_all[start_tilt:end_tilt].copy()[
                    shuffled_indices
                ]
                initial_positions_px = self._positions_px_initial_all[
                    start_tilt:end_tilt
                ].copy()[shuffled_indices]

                for start, end in generate_batches(
                    num_diffraction_patterns, max_batch=max_batch_size
                ):
                    # batch indices
                    self._positions_px = positions_px[start:end]
                    self._positions_px_initial = initial_positions_px[start:end]
                    self._positions_px_com = xp.mean(self._positions_px, axis=0)
                    self._positions_px_fractional = self._positions_px - xp.round(
                        self._positions_px
                    )

                    (
                        self._vectorized_patch_indices_row,
                        self._vectorized_patch_indices_col,
                    ) = self._extract_vectorized_patch_indices()

                    amplitudes = self._amplitudes[start_tilt:end_tilt][
                        shuffled_indices[start:end]
                    ]

                    # forward operator
                    (
                        shifted_probes,
                        object_patches,
                        overlap,
                        self._exit_waves,
                        batch_error,
                    ) = self._forward(
                        object_sliced,
                        _probe,
                        amplitudes,
                        self._exit_waves,
                        use_projection_scheme,
                        projection_a,
                        projection_b,
                        projection_c,
                    )

                    # adjoint operator
                    object_sliced, _probe = self._adjoint(
                        object_sliced,
                        _probe,
                        object_patches,
                        shifted_probes,
                        self._exit_waves,
                        use_projection_scheme=use_projection_scheme,
                        step_size=step_size,
                        normalization_min=normalization_min,
                        fix_probe=a0 < fix_probe_iter,
                    )

                    # position correction
                    if a0 >= fix_positions_iter:
                        positions_px[start:end] = self._position_correction(
                            object_sliced,
                            _probe,
                            overlap,
                            amplitudes,
                            self._positions_px,
                            self._positions_px_initial,
                            positions_step_size,
                            max_position_update_distance,
                            max_position_total_distance,
                        )

                    tilt_error += batch_error

                if not use_projection_scheme:
                    object_sliced -= object_sliced_old

                object_update = self._expand_sliced_object(
                    object_sliced, self._num_voxels
                )

                if collective_tilt_updates:
                    collective_object += self._rotate_zxy_volume(
                        object_update, rot_matrix.T
                    )
                else:
                    self._object += object_update

                old_rot_matrix = rot_matrix

                # Normalize Error
                tilt_error /= (
                    self._mean_diffraction_intensity[self._active_tilt_index]
                    * num_diffraction_patterns
                )
                error += tilt_error

                # constraints
                self._positions_px_all[start_tilt:end_tilt] = positions_px.copy()[
                    unshuffled_indices
                ]

                if not collective_tilt_updates:
                    (
                        self._object,
                        _probe,
                        self._positions_px_all[start_tilt:end_tilt],
                    ) = self._constraints(
                        self._object,
                        _probe,
                        self._positions_px_all[start_tilt:end_tilt],
                        fix_com=fix_com and a0 >= fix_probe_iter,
                        constrain_probe_amplitude=a0 < constrain_probe_amplitude_iter
                        and a0 >= fix_probe_iter,
                        constrain_probe_amplitude_relative_radius=constrain_probe_amplitude_relative_radius,
                        constrain_probe_amplitude_relative_width=constrain_probe_amplitude_relative_width,
                        constrain_probe_fourier_amplitude=a0
                        < constrain_probe_fourier_amplitude_iter
                        and a0 >= fix_probe_iter,
                        constrain_probe_fourier_amplitude_max_width_pixels=constrain_probe_fourier_amplitude_max_width_pixels,
                        constrain_probe_fourier_amplitude_constant_intensity=constrain_probe_fourier_amplitude_constant_intensity,
                        fit_probe_aberrations=a0 < fit_probe_aberrations_iter
                        and a0 >= fix_probe_iter,
                        fit_probe_aberrations_max_angular_order=fit_probe_aberrations_max_angular_order,
                        fit_probe_aberrations_max_radial_order=fit_probe_aberrations_max_radial_order,
                        fix_probe_aperture=a0 < fix_probe_aperture_iter,
                        initial_probe_aperture=_probe_initial_aperture,
                        fix_positions=a0 < fix_positions_iter,
                        global_affine_transformation=global_affine_transformation,
                        gaussian_filter=a0 < gaussian_filter_iter
                        and gaussian_filter_sigma is not None,
                        gaussian_filter_sigma=gaussian_filter_sigma,
                        butterworth_filter=a0 < butterworth_filter_iter
                        and (q_lowpass is not None or q_highpass is not None),
                        q_lowpass=q_lowpass,
                        q_highpass=q_highpass,
                        butterworth_order=butterworth_order,
                        object_positivity=object_positivity,
                        shrinkage_rad=shrinkage_rad,
                        object_mask=self._object_fov_mask_inverse
                        if fix_potential_baseline
                        and self._object_fov_mask_inverse.sum() > 0
                        else None,
                        tv_denoise=a0 < tv_denoise_iter
                        and tv_denoise_weights is not None,
                        tv_denoise_weights=tv_denoise_weights,
                        tv_denoise_inner_iter=tv_denoise_inner_iter,
                    )

            self._object = self._rotate_zxy_volume(self._object, old_rot_matrix.T)

            # Normalize Error Over Tilts
            error /= self._num_tilts

            if collective_tilt_updates:
                self._object += collective_object / self._num_tilts

                (
                    self._object,
                    _probe,
                    _,
                ) = self._constraints(
                    self._object,
                    _probe,
                    None,
                    fix_com=fix_com and a0 >= fix_probe_iter,
                    constrain_probe_amplitude=a0 < constrain_probe_amplitude_iter
                    and a0 >= fix_probe_iter,
                    constrain_probe_amplitude_relative_radius=constrain_probe_amplitude_relative_radius,
                    constrain_probe_amplitude_relative_width=constrain_probe_amplitude_relative_width,
                    constrain_probe_fourier_amplitude=a0
                    < constrain_probe_fourier_amplitude_iter
                    and a0 >= fix_probe_iter,
                    constrain_probe_fourier_amplitude_max_width_pixels=constrain_probe_fourier_amplitude_max_width_pixels,
                    constrain_probe_fourier_amplitude_constant_intensity=constrain_probe_fourier_amplitude_constant_intensity,
                    fit_probe_aberrations=a0 < fit_probe_aberrations_iter
                    and a0 >= fix_probe_iter,
                    fit_probe_aberrations_max_angular_order=fit_probe_aberrations_max_angular_order,
                    fit_probe_aberrations_max_radial_order=fit_probe_aberrations_max_radial_order,
                    fix_probe_aperture=a0 < fix_probe_aperture_iter,
                    initial_probe_aperture=_probe_initial_aperture,
                    fix_positions=True,
                    global_affine_transformation=global_affine_transformation,
                    gaussian_filter=a0 < gaussian_filter_iter
                    and gaussian_filter_sigma is not None,
                    gaussian_filter_sigma=gaussian_filter_sigma,
                    butterworth_filter=a0 < butterworth_filter_iter
                    and (q_lowpass is not None or q_highpass is not None),
                    q_lowpass=q_lowpass,
                    q_highpass=q_highpass,
                    butterworth_order=butterworth_order,
                    object_positivity=object_positivity,
                    shrinkage_rad=shrinkage_rad,
                    object_mask=self._object_fov_mask_inverse
                    if fix_potential_baseline
                    and self._object_fov_mask_inverse.sum() > 0
                    else None,
                    tv_denoise=a0 < tv_denoise_iter and tv_denoise_weights is not None,
                    tv_denoise_weights=tv_denoise_weights,
                    tv_denoise_inner_iter=tv_denoise_inner_iter,
                )

            self.error_iterations.append(error.item())

            if store_iterations:
                self.object_iterations.append(asnumpy(self._object.copy()))
                self.probe_iterations.append(self.probe_centered)

        # store result
        self.object = asnumpy(self._object)
        self.probe = self.probe_centered
        self.error = error.item()

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def _visualize_last_iteration(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        plot_fourier_probe: bool,
        remove_initial_probe_aberrations: bool,
        projection_angle_deg: float,
        projection_axes: Tuple[int, int],
        x_lims: Tuple[int, int],
        y_lims: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays last reconstructed object and probe iterations.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        asnumpy = self._asnumpy

        figsize = kwargs.pop("figsize", (8, 5))
        cmap = kwargs.pop("cmap", "magma")

        chroma_boost = kwargs.pop("chroma_boost", 1)

        asnumpy = self._asnumpy

        if projection_angle_deg is not None:
            rotated_3d_obj = self._rotate(
                self._object,
                projection_angle_deg,
                axes=projection_axes,
                reshape=False,
                order=2,
            )
            rotated_3d_obj = asnumpy(rotated_3d_obj)
        else:
            rotated_3d_obj = self.object

        rotated_object = self._crop_rotate_object_manually(
            rotated_3d_obj.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
        )
        rotated_shape = rotated_object.shape

        extent = [
            0,
            self.sampling[1] * rotated_shape[1],
            self.sampling[0] * rotated_shape[0],
            0,
        ]

        if plot_fourier_probe:
            probe_extent = [
                -self.angular_sampling[1] * self._region_of_interest_shape[1] / 2,
                self.angular_sampling[1] * self._region_of_interest_shape[1] / 2,
                self.angular_sampling[0] * self._region_of_interest_shape[0] / 2,
                -self.angular_sampling[0] * self._region_of_interest_shape[0] / 2,
            ]
        elif plot_probe:
            probe_extent = [
                0,
                self.sampling[1] * self._region_of_interest_shape[1],
                self.sampling[0] * self._region_of_interest_shape[0],
                0,
            ]

        if plot_convergence:
            if plot_probe or plot_fourier_probe:
                spec = GridSpec(
                    ncols=2,
                    nrows=2,
                    height_ratios=[4, 1],
                    hspace=0.15,
                    width_ratios=[
                        (extent[1] / extent[2]) / (probe_extent[1] / probe_extent[2]),
                        1,
                    ],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            if plot_probe or plot_fourier_probe:
                spec = GridSpec(
                    ncols=2,
                    nrows=1,
                    width_ratios=[
                        (extent[1] / extent[2]) / (probe_extent[1] / probe_extent[2]),
                        1,
                    ],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=1, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)

        if plot_probe or plot_fourier_probe:
            # Object
            ax = fig.add_subplot(spec[0, 0])
            im = ax.imshow(
                rotated_object,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed object projection")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Probe
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)

            ax = fig.add_subplot(spec[0, 1])
            if plot_fourier_probe:
                if remove_initial_probe_aberrations:
                    probe_array = self.probe_fourier_residual
                else:
                    probe_array = self.probe_fourier

                probe_array = Complex2RGB(
                    probe_array,
                    chroma_boost=chroma_boost,
                )

                ax.set_title("Reconstructed Fourier probe")
                ax.set_ylabel("kx [mrad]")
                ax.set_xlabel("ky [mrad]")
            else:
                probe_array = Complex2RGB(
                    self.probe,
                    power=2,
                    chroma_boost=chroma_boost,
                )
                ax.set_title("Reconstructed probe intensity")
                ax.set_ylabel("x [A]")
                ax.set_xlabel("y [A]")

            im = ax.imshow(
                probe_array,
                extent=probe_extent,
                **kwargs,
            )

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                add_colorbar_arg(
                    ax_cb,
                    chroma_boost=chroma_boost,
                )
        else:
            ax = fig.add_subplot(spec[0])
            im = ax.imshow(
                rotated_object,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed object projection")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "error_iterations"):
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            errors = np.array(self.error_iterations)
            if plot_probe:
                ax = fig.add_subplot(spec[1, :])
            else:
                ax = fig.add_subplot(spec[1])
            ax.semilogy(np.arange(errors.shape[0]), errors, **kwargs)
            ax.set_ylabel("NMSE")
            ax.set_xlabel("Iteration number")
            ax.yaxis.tick_right()

        fig.suptitle(f"Normalized mean squared error: {self.error:.3e}")
        spec.tight_layout(fig)

    def _visualize_all_iterations(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        plot_fourier_probe: bool,
        remove_initial_probe_aberrations: bool,
        iterations_grid: Tuple[int, int],
        projection_angle_deg: float,
        projection_axes: Tuple[int, int],
        x_lims: Tuple[int, int],
        y_lims: Tuple[int, int],
        **kwargs,
    ):
        """
        Displays all reconstructed object and probe iterations.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        asnumpy = self._asnumpy

        if not hasattr(self, "object_iterations"):
            raise ValueError(
                (
                    "Object and probe iterations were not saved during reconstruction. "
                    "Please re-run using store_iterations=True."
                )
            )

        if iterations_grid == "auto":
            num_iter = len(self.error_iterations)

            if num_iter == 1:
                return self._visualize_last_iteration(
                    fig=fig,
                    plot_convergence=plot_convergence,
                    plot_probe=plot_probe,
                    plot_fourier_probe=plot_fourier_probe,
                    cbar=cbar,
                    projection_angle_deg=projection_angle_deg,
                    projection_axes=projection_axes,
                    x_lims=x_lims,
                    y_lims=y_lims,
                    **kwargs,
                )
            elif plot_probe or plot_fourier_probe:
                iterations_grid = (2, 4) if num_iter > 4 else (2, num_iter)
            else:
                iterations_grid = (2, 4) if num_iter > 8 else (2, num_iter // 2)
        else:
            if (plot_probe or plot_fourier_probe) and iterations_grid[0] != 2:
                raise ValueError()

        auto_figsize = (
            (3 * iterations_grid[1], 3 * iterations_grid[0] + 1)
            if plot_convergence
            else (3 * iterations_grid[1], 3 * iterations_grid[0])
        )
        figsize = kwargs.pop("figsize", auto_figsize)
        cmap = kwargs.pop("cmap", "magma")

        chroma_boost = kwargs.pop("chroma_boost", 1)

        errors = np.array(self.error_iterations)

        if projection_angle_deg is not None:
            objects = [
                self._crop_rotate_object_manually(
                    rotate_np(
                        obj,
                        projection_angle_deg,
                        axes=projection_axes,
                        reshape=False,
                        order=2,
                    ).sum(0),
                    angle=None,
                    x_lims=x_lims,
                    y_lims=y_lims,
                )
                for obj in self.object_iterations
            ]
        else:
            objects = [
                self._crop_rotate_object_manually(
                    obj.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
                )
                for obj in self.object_iterations
            ]

        if plot_probe or plot_fourier_probe:
            total_grids = (np.prod(iterations_grid) / 2).astype("int")
            probes = self.probe_iterations
        else:
            total_grids = np.prod(iterations_grid)
        max_iter = len(objects) - 1
        grid_range = range(0, max_iter + 1, max_iter // (total_grids - 1))

        extent = [
            0,
            self.sampling[1] * objects[0].shape[1],
            self.sampling[0] * objects[0].shape[0],
            0,
        ]

        if plot_fourier_probe:
            probe_extent = [
                -self.angular_sampling[1] * self._region_of_interest_shape[1] / 2,
                self.angular_sampling[1] * self._region_of_interest_shape[1] / 2,
                self.angular_sampling[0] * self._region_of_interest_shape[0] / 2,
                -self.angular_sampling[0] * self._region_of_interest_shape[0] / 2,
            ]
        elif plot_probe:
            probe_extent = [
                0,
                self.sampling[1] * self._region_of_interest_shape[1],
                self.sampling[0] * self._region_of_interest_shape[0],
                0,
            ]

        if plot_convergence:
            if plot_probe or plot_fourier_probe:
                spec = GridSpec(ncols=1, nrows=3, height_ratios=[4, 4, 1], hspace=0)
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0)
        else:
            if plot_probe or plot_fourier_probe:
                spec = GridSpec(ncols=1, nrows=2)
            else:
                spec = GridSpec(ncols=1, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)

        grid = ImageGrid(
            fig,
            spec[0],
            nrows_ncols=(1, iterations_grid[1]) if plot_probe else iterations_grid,
            axes_pad=(0.75, 0.5) if cbar else 0.5,
            cbar_mode="each" if cbar else None,
            cbar_pad="2.5%" if cbar else None,
        )

        for n, ax in enumerate(grid):
            im = ax.imshow(
                objects[grid_range[n]],
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_title(f"Iter: {grid_range[n]} Object")

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            if cbar:
                grid.cbar_axes[n].colorbar(im)

        if plot_probe or plot_fourier_probe:
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            grid = ImageGrid(
                fig,
                spec[1],
                nrows_ncols=(1, iterations_grid[1]),
                axes_pad=(0.75, 0.5) if cbar else 0.5,
                cbar_mode="each" if cbar else None,
                cbar_pad="2.5%" if cbar else None,
            )

            for n, ax in enumerate(grid):
                if plot_fourier_probe:
                    probe_array = asnumpy(
                        self._return_fourier_probe_from_centered_probe(
                            probes[grid_range[n]],
                            remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                        )
                    )

                    probe_array = Complex2RGB(probe_array, chroma_boost=chroma_boost)

                    ax.set_title(f"Iter: {grid_range[n]} Fourier probe")
                    ax.set_ylabel("kx [mrad]")
                    ax.set_xlabel("ky [mrad]")
                else:
                    probe_array = Complex2RGB(
                        probes[grid_range[n]],
                        power=2,
                        chroma_boost=chroma_boost,
                    )
                    ax.set_title(f"Iter: {grid_range[n]} probe intensity")
                    ax.set_ylabel("x [A]")
                    ax.set_xlabel("y [A]")

                im = ax.imshow(
                    probe_array,
                    extent=probe_extent,
                )

                if cbar:
                    add_colorbar_arg(
                        grid.cbar_axes[n],
                        chroma_boost=chroma_boost,
                    )

        if plot_convergence:
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            if plot_probe:
                ax2 = fig.add_subplot(spec[2])
            else:
                ax2 = fig.add_subplot(spec[1])
            ax2.semilogy(np.arange(errors.shape[0]), errors, **kwargs)
            ax2.set_ylabel("NMSE")
            ax2.set_xlabel("Iteration number")
            ax2.yaxis.tick_right()

        spec.tight_layout(fig)

    def visualize(
        self,
        fig=None,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        plot_probe: bool = True,
        plot_fourier_probe: bool = False,
        remove_initial_probe_aberrations: bool = False,
        cbar: bool = True,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims=(None, None),
        y_lims=(None, None),
        **kwargs,
    ):
        """
        Displays reconstructed object and probe.

        Parameters
        --------
        fig: Figure
            Matplotlib figure to place Gridspec in
        plot_convergence: bool, optional
            If true, the normalized mean squared error (NMSE) plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices

        Returns
        --------
        self: OverlapTomographicReconstruction
            Self to accommodate chaining
        """

        if iterations_grid is None:
            self._visualize_last_iteration(
                fig=fig,
                plot_convergence=plot_convergence,
                plot_probe=plot_probe,
                plot_fourier_probe=plot_fourier_probe,
                remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                cbar=cbar,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
                **kwargs,
            )
        else:
            self._visualize_all_iterations(
                fig=fig,
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                plot_probe=plot_probe,
                plot_fourier_probe=plot_fourier_probe,
                remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                cbar=cbar,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
                **kwargs,
            )

        return self

    @property
    def positions(self):
        """Probe positions [A]"""

        if self.angular_sampling is None:
            return None

        asnumpy = self._asnumpy
        positions_all = []
        for tilt_index in range(self._num_tilts):
            positions = self._positions_px_all[
                self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                    tilt_index + 1
                ]
            ].copy()
            positions[:, 0] *= self.sampling[0]
            positions[:, 1] *= self.sampling[1]
            positions_all.append(asnumpy(positions))

        return np.asarray(positions_all)
