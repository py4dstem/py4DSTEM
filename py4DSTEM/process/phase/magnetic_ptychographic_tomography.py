"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely magnetic ptychographic tomography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np

from emdfile import Custom, tqdmnd
from py4DSTEM import DataCube
from py4DSTEM.process.phase.phase_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.ptychographic_constraints import (
    Object2p5DConstraintsMixin,
    Object3DConstraintsMixin,
    ObjectNDConstraintsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
)
from py4DSTEM.process.phase.ptychographic_methods import (
    MultipleMeasurementsMethodsMixin,
    Object2p5DMethodsMixin,
    Object2p5DProbeMethodsMixin,
    Object3DMethodsMixin,
    ObjectNDMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeMethodsMixin,
)
from py4DSTEM.process.phase.ptychographic_visualizations import VisualizationsMixin
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
    project_vector_field_divergence_periodic_3D,
)

warnings.simplefilter(action="always", category=UserWarning)


class MagneticPtychographicTomography(
    VisualizationsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
    Object3DConstraintsMixin,
    Object2p5DConstraintsMixin,
    ObjectNDConstraintsMixin,
    MultipleMeasurementsMethodsMixin,
    Object2p5DProbeMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeMethodsMixin,
    Object3DMethodsMixin,
    Object2p5DMethodsMixin,
    ObjectNDMethodsMixin,
    PtychographicReconstruction,
):
    """
    Magnetic Ptychographic Tomography Reconstruction Class.

    List of diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions              : (Sx,Sy)
    Reconstructed object dimensions             : (Px,Py,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our probe
    and (Px,Py,Py) is the padded-object electrostatic potential volume,
    where x-axis is the tilt.

    Parameters
    ----------
    datacube: List of DataCubes
        Input list of 4D diffraction pattern intensities for different tilts
    energy: float
        The electron energy of the wave functions in eV
    num_slices: int
        Number of super-slices to use in the forward model
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
        Boolean real space mask to select positions in datacube to skip for reconstruction
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
        name: str = "magnetic-ptychographic-tomography_reconstruction",
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
        self._num_measurements = num_tilts

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
        progress_bar: bool = True,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        object_fov_mask: np.ndarray = None,
        crop_patterns: bool = False,
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
            if True, crop patterns to avoid wrap around of patterns when centering

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
                    self._positions_mask, (self._num_measurements, 1, 1)
                )

            num_probes_per_measurement = np.insert(
                self._positions_mask.sum(axis=(-2, -1)), 0, 0
            )

        else:
            self._positions_mask = [None] * self._num_measurements
            num_probes_per_measurement = [0] + [dc.R_N for dc in self._datacube]
            num_probes_per_measurement = np.array(num_probes_per_measurement)

        # prepopulate relevant arrays
        self._mean_diffraction_intensity = []
        self._num_diffraction_patterns = num_probes_per_measurement.sum()
        self._cum_probes_per_measurement = np.cumsum(num_probes_per_measurement)
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
            force_com_shifts = [None] * self._num_measurements

        self._rotation_best_rad = np.deg2rad(diffraction_patterns_rotate_degrees)
        self._rotation_best_transpose = diffraction_patterns_transpose

        # loop over DPs for preprocessing
        for index in tqdmnd(
            self._num_measurements,
            desc="Preprocessing data",
            unit="tilt",
            disable=not progress_bar,
        ):
            # preprocess datacube, vacuum and masks only for first tilt
            if index == 0:
                (
                    self._datacube[index],
                    self._vacuum_probe_intensity,
                    self._dp_mask,
                    force_com_shifts[index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    probe_roi_shape=self._probe_roi_shape,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    dp_mask=self._dp_mask,
                    com_shifts=force_com_shifts[index],
                )

            else:
                (
                    self._datacube[index],
                    _,
                    _,
                    force_com_shifts[index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    probe_roi_shape=self._probe_roi_shape,
                    vacuum_probe_intensity=None,
                    dp_mask=None,
                    com_shifts=force_com_shifts[index],
                )

            # calibrations
            intensities = self._extract_intensities_and_calibrations_from_datacube(
                self._datacube[index],
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
                com_shifts=force_com_shifts[index],
            )

            # corner-center amplitudes
            idx_start = self._cum_probes_per_measurement[index]
            idx_end = self._cum_probes_per_measurement[index + 1]
            (
                self._amplitudes[idx_start:idx_end],
                mean_diffraction_intensity_temp,
                self._crop_mask,
            ) = self._normalize_diffraction_intensities(
                intensities,
                com_fitted_x,
                com_fitted_y,
                self._positions_mask[index],
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
                self._scan_positions[index],
                self._positions_mask[index],
                self._object_padding_px,
            )

        # handle semiangle specified in pixels
        if self._semiangle_cutoff_pixels:
            self._semiangle_cutoff = (
                self._semiangle_cutoff_pixels * self._angular_sampling[0]
            )

        # initialize object
        obj = self._initialize_object(
            self._object,
            self._positions_px_all,
            self._object_type,
            main_tilt_axis=None,
        )

        if self._object is None:
            self._object = xp.full((4,) + obj.shape, obj)
        else:
            self._object = obj

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[1]

        # center probe positions
        self._positions_px_all = xp.asarray(self._positions_px_all, dtype=xp.float32)

        for index in range(self._num_measurements):
            idx_start = self._cum_probes_per_measurement[index]
            idx_end = self._cum_probes_per_measurement[index + 1]
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

        for index in range(self._num_measurements):
            _probe, self._semiangle_cutoff = self._initialize_probe(
                self._probe_init[index] if list_Q else self._probe_init,
                self._vacuum_probe_intensity,
                self._mean_diffraction_intensity[index],
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
        thickness_h = self._object_shape[1] * self.sampling[1]
        thickness_v = self._object_shape[0] * self.sampling[0]
        thickness = max(thickness_h, thickness_v)

        self._slice_thicknesses = np.tile(
            thickness / self._num_slices, self._num_slices - 1
        )
        self._propagator_arrays = self._precompute_propagator_arrays(
            self._region_of_interest_shape,
            self.sampling,
            self._energy,
            self._slice_thicknesses,
        )

        # overlaps
        if object_fov_mask is None:
            probe_overlap_3D = xp.zeros_like(self._object[0])
            old_rot_matrix = np.eye(3)  # identity

            for index in range(self._num_measurements):
                idx_start = self._cum_probes_per_measurement[index]
                idx_end = self._cum_probes_per_measurement[index + 1]
                rot_matrix = self._tilt_orientation_matrices[index]

                probe_overlap_3D = self._rotate_zxy_volume(
                    probe_overlap_3D,
                    rot_matrix @ old_rot_matrix.T,
                )

                self._positions_px = self._positions_px_all[idx_start:idx_end]
                self._positions_px_fractional = self._positions_px - xp.round(
                    self._positions_px
                )
                shifted_probes = fft_shift(
                    self._probes_all[index], self._positions_px_fractional, xp
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

            self._positions_px = self._positions_px_all[
                : self._cum_probes_per_measurement[1]
            ]
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
                self.probe_centered[0],
                power=power,
                chroma_boost=chroma_boost,
            )

            # propagated
            propagated_probe = self._probes_all[0].copy()

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

    def _divergence_free_constraint(self, vector_field):
        """
        Leray projection operator

        Parameters
        --------
        vector_field: np.ndarray
            Current object vector as Ax, Ay, Az

        Returns
        --------
        projected_vector_field: np.ndarray
            Divergence-less object vector as Ax, Ay, Az
        """
        xp = self._xp

        vector_field = project_vector_field_divergence_periodic_3D(vector_field, xp=xp)

        return vector_field

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
        gaussian_filter_sigma_e,
        gaussian_filter_sigma_m,
        butterworth_filter,
        q_lowpass_e,
        q_lowpass_m,
        q_highpass_e,
        q_highpass_m,
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
        Calls _threshold_object_constraint() and _probe_center_of_mass_constraint()

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
        gaussian_filter_sigma_e: float
            Standard deviation of gaussian kernel for electrostatic object in A
        gaussian_filter_sigma_m: float
            Standard deviation of gaussian kernel for magnetic object in A
        butterworth_filter: bool
            If True, applies high-pass butteworth filter
        q_lowpass_e: float
            Cut-off frequency in A^-1 for low-pass filtering electrostatic object
        q_lowpass_m: float
            Cut-off frequency in A^-1 for low-pass filtering magnetic object
        q_highpass_e: float
            Cut-off frequency in A^-1 for high-pass filtering electrostatic object
        q_highpass_m: float
            Cut-off frequency in A^-1 for high-pass filtering magnetic object
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
            current_object[0] = self._object_gaussian_constraint(
                current_object[0], gaussian_filter_sigma_e, pure_phase_object=False
            )
            current_object[1] = self._object_gaussian_constraint(
                current_object[1], gaussian_filter_sigma_m, pure_phase_object=False
            )
            current_object[2] = self._object_gaussian_constraint(
                current_object[2], gaussian_filter_sigma_m, pure_phase_object=False
            )
            current_object[3] = self._object_gaussian_constraint(
                current_object[3], gaussian_filter_sigma_m, pure_phase_object=False
            )

        if butterworth_filter:
            current_object[0] = self._object_butterworth_constraint(
                current_object[0],
                q_lowpass_e,
                q_highpass_e,
                butterworth_order,
            )
            current_object[1] = self._object_butterworth_constraint(
                current_object[1],
                q_lowpass_m,
                q_highpass_m,
                butterworth_order,
            )
            current_object[2] = self._object_butterworth_constraint(
                current_object[2],
                q_lowpass_m,
                q_highpass_m,
                butterworth_order,
            )
            current_object[3] = self._object_butterworth_constraint(
                current_object[3],
                q_lowpass_m,
                q_highpass_m,
                butterworth_order,
            )

        elif tv_denoise:
            current_object[0] = self._object_denoise_tv_pylops(
                current_object[0],
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

            current_object[1] = self._object_denoise_tv_pylops(
                current_object[1],
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

            current_object[2] = self._object_denoise_tv_pylops(
                current_object[2],
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

            current_object[3] = self._object_denoise_tv_pylops(
                current_object[3],
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object[0] = self._object_shrinkage_constraint(
                current_object[0],
                shrinkage_rad,
                object_mask,
            )

        if object_positivity:
            current_object[0] = self._object_positivity_constraint(current_object[0])

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
        max_iter: int = 64,
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
        constrain_position_distance: float = None,
        global_affine_transformation: bool = True,
        gaussian_filter_sigma_e: float = None,
        gaussian_filter_sigma_m: float = None,
        gaussian_filter_iter: int = np.inf,
        fit_probe_aberrations_iter: int = 0,
        fit_probe_aberrations_max_angular_order: int = 4,
        fit_probe_aberrations_max_radial_order: int = 4,
        butterworth_filter_iter: int = np.inf,
        q_lowpass_e: float = None,
        q_lowpass_m: float = None,
        q_highpass_e: float = None,
        q_highpass_m: float = None,
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
        constrain_position_distance: float, optional
            Distance to constrain position correction within original
            field of view in A
        global_affine_transformation: bool, optional
            If True, positions are assumed to be a global affine transform from initial scan
        gaussian_filter_sigma_e: float
            Standard deviation of gaussian kernel for electrostatic object in A
        gaussian_filter_sigma_m: float
            Standard deviation of gaussian kernel for magnetic object in A
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
        self: OverlapMagneticTomographicReconstruction
            Self to accommodate chaining
        """
        asnumpy = self._asnumpy
        xp = self._xp

        # Reconstruction method

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

        if self._verbose:
            if max_batch_size is not None:
                if use_projection_scheme:
                    raise ValueError(
                        (
                            "Stochastic object/probe updating is inconsistent with 'DM_AP', 'RAAR', 'RRR', and 'SUPERFLIP'. "
                            "Use reconstruction_method='GD' or set max_batch_size=None."
                        )
                    )
                else:
                    print(
                        (
                            f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                            f"with normalization_min: {normalization_min} and step _size: {step_size}, "
                            f"in batches of max {max_batch_size} measurements."
                        )
                    )
            else:
                if reconstruction_parameter is not None:
                    if np.array(reconstruction_parameter).shape == (3,):
                        print(
                            (
                                f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and (a,b,c): {reconstruction_parameter}."
                            )
                        )
                    else:
                        print(
                            (
                                f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and α: {reconstruction_parameter}."
                            )
                        )
                else:
                    if step_size is not None:
                        print(
                            (
                                f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min}."
                            )
                        )
                    else:
                        print(
                            (
                                f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and step _size: {step_size}."
                            )
                        )

        # Position Correction + Collective Updates not yet implemented
        if fix_positions_iter < max_iter:
            raise NotImplementedError(
                "Position correction is currently incompatible with collective updates."
            )

        # Batching

        if max_batch_size is not None:
            xp.random.seed(seed_random)

        # initialization
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []

        if reset:
            self._object = self._object_initial.copy()
            self.error_iterations = []
            self._probe = self._probe_initial.copy()
            self._positions_px_all = self._positions_px_initial_all.copy()
            if hasattr(self, "_tf"):
                del self._tf

            if use_projection_scheme:
                self._exit_waves = [None] * self._num_tilts
            else:
                self._exit_waves = None
        elif reset is None:
            if hasattr(self, "error"):
                warnings.warn(
                    (
                        "Continuing reconstruction from previous result. "
                        "Use reset=True for a fresh start."
                    ),
                    UserWarning,
                )
            else:
                self.error_iterations = []
                if use_projection_scheme:
                    self._exit_waves = [None] * self._num_tilts
                else:
                    self._exit_waves = None

        if gaussian_filter_sigma_m is None:
            gaussian_filter_sigma_m = gaussian_filter_sigma_e

        if q_lowpass_m is None:
            q_lowpass_m = q_lowpass_e

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

            for tilt_index in tilt_indices:
                tilt_error = 0.0
                self._active_tilt_index = tilt_index

                alpha_deg, beta_deg = self._tilt_angles_deg[self._active_tilt_index]
                alpha, beta = np.deg2rad([alpha_deg, beta_deg])

                # V
                self._object[0] = self._euler_angle_rotate_volume(
                    self._object[0],
                    alpha_deg,
                    beta_deg,
                )

                # Az
                self._object[1] = self._euler_angle_rotate_volume(
                    self._object[1],
                    alpha_deg,
                    beta_deg,
                )

                # Ax
                self._object[2] = self._euler_angle_rotate_volume(
                    self._object[2],
                    alpha_deg,
                    beta_deg,
                )

                # Ay
                self._object[3] = self._euler_angle_rotate_volume(
                    self._object[3],
                    alpha_deg,
                    beta_deg,
                )

                object_A = self._object[1] * np.cos(beta) + np.sin(beta) * (
                    self._object[3] * np.cos(alpha) - self._object[2] * np.sin(alpha)
                )

                object_sliced_V = self._project_sliced_object(
                    self._object[0], self._num_slices
                )

                object_sliced_A = self._project_sliced_object(
                    object_A, self._num_slices
                )

                if not use_projection_scheme:
                    object_sliced_old_V = object_sliced_V.copy()
                    object_sliced_old_A = object_sliced_A.copy()

                start_tilt = self._cum_probes_per_tilt[self._active_tilt_index]
                end_tilt = self._cum_probes_per_tilt[self._active_tilt_index + 1]

                num_diffraction_patterns = end_tilt - start_tilt
                shuffled_indices = np.arange(num_diffraction_patterns)
                unshuffled_indices = np.zeros_like(shuffled_indices)

                if max_batch_size is None:
                    current_max_batch_size = num_diffraction_patterns
                else:
                    current_max_batch_size = max_batch_size

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
                    num_diffraction_patterns, max_batch=current_max_batch_size
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
                        propagated_probes,
                        object_patches,
                        transmitted_probes,
                        self._exit_waves,
                        batch_error,
                    ) = self._forward(
                        object_sliced_V,
                        object_sliced_A,
                        self._probe,
                        amplitudes,
                        self._exit_waves,
                        use_projection_scheme,
                        projection_a,
                        projection_b,
                        projection_c,
                    )

                    # adjoint operator
                    object_sliced_V, object_sliced_A, self._probe = self._adjoint(
                        object_sliced_V,
                        object_sliced_A,
                        self._probe,
                        object_patches,
                        propagated_probes,
                        self._exit_waves,
                        use_projection_scheme=use_projection_scheme,
                        step_size=step_size,
                        normalization_min=normalization_min,
                        fix_probe=a0 < fix_probe_iter,
                    )

                    # position correction
                    if a0 >= fix_positions_iter:
                        positions_px[start:end] = self._position_correction(
                            object_sliced_V,
                            self._probe,
                            transmitted_probes,
                            amplitudes,
                            self._positions_px,
                            positions_step_size,
                            constrain_position_distance,
                        )

                    tilt_error += batch_error

                if not use_projection_scheme:
                    object_sliced_V -= object_sliced_old_V
                    object_sliced_A -= object_sliced_old_A

                object_update_V = self._expand_sliced_object(
                    object_sliced_V, self._num_voxels
                )
                object_update_A = self._expand_sliced_object(
                    object_sliced_A, self._num_voxels
                )

                if collective_tilt_updates:
                    collective_object[0] += self._euler_angle_rotate_volume(
                        object_update_V,
                        alpha_deg,
                        -beta_deg,
                    )
                    collective_object[1] += self._euler_angle_rotate_volume(
                        object_update_A * np.cos(beta),
                        alpha_deg,
                        -beta_deg,
                    )
                    collective_object[2] -= self._euler_angle_rotate_volume(
                        object_update_A * np.sin(alpha) * np.sin(beta),
                        alpha_deg,
                        -beta_deg,
                    )
                    collective_object[3] += self._euler_angle_rotate_volume(
                        object_update_A * np.cos(alpha) * np.sin(beta),
                        alpha_deg,
                        -beta_deg,
                    )
                else:
                    self._object[0] += object_update_V
                    self._object[1] += object_update_A * np.cos(beta)
                    self._object[2] -= object_update_A * np.sin(alpha) * np.sin(beta)
                    self._object[3] += object_update_A * np.cos(alpha) * np.sin(beta)

                self._object[0] = self._euler_angle_rotate_volume(
                    self._object[0],
                    alpha_deg,
                    -beta_deg,
                )

                self._object[1] = self._euler_angle_rotate_volume(
                    self._object[1],
                    alpha_deg,
                    -beta_deg,
                )

                self._object[2] = self._euler_angle_rotate_volume(
                    self._object[2],
                    alpha_deg,
                    -beta_deg,
                )

                self._object[3] = self._euler_angle_rotate_volume(
                    self._object[3],
                    alpha_deg,
                    -beta_deg,
                )

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
                        self._probe,
                        self._positions_px_all[start_tilt:end_tilt],
                    ) = self._constraints(
                        self._object,
                        self._probe,
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
                        initial_probe_aperture=self._probe_initial_aperture,
                        fix_positions=a0 < fix_positions_iter,
                        global_affine_transformation=global_affine_transformation,
                        gaussian_filter=a0 < gaussian_filter_iter
                        and gaussian_filter_sigma_m is not None,
                        gaussian_filter_sigma_e=gaussian_filter_sigma_e,
                        gaussian_filter_sigma_m=gaussian_filter_sigma_m,
                        butterworth_filter=a0 < butterworth_filter_iter
                        and (q_lowpass_m is not None or q_highpass_m is not None),
                        q_lowpass_e=q_lowpass_e,
                        q_lowpass_m=q_lowpass_m,
                        q_highpass_e=q_highpass_e,
                        q_highpass_m=q_highpass_m,
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

            # Normalize Error Over Tilts
            error /= self._num_tilts

            self._object[1:] = self._divergence_free_constraint(self._object[1:])

            if collective_tilt_updates:
                self._object += collective_object / self._num_tilts

                (
                    self._object,
                    self._probe,
                    _,
                ) = self._constraints(
                    self._object,
                    self._probe,
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
                    initial_probe_aperture=self._probe_initial_aperture,
                    fix_positions=True,
                    global_affine_transformation=global_affine_transformation,
                    gaussian_filter=a0 < gaussian_filter_iter
                    and gaussian_filter_sigma_m is not None,
                    gaussian_filter_sigma_e=gaussian_filter_sigma_e,
                    gaussian_filter_sigma_m=gaussian_filter_sigma_m,
                    butterworth_filter=a0 < butterworth_filter_iter
                    and (q_lowpass_m is not None or q_highpass_m is not None),
                    q_lowpass_e=q_lowpass_e,
                    q_lowpass_m=q_lowpass_m,
                    q_highpass_e=q_highpass_e,
                    q_highpass_m=q_highpass_m,
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
