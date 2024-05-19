"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely magnetic ptychographic tomography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.visualize.vis_special import (
    Complex2RGB,
    add_colorbar_arg,
    return_scaled_histogram_ordering,
)

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
    copy_to_device,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
    project_vector_field_divergence_periodic_3D,
)


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
    positions_offset_ang: list of np.ndarray, optional
        Offset of positions in A
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    object_type: str, optional
        The object can be reconstructed as a real potential ('potential') or a complex
        object ('complex')
    positions_mask: np.ndarray, optional
        Boolean real space mask to select positions in datacube to skip for reconstruction
    name: str, optional
        Class name
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    storage: str, optional
        Device non-frequent arrays will be stored on. Must be 'cpu' or 'gpu'
    clear_fft_cache: bool, optional
        If True, and device = 'gpu', clears the cached fft plan at the end of function calls
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = (
        "_num_slices",
        "_tilt_orientation_matrices",
        "_num_measurements",
    )

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
        positions_offset_ang: Sequence[np.ndarray] = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = None,
        clear_fft_cache: bool = True,
        name: str = "magnetic-ptychographic-tomography_reconstruction",
        **kwargs,
    ):
        Custom.__init__(self, name=name)

        if storage is None:
            storage = device

        self.set_device(device, clear_fft_cache)
        self.set_storage(storage)

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
        self._positions_offset_ang = positions_offset_ang
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._semiangle_cutoff_pixels = semiangle_cutoff_pixels
        self._rolloff = rolloff
        self._object_type = object_type
        self._object_padding_px = object_padding_px
        self._positions_mask = positions_mask
        self._verbose = verbose
        self._preprocessed = False

        # Class-specific Metadata
        self._num_slices = num_slices
        self._tilt_orientation_matrices = tuple(tilt_orientation_matrices)
        self._num_measurements = num_tilts

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "bilinear",
        padded_diffraction_intensities_shape: Tuple[int, int] = None,
        region_of_interest_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_probe_overlaps: bool = True,
        rotation_real_space_degrees: float = None,
        diffraction_patterns_rotate_degrees: float = None,
        diffraction_patterns_transpose: bool = None,
        force_com_shifts: Sequence[float] = None,
        force_com_measured: Sequence[np.ndarray] = None,
        vectorized_com_calculation: bool = True,
        progress_bar: bool = True,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        object_fov_mask: np.ndarray = True,
        crop_patterns: bool = False,
        store_initial_arrays: bool = True,
        device: str = None,
        clear_fft_cache: bool = None,
        max_batch_size: int = None,
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
        padded_diffraction_intensities_shape: (int,int), optional
            Padded diffraction intensities shape.
            If None, no padding is performed
        region_of_interest_shape: (int,int), optional
            If not None, explicitly sets region_of_interest_shape and resamples exit_waves
            at the diffraction plane to allow comparison with experimental data
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
        force_com_measured: tuple of ndarrays (CoMx measured, CoMy measured)
            Force CoM measured shifts
        vectorized_com_calculation: bool, optional
            If True (default), the memory-intensive CoM calculation is vectorized
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
        store_initial_arrays: bool
            If True, preprocesed object and probe arrays are stored allowing reset=True in reconstruct.
        device: str, optional
            if not none, overwrites self._device to set device preprocess will be perfomed on.
        clear_fft_cache: bool, optional
            if true, and device = 'gpu', clears the cached fft plan at the end of function calls
        max_batch_size: int, optional
            Max number of probes to use at once in computing probe overlaps

        Returns
        --------
        self: OverlapTomographicReconstruction
            Self to accommodate chaining
        """
        # handle device/storage
        self.set_device(device, clear_fft_cache)

        xp = self._xp
        device = self._device
        xp_storage = self._xp_storage
        storage = self._storage
        asnumpy = self._asnumpy

        # set additional metadata
        self._diffraction_intensities_shape = diffraction_intensities_shape
        self._reshaping_method = reshaping_method
        self._padded_diffraction_intensities_shape = (
            padded_diffraction_intensities_shape
        )
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
        if padded_diffraction_intensities_shape is not None:
            roi_shape = tuple(
                max(q, s)
                for q, s in zip(roi_shape, padded_diffraction_intensities_shape)
            )

        self._amplitudes = xp_storage.empty(
            (self._num_diffraction_patterns,) + roi_shape
        )

        self._amplitudes_shape = np.array(self._amplitudes.shape[-2:])
        if region_of_interest_shape is not None:
            self._resample_exit_waves = True
            self._region_of_interest_shape = np.array(region_of_interest_shape)
        else:
            self._resample_exit_waves = False
            self._region_of_interest_shape = np.array(self._amplitudes.shape[-2:])

        # TO-DO: generalize this
        if force_com_shifts is None:
            force_com_shifts = [None] * self._num_measurements

        if force_com_measured is None:
            force_com_measured = [None] * self._num_measurements

        if self._positions_offset_ang is None:
            self._positions_offset_ang = [None] * self._num_measurements

        self._rotation_best_rad = np.deg2rad(diffraction_patterns_rotate_degrees)
        self._rotation_best_transpose = diffraction_patterns_transpose

        if progress_bar:
            # turn off verbosity to play nice with tqdm
            verbose = self._verbose
            self._verbose = False

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
                    force_com_measured[index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    padded_diffraction_intensities_shape=self._padded_diffraction_intensities_shape,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    dp_mask=self._dp_mask,
                    com_shifts=force_com_shifts[index],
                    com_measured=force_com_measured[index],
                )

            else:
                (
                    self._datacube[index],
                    _,
                    _,
                    force_com_shifts[index],
                    force_com_measured[index],
                ) = self._preprocess_datacube_and_vacuum_probe(
                    self._datacube[index],
                    diffraction_intensities_shape=self._diffraction_intensities_shape,
                    reshaping_method=self._reshaping_method,
                    padded_diffraction_intensities_shape=self._padded_diffraction_intensities_shape,
                    vacuum_probe_intensity=None,
                    dp_mask=None,
                    com_shifts=force_com_shifts[index],
                    com_measured=force_com_measured[index],
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
                vectorized_calculation=vectorized_com_calculation,
                com_measured=force_com_measured[index],
            )

            # corner-center amplitudes
            idx_start = self._cum_probes_per_measurement[index]
            idx_end = self._cum_probes_per_measurement[index + 1]
            (
                amplitudes,
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

            # explicitly transfer arrays to storage
            self._amplitudes[idx_start:idx_end] = copy_to_device(amplitudes, storage)

            del (
                intensities,
                amplitudes,
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
                self._positions_offset_ang[index],
            )

        if progress_bar:
            # reset verbosity
            self._verbose = verbose

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

        if store_initial_arrays:
            self._object_initial = self._object.copy()
            self._object_type_initial = self._object_type

        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[1]

        # center probe positions
        self._positions_px_all = xp_storage.asarray(
            self._positions_px_all, dtype=xp_storage.float32
        )

        for index in range(self._num_measurements):
            idx_start = self._cum_probes_per_measurement[index]
            idx_end = self._cum_probes_per_measurement[index + 1]

            positions_px = self._positions_px_all[idx_start:idx_end]
            positions_px_com = positions_px.mean(0)
            positions_px -= positions_px_com - xp_storage.array(self._object_shape) / 2
            self._positions_px_all[idx_start:idx_end] = positions_px.copy()

        self._positions_px_initial_all = self._positions_px_all.copy()
        self._positions_initial_all = self._positions_px_initial_all.copy()
        self._positions_initial_all[:, 0] *= self.sampling[0]
        self._positions_initial_all[:, 1] *= self.sampling[1]

        self._positions_initial = self._return_average_positions()
        if self._positions_initial is not None:
            self._positions_initial[:, 0] *= self.sampling[0]
            self._positions_initial[:, 1] *= self.sampling[1]

        # initialize probe
        self._probes_all = []
        list_Q = isinstance(self._probe_init, (list, tuple))

        if store_initial_arrays:
            self._probes_all_initial = []
            self._probes_all_initial_aperture = []
        else:
            self._probes_all_initial_aperture = [None] * self._num_measurements

        for index in range(self._num_measurements):
            _probe, self._semiangle_cutoff = self._initialize_probe(
                self._probe_init[index] if list_Q else self._probe_init,
                self._vacuum_probe_intensity,
                self._mean_diffraction_intensity[index],
                self._semiangle_cutoff,
                crop_patterns,
            )

            self._probes_all.append(_probe)
            if store_initial_arrays:
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

        if object_fov_mask is not True:
            raise NotImplementedError()
        else:
            self._object_fov_mask = np.full(self._object_shape, True)
        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        # plot probe overlaps
        if plot_probe_overlaps:
            if max_batch_size is None:
                max_batch_size = self._num_diffraction_patterns

            probe_overlap = xp.zeros(self._object_shape, dtype=xp.float32)

            for start, end in generate_batches(
                self._cum_probes_per_measurement[1], max_batch=max_batch_size
            ):
                # batch indices
                positions_px = self._positions_px_all[start:end]
                positions_px_fractional = positions_px - xp_storage.round(positions_px)

                shifted_probes = fft_shift(
                    self._probes_all[0], positions_px_fractional, xp
                )
                probe_overlap += self._sum_overlapping_patches_bincounts(
                    xp.abs(shifted_probes) ** 2, positions_px
                )

            del shifted_probes
            probe_overlap = asnumpy(probe_overlap)

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
                probe_overlap,
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
        self.clear_device_mem(self._device, self._clear_fft_cache)

        return self

    def _object_constraints_vector(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma_e,
        gaussian_filter_sigma_m,
        butterworth_filter,
        butterworth_order,
        q_lowpass_e,
        q_lowpass_m,
        q_highpass_e,
        q_highpass_m,
        tv_denoise,
        tv_denoise_weights,
        tv_denoise_inner_iter,
        object_positivity,
        shrinkage_rad,
        object_mask,
        **kwargs,
    ):
        """Calls Object3DConstraints _object_constraints for each object."""
        xp = self._xp

        # electrostatic
        current_object[0] = self._object_constraints(
            current_object[0],
            gaussian_filter,
            gaussian_filter_sigma_e,
            butterworth_filter,
            butterworth_order,
            q_lowpass_e,
            q_highpass_e,
            tv_denoise,
            tv_denoise_weights,
            tv_denoise_inner_iter,
            object_positivity,
            shrinkage_rad,
            object_mask,
            **kwargs,
        )

        # magnetic
        for index in range(1, 4):
            current_object[index] = self._object_constraints(
                current_object[index],
                gaussian_filter,
                gaussian_filter_sigma_m,
                butterworth_filter,
                butterworth_order,
                q_lowpass_m,
                q_highpass_m,
                tv_denoise,
                tv_denoise_weights,
                tv_denoise_inner_iter,
                False,
                0.0,
                None,
                **kwargs,
            )

        # divergence-free
        current_object[1:] = project_vector_field_divergence_periodic_3D(
            current_object[1:], xp=xp
        )

        return current_object

    def _constraints(self, current_object, current_probe, current_positions, **kwargs):
        """Wrapper function to bypass _object_constraints"""

        current_object = self._object_constraints_vector(current_object, **kwargs)
        current_probe = self._probe_constraints(current_probe, **kwargs)
        current_positions = self._positions_constraints(current_positions, **kwargs)

        return current_object, current_probe, current_positions

    def reconstruct(
        self,
        num_iter: int = 8,
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
        fix_probe_com: bool = True,
        fix_probe: bool = False,
        fix_probe_aperture: bool = False,
        constrain_probe_amplitude: bool = False,
        constrain_probe_amplitude_relative_radius: float = 0.5,
        constrain_probe_amplitude_relative_width: float = 0.05,
        constrain_probe_fourier_amplitude: bool = False,
        constrain_probe_fourier_amplitude_max_width_pixels: float = 3.0,
        constrain_probe_fourier_amplitude_constant_intensity: bool = False,
        fix_positions: bool = True,
        fix_positions_com: bool = True,
        max_position_update_distance: float = None,
        max_position_total_distance: float = None,
        global_affine_transformation: bool = False,
        gaussian_filter_sigma_e: float = None,
        gaussian_filter_sigma_m: float = None,
        gaussian_filter: bool = True,
        fit_probe_aberrations: bool = False,
        fit_probe_aberrations_max_angular_order: int = 4,
        fit_probe_aberrations_max_radial_order: int = 4,
        fit_probe_aberrations_remove_initial: bool = False,
        fit_probe_aberrations_using_scikit_image: bool = True,
        butterworth_filter: bool = True,
        q_lowpass_e: float = None,
        q_lowpass_m: float = None,
        q_highpass_e: float = None,
        q_highpass_m: float = None,
        butterworth_order: float = 2,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
        detector_fourier_mask: np.ndarray = None,
        tv_denoise: bool = True,
        tv_denoise_weights=None,
        tv_denoise_inner_iter=40,
        collective_measurement_updates: bool = True,
        store_iterations: bool = False,
        progress_bar: bool = True,
        reset: bool = None,
        device: str = None,
        clear_fft_cache: bool = None,
    ):
        """
        Ptychographic reconstruction main method.

        Parameters
        --------
        num_iter: int, optional
            Number of iterations to run
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
        fix_probe_com: bool, optional
            If True, fixes center of mass of probe
        fix_probe: bool, optional
            If True, probe is fixed
        fix_probe_aperture: bool, optional
            If True, vaccum probe is used to fix Fourier amplitude
        constrain_probe_amplitude: bool, optional
            If True, real-space probe is constrained with a top-hat support.
        constrain_probe_amplitude_relative_radius: float
            Relative location of top-hat inflection point, between 0 and 0.5
        constrain_probe_amplitude_relative_width: float
            Relative width of top-hat sigmoid, between 0 and 0.5
        constrain_probe_fourier_amplitude: bool, optional
            If True, Fourier-probe is constrained by fitting a sigmoid for each angular frequency
        constrain_probe_fourier_amplitude_max_width_pixels: float
            Maximum pixel width of fitted sigmoid functions.
        constrain_probe_fourier_amplitude_constant_intensity: bool
            If True, the probe aperture is additionally constrained to a constant intensity.
        fix_positions: bool, optional
            If True, probe-positions are fixed
        fix_positions_com: bool, optional
            If True, fixes the positions CoM to the middle of the fov
        max_position_update_distance: float, optional
            Maximum allowed distance for update in A
        max_position_total_distance: float, optional
            Maximum allowed distance from initial positions
        global_affine_transformation: bool, optional
            If True, positions are assumed to be a global affine transform from initial scan
        gaussian_filter_sigma_e: float
            Standard deviation of gaussian kernel for electrostatic object in A
        gaussian_filter_sigma_m: float
            Standard deviation of gaussian kernel for magnetic object in A
        gaussian_filter: bool, optional
            If True and gaussian_filter_sigma is not None, object is smoothed using gaussian filtering
        fit_probe_aberrations: bool, optional
            If True, probe aberrations are fitted to a low-order expansion
        fit_probe_aberrations_max_angular_order: bool
            Max angular order of probe aberrations basis functions
        fit_probe_aberrations_max_radial_order: bool
            Max radial order of probe aberrations basis functions
        fit_probe_aberrations_remove_initial: bool
            If true, initial probe aberrations are removed before fitting
        fit_probe_aberrations_using_scikit_image: bool
            If true, the necessary phase unwrapping is performed using scikit-image. This is more stable, but occasionally leads
            to a documented bug where the kernel hangs..
            If false, a poisson-based solver is used for phase unwrapping. This won't hang, but tends to underestimate aberrations.
        butterworth_filter: bool, optional
            If True and q_lowpass or q_highpass is not None, object is smoothed using butterworth filtering
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        object_positivity: bool, optional
            If True, forces object to be positive
        tv_denoise: bool, optional
            If True and tv_denoise_weight is not None, object is smoothed using TV denoising
        tv_denoise_weights: [float,float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        collective_measurement_updates: bool
            if True perform collective tilt updates
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        fix_potential_baseline: bool
            If true, the potential mean outside the FOV is forced to zero at each iteration
        detector_fourier_mask: np.ndarray
            Corner-centered mask to multiply the detector-plane gradients with (a value of zero supresses those pixels).
            Useful when detector has artifacts such as dead-pixels. Usually binary.
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        reset: bool, optional
            If True, previous reconstructions are ignored
        device: str, optional
            if not none, overwrites self._device to set device preprocess will be perfomed on.
        clear_fft_cache: bool, optional
            if true, and device = 'gpu', clears the cached fft plan at the end of function calls

        Returns
        --------
        self: OverlapMagneticTomographicReconstruction
            Self to accommodate chaining
        """
        # handle device/storage
        self.set_device(device, clear_fft_cache)

        if device is not None:
            attrs = [
                "_known_aberrations_array",
                "_object",
                "_object_initial",
                "_probes_all",
                "_probes_all_initial",
                "_probes_all_initial_aperture",
                "_propagator_arrays",
            ]
            self.copy_attributes_to_device(attrs, device)

        xp = self._xp
        xp_storage = self._xp_storage
        device = self._device
        asnumpy = self._asnumpy

        if not collective_measurement_updates and self._verbose:
            warnings.warn(
                "Magnetic ptychography is much more robust with `collective_measurement_updates=True`.",
                UserWarning,
            )

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

        if use_projection_scheme:
            raise NotImplementedError(
                "Magnetic ptychographic tomography is currently only implemented for gradient descent."
            )

        # initialization
        self._reset_reconstruction(store_iterations, reset, use_projection_scheme)

        if self._verbose:
            self._report_reconstruction_summary(
                num_iter,
                use_projection_scheme,
                reconstruction_method,
                reconstruction_parameter,
                projection_a,
                projection_b,
                projection_c,
                normalization_min,
                max_batch_size,
                step_size,
            )

        if max_batch_size is not None:
            np.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        if detector_fourier_mask is not None:
            detector_fourier_mask = xp.asarray(detector_fourier_mask)

        if gaussian_filter_sigma_m is None:
            gaussian_filter_sigma_m = gaussian_filter_sigma_e

        if q_lowpass_m is None:
            q_lowpass_m = q_lowpass_e

        # main loop
        for a0 in tqdmnd(
            num_iter,
            desc="Reconstructing object and probe",
            unit=" iter",
            disable=not progress_bar,
        ):
            error = 0.0

            if collective_measurement_updates:
                collective_object = xp.zeros_like(self._object)

            indices = np.arange(self._num_measurements)
            np.random.shuffle(indices)

            old_rot_matrix = np.eye(3)  # identity

            for index in indices:
                self._active_measurement_index = index

                measurement_error = 0.0

                rot_matrix = self._tilt_orientation_matrices[
                    self._active_measurement_index
                ]
                self._object = self._rotate_zxy_volume_util(
                    self._object,
                    rot_matrix @ old_rot_matrix.T,
                )
                object_V = self._object[0]

                # last transformation matrix row
                weight_x, weight_y, weight_z = rot_matrix[-1]
                object_A = (
                    weight_x * self._object[2]
                    + weight_y * self._object[3]
                    + weight_z * self._object[1]
                )

                object_sliced = self._project_sliced_object(
                    object_V + object_A, self._num_slices
                )

                _probe = self._probes_all[self._active_measurement_index]
                _probe_initial_aperture = self._probes_all_initial_aperture[
                    self._active_measurement_index
                ]

                if not use_projection_scheme:
                    object_sliced_old = object_sliced.copy()

                start_idx = self._cum_probes_per_measurement[
                    self._active_measurement_index
                ]
                end_idx = self._cum_probes_per_measurement[
                    self._active_measurement_index + 1
                ]

                num_diffraction_patterns = end_idx - start_idx
                shuffled_indices = np.arange(start_idx, end_idx)

                # randomize
                if not use_projection_scheme:
                    np.random.shuffle(shuffled_indices)

                for start, end in generate_batches(
                    num_diffraction_patterns, max_batch=max_batch_size
                ):
                    # batch indices
                    batch_indices = shuffled_indices[start:end]
                    positions_px = self._positions_px_all[batch_indices]
                    positions_px_initial = self._positions_px_initial_all[batch_indices]
                    positions_px_fractional = positions_px - xp_storage.round(
                        positions_px
                    )

                    (
                        vectorized_patch_indices_row,
                        vectorized_patch_indices_col,
                    ) = self._extract_vectorized_patch_indices(positions_px)

                    amplitudes_device = copy_to_device(
                        self._amplitudes[batch_indices], device
                    )

                    # forward operator
                    (
                        shifted_probes,
                        object_patches,
                        overlap,
                        self._exit_waves,
                        batch_error,
                    ) = self._forward(
                        object_sliced,
                        vectorized_patch_indices_row,
                        vectorized_patch_indices_col,
                        _probe,
                        positions_px_fractional,
                        amplitudes_device,
                        self._exit_waves,
                        detector_fourier_mask,
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
                        positions_px,
                        self._exit_waves,
                        use_projection_scheme=use_projection_scheme,
                        step_size=step_size,
                        normalization_min=normalization_min,
                        fix_probe=fix_probe,
                    )

                    # position correction
                    if not fix_positions and a0 > 0:
                        self._positions_px_all[batch_indices] = (
                            self._position_correction(
                                object_sliced,
                                vectorized_patch_indices_row,
                                vectorized_patch_indices_col,
                                shifted_probes,
                                overlap,
                                amplitudes_device,
                                positions_px,
                                positions_px_initial,
                                positions_step_size,
                                max_position_update_distance,
                                max_position_total_distance,
                            )
                        )

                    measurement_error += batch_error

                if not use_projection_scheme:
                    object_sliced -= object_sliced_old

                object_update = self._expand_sliced_object(
                    object_sliced, self._num_voxels
                )

                weights = (1, weight_z, weight_x, weight_y)
                for index, weight in zip(range(4), weights):
                    if collective_measurement_updates:
                        collective_object[index] += self._rotate_zxy_volume(
                            object_update * weight,
                            rot_matrix.T,
                        )
                    else:
                        self._object[index] += object_update * weight

                old_rot_matrix = rot_matrix

                # Normalize Error
                measurement_error /= (
                    self._mean_diffraction_intensity[self._active_measurement_index]
                    * num_diffraction_patterns
                )
                error += measurement_error

                # constraints

                if collective_measurement_updates:
                    # probe and positions
                    _probe = self._probe_constraints(
                        _probe,
                        fix_probe_com=fix_probe_com and not fix_probe,
                        constrain_probe_amplitude=constrain_probe_amplitude
                        and not fix_probe,
                        constrain_probe_amplitude_relative_radius=constrain_probe_amplitude_relative_radius,
                        constrain_probe_amplitude_relative_width=constrain_probe_amplitude_relative_width,
                        constrain_probe_fourier_amplitude=constrain_probe_fourier_amplitude
                        and not fix_probe,
                        constrain_probe_fourier_amplitude_max_width_pixels=constrain_probe_fourier_amplitude_max_width_pixels,
                        constrain_probe_fourier_amplitude_constant_intensity=constrain_probe_fourier_amplitude_constant_intensity,
                        fit_probe_aberrations=fit_probe_aberrations and not fix_probe,
                        fit_probe_aberrations_max_angular_order=fit_probe_aberrations_max_angular_order,
                        fit_probe_aberrations_max_radial_order=fit_probe_aberrations_max_radial_order,
                        fit_probe_aberrations_remove_initial=fit_probe_aberrations_remove_initial,
                        fit_probe_aberrations_using_scikit_image=fit_probe_aberrations_using_scikit_image,
                        fix_probe_aperture=fix_probe_aperture and not fix_probe,
                        initial_probe_aperture=_probe_initial_aperture,
                    )

                    self._positions_px_all[batch_indices] = self._positions_constraints(
                        self._positions_px_all[batch_indices],
                        self._positions_px_initial_all[batch_indices],
                        fix_positions=fix_positions,
                        fix_positions_com=fix_positions_com and not fix_positions,
                        global_affine_transformation=global_affine_transformation,
                    )

                else:
                    # object, probe, and positions
                    (
                        self._object,
                        _probe,
                        self._positions_px_all[batch_indices],
                    ) = self._constraints(
                        self._object,
                        _probe,
                        self._positions_px_all[batch_indices],
                        self._positions_px_initial_all[batch_indices],
                        fix_probe_com=fix_probe_com and not fix_probe,
                        constrain_probe_amplitude=constrain_probe_amplitude
                        and not fix_probe,
                        constrain_probe_amplitude_relative_radius=constrain_probe_amplitude_relative_radius,
                        constrain_probe_amplitude_relative_width=constrain_probe_amplitude_relative_width,
                        constrain_probe_fourier_amplitude=constrain_probe_fourier_amplitude
                        and not fix_probe,
                        constrain_probe_fourier_amplitude_max_width_pixels=constrain_probe_fourier_amplitude_max_width_pixels,
                        constrain_probe_fourier_amplitude_constant_intensity=constrain_probe_fourier_amplitude_constant_intensity,
                        fit_probe_aberrations=fit_probe_aberrations and not fix_probe,
                        fit_probe_aberrations_max_angular_order=fit_probe_aberrations_max_angular_order,
                        fit_probe_aberrations_max_radial_order=fit_probe_aberrations_max_radial_order,
                        fit_probe_aberrations_remove_initial=fit_probe_aberrations_remove_initial,
                        fit_probe_aberrations_using_scikit_image=fit_probe_aberrations_using_scikit_image,
                        fix_probe_aperture=fix_probe_aperture and not fix_probe,
                        initial_probe_aperture=_probe_initial_aperture,
                        fix_positions=fix_positions,
                        fix_positions_com=fix_positions_com and not fix_positions,
                        global_affine_transformation=global_affine_transformation,
                        gaussian_filter=gaussian_filter
                        and gaussian_filter_sigma_m is not None,
                        gaussian_filter_sigma_e=gaussian_filter_sigma_e,
                        gaussian_filter_sigma_m=gaussian_filter_sigma_m,
                        butterworth_filter=butterworth_filter
                        and (q_lowpass_m is not None or q_highpass_m is not None),
                        q_lowpass_e=q_lowpass_e,
                        q_lowpass_m=q_lowpass_m,
                        q_highpass_e=q_highpass_e,
                        q_highpass_m=q_highpass_m,
                        butterworth_order=butterworth_order,
                        object_positivity=object_positivity,
                        shrinkage_rad=shrinkage_rad,
                        object_mask=(
                            self._object_fov_mask_inverse
                            if fix_potential_baseline
                            and self._object_fov_mask_inverse.sum() > 0
                            else None
                        ),
                        tv_denoise=tv_denoise and tv_denoise_weights is not None,
                        tv_denoise_weights=tv_denoise_weights,
                        tv_denoise_inner_iter=tv_denoise_inner_iter,
                    )

            self._object = self._rotate_zxy_volume_util(self._object, old_rot_matrix.T)

            # Normalize Error Over Tilts
            error /= self._num_measurements

            if collective_measurement_updates:
                self._object += collective_object / self._num_measurements

                # object only
                self._object = self._object_constraints_vector(
                    self._object,
                    gaussian_filter=gaussian_filter
                    and gaussian_filter_sigma_m is not None,
                    gaussian_filter_sigma_e=gaussian_filter_sigma_e,
                    gaussian_filter_sigma_m=gaussian_filter_sigma_m,
                    butterworth_filter=butterworth_filter
                    and (q_lowpass_m is not None or q_highpass_m is not None),
                    q_lowpass_e=q_lowpass_e,
                    q_lowpass_m=q_lowpass_m,
                    q_highpass_e=q_highpass_e,
                    q_highpass_m=q_highpass_m,
                    butterworth_order=butterworth_order,
                    object_positivity=object_positivity,
                    shrinkage_rad=shrinkage_rad,
                    object_mask=(
                        self._object_fov_mask_inverse
                        if fix_potential_baseline
                        and self._object_fov_mask_inverse.sum() > 0
                        else None
                    ),
                    tv_denoise=tv_denoise and tv_denoise_weights is not None,
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

        # remove _exit_waves attr from self for GD
        if not use_projection_scheme:
            self._exit_waves = None

        self.clear_device_mem(self._device, self._clear_fft_cache)

        return self

    def _visualize_all_iterations(self, **kwargs):
        raise NotImplementedError()

    def _visualize_last_iteration(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        orientation_matrix=None,
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
        plot_probe: bool, optional
            If true, the reconstructed complex probe is displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        """

        asnumpy = self._asnumpy

        # get scaled arrays

        if orientation_matrix is not None:
            ordered_obj = self._rotate_zxy_volume_vector(
                self._object,
                orientation_matrix,
            )

            # V(z,x,y), Ax(z,x,y), Ay(z,x,y), Az(z,x,y)
            ordered_obj = asnumpy(ordered_obj)
            ordered_obj[1:] = np.roll(ordered_obj[1:], -1, axis=0)

        else:
            # V(z,x,y), Ax(z,x,y), Ay(z,x,y), Az(z,x,y)
            ordered_obj = self.object.copy()
            ordered_obj[1:] = np.roll(ordered_obj[1:], -1, axis=0)

        _, nz, nx, ny = ordered_obj.shape
        img_array = np.zeros((nx + nx + nz, ny * 4), dtype=ordered_obj.dtype)

        axes = [1, 2, 0]
        transposes = [False, True, False]
        labels = [("z [A]", "y [A]"), ("x [A]", "z [A]"), ("x [A]", "y [A]")]
        limits_v = [(0, nz), (nz, nz + nx), (nz + nx, nz + nx + nx)]
        limits_h = [(0, ny), (0, nz), (0, ny)]

        titles = [
            [
                r"$V$ projected along $\hat{x}$",
                r"$A_x$ projected along $\hat{x}$",
                r"$A_y$ projected along $\hat{x}$",
                r"$A_z$ projected along $\hat{x}$",
            ],
            [
                r"$V$ projected along $\hat{y}$",
                r"$A_x$ projected along $\hat{y}$",
                r"$A_y$ projected along $\hat{y}$",
                r"$A_z$ projected along $\hat{y}$",
            ],
            [
                r"$V$ projected along $\hat{z}$",
                r"$A_x$ projected along $\hat{z}$",
                r"$A_y$ projected along $\hat{z}$",
                r"$A_z$ projected along $\hat{z}$",
            ],
        ]

        for index in range(4):
            for axis, transpose, limit_v, limit_h in zip(
                axes, transposes, limits_v, limits_h
            ):
                start_v, end_v = limit_v
                start_h, end_h = np.array(limit_h) + index * ny

                subarray = ordered_obj[index].sum(axis)
                if transpose:
                    subarray = subarray.T

                img_array[start_v:end_v, start_h:end_h] = subarray

        if plot_convergence:
            auto_figsize = (ny * 4 * 4 / nx, (nx + nx + nz) * 3.5 / nx + 1)
        else:
            auto_figsize = (ny * 4 * 4 / nx, (nx + nx + nz) * 3.5 / nx)

        figsize = kwargs.pop("figsize", auto_figsize)
        cmap_e = kwargs.pop("cmap_e", "magma")
        cmap_m = kwargs.pop("cmap_m", "PuOr")
        vmin_e = kwargs.pop("vmin_e", None)
        vmax_e = kwargs.pop("vmax_e", None)

        # remove common unused kwargs
        kwargs.pop("plot_probe", None)
        kwargs.pop("plot_fourier_probe", None)
        kwargs.pop("remove_initial_probe_aberrations", None)
        kwargs.pop("vertical_lims", None)
        kwargs.pop("horizontal_lims", None)

        _, vmin_e, vmax_e = return_scaled_histogram_ordering(
            img_array[:, :ny], vmin_e, vmax_e
        )

        _, _, _vmax_m = return_scaled_histogram_ordering(np.abs(img_array[:, ny:]))
        vmin_m = kwargs.pop("vmin_m", -_vmax_m)
        vmax_m = kwargs.pop("vmax_m", _vmax_m)

        if plot_convergence:
            spec = GridSpec(
                ncols=4,
                nrows=4,
                height_ratios=[nx, nz, nx, nx / 4],
                hspace=0.15,
                wspace=0.35,
            )
        else:
            spec = GridSpec(
                ncols=4, nrows=3, height_ratios=[nx, nz, nx], hspace=0.15, wspace=0.35
            )

        if fig is None:
            fig = plt.figure(figsize=figsize)

        for sp in spec:
            row, col = np.unravel_index(sp.num1, (4, 4))

            if row < 3:
                ax = fig.add_subplot(sp)

                start_v, end_v = limits_v[row]
                start_h, end_h = np.array(limits_h[row]) + col * ny
                subarray = img_array[start_v:end_v, start_h:end_h]

                extent = [
                    0,
                    self.sampling[1] * subarray.shape[1],
                    self.sampling[0] * subarray.shape[0],
                    0,
                ]

                im = ax.imshow(
                    subarray,
                    cmap=cmap_e if sp.is_first_col() else cmap_m,
                    vmin=vmin_e if sp.is_first_col() else vmin_m,
                    vmax=vmax_e if sp.is_first_col() else vmax_m,
                    extent=extent,
                    **kwargs,
                )

                if cbar:
                    divider = make_axes_locatable(ax)
                    ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                    fig.add_axes(ax_cb)
                    fig.colorbar(im, cax=ax_cb)

                ax.set_title(titles[row][col])

                y_label, x_label = labels[row]
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

        if plot_convergence and hasattr(self, "error_iterations"):
            errors = np.array(self.error_iterations)

            ax = fig.add_subplot(spec[-1, :])
            ax.semilogy(np.arange(errors.shape[0]), errors, **kwargs)
            ax.set_ylabel("NMSE")
            ax.set_xlabel("Iteration number")
            ax.yaxis.tick_right()

        fig.suptitle(f"Normalized mean squared error: {self.error:.3e}")
        spec.tight_layout(fig)

    def _rotate_zxy_volume_util(
        self,
        current_object,
        rot_matrix,
    ):
        """ """
        for index in range(4):
            current_object[index] = self._rotate_zxy_volume(
                current_object[index], rot_matrix
            )

        return current_object

    def _rotate_zxy_volume_vector(self, current_object, rot_matrix):
        """Rotates vector field consistently. Note this is very expensive"""

        xp = self._xp
        swap_zxy_to_xyz = self._swap_zxy_to_xyz

        if xp is np:
            from scipy.interpolate import RegularGridInterpolator

            current_object = self._asnumpy(current_object)
        else:
            try:
                from cupyx.scipy.interpolate import RegularGridInterpolator
            except ModuleNotFoundError:
                from scipy.interpolate import RegularGridInterpolator

                xp = np  # force xp to np for cupy <12.0
                current_object = self._asnumpy(current_object)

        _, nz, nx, ny = current_object.shape

        z, x, y = [xp.linspace(-1, 1, s, endpoint=False) for s in (nx, ny, nz)]
        Z, X, Y = xp.meshgrid(z, x, y, indexing="ij")
        coords = xp.array([Z.ravel(), X.ravel(), Y.ravel()])

        tf = xp.asarray(swap_zxy_to_xyz.T @ rot_matrix @ swap_zxy_to_xyz)
        rotated_vecs = tf.T.dot(coords).T

        Az = RegularGridInterpolator(
            (z, x, y), current_object[1], bounds_error=False, fill_value=0
        )
        Ax = RegularGridInterpolator(
            (z, x, y), current_object[2], bounds_error=False, fill_value=0
        )
        Ay = RegularGridInterpolator(
            (z, x, y), current_object[3], bounds_error=False, fill_value=0
        )

        xp = self._xp  # switch back to device
        obj = xp.zeros_like(current_object)
        obj[0] = self._rotate_zxy_volume(xp.asarray(current_object[0]), rot_matrix)

        obj[1] = xp.asarray(Az(rotated_vecs).reshape(nz, nx, ny))
        obj[2] = xp.asarray(Ax(rotated_vecs).reshape(nz, nx, ny))
        obj[3] = xp.asarray(Ay(rotated_vecs).reshape(nz, nx, ny))

        return obj
