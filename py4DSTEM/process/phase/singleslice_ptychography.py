"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely (single-slice) ptychography.
"""

from typing import Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from emdfile import Custom, tqdmnd
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.phase.phase_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.ptychographic_constraints import (
    ObjectNDConstraintsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
)
from py4DSTEM.process.phase.ptychographic_methods import (
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
)


class SingleslicePtychography(
    VisualizationsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
    ObjectNDConstraintsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeMethodsMixin,
    ObjectNDMethodsMixin,
    PtychographicReconstruction,
):
    """
    Iterative Ptychographic Reconstruction Class.

    Diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions      : (Sx,Sy)
    Reconstructed object dimensions     : (Px,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our probe
    and (Px,Py) is the padded-object size we position our ROI around in.

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions in eV
    datacube: DataCube
        Input 4D diffraction pattern intensities
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
        Initial guess for complex-valued object of dimensions (Px,Py)
        If None, initialized to 1.0j
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy). If None,
        initialized to ComplexProbe with semiangle_cutoff, energy, and aberrations
    initial_scan_positions: np.ndarray, optional
        Probe positions in Å for each diffraction intensity
        If None, initialized to a grid scan
    positions_offset_ang: np.ndarray, optional
        Offset of positions in A
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    object_type: str, optional
        The object can be reconstructed as a real potential ('potential') or a complex
        object ('complex')
    positions_mask: np.ndarray, optional
        Boolean real space mask to select positions in datacube to skip for reconstruction
    device: str, optional
        Device calculation will be perfomed on. Must be 'cpu' or 'gpu'
    storage: str, optional
        Device non-frequent arrays will be stored on. Must be 'cpu' or 'gpu'
    clear_fft_cache: bool, optional
        If True, and device = 'gpu', clears the cached fft plan at the end of function calls
    name: str, optional
        Class name
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = ()

    def __init__(
        self,
        energy: float,
        datacube: DataCube = None,
        semiangle_cutoff: float = None,
        semiangle_cutoff_pixels: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        initial_scan_positions: np.ndarray = None,
        positions_offset_ang: np.ndarray = None,
        object_padding_px: Tuple[int, int] = None,
        object_type: str = "complex",
        positions_mask: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        storage: str = None,
        clear_fft_cache: bool = True,
        name: str = "ptychographic_reconstruction",
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

        if object_type != "potential" and object_type != "complex":
            raise ValueError(
                f"object_type must be either 'potential' or 'complex', not {object_type}"
            )

        self.set_save_defaults()

        # Data
        self._datacube = datacube
        self._object = initial_object_guess
        self._probe = initial_probe_guess

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

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "bilinear",
        padded_diffraction_intensities_shape: Tuple[int, int] = None,
        region_of_interest_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_center_of_mass: str = "default",
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = None,
        plot_probe_overlaps: bool = True,
        force_com_rotation: float = None,
        force_com_transpose: float = None,
        force_com_shifts: Union[Sequence[np.ndarray], Sequence[float]] = None,
        force_com_measured: Sequence[np.ndarray] = None,
        vectorized_com_calculation: bool = True,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        object_fov_mask: np.ndarray = None,
        crop_patterns: bool = False,
        store_initial_arrays: bool = True,
        device: str = None,
        clear_fft_cache: bool = None,
        max_batch_size: int = None,
        **kwargs,
    ):
        """
        Ptychographic preprocessing step.

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
        plot_center_of_mass: str, optional
            If 'default', the corrected CoM arrays will be displayed
            If 'all', the computed and fitted CoM arrays will be displayed
        plot_rotation: bool, optional
            If True, the CoM curl minimization search result will be displayed
        maximize_divergence: bool, optional
            If True, the divergence of the CoM gradient vector field is maximized
        rotation_angles_deg: np.darray, optional
            Array of angles in degrees to perform curl minimization over
        plot_probe_overlaps: bool, optional
            If True, initial probe overlaps scanned over the object will be displayed
        force_com_rotation: float (degrees), optional
            Force relative rotation angle between real and reciprocal space
        force_com_transpose: bool, optional
            Force whether diffraction intensities need to be transposed.
        force_com_shifts: tuple of ndarrays (CoMx, CoMy)
            Amplitudes come from diffraction patterns shifted with
            the CoM in the upper left corner for each probe unless
            shift is overwritten.
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
        self: PtychographicReconstruction
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

        # preprocess datacube
        (
            self._datacube,
            self._vacuum_probe_intensity,
            self._dp_mask,
            force_com_shifts,
            force_com_measured,
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube,
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            padded_diffraction_intensities_shape=self._padded_diffraction_intensities_shape,
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            dp_mask=self._dp_mask,
            com_shifts=force_com_shifts,
            com_measured=force_com_measured,
        )

        # calibrations
        _intensities = self._extract_intensities_and_calibrations_from_datacube(
            self._datacube,
            require_calibrations=True,
            force_scan_sampling=force_scan_sampling,
            force_angular_sampling=force_angular_sampling,
            force_reciprocal_sampling=force_reciprocal_sampling,
        )

        # handle semiangle specified in pixels
        if self._semiangle_cutoff_pixels:
            self._semiangle_cutoff = (
                self._semiangle_cutoff_pixels * self._angular_sampling[0]
            )

        # calculate CoM
        (
            self._com_measured_x,
            self._com_measured_y,
            self._com_fitted_x,
            self._com_fitted_y,
            self._com_normalized_x,
            self._com_normalized_y,
        ) = self._calculate_intensities_center_of_mass(
            _intensities,
            dp_mask=self._dp_mask,
            fit_function=fit_function,
            com_shifts=force_com_shifts,
            vectorized_calculation=vectorized_com_calculation,
            com_measured=force_com_measured,
        )

        # estimate rotation / transpose
        (
            self._rotation_best_rad,
            self._rotation_best_transpose,
            self._com_x,
            self._com_y,
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

        # explicitly transfer arrays to storage
        attrs = [
            "_com_measured_x",
            "_com_measured_y",
            "_com_fitted_x",
            "_com_fitted_y",
            "_com_normalized_x",
            "_com_normalized_y",
            "_com_x",
            "_com_y",
        ]
        self.copy_attributes_to_device(attrs, storage)

        # corner-center amplitudes
        (
            self._amplitudes,
            self._mean_diffraction_intensity,
            self._crop_mask,
        ) = self._normalize_diffraction_intensities(
            _intensities,
            self._com_fitted_x,
            self._com_fitted_y,
            self._positions_mask,
            crop_patterns,
        )

        # explicitly transfer arrays to storage
        self._amplitudes = copy_to_device(self._amplitudes, storage)
        del _intensities

        self._num_diffraction_patterns = self._amplitudes.shape[0]
        self._amplitudes_shape = np.array(self._amplitudes.shape[-2:])

        if region_of_interest_shape is not None:
            self._resample_exit_waves = True
            self._region_of_interest_shape = np.array(region_of_interest_shape)
        else:
            self._resample_exit_waves = False
            self._region_of_interest_shape = np.array(self._amplitudes.shape[-2:])

        # initialize probe positions
        (
            self._positions_px,
            self._object_padding_px,
        ) = self._calculate_scan_positions_in_pixels(
            self._scan_positions,
            self._positions_mask,
            self._object_padding_px,
            self._positions_offset_ang,
        )

        # initialize object
        self._object = self._initialize_object(
            self._object,
            self._positions_px,
            self._object_type,
        )

        if store_initial_arrays:
            self._object_initial = self._object.copy()
            self._object_type_initial = self._object_type

        self._object_shape = self._object.shape

        # center probe positions
        self._positions_px = xp_storage.asarray(
            self._positions_px, dtype=xp_storage.float32
        )
        self._positions_px_initial_com = self._positions_px.mean(0)
        self._positions_px -= (
            self._positions_px_initial_com - xp_storage.array(self._object_shape) / 2
        )
        self._positions_px_initial_com = self._positions_px.mean(0)

        self._positions_px_initial = self._positions_px.copy()
        self._positions_initial = self._positions_px_initial.copy()
        self._positions_initial[:, 0] *= self.sampling[0]
        self._positions_initial[:, 1] *= self.sampling[1]

        # initialize probe
        self._probe, self._semiangle_cutoff = self._initialize_probe(
            self._probe,
            self._vacuum_probe_intensity,
            self._mean_diffraction_intensity,
            self._semiangle_cutoff,
            crop_patterns,
        )

        # initialize aberrations
        self._known_aberrations_array = ComplexProbe(
            energy=self._energy,
            gpts=self._region_of_interest_shape,
            sampling=self.sampling,
            parameters=self._polar_parameters,
            device=device,
        )._evaluate_ctf()

        if store_initial_arrays:
            self._probe_initial = self._probe.copy()
            self._probe_initial_aperture = xp.abs(xp.fft.fft2(self._probe))
        else:
            self._probe_initial_aperture = None

        if object_fov_mask is None or plot_probe_overlaps:
            # overlaps
            if max_batch_size is None:
                max_batch_size = self._num_diffraction_patterns

            probe_overlap = xp.zeros(self._object_shape, dtype=xp.float32)

            for start, end in generate_batches(
                self._num_diffraction_patterns, max_batch=max_batch_size
            ):
                # batch indices
                positions_px = self._positions_px[start:end]
                positions_px_fractional = positions_px - xp_storage.round(positions_px)

                shifted_probes = fft_shift(self._probe, positions_px_fractional, xp)
                probe_overlap += self._sum_overlapping_patches_bincounts(
                    xp.abs(shifted_probes) ** 2, positions_px
                )

            del shifted_probes

        # initialize object_fov_mask
        if object_fov_mask is None:
            gaussian_filter = self._scipy.ndimage.gaussian_filter
            probe_overlap_blurred = gaussian_filter(probe_overlap, 1.0)
            self._object_fov_mask = asnumpy(
                probe_overlap_blurred > 0.25 * probe_overlap_blurred.max()
            )
            del probe_overlap_blurred
        elif object_fov_mask is True:
            self._object_fov_mask = np.full(self._object_shape, True)
        else:
            self._object_fov_mask = np.asarray(object_fov_mask)
        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        # plot probe overlaps
        if plot_probe_overlaps:
            probe_overlap = asnumpy(probe_overlap)

            figsize = kwargs.pop("figsize", (9, 4))
            chroma_boost = kwargs.pop("chroma_boost", 1)
            power = kwargs.pop("power", 2)

            # initial probe
            complex_probe_rgb = Complex2RGB(
                self.probe_centered,
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            ax1.imshow(
                complex_probe_rgb,
                extent=probe_extent,
            )

            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad="2.5%")
            add_colorbar_arg(cax1, chroma_boost=chroma_boost)
            ax1.set_ylabel("x [A]")
            ax1.set_xlabel("y [A]")
            ax1.set_title("Initial probe intensity")

            ax2.imshow(
                probe_overlap,
                extent=extent,
                cmap="gray",
            )
            ax2.scatter(
                self.positions[:, 1],
                self.positions[:, 0],
                s=2.5,
                color=(1, 0, 0, 1),
            )
            ax2.set_ylabel("x [A]")
            ax2.set_xlabel("y [A]")
            ax2.set_xlim((extent[0], extent[1]))
            ax2.set_ylim((extent[2], extent[3]))
            ax2.set_title("Object field of view")

            fig.tight_layout()

        self._preprocessed = True
        self.clear_device_mem(self._device, self._clear_fft_cache)

        return self

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
        positions_step_size: float = 0.5,
        pure_phase_object: bool = False,
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
        gaussian_filter_sigma: float = None,
        gaussian_filter: bool = True,
        fit_probe_aberrations: bool = False,
        fit_probe_aberrations_max_angular_order: int = 4,
        fit_probe_aberrations_max_radial_order: int = 4,
        fit_probe_aberrations_remove_initial: bool = False,
        fit_probe_aberrations_using_scikit_image: bool = True,
        butterworth_filter: bool = True,
        q_lowpass: float = None,
        q_highpass: float = None,
        butterworth_order: float = 2,
        tv_denoise: bool = True,
        tv_denoise_weight: float = None,
        tv_denoise_inner_iter: float = 40,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
        detector_fourier_mask: np.ndarray = None,
        store_iterations: bool = False,
        progress_bar: bool = True,
        reset: bool = None,
        device: str = None,
        clear_fft_cache: bool = None,
        object_type: str = None,
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
        pure_phase_object: bool, optional
            If True, object amplitude is set to unity
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
        gaussian_filter_sigma: float, optional
            Standard deviation of gaussian kernel in A
        gaussian_filter: bool, optional
            If True and gaussian_filter_sigma is not None, object is smoothed using gaussian filtering
        fit_probe_aberrations: bool, optional
            If True, probe aberrations are fitted to a low-order expansion
        fit_probe_aberrations_max_angular_order: int
            Max angular order of probe aberrations basis functions
        fit_probe_aberrations_max_radial_order: int
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
        tv_denoise: bool, optional
            If True and tv_denoise_weight is not None, object is smoothed using TV denoising
        tv_denoise_weight: float
            Denoising weight. The greater `weight`, the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        object_positivity: bool, optional
            If True, forces object to be positive
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
            If not none, overwrites self._device to set device preprocess will be perfomed on.
        clear_fft_cache: bool, optional
            If true, and device = 'gpu', clears the cached fft plan at the end of function calls
        object_type: str, optional
            Overwrites self._object_type

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        # handle device/storage
        self.set_device(device, clear_fft_cache)

        if device is not None:
            attrs = [
                "_known_aberrations_array",
                "_object",
                "_object_initial",
                "_probe",
                "_probe_initial",
                "_probe_initial_aperture",
            ]
            self.copy_attributes_to_device(attrs, device)

        # initialization
        self._reset_reconstruction(store_iterations, reset)

        if object_type is not None:
            self._switch_object_type(object_type)

        xp = self._xp
        xp_storage = self._xp_storage
        device = self._device
        asnumpy = self._asnumpy

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

        # batching
        shuffled_indices = np.arange(self._num_diffraction_patterns)

        if max_batch_size is not None:
            np.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        if detector_fourier_mask is not None:
            detector_fourier_mask = xp.asarray(detector_fourier_mask)

        # main loop
        for a0 in tqdmnd(
            num_iter,
            desc="Reconstructing object and probe",
            unit=" iter",
            disable=not progress_bar,
        ):
            error = 0.0

            # randomize
            if not use_projection_scheme:
                np.random.shuffle(shuffled_indices)

            for start, end in generate_batches(
                self._num_diffraction_patterns, max_batch=max_batch_size
            ):
                # batch indices
                batch_indices = shuffled_indices[start:end]
                positions_px = self._positions_px[batch_indices]
                positions_px_initial = self._positions_px_initial[batch_indices]
                positions_px_fractional = positions_px - xp_storage.round(positions_px)

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
                    self._object,
                    vectorized_patch_indices_row,
                    vectorized_patch_indices_col,
                    self._probe,
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
                self._object, self._probe = self._adjoint(
                    self._object,
                    self._probe,
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
                if not fix_positions:
                    self._positions_px[batch_indices] = self._position_correction(
                        self._object,
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

                error += batch_error

            # Normalize Error
            error /= self._mean_diffraction_intensity * self._num_diffraction_patterns

            # constraints
            self._object, self._probe, self._positions_px = self._constraints(
                self._object,
                self._probe,
                self._positions_px,
                self._positions_px_initial,
                fix_probe_com=fix_probe_com and not fix_probe,
                constrain_probe_amplitude=constrain_probe_amplitude and not fix_probe,
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
                initial_probe_aperture=self._probe_initial_aperture,
                fix_positions=fix_positions,
                fix_positions_com=fix_positions_com and not fix_positions,
                global_affine_transformation=global_affine_transformation,
                gaussian_filter=gaussian_filter and gaussian_filter_sigma is not None,
                gaussian_filter_sigma=gaussian_filter_sigma,
                butterworth_filter=butterworth_filter
                and (q_lowpass is not None or q_highpass is not None),
                q_lowpass=q_lowpass,
                q_highpass=q_highpass,
                butterworth_order=butterworth_order,
                tv_denoise=tv_denoise and tv_denoise_weight is not None,
                tv_denoise_weight=tv_denoise_weight,
                tv_denoise_inner_iter=tv_denoise_inner_iter,
                object_positivity=object_positivity,
                shrinkage_rad=shrinkage_rad,
                object_mask=(
                    self._object_fov_mask_inverse
                    if fix_potential_baseline
                    and self._object_fov_mask_inverse.sum() > 0
                    else None
                ),
                pure_phase_object=pure_phase_object and self._object_type == "complex",
            )

            self.error_iterations.append(error.item())

            if store_iterations:
                self.object_iterations.append(asnumpy(self._object).copy())
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
