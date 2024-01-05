"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely magnetic ptychography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from emdfile import Custom, tqdmnd
from py4DSTEM import DataCube
from py4DSTEM.process.phase.iterative_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.iterative_ptychographic_constraints import (
    ObjectNDConstraintsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
)
from py4DSTEM.process.phase.iterative_ptychographic_methods import (
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


class MagneticPtychographicReconstruction(
    VisualizationsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
    ObjectNDConstraintsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeListMethodsMixin,
    ProbeMethodsMixin,
    ObjectNDMethodsMixin,
    PtychographicReconstruction,
):
    """
    Iterative Magnetic Ptychographic Reconstruction Class.

    Diffraction intensities dimensions         : (Rx,Ry,Qx,Qy) (for each measurement)
    Reconstructed probe dimensions             : (Sx,Sy)
    Reconstructed electrostatic dimensions     : (Px,Py)
    Reconstructed magnetic dimensions          : (Px,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our probe
    and (Px,Py) is the padded-object size we position our ROI around in.

    Parameters
    ----------
    datacube: Sequence[DataCube]
        Tuple of input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    magnetic_contribution_sign: str, optional
        One of '-+', '-0+', '0+'
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
        Pixel dimensions to pad objects with
        If None, the padding is set to half the probe ROI dimensions
    positions_mask: np.ndarray, optional
        Boolean real space mask to select positions in datacube to skip for reconstruction
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (2,Px,Py)
        If None, initialized to 1.0j for complex objects and 0.0 for potential objects
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy). If None,
        initialized to ComplexProbe with semiangle_cutoff, energy, and aberrations
    initial_scan_positions: np.ndarray, optional
        Probe positions in Å for each diffraction intensity
        If None, initialized to a grid scan
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    object_type: str, optional
        The object can be reconstructed as a real potential ('potential') or a complex
        object ('complex')
    name: str, optional
        Class name
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = ("_magnetic_contribution_sign",)

    def __init__(
        self,
        energy: float,
        datacube: Sequence[DataCube] = None,
        magnetic_contribution_sign: str = "-+",
        semiangle_cutoff: float = None,
        semiangle_cutoff_pixels: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        object_padding_px: Tuple[int, int] = None,
        positions_mask: np.ndarray = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        initial_scan_positions: np.ndarray = None,
        object_type: str = "complex",
        verbose: bool = True,
        device: str = "cpu",
        name: str = "magnetic_ptychographic_reconstruction",
        **kwargs,
    ):
        Custom.__init__(self, name=name)

        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
            from scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
            from scipy.special import erf

            self._erf = erf
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
            from cupyx.scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
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

        if object_type != "potential" and object_type != "complex":
            raise ValueError(
                f"object_type must be either 'potential' or 'complex', not {object_type}"
            )

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
        self._magnetic_contribution_sign = magnetic_contribution_sign

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "fourier",
        probe_roi_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = None,
        plot_probe_overlaps: bool = True,
        force_com_rotation: float = None,
        force_com_transpose: float = None,
        force_com_shifts: float = None,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        progress_bar: bool = True,
        object_fov_mask: np.ndarray = None,
        crop_patterns: bool = False,
        **kwargs,
    ):
        """
        Ptychographic preprocessing step.
        Calls the base class methods:

        _extract_intensities_and_calibrations_from_datacube,
        _compute_center_of_mass(),
        _solve_CoM_rotation(),
        _normalize_diffraction_intensities()
        _calculate_scan_positions_in_px()

        Additionally, it initializes an (Px,Py) array of 1.0j
        and a complex probe using the specified polar parameters.

        Parameters
        ----------
        diffraction_intensities_shape: Tuple[int,int], optional
            Pixel dimensions (Qx',Qy') of the resampled diffraction intensities
            If None, no resampling of diffraction intenstities is performed
        reshaping_method: str, optional
            Method to use for reshaping, either 'bin', 'bilinear', or 'fourier' (default)
        probe_roi_shape, (int,int), optional
            Padded diffraction intensities shape.
            If None, no padding is performed
        dp_mask: ndarray, optional
            Mask for datacube intensities (Qx,Qy)
        fit_function: str, optional
            2D fitting function for CoM fitting. One of 'plane','parabola','bezier_two'
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
        force_com_shifts: sequence of tuples of ndarrays (CoMx, CoMy)
            Amplitudes come from diffraction patterns shifted with
            the CoM in the upper left corner for each probe unless
            shift is overwritten.
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
        self: PtychographicReconstruction
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

        if self._magnetic_contribution_sign == "-+":
            self._recon_mode = 0
            self._num_measurements = 2
            magnetic_contribution_msg = (
                "Magnetic vector potential sign in first meaurement assumed to be negative.\n"
                "Magnetic vector potential sign in second meaurement assumed to be positive."
            )

        elif self._magnetic_contribution_sign == "-0+":
            self._recon_mode = 1
            self._num_measurements = 3
            magnetic_contribution_msg = (
                "Magnetic vector potential sign in first meaurement assumed to be negative.\n"
                "Magnetic vector potential assumed to be zero in second meaurement.\n"
                "Magnetic vector potential sign in third meaurement assumed to be positive."
            )

        elif self._magnetic_contribution_sign == "0+":
            self._recon_mode = 2
            self._num_measurements = 2
            magnetic_contribution_msg = (
                "Magnetic vector potential assumed to be zero in first meaurement.\n"
                "Magnetic vector potential sign in second meaurement assumed to be positive."
            )
        else:
            raise ValueError(
                f"magnetic_contribution_sign must be either '-+', '-0+', or '0+', not {self._magnetic_contribution_sign}"
            )

        if self._verbose:
            print(magnetic_contribution_msg)

        if len(self._datacube) != self._num_measurements:
            raise ValueError(
                f"datacube must be the same length as magnetic_contribution_sign, not length {len(self._datacube)}."
            )

        dc_shapes = [dc.shape for dc in self._datacube]
        if dc_shapes.count(dc_shapes[0]) != self._num_measurements:
            raise ValueError("datacube intensities must be the same size.")

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

            num_probes_per_tilt = np.insert(
                self._positions_mask.sum(axis=(-2, -1)), 0, 0
            )

        else:
            self._positions_mask = [None] * self._num_measurements
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
            force_com_shifts = [None] * self._num_measurements

        if self._scan_positions is None:
            self._scan_positions = [None] * self._num_measurements

        # Ensure plot_center_of_mass is not in kwargs
        kwargs.pop("plot_center_of_mass", None)

        # prepopulate relevant arrays
        self._mean_diffraction_intensity = []
        self._num_diffraction_patterns = num_probes_per_tilt.sum()
        self._cum_probes_per_tilt = np.cumsum(num_probes_per_tilt)
        self._positions_px_all = np.empty((self._num_diffraction_patterns, 2))

        # loop over DPs for preprocessing
        for index in tqdmnd(
            self._num_measurements,
            desc="Preprocessing data",
            unit="measurement",
            disable=not progress_bar,
        ):
            # preprocess datacube, vacuum and masks only for first measurement
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

            # estimate rotation / transpose using first measurement
            if index == 0:
                # silence warnings to play nice with progress bar
                verbose = self._verbose
                self._verbose = False

                (
                    self._rotation_best_rad,
                    self._rotation_best_transpose,
                    _com_x,
                    _com_y,
                    com_x,
                    com_y,
                ) = self._solve_for_center_of_mass_relative_rotation(
                    com_measured_x,
                    com_measured_y,
                    com_normalized_x,
                    com_normalized_y,
                    rotation_angles_deg=rotation_angles_deg,
                    plot_rotation=plot_rotation,
                    plot_center_of_mass=False,
                    maximize_divergence=maximize_divergence,
                    force_com_rotation=force_com_rotation,
                    force_com_transpose=force_com_transpose,
                    **kwargs,
                )
                self._verbose = verbose

            # corner-center amplitudes
            idx_start = self._cum_probes_per_tilt[index]
            idx_end = self._cum_probes_per_tilt[index + 1]
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

        # Object Initialization
        obj = self._initialize_object(
            self._object,
            self._positions_px_all,
            self._object_type,
        )

        if self._object is None:
            self._object = xp.full((2,) + obj.shape, obj)
        else:
            self._object = obj

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]

        # center probe positions
        self._positions_px_all = xp.asarray(self._positions_px_all, dtype=xp.float32)

        for index in range(self._num_measurements):
            idx_start = self._cum_probes_per_tilt[index]
            idx_end = self._cum_probes_per_tilt[index + 1]
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

        # overlaps
        idx_end = self._cum_probes_per_tilt[1]
        self._positions_px = self._positions_px_all[0:idx_end]
        self._positions_px_fractional = self._positions_px - xp.round(
            self._positions_px
        )
        shifted_probes = fft_shift(
            self._probes_all[0], self._positions_px_fractional, xp
        )
        probe_intensities = xp.abs(shifted_probes) ** 2
        probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)

        # initialize object_fov_mask
        if object_fov_mask is None:
            probe_overlap_blurred = self._gaussian_filter(probe_overlap, 1.0)
            self._object_fov_mask = asnumpy(
                probe_overlap_blurred > 0.25 * probe_overlap_blurred.max()
            )
        else:
            self._object_fov_mask = np.asarray(object_fov_mask)
        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        # plot probe overlaps
        if plot_probe_overlaps:
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
                asnumpy(probe_overlap),
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

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def _overlap_projection(self, current_object, shifted_probes):
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

        if self._object_type == "potential":
            complex_object = xp.exp(1j * current_object)
        else:
            complex_object = current_object

        object_patches = xp.empty(
            (self._num_measurements,) + shifted_probes.shape, dtype=xp.complex64
        )
        object_patches[0] = complex_object[
            0, self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]
        object_patches[1] = complex_object[
            1, self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]

        overlap_base = shifted_probes * object_patches[0]

        match (self._recon_mode, self._active_measurement_index):
            case (0, 0) | (1, 0):  # reverse
                overlap = overlap_base * xp.conj(object_patches[1])
            case (0, 1) | (1, 2) | (2, 1):  # forward
                overlap = overlap_base * object_patches[1]
            case (1, 1) | (2, 0):  # neutral
                overlap = overlap_base
            case _:
                raise ValueError()

        return shifted_probes, object_patches, overlap

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
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

        probe_conj = xp.conj(shifted_probes)  # P*
        electrostatic_conj = xp.conj(object_patches[0])  # V* = exp(-i v)

        probe_electrostatic_abs = xp.abs(shifted_probes * object_patches[0])
        probe_electrostatic_normalization = self._sum_overlapping_patches_bincounts(
            probe_electrostatic_abs**2
        )
        probe_electrostatic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_electrostatic_normalization) ** 2
            + (normalization_min * xp.max(probe_electrostatic_normalization)) ** 2
        )

        probe_magnetic_abs = xp.abs(shifted_probes * object_patches[1])
        probe_magnetic_normalization = self._sum_overlapping_patches_bincounts(
            probe_magnetic_abs**2
        )
        probe_magnetic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_magnetic_normalization) ** 2
            + (normalization_min * xp.max(probe_magnetic_normalization)) ** 2
        )

        if not fix_probe:
            electrostatic_magnetic_abs = xp.abs(object_patches[0] * object_patches[1])
            electrostatic_magnetic_normalization = xp.sum(
                electrostatic_magnetic_abs**2,
                axis=0,
            )
            electrostatic_magnetic_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * electrostatic_magnetic_normalization) ** 2
                + (normalization_min * xp.max(electrostatic_magnetic_normalization))
                ** 2
            )

            if self._recon_mode > 0:
                electrostatic_abs = xp.abs(object_patches[0])
                electrostatic_normalization = xp.sum(
                    electrostatic_abs**2,
                    axis=0,
                )
                electrostatic_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * electrostatic_normalization) ** 2
                    + (normalization_min * xp.max(electrostatic_normalization)) ** 2
                )

        match (self._recon_mode, self._active_measurement_index):
            case (0, 0) | (1, 0):  # reverse
                if self._object_type == "potential":
                    # -i exp(-i v) exp(i m) P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * object_patches[1]
                            * electrostatic_conj
                            * probe_conj
                            * exit_waves
                        )
                    )

                    # i exp(-i v) exp(i m) P*
                    magnetic_update = -electrostatic_update

                else:
                    # M P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        probe_conj * object_patches[1] * exit_waves
                    )

                    # V* P*
                    magnetic_update = xp.conj(
                        self._sum_overlapping_patches_bincounts(
                            probe_conj * electrostatic_conj * exit_waves
                        )
                    )

                current_object[0] += (
                    step_size * electrostatic_update * probe_magnetic_normalization
                )
                current_object[1] += (
                    step_size * magnetic_update * probe_electrostatic_normalization
                )

                if not fix_probe:
                    # M V*
                    current_probe += step_size * (
                        xp.sum(
                            electrostatic_conj * object_patches[1] * exit_waves,
                            axis=0,
                        )
                        * electrostatic_magnetic_normalization
                    )

            case (0, 1) | (1, 2) | (2, 1):  # forward
                magnetic_conj = xp.conj(object_patches[1])  # M* = exp(-i m)

                if self._object_type == "potential":
                    # -i exp(-i v) exp(-i m) P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * magnetic_conj
                            * electrostatic_conj
                            * probe_conj
                            * exit_waves
                        )
                    )

                    # -i exp(-i v) exp(-i m) P*
                    magnetic_update = electrostatic_update

                else:
                    # M* P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        probe_conj * magnetic_conj * exit_waves
                    )

                    # V* P*
                    magnetic_update = self._sum_overlapping_patches_bincounts(
                        probe_conj * electrostatic_conj * exit_waves
                    )

                current_object[0] += (
                    step_size * electrostatic_update * probe_magnetic_normalization
                )
                current_object[1] += (
                    step_size * magnetic_update * probe_electrostatic_normalization
                )

                if not fix_probe:
                    # M* V*
                    current_probe += step_size * (
                        xp.sum(
                            electrostatic_conj * magnetic_conj * exit_waves,
                            axis=0,
                        )
                        * electrostatic_magnetic_normalization
                    )

            case (1, 1) | (2, 0):  # neutral
                probe_abs = xp.abs(shifted_probes)
                probe_normalization = self._sum_overlapping_patches_bincounts(
                    probe_abs**2
                )
                probe_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * probe_normalization) ** 2
                    + (normalization_min * xp.max(probe_normalization)) ** 2
                )

                if self._object_type == "potential":
                    # -i exp(-i v) P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        xp.real(-1j * electrostatic_conj * probe_conj * exit_waves)
                    )

                else:
                    # P*
                    electrostatic_update = self._sum_overlapping_patches_bincounts(
                        probe_conj * exit_waves
                    )

                current_object[0] += (
                    step_size * electrostatic_update * probe_normalization
                )

                if not fix_probe:
                    # V*
                    current_probe += step_size * (
                        xp.sum(
                            electrostatic_conj * exit_waves,
                            axis=0,
                        )
                        * electrostatic_normalization
                    )

            case _:
                raise ValueError()

        return current_object, current_probe

    def _object_constraints(
        self,
        current_object,
        pure_phase_object,
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
        tv_denoise_weight,
        tv_denoise_inner_iter,
        object_positivity,
        shrinkage_rad,
        object_mask,
        **kwargs,
    ):
        """MagneticObjectNDConstraints wrapper function"""

        # smoothness
        if gaussian_filter:
            current_object[0] = self._object_gaussian_constraint(
                current_object[0], gaussian_filter_sigma_e, pure_phase_object
            )
            current_object[1] = self._object_gaussian_constraint(
                current_object[1], gaussian_filter_sigma_m, True
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
        if tv_denoise:
            current_object[0] = self._object_denoise_tv_pylops(
                current_object[0], tv_denoise_weight, tv_denoise_inner_iter
            )

        # L1-norm pushing vacuum to zero
        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object[0] = self._object_shrinkage_constraint(
                current_object[0],
                shrinkage_rad,
                object_mask,
            )

        # amplitude threshold (complex) or positivity (potential)
        if self._object_type == "complex":
            current_object[0] = self._object_threshold_constraint(
                current_object[0], pure_phase_object
            )
            current_object[1] = self._object_threshold_constraint(
                current_object[1], True
            )
        elif object_positivity:
            current_object[0] = self._object_positivity_constraint(current_object[0])

        return current_object

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
        pure_phase_object_iter: int = 0,
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
        tv_denoise_iter: int = np.inf,
        tv_denoise_weight: float = None,
        tv_denoise_inner_iter: float = 40,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
        switch_object_iter: int = np.inf,
        store_iterations: bool = False,
        collective_measurement_updates: bool = True,
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
        pure_phase_object_iter: float, optional
            Number of iterations where object amplitude is set to unity
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
        tv_denoise_iter: int, optional
            Number of iterations to run using tv denoise filter on object
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
        switch_object_iter: int, optional
            Iteration to switch object type between 'complex' and 'potential' or between
            'potential' and 'complex'
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration
        collective_measurement_updates: bool
            if True perform collective updates for all measurements
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        reset: bool, optional
            If True, previous reconstructions are ignored

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        asnumpy = self._asnumpy
        xp = self._xp

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
                "Magnetic ptychography is currently only implemented for gradient descent."
            )

        if self._verbose:
            self._report_reconstruction_summary(
                max_iter,
                switch_object_iter,
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

        if max_batch_size is not None:
            xp.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        # initialization
        self._reset_reconstruction(store_iterations, reset, use_projection_scheme)

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

            if a0 == switch_object_iter:
                if self._object_type == "potential":
                    self._object_type = "complex"
                    self._object = xp.exp(1j * self._object)
                else:
                    self._object_type = "potential"
                    self._object = xp.angle(self._object)

            if collective_measurement_updates:
                collective_object = xp.zeros_like(self._object)

            measurement_indices = np.arange(self._num_measurements)
            np.random.shuffle(measurement_indices)

            for measurement_index in measurement_indices:
                self._active_measurement_index = measurement_index

                measurement_error = 0.0

                _probe = self._probes_all[self._active_measurement_index]
                _probe_initial_aperture = self._probes_all_initial_aperture[
                    self._active_measurement_index
                ]

                start_idx = self._cum_probes_per_tilt[self._active_measurement_index]
                end_idx = self._cum_probes_per_tilt[self._active_measurement_index + 1]

                num_diffraction_patterns = end_idx - start_idx
                shuffled_indices = np.arange(num_diffraction_patterns)
                unshuffled_indices = np.zeros_like(shuffled_indices)

                # randomize
                if not use_projection_scheme:
                    np.random.shuffle(shuffled_indices)

                unshuffled_indices[shuffled_indices] = np.arange(
                    num_diffraction_patterns
                )

                positions_px = self._positions_px_all[start_idx:end_idx].copy()[
                    shuffled_indices
                ]
                initial_positions_px = self._positions_px_initial_all[
                    start_idx:end_idx
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

                    amplitudes = self._amplitudes[start_idx:end_idx][
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
                        self._object,
                        _probe,
                        amplitudes,
                        self._exit_waves,
                        use_projection_scheme=use_projection_scheme,
                        projection_a=projection_a,
                        projection_b=projection_b,
                        projection_c=projection_c,
                    )

                    # adjoint operator
                    object_update, _probe = self._adjoint(
                        self._object.copy(),
                        _probe,
                        object_patches,
                        shifted_probes,
                        self._exit_waves,
                        use_projection_scheme=use_projection_scheme,
                        step_size=step_size,
                        normalization_min=normalization_min,
                        fix_probe=a0 < fix_probe_iter,
                    )

                    object_update -= self._object

                    # position correction
                    if a0 >= fix_positions_iter:
                        positions_px[start:end] = self._position_correction(
                            self._object,
                            shifted_probes,
                            overlap,
                            amplitudes,
                            self._positions_px,
                            self._positions_px_initial,
                            positions_step_size,
                            max_position_update_distance,
                            max_position_total_distance,
                        )

                    measurement_error += batch_error

                if collective_measurement_updates:
                    collective_object += object_update
                else:
                    self._object += object_update

                # Normalize Error
                measurement_error /= (
                    self._mean_diffraction_intensity[self._active_measurement_index]
                    * num_diffraction_patterns
                )
                error += measurement_error

                # constraints
                self._positions_px_all[start_idx:end_idx] = positions_px.copy()[
                    unshuffled_indices
                ]

                if collective_measurement_updates:
                    # probe and positions
                    _probe = self._probe_constraints(
                        _probe,
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
                    )

                    self._positions_px_all[
                        start_idx:end_idx
                    ] = self._positions_constraints(
                        self._positions_px_all[start_idx:end_idx],
                        fix_positions=a0 < fix_positions_iter,
                        global_affine_transformation=global_affine_transformation,
                    )

                else:
                    # object, probe, and positions
                    (
                        self._object,
                        _probe,
                        self._positions_px_all[start_idx:end_idx],
                    ) = self._constraints(
                        self._object,
                        _probe,
                        self._positions_px_all[start_idx:end_idx],
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
                        tv_denoise=a0 < tv_denoise_iter
                        and tv_denoise_weight is not None,
                        tv_denoise_weight=tv_denoise_weight,
                        tv_denoise_inner_iter=tv_denoise_inner_iter,
                        object_positivity=object_positivity,
                        shrinkage_rad=shrinkage_rad,
                        object_mask=self._object_fov_mask_inverse
                        if fix_potential_baseline
                        and self._object_fov_mask_inverse.sum() > 0
                        else None,
                        pure_phase_object=a0 < pure_phase_object_iter
                        and self._object_type == "complex",
                    )

            # Normalize Error Over Tilts
            error /= self._num_measurements

            if collective_measurement_updates:
                self._object += collective_object / self._num_measurements

                # object only
                self._object = self._object_constraints(
                    self._object,
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
                    tv_denoise=a0 < tv_denoise_iter and tv_denoise_weight is not None,
                    tv_denoise_weight=tv_denoise_weight,
                    tv_denoise_inner_iter=tv_denoise_inner_iter,
                    object_positivity=object_positivity,
                    shrinkage_rad=shrinkage_rad,
                    object_mask=self._object_fov_mask_inverse
                    if fix_potential_baseline
                    and self._object_fov_mask_inverse.sum() > 0
                    else None,
                    pure_phase_object=a0 < pure_phase_object_iter
                    and self._object_type == "complex",
                )

            self.error_iterations.append(error.item())

            if store_iterations:
                self.object_iterations.append(asnumpy(self._object.copy()))
                self.probe_iterations.append(
                    [
                        asnumpy(self._return_centered_probe(pr.copy()))
                        for pr in self._probes_all
                    ]
                )

        # store result
        self.object = asnumpy(self._object)
        self.probe = self.probe_centered
        self.error = error.item()

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def _visualize_last_iteration_figax(
        self,
        fig,
        object_ax,
        convergence_ax: None,
        cbar: bool,
        padding: int = 0,
        **kwargs,
    ):
        """
        Displays last reconstructed object on a given fig/ax.

        Parameters
        --------
        fig: Figure
            Matplotlib figure object_ax lives in
        object_ax: Axes
            Matplotlib axes to plot reconstructed object in
        convergence_ax: Axes, optional
            Matplotlib axes to plot convergence plot in
        cbar: bool, optional
            If true, displays a colorbar
        padding : int, optional
            Pixels to pad by post rotating-cropping object
        """
        cmap = kwargs.pop("cmap", "magma")

        if self._object_type == "complex":
            obj = np.angle(self.object[0])
        else:
            obj = self.object[0]

        rotated_object = self._crop_rotate_object_fov(obj, padding=padding)
        rotated_shape = rotated_object.shape

        extent = [
            0,
            self.sampling[1] * rotated_shape[1],
            self.sampling[0] * rotated_shape[0],
            0,
        ]

        im = object_ax.imshow(
            rotated_object,
            extent=extent,
            cmap=cmap,
            **kwargs,
        )

        if cbar:
            divider = make_axes_locatable(object_ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            fig.add_axes(ax_cb)
            fig.colorbar(im, cax=ax_cb)

        if convergence_ax is not None and hasattr(self, "error_iterations"):
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            errors = self.error_iterations
            convergence_ax.semilogy(np.arange(len(errors)), errors, **kwargs)

    def _visualize_last_iteration(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        plot_fourier_probe: bool,
        remove_initial_probe_aberrations: bool,
        padding: int,
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
        padding : int, optional
            Pixels to pad by post rotating-cropping object
        """
        figsize = kwargs.pop("figsize", (12, 5))
        cmap_e = kwargs.pop("cmap_e", "magma")
        cmap_m = kwargs.pop("cmap_m", "PuOr")

        if self._object_type == "complex":
            obj_e = np.angle(self.object[0])
            obj_m = self.object[1]
        else:
            obj_e, obj_m = self.object

        rotated_electrostatic = self._crop_rotate_object_fov(obj_e, padding=padding)
        rotated_magnetic = self._crop_rotate_object_fov(obj_m, padding=padding)
        rotated_shape = rotated_electrostatic.shape

        min_e = rotated_electrostatic.min()
        max_e = rotated_electrostatic.max()
        max_m = np.abs(rotated_magnetic).max()
        min_m = -max_m

        vmin_e = kwargs.pop("vmin_e", min_e)
        vmax_e = kwargs.pop("vmax_e", max_e)
        vmin_m = kwargs.pop("vmin_m", min_m)
        vmax_m = kwargs.pop("vmax_m", max_m)

        chroma_boost = kwargs.pop("chroma_boost", 1)

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
                    ncols=3,
                    nrows=2,
                    height_ratios=[4, 1],
                    hspace=0.15,
                    width_ratios=[
                        1,
                        1,
                        (probe_extent[1] / probe_extent[2]) / (extent[1] / extent[2]),
                    ],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=2, nrows=2, height_ratios=[4, 1], hspace=0.15)
        else:
            if plot_probe or plot_fourier_probe:
                spec = GridSpec(
                    ncols=3,
                    nrows=1,
                    width_ratios=[
                        1,
                        1,
                        (probe_extent[1] / probe_extent[2]) / (extent[1] / extent[2]),
                    ],
                    wspace=0.35,
                )
            else:
                spec = GridSpec(ncols=2, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)

        if plot_probe or plot_fourier_probe:
            # Electrostatic Object
            ax = fig.add_subplot(spec[0, 0])
            im = ax.imshow(
                rotated_electrostatic,
                extent=extent,
                cmap=cmap_e,
                vmin=vmin_e,
                vmax=vmax_e,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            if self._object_type == "potential":
                ax.set_title("Reconstructed electrostatic potential")
            elif self._object_type == "complex":
                ax.set_title("Reconstructed electrostatic phase")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Magnetic Object
            ax = fig.add_subplot(spec[0, 1])
            im = ax.imshow(
                rotated_magnetic,
                extent=extent,
                cmap=cmap_m,
                vmin=vmin_m,
                vmax=vmax_m,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed magnetic potential")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Probe
            ax = fig.add_subplot(spec[0, 2])
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
                    self.probe, power=2, chroma_boost=chroma_boost
                )
                ax.set_title("Reconstructed probe intensity")
                ax.set_ylabel("x [A]")
                ax.set_xlabel("y [A]")

            im = ax.imshow(
                probe_array,
                extent=probe_extent,
            )

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                add_colorbar_arg(ax_cb, chroma_boost=chroma_boost)

        else:
            # Electrostatic Object
            ax = fig.add_subplot(spec[0, 0])
            im = ax.imshow(
                rotated_electrostatic,
                extent=extent,
                cmap=cmap_e,
                vmin=vmin_e,
                vmax=vmax_e,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            if self._object_type == "potential":
                ax.set_title("Reconstructed electrostatic potential")
            elif self._object_type == "complex":
                ax.set_title("Reconstructed electrostatic phase")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Magnetic Object
            ax = fig.add_subplot(spec[0, 1])
            im = ax.imshow(
                rotated_magnetic,
                extent=extent,
                cmap=cmap_m,
                vmin=vmin_m,
                vmax=vmax_m,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Reconstructed magnetic potential")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "error_iterations"):
            errors = np.array(self.error_iterations)
            ax = fig.add_subplot(spec[1, :])
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
        padding: int,
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
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool, optional
            If true, the reconstructed complex probe is displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        padding : int, optional
            Pixels to pad by post rotating-cropping object
        """
        raise NotImplementedError()

    def visualize(
        self,
        fig=None,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        plot_probe: bool = True,
        plot_fourier_probe: bool = False,
        remove_initial_probe_aberrations: bool = False,
        cbar: bool = True,
        padding: int = 0,
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
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool, optional
            If true, the reconstructed complex probe is displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        padding : int, optional
            Pixels to pad by post rotating-cropping object

        Returns
        --------
        self: PtychographicReconstruction
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
                padding=padding,
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
                padding=padding,
                **kwargs,
            )

        return self

    @property
    def self_consistency_errors(self):
        """Compute the self-consistency errors for each probe position"""

        xp = self._xp
        asnumpy = self._asnumpy

        # Re-initialize fractional positions and vector patches, max_batch_size = None
        self._positions_px_fractional = self._positions_px - xp.round(
            self._positions_px
        )

        (
            self._vectorized_patch_indices_row,
            self._vectorized_patch_indices_col,
        ) = self._extract_vectorized_patch_indices()

        # Overlaps
        _, _, overlap = self._warmup_overlap_projection(self._object, self._probe)
        fourier_overlap = xp.fft.fft2(overlap[0])

        # Normalized mean-squared errors
        error = xp.sum(
            xp.abs(self._amplitudes[0] - xp.abs(fourier_overlap)) ** 2, axis=(-2, -1)
        )
        error /= self._mean_diffraction_intensity

        return asnumpy(error)

    def _return_self_consistency_errors(
        self,
        max_batch_size=None,
    ):
        """Compute the self-consistency errors for each probe position"""

        xp = self._xp
        asnumpy = self._asnumpy

        # Batch-size
        if max_batch_size is None:
            max_batch_size = self._num_diffraction_patterns

        # Re-initialize fractional positions and vector patches
        errors = np.array([])
        positions_px = self._positions_px.copy()

        for start, end in generate_batches(
            self._num_diffraction_patterns, max_batch=max_batch_size
        ):
            # batch indices
            self._positions_px = positions_px[start:end]
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            (
                self._vectorized_patch_indices_row,
                self._vectorized_patch_indices_col,
            ) = self._extract_vectorized_patch_indices()
            amplitudes = self._amplitudes[0][start:end]

            # Overlaps
            _, _, overlap = self._warmup_overlap_projection(self._object, self._probe)
            fourier_overlap = xp.fft.fft2(overlap[0])

            # Normalized mean-squared errors
            batch_errors = xp.sum(
                xp.abs(amplitudes - xp.abs(fourier_overlap)) ** 2, axis=(-2, -1)
            )
            errors = np.hstack((errors, batch_errors))

        self._positions_px = positions_px.copy()
        errors /= self._mean_diffraction_intensity

        return asnumpy(errors)

    def _return_projected_cropped_potential(
        self,
    ):
        """Utility function to accommodate multiple classes"""
        if self._object_type == "complex":
            projected_cropped_potential = np.angle(self.object_cropped[0])
        else:
            projected_cropped_potential = self.object_cropped[0]

        return projected_cropped_potential

    @property
    def object_cropped(self):
        """Cropped and rotated object"""

        obj_e, obj_m = self._object
        obj_e = self._crop_rotate_object_fov(obj_e)
        obj_m = self._crop_rotate_object_fov(obj_m)
        return (obj_e, obj_m)
