"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely multislice ptychography.
"""

import warnings
from typing import Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg, show_complex

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = None

from emdfile import Custom, tqdmnd
from py4DSTEM import DataCube
from py4DSTEM.process.phase.iterative_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.iterative_ptychographic_constraints import (
    Object2p5DConstraintsMixin,
    ObjectNDConstraintsMixin,
    PositionsConstraintsMixin,
    ProbeConstraintsMixin,
    ProbeMixedConstraintsMixin,
)
from py4DSTEM.process.phase.iterative_ptychographic_methods import (
    Object2p5DMethodsMixin,
    Object2p5DProbeMixedMethodsMixin,
    ObjectNDMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ObjectNDProbeMixedMethodsMixin,
    ProbeMethodsMixin,
    ProbeMixedMethodsMixin,
)
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
)

warnings.simplefilter(action="always", category=UserWarning)


class MixedstateMultislicePtychographicReconstruction(
    PositionsConstraintsMixin,
    ProbeMixedConstraintsMixin,
    ProbeConstraintsMixin,
    Object2p5DConstraintsMixin,
    ObjectNDConstraintsMixin,
    Object2p5DProbeMixedMethodsMixin,
    ObjectNDProbeMixedMethodsMixin,
    ObjectNDProbeMethodsMixin,
    ProbeMixedMethodsMixin,
    ProbeMethodsMixin,
    Object2p5DMethodsMixin,
    ObjectNDMethodsMixin,
    PtychographicReconstruction,
):
    """
    Mixed-State Multislice Ptychographic Reconstruction Class.

    Diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions      : (N,Sx,Sy)
    Reconstructed object dimensions     : (T,Px,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our N probes
    and (Px,Py) is the padded-object size we position our ROI around in
    each of the T slices.

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions in eV
    num_probes: int, optional
        Number of mixed-state probes
    num_slices: int
        Number of slices to use in the forward model
    slice_thicknesses: float or Sequence[float]
        Slice thicknesses in angstroms. If float, all slices are assigned the same thickness
    datacube: DataCube, optional
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
    theta_x: float
        x tilt of propagator (in degrees)
    theta_y: float
        y tilt of propagator (in degrees)
    middle_focus: bool
        if True, adds half the sample thickness to the defocus
    object_type: str, optional
        The object can be reconstructed as a real potential ('potential') or a complex
        object ('complex')
    positions_mask: np.ndarray, optional
        Boolean real space mask to select positions in datacube to skip for reconstruction
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    name: str, optional
        Class name
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = ("_num_probes", "_num_slices", "_slice_thicknesses")

    def __init__(
        self,
        energy: float,
        num_slices: int,
        slice_thicknesses: Union[float, Sequence[float]],
        num_probes: int = None,
        datacube: DataCube = None,
        semiangle_cutoff: float = None,
        semiangle_cutoff_pixels: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        object_padding_px: Tuple[int, int] = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        initial_scan_positions: np.ndarray = None,
        theta_x: float = 0,
        theta_y: float = 0,
        middle_focus: bool = False,
        object_type: str = "complex",
        positions_mask: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        name: str = "multi-slice_ptychographic_reconstruction",
        **kwargs,
    ):
        Custom.__init__(self, name=name)

        if initial_probe_guess is None or isinstance(initial_probe_guess, ComplexProbe):
            if num_probes is None:
                raise ValueError(
                    (
                        "If initial_probe_guess is None, or a ComplexProbe object, "
                        "num_probes must be specified."
                    )
                )
        else:
            if len(initial_probe_guess.shape) != 3:
                raise ValueError(
                    "Specified initial_probe_guess must have dimensions (N,Sx,Sy)."
                )
            num_probes = initial_probe_guess.shape[0]

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

        if np.isscalar(slice_thicknesses):
            mean_slice_thickness = slice_thicknesses
        else:
            mean_slice_thickness = np.mean(slice_thicknesses)

        if middle_focus:
            if "defocus" in kwargs:
                kwargs["defocus"] += mean_slice_thickness * num_slices / 2
            elif "C10" in kwargs:
                kwargs["C10"] -= mean_slice_thickness * num_slices / 2
            elif polar_parameters is not None and "defocus" in polar_parameters:
                polar_parameters["defocus"] = (
                    polar_parameters["defocus"] + mean_slice_thickness * num_slices / 2
                )
            elif polar_parameters is not None and "C10" in polar_parameters:
                polar_parameters["C10"] = (
                    polar_parameters["C10"] - mean_slice_thickness * num_slices / 2
                )

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))

        if polar_parameters is None:
            polar_parameters = {}

        polar_parameters.update(kwargs)
        self._set_polar_parameters(polar_parameters)

        slice_thicknesses = np.array(slice_thicknesses)
        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, num_slices - 1)
        elif slice_thicknesses.shape[0] != (num_slices - 1):
            raise ValueError(
                (
                    f"slice_thicknesses must have length {num_slices - 1}, "
                    f"not {slice_thicknesses.shape[0]}."
                )
            )

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
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._semiangle_cutoff_pixels = semiangle_cutoff_pixels
        self._rolloff = rolloff
        self._object_type = object_type
        self._positions_mask = positions_mask
        self._object_padding_px = object_padding_px
        self._verbose = verbose
        self._device = device
        self._preprocessed = False

        # Class-specific Metadata
        self._num_probes = num_probes
        self._num_slices = num_slices
        self._slice_thicknesses = slice_thicknesses
        self._theta_x = theta_x
        self._theta_y = theta_y

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "fourier",
        probe_roi_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_center_of_mass: str = "default",
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

        Additionally, it initializes an (T,Px,Py) array of 1.0j
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
        self: MixedstateMultislicePtychographicReconstruction
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

        # preprocess datacube
        (
            self._datacube,
            self._vacuum_probe_intensity,
            self._dp_mask,
            force_com_shifts,
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube,
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            probe_roi_shape=self._probe_roi_shape,
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            dp_mask=self._dp_mask,
            com_shifts=force_com_shifts,
        )

        # calibrations
        self._intensities = self._extract_intensities_and_calibrations_from_datacube(
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
            self._intensities,
            dp_mask=self._dp_mask,
            fit_function=fit_function,
            com_shifts=force_com_shifts,
        )

        # estimate rotation / transpose
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

        # corner-center amplitudes
        (
            self._amplitudes,
            self._mean_diffraction_intensity,
            self._crop_mask,
        ) = self._normalize_diffraction_intensities(
            self._intensities,
            self._com_fitted_x,
            self._com_fitted_y,
            self._positions_mask,
            crop_patterns,
        )

        # explicitly delete namespace
        self._num_diffraction_patterns = self._amplitudes.shape[0]
        self._region_of_interest_shape = np.array(self._amplitudes.shape[-2:])
        del self._intensities

        # initialize probe positions
        (
            self._positions_px,
            self._object_padding_px,
        ) = self._calculate_scan_positions_in_pixels(
            self._scan_positions,
            self._positions_mask,
            self._object_padding_px,
        )

        # initialize object
        self._object = self._initialize_object(
            self._object,
            self._num_slices,
            self._positions_px,
            self._object_type,
        )

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]

        # center probe positions
        self._positions_px = xp.asarray(self._positions_px, dtype=xp.float32)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self._positions_px -= self._positions_px_com - xp.array(self._object_shape) / 2
        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self._positions_px_fractional = self._positions_px - xp.round(
            self._positions_px
        )

        self._positions_px_initial = self._positions_px.copy()
        self._positions_initial = self._positions_px_initial.copy()
        self._positions_initial[:, 0] *= self.sampling[0]
        self._positions_initial[:, 1] *= self.sampling[1]

        # set vectorized patches
        (
            self._vectorized_patch_indices_row,
            self._vectorized_patch_indices_col,
        ) = self._extract_vectorized_patch_indices()

        # initialize probe
        self._probe, self._semiangle_cutoff = self._initialize_probe(
            self._probe,
            self._vacuum_probe_intensity,
            self._mean_diffraction_intensity,
            self._semiangle_cutoff,
            crop_patterns,
        )

        self._probe_initial = self._probe.copy()
        self._probe_initial_aperture = xp.abs(xp.fft.fft2(self._probe))

        # initialize aberrations
        self._known_aberrations_array = ComplexProbe(
            energy=self._energy,
            gpts=self._region_of_interest_shape,
            sampling=self.sampling,
            parameters=self._polar_parameters,
            device=self._device,
        )._evaluate_ctf()

        # precompute propagator arrays
        self._propagator_arrays = self._precompute_propagator_arrays(
            self._region_of_interest_shape,
            self.sampling,
            self._energy,
            self._slice_thicknesses,
            self._theta_x,
            self._theta_y,
        )

        # overlaps
        shifted_probes = fft_shift(self._probe[0], self._positions_px_fractional, xp)
        probe_intensities = xp.abs(shifted_probes) ** 2
        probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)

        if object_fov_mask is None:
            probe_overlap_blurred = self._gaussian_filter(probe_overlap, 1.0)
            self._object_fov_mask = asnumpy(
                probe_overlap_blurred > 0.25 * probe_overlap_blurred.max()
            )
        else:
            self._object_fov_mask = np.asarray(object_fov_mask)
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
            propagated_probe = self._probe[0].copy()

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
            ax1.set_title("Initial probe[0] intensity")

            ax2.imshow(
                complex_propagated_rgb,
                extent=probe_extent,
            )

            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="5%", pad="2.5%")
            add_colorbar_arg(cax2, chroma_boost=chroma_boost)
            ax2.set_ylabel("x [A]")
            ax2.set_xlabel("y [A]")
            ax2.set_title("Propagated probe[0] intensity")

            ax3.imshow(
                asnumpy(probe_overlap),
                extent=extent,
                cmap="Greys_r",
            )
            ax3.scatter(
                self.positions[:, 1],
                self.positions[:, 0],
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

    def _position_correction(
        self,
        current_object,
        current_probe,
        transmitted_probes,
        amplitudes,
        current_positions,
        positions_step_size,
        constrain_position_distance,
    ):
        """
        Position correction using estimated intensity gradient.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe:np.ndarray
            fractionally-shifted probes
        transmitted_probes: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions
        amplitudes: np.ndarray
            Measured amplitudes
        current_positions: np.ndarray
            Current positions estimate
        positions_step_size: float
            Positions step size
        constrain_position_distance: float
            Distance to constrain position correction within original
            field of view in A

        Returns
        --------
        updated_positions: np.ndarray
            Updated positions estimate
        """

        xp = self._xp

        # Intensity gradient
        exit_waves_fft = xp.fft.fft2(transmitted_probes)
        exit_waves_fft_conj = xp.conj(exit_waves_fft)
        estimated_intensity = xp.abs(exit_waves_fft) ** 2
        measured_intensity = amplitudes**2

        flat_shape = (transmitted_probes.shape[0], -1)
        difference_intensity = (measured_intensity - estimated_intensity).reshape(
            flat_shape
        )

        # Computing perturbed exit waves one at a time to save on memory

        if self._object_type == "potential":
            complex_object = xp.exp(1j * current_object)
        else:
            complex_object = current_object

        # dx
        obj_rolled_patches = complex_object[
            :,
            (self._vectorized_patch_indices_row + 1) % self._object_shape[0],
            self._vectorized_patch_indices_col,
        ]

        propagated_probes_perturbed = xp.empty_like(obj_rolled_patches)
        propagated_probes_perturbed[0] = fft_shift(
            current_probe, self._positions_px_fractional, xp
        )

        for s in range(self._num_slices):
            # transmit
            transmitted_probes_perturbed = (
                obj_rolled_patches[s] * propagated_probes_perturbed[s]
            )

            # propagate
            if s + 1 < self._num_slices:
                propagated_probes_perturbed[s + 1] = self._propagate_array(
                    transmitted_probes_perturbed, self._propagator_arrays[s]
                )

        exit_waves_dx_fft = exit_waves_fft - xp.fft.fft2(transmitted_probes_perturbed)

        # dy
        obj_rolled_patches = complex_object[
            :,
            self._vectorized_patch_indices_row,
            (self._vectorized_patch_indices_col + 1) % self._object_shape[1],
        ]

        propagated_probes_perturbed = xp.empty_like(obj_rolled_patches)
        propagated_probes_perturbed[0] = fft_shift(
            current_probe, self._positions_px_fractional, xp
        )

        for s in range(self._num_slices):
            # transmit
            transmitted_probes_perturbed = (
                obj_rolled_patches[s] * propagated_probes_perturbed[s]
            )

            # propagate
            if s + 1 < self._num_slices:
                propagated_probes_perturbed[s + 1] = self._propagate_array(
                    transmitted_probes_perturbed, self._propagator_arrays[s]
                )

        exit_waves_dy_fft = exit_waves_fft - xp.fft.fft2(transmitted_probes_perturbed)

        partial_intensity_dx = 2 * xp.real(
            exit_waves_dx_fft * exit_waves_fft_conj
        ).reshape(flat_shape)
        partial_intensity_dy = 2 * xp.real(
            exit_waves_dy_fft * exit_waves_fft_conj
        ).reshape(flat_shape)

        coefficients_matrix = xp.dstack((partial_intensity_dx, partial_intensity_dy))

        # positions_update = xp.einsum(
        #    "idk,ik->id", xp.linalg.pinv(coefficients_matrix), difference_intensity
        # )

        coefficients_matrix_T = coefficients_matrix.conj().swapaxes(-1, -2)
        positions_update = (
            xp.linalg.inv(coefficients_matrix_T @ coefficients_matrix)
            @ coefficients_matrix_T
            @ difference_intensity[..., None]
        )

        if constrain_position_distance is not None:
            constrain_position_distance /= xp.sqrt(
                self.sampling[0] ** 2 + self.sampling[1] ** 2
            )
            x1 = (current_positions - positions_step_size * positions_update[..., 0])[
                :, 0
            ]
            y1 = (current_positions - positions_step_size * positions_update[..., 0])[
                :, 1
            ]
            x0 = self._positions_px_initial[:, 0]
            y0 = self._positions_px_initial[:, 1]
            if self._rotation_best_transpose:
                x0, y0 = xp.array([y0, x0])
                x1, y1 = xp.array([y1, x1])

            if self._rotation_best_rad is not None:
                rotation_angle = self._rotation_best_rad
                x0, y0 = x0 * xp.cos(-rotation_angle) + y0 * xp.sin(
                    -rotation_angle
                ), -x0 * xp.sin(-rotation_angle) + y0 * xp.cos(-rotation_angle)
                x1, y1 = x1 * xp.cos(-rotation_angle) + y1 * xp.sin(
                    -rotation_angle
                ), -x1 * xp.sin(-rotation_angle) + y1 * xp.cos(-rotation_angle)

            outlier_ind = (x1 > (xp.max(x0) + constrain_position_distance)) + (
                x1 < (xp.min(x0) - constrain_position_distance)
            ) + (y1 > (xp.max(y0) + constrain_position_distance)) + (
                y1 < (xp.min(y0) - constrain_position_distance)
            ) > 0

            positions_update[..., 0][outlier_ind] = 0

        current_positions -= positions_step_size * positions_update[..., 0]

        return current_positions

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
        kz_regularization_filter,
        kz_regularization_gamma,
        identical_slices,
        object_positivity,
        shrinkage_rad,
        object_mask,
        pure_phase_object,
        tv_denoise_chambolle,
        tv_denoise_weight_chambolle,
        tv_denoise_pad_chambolle,
        tv_denoise,
        tv_denoise_weights,
        tv_denoise_inner_iter,
        orthogonalize_probe,
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
        fix_probe_aperture: bool
            If True, probe Fourier amplitude is replaced by initial_probe_aperture
        initial_probe_aperture: np.ndarray
            Initial probe aperture to use in replacing probe Fourier amplitude
        fix_positions: bool
            If True, positions are not updated
        gaussian_filter: bool
            If True, applies real-space gaussian filter in A
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel
        butterworth_filter: bool
            If True, applies fourier-space butterworth filter
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        kz_regularization_filter: bool
            If True, applies fourier-space arctan regularization filter
        kz_regularization_gamma: float
            Slice regularization strength
        identical_slices: bool
            If True, forces all object slices to be identical
        object_positivity: bool
            If True, forces object to be positive
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        object_mask: np.ndarray (boolean)
            If not None, used to calculate additional shrinkage using masked-mean of object
        pure_phase_object: bool
            If True, object amplitude is set to unity
        tv_denoise_chambolle: bool
            If True, performs TV denoising along z
        tv_denoise_weight_chambolle: float
            weight of tv denoising constraint
        tv_denoise_pad_chambolle: int
            If not None, pads object at top and bottom with this many zeros before applying denoising
        tv_denoise: bool
            If True, applies TV denoising on object
        tv_denoise_weights: [float,float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        orthogonalize_probe: bool
            If True, probe will be orthogonalized

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
                current_object, gaussian_filter_sigma, pure_phase_object
            )

        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )

        if identical_slices:
            current_object = self._object_identical_slices_constraint(current_object)
        elif kz_regularization_filter:
            current_object = self._object_kz_regularization_constraint(
                current_object,
                kz_regularization_gamma,
                z_padding=1,
            )
        elif tv_denoise:
            current_object = self._object_denoise_tv_pylops(
                current_object,
                tv_denoise_weights,
                tv_denoise_inner_iter,
                z_padding=1,
            )
        elif tv_denoise_chambolle:
            current_object = self._object_denoise_tv_chambolle(
                current_object,
                tv_denoise_weight_chambolle,
                axis=0,
                padding=tv_denoise_pad_chambolle,
            )

        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object = self._object_shrinkage_constraint(
                current_object,
                shrinkage_rad,
                object_mask,
            )

        if self._object_type == "complex":
            current_object = self._object_threshold_constraint(
                current_object, pure_phase_object
            )
        elif object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        if fix_com:
            current_probe = self._probe_center_of_mass_constraint(current_probe)

        # These constraints don't _really_ make sense for mixed-state
        if fix_probe_aperture:
            raise NotImplementedError()
        elif constrain_probe_fourier_amplitude:
            raise NotImplementedError()
        if fit_probe_aberrations:
            raise NotImplementedError()
        if constrain_probe_amplitude:
            raise NotImplementedError()

        if orthogonalize_probe:
            current_probe = self._probe_orthogonalization_constraint(current_probe)

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
        orthogonalize_probe: bool = True,
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
        gaussian_filter_sigma: float = None,
        gaussian_filter_iter: int = np.inf,
        fit_probe_aberrations_iter: int = 0,
        fit_probe_aberrations_max_angular_order: int = 4,
        fit_probe_aberrations_max_radial_order: int = 4,
        butterworth_filter_iter: int = np.inf,
        q_lowpass: float = None,
        q_highpass: float = None,
        butterworth_order: float = 2,
        kz_regularization_filter_iter: int = np.inf,
        kz_regularization_gamma: Union[float, np.ndarray] = None,
        identical_slices_iter: int = 0,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
        pure_phase_object_iter: int = 0,
        tv_denoise_iter_chambolle=np.inf,
        tv_denoise_weight_chambolle=None,
        tv_denoise_pad_chambolle=1,
        tv_denoise_iter=np.inf,
        tv_denoise_weights=None,
        tv_denoise_inner_iter=40,
        switch_object_iter: int = np.inf,
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
        kz_regularization_filter_iter: int, optional
            Number of iterations to run using kz regularization filter
        kz_regularization_gamma, float, optional
            kz regularization strength
        identical_slices_iter: int, optional
            Number of iterations to run using identical slices
        object_positivity: bool, optional
            If True, forces object to be positive
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        fix_potential_baseline: bool
            If true, the potential mean outside the FOV is forced to zero at each iteration
        pure_phase_object_iter: int, optional
            Number of iterations where object amplitude is set to unity
        tv_denoise_iter_chambolle: bool
            Number of iterations with TV denoisining
        tv_denoise_weight_chambolle: float
            weight of tv denoising constraint
        tv_denoise_pad_chambolle: int
            If not None, pads object at top and bottom with this many zeros before applying denoising
        tv_denoise: bool
            If True, applies TV denoising on object
        tv_denoise_weights: [float,float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        switch_object_iter: int, optional
            Iteration to switch object type between 'complex' and 'potential' or between
            'potential' and 'complex'
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        reset: bool, optional
            If True, previous reconstructions are ignored

        Returns
        --------
        self: MultislicePtychographicReconstruction
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

        # batching
        shuffled_indices = np.arange(self._num_diffraction_patterns)
        unshuffled_indices = np.zeros_like(shuffled_indices)

        if max_batch_size is not None:
            xp.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        # initialization
        self._reset_reconstruction(store_iterations, reset)

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

            # randomize
            if not use_projection_scheme:
                np.random.shuffle(shuffled_indices)

            unshuffled_indices[shuffled_indices] = np.arange(
                self._num_diffraction_patterns
            )

            positions_px = self._positions_px.copy()[shuffled_indices]

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
                amplitudes = self._amplitudes[shuffled_indices[start:end]]

                # forward operator
                (
                    shifted_probes,
                    object_patches,
                    overlap,
                    self._exit_waves,
                    batch_error,
                ) = self._forward(
                    self._object,
                    self._probe,
                    amplitudes,
                    self._exit_waves,
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
                    self._exit_waves,
                    use_projection_scheme=use_projection_scheme,
                    step_size=step_size,
                    normalization_min=normalization_min,
                    fix_probe=a0 < fix_probe_iter,
                )

                # position correction
                if a0 >= fix_positions_iter:
                    positions_px[start:end] = self._position_correction(
                        self._object,
                        self._probe[0],
                        overlap[:, 0],
                        amplitudes,
                        self._positions_px,
                        positions_step_size,
                        constrain_position_distance,
                    )

                error += batch_error

            # Normalize Error
            error /= self._mean_diffraction_intensity * self._num_diffraction_patterns

            # constraints
            self._positions_px = positions_px.copy()[unshuffled_indices]
            self._object, self._probe, self._positions_px = self._constraints(
                self._object,
                self._probe,
                self._positions_px,
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
                and gaussian_filter_sigma is not None,
                gaussian_filter_sigma=gaussian_filter_sigma,
                butterworth_filter=a0 < butterworth_filter_iter
                and (q_lowpass is not None or q_highpass is not None),
                q_lowpass=q_lowpass,
                q_highpass=q_highpass,
                butterworth_order=butterworth_order,
                kz_regularization_filter=a0 < kz_regularization_filter_iter
                and kz_regularization_gamma is not None,
                kz_regularization_gamma=kz_regularization_gamma[a0]
                if kz_regularization_gamma is not None
                and isinstance(kz_regularization_gamma, np.ndarray)
                else kz_regularization_gamma,
                identical_slices=a0 < identical_slices_iter,
                object_positivity=object_positivity,
                shrinkage_rad=shrinkage_rad,
                object_mask=self._object_fov_mask_inverse
                if fix_potential_baseline and self._object_fov_mask_inverse.sum() > 0
                else None,
                pure_phase_object=a0 < pure_phase_object_iter
                and self._object_type == "complex",
                tv_denoise_chambolle=a0 < tv_denoise_iter_chambolle
                and tv_denoise_weight_chambolle is not None,
                tv_denoise_weight_chambolle=tv_denoise_weight_chambolle,
                tv_denoise_pad_chambolle=tv_denoise_pad_chambolle,
                tv_denoise=a0 < tv_denoise_iter and tv_denoise_weights is not None,
                tv_denoise_weights=tv_denoise_weights,
                tv_denoise_inner_iter=tv_denoise_inner_iter,
                orthogonalize_probe=orthogonalize_probe,
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

    def _visualize_last_iteration_figax(
        self,
        fig,
        object_ax,
        convergence_ax,
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
            obj = np.angle(self.object)
        else:
            obj = self.object

        rotated_object = self._crop_rotate_object_fov(
            np.sum(obj, axis=0), padding=padding
        )
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
            errors = np.array(self.error_iterations)
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
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
        padding : int, optional
            Pixels to pad by post rotating-cropping object

        """
        figsize = kwargs.pop("figsize", (8, 5))
        cmap = kwargs.pop("cmap", "magma")

        chroma_boost = kwargs.pop("chroma_boost", 1)

        if self._object_type == "complex":
            obj = np.angle(self.object)
        else:
            obj = self.object

        rotated_object = self._crop_rotate_object_fov(
            np.sum(obj, axis=0), padding=padding
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
            if self._object_type == "potential":
                ax.set_title("Reconstructed object potential")
            elif self._object_type == "complex":
                ax.set_title("Reconstructed object phase")

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
                    probe_array = self.probe_fourier_residual[0]
                else:
                    probe_array = self.probe_fourier[0]

                probe_array = Complex2RGB(
                    probe_array,
                    chroma_boost=chroma_boost,
                )

                ax.set_title("Reconstructed Fourier probe[0]")
                ax.set_ylabel("kx [mrad]")
                ax.set_xlabel("ky [mrad]")
            else:
                probe_array = Complex2RGB(
                    self.probe[0], power=2, chroma_boost=chroma_boost
                )
                ax.set_title("Reconstructed probe[0] intensity")
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
            ax = fig.add_subplot(spec[0])
            im = ax.imshow(
                rotated_object,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            if self._object_type == "potential":
                ax.set_title("Reconstructed object potential")
            elif self._object_type == "complex":
                ax.set_title("Reconstructed object phase")

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
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        plot_fourier_probe: bool, optional
            If true, the reconstructed complex Fourier probe is displayed
        remove_initial_probe_aberrations: bool, optional
            If true, when plotting fourier probe, removes initial probe
            to visualize changes
        padding : int, optional
            Pixels to pad by post rotating-cropping object
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
                    padding=padding,
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

        objects = []
        object_type = []

        for obj in self.object_iterations:
            if np.iscomplexobj(obj):
                obj = np.angle(obj)
                object_type.append("phase")
            else:
                object_type.append("potential")
            objects.append(
                self._crop_rotate_object_fov(np.sum(obj, axis=0), padding=padding)
            )

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
            nrows_ncols=(1, iterations_grid[1])
            if (plot_probe or plot_fourier_probe)
            else iterations_grid,
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
            ax.set_title(f"Iter: {grid_range[n]} {object_type[grid_range[n]]}")
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
                            probes[grid_range[n]][0],
                            remove_initial_probe_aberrations=remove_initial_probe_aberrations,
                        )
                    )

                    probe_array = Complex2RGB(probe_array, chroma_boost=chroma_boost)

                    ax.set_title(f"Iter: {grid_range[n]} Fourier probe[0]")
                    ax.set_ylabel("kx [mrad]")
                    ax.set_xlabel("ky [mrad]")
                else:
                    probe_array = Complex2RGB(
                        probes[grid_range[n]][0],
                        power=2,
                        chroma_boost=chroma_boost,
                    )
                    ax.set_title(f"Iter: {grid_range[n]} probe[0]")
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
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
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

    def show_transmitted_probe(
        self,
        plot_fourier_probe: bool = False,
        remove_initial_probe_aberrations=False,
        **kwargs,
    ):
        """
        Plots the min, max, and mean transmitted probe after propagation and transmission.

        Parameters
        ----------
        plot_fourier_probe: boolean, optional
            If True, the transmitted probes are also plotted in Fourier space
        kwargs:
            Passed to show_complex
        """

        xp = self._xp
        asnumpy = self._asnumpy

        transmitted_probe_intensities = xp.sum(
            xp.abs(self._transmitted_probes[:, 0]) ** 2, axis=(-2, -1)
        )
        min_intensity_transmitted = self._transmitted_probes[
            xp.argmin(transmitted_probe_intensities), 0
        ]
        max_intensity_transmitted = self._transmitted_probes[
            xp.argmax(transmitted_probe_intensities), 0
        ]
        mean_transmitted = self._transmitted_probes[:, 0].mean(0)
        probes = [
            asnumpy(self._return_centered_probe(probe))
            for probe in [
                mean_transmitted,
                min_intensity_transmitted,
                max_intensity_transmitted,
            ]
        ]
        title = [
            "Mean Transmitted Probe",
            "Min Intensity Transmitted Probe",
            "Max Intensity Transmitted Probe",
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
                "Mean Transmitted Fourier Probe",
                "Min Intensity Transmitted Fourier Probe",
                "Max Intensity Transmitted Fourier Probe",
            ]

        title = kwargs.get("title", title)
        show_complex(
            probes,
            title=title,
            **kwargs,
        )

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
            amplitudes = self._amplitudes[start:end]

            # Overlaps
            _, _, overlap = self._overlap_projection(self._object, self._probe)
            fourier_overlap = xp.fft.fft2(overlap)
            intensity_norm = xp.sqrt(xp.sum(xp.abs(fourier_overlap) ** 2, axis=1))

            # Normalized mean-squared errors
            batch_errors = xp.sum(
                xp.abs(amplitudes - intensity_norm) ** 2, axis=(-2, -1)
            )
            errors = np.hstack((errors, batch_errors))

        self._positions_px = positions_px.copy()
        errors /= self._mean_diffraction_intensity

        return asnumpy(errors)
