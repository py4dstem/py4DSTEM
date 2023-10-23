"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely joint ptychography.
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
except ModuleNotFoundError:
    cp = np

from emdfile import Custom, tqdmnd
from py4DSTEM import DataCube
from py4DSTEM.process.phase.iterative_base_class import PtychographicReconstruction
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
)
from py4DSTEM.process.utils import get_CoM, get_shifted_ar

warnings.simplefilter(action="always", category=UserWarning)


class SimultaneousPtychographicReconstruction(PtychographicReconstruction):
    """
    Iterative Simultaneous Ptychographic Reconstruction Class.

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
    simultaneous_measurements_mode: str, optional
        One of '-+', '-0+', '0+', where -/0/+ refer to the sign of the magnetic potential
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
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (Px,Py)
        If None, initialized to 1.0j
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
    _class_specific_metadata = ("_simultaneous_measurements_mode",)

    def __init__(
        self,
        energy: float,
        datacube: Sequence[DataCube] = None,
        simultaneous_measurements_mode: str = "-+",
        semiangle_cutoff: float = None,
        semiangle_cutoff_pixels: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        object_padding_px: Tuple[int, int] = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        initial_scan_positions: np.ndarray = None,
        object_type: str = "complex",
        verbose: bool = True,
        device: str = "cpu",
        name: str = "simultaneous_ptychographic_reconstruction",
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
        self._probe = initial_probe_guess

        # Common Metadata
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._scan_positions = initial_scan_positions
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._semiangle_cutoff_pixels = semiangle_cutoff_pixels
        self._rolloff = rolloff
        self._object_type = object_type
        self._object_padding_px = object_padding_px
        self._verbose = verbose
        self._device = device
        self._preprocessed = False

        # Class-specific Metadata
        self._simultaneous_measurements_mode = simultaneous_measurements_mode

    def preprocess(
        self,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "fourier",
        probe_roi_shape: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        fit_function: str = "plane",
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
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

        if self._simultaneous_measurements_mode == "-+":
            self._sim_recon_mode = 0
            self._num_sim_measurements = 2
            if self._verbose:
                print(
                    (
                        "Magnetic vector potential sign in first meaurement assumed to be negative.\n"
                        "Magnetic vector potential sign in second meaurement assumed to be positive."
                    )
                )
            if len(self._datacube) != 2:
                raise ValueError(
                    f"datacube must be a set of two measurements, not length {len(self._datacube)}."
                )
            if self._datacube[0].shape != self._datacube[1].shape:
                raise ValueError("datacube intensities must be the same size.")
        elif self._simultaneous_measurements_mode == "-0+":
            self._sim_recon_mode = 1
            self._num_sim_measurements = 3
            if self._verbose:
                print(
                    (
                        "Magnetic vector potential sign in first meaurement assumed to be negative.\n"
                        "Magnetic vector potential assumed to be zero in second meaurement.\n"
                        "Magnetic vector potential sign in third meaurement assumed to be positive."
                    )
                )
            if len(self._datacube) != 3:
                raise ValueError(
                    f"datacube must be a set of three measurements, not length {len(self._datacube)}."
                )
            if (
                self._datacube[0].shape != self._datacube[1].shape
                or self._datacube[0].shape != self._datacube[2].shape
            ):
                raise ValueError("datacube intensities must be the same size.")
        elif self._simultaneous_measurements_mode == "0+":
            self._sim_recon_mode = 2
            self._num_sim_measurements = 2
            if self._verbose:
                print(
                    (
                        "Magnetic vector potential assumed to be zero in first meaurement.\n"
                        "Magnetic vector potential sign in second meaurement assumed to be positive."
                    )
                )
            if len(self._datacube) != 2:
                raise ValueError(
                    f"datacube must be a set of two measurements, not length {len(self._datacube)}."
                )
            if self._datacube[0].shape != self._datacube[1].shape:
                raise ValueError("datacube intensities must be the same size.")
        else:
            raise ValueError(
                f"simultaneous_measurements_mode must be either '-+', '-0+', or '0+', not {self._simultaneous_measurements_mode}"
            )

        if force_com_shifts is None:
            force_com_shifts = [None, None, None]
        elif len(force_com_shifts) == self._num_sim_measurements:
            force_com_shifts = list(force_com_shifts)
        else:
            raise ValueError(
                (
                    "force_com_shifts must be a sequence of tuples "
                    "with the same length as the datasets."
                )
            )

        # 1st measurement sets rotation angle and transposition
        (
            measurement_0,
            self._vacuum_probe_intensity,
            self._dp_mask,
            force_com_shifts[0],
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube[0],
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            probe_roi_shape=self._probe_roi_shape,
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            dp_mask=self._dp_mask,
            com_shifts=force_com_shifts[0],
        )

        intensities_0 = self._extract_intensities_and_calibrations_from_datacube(
            measurement_0,
            require_calibrations=True,
            force_scan_sampling=force_scan_sampling,
            force_angular_sampling=force_angular_sampling,
            force_reciprocal_sampling=force_reciprocal_sampling,
        )

        (
            com_measured_x_0,
            com_measured_y_0,
            com_fitted_x_0,
            com_fitted_y_0,
            com_normalized_x_0,
            com_normalized_y_0,
        ) = self._calculate_intensities_center_of_mass(
            intensities_0,
            dp_mask=self._dp_mask,
            fit_function=fit_function,
            com_shifts=force_com_shifts[0],
        )

        (
            self._rotation_best_rad,
            self._rotation_best_transpose,
            _com_x_0,
            _com_y_0,
            com_x_0,
            com_y_0,
        ) = self._solve_for_center_of_mass_relative_rotation(
            com_measured_x_0,
            com_measured_y_0,
            com_normalized_x_0,
            com_normalized_y_0,
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            plot_center_of_mass=False,
            maximize_divergence=maximize_divergence,
            force_com_rotation=force_com_rotation,
            force_com_transpose=force_com_transpose,
            **kwargs,
        )

        (
            amplitudes_0,
            mean_diffraction_intensity_0,
        ) = self._normalize_diffraction_intensities(
            intensities_0, com_fitted_x_0, com_fitted_y_0, crop_patterns
        )

        # explicitly delete namescapes
        del (
            intensities_0,
            com_measured_x_0,
            com_measured_y_0,
            com_fitted_x_0,
            com_fitted_y_0,
            com_normalized_x_0,
            com_normalized_y_0,
            _com_x_0,
            _com_y_0,
            com_x_0,
            com_y_0,
        )

        # 2nd measurement
        (
            measurement_1,
            _,
            _,
            force_com_shifts[1],
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube[1],
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            probe_roi_shape=self._probe_roi_shape,
            vacuum_probe_intensity=None,
            dp_mask=None,
            com_shifts=force_com_shifts[1],
        )

        intensities_1 = self._extract_intensities_and_calibrations_from_datacube(
            measurement_1,
            require_calibrations=True,
            force_scan_sampling=force_scan_sampling,
            force_angular_sampling=force_angular_sampling,
            force_reciprocal_sampling=force_reciprocal_sampling,
        )

        (
            com_measured_x_1,
            com_measured_y_1,
            com_fitted_x_1,
            com_fitted_y_1,
            com_normalized_x_1,
            com_normalized_y_1,
        ) = self._calculate_intensities_center_of_mass(
            intensities_1,
            dp_mask=self._dp_mask,
            fit_function=fit_function,
            com_shifts=force_com_shifts[1],
        )

        (
            _,
            _,
            _com_x_1,
            _com_y_1,
            com_x_1,
            com_y_1,
        ) = self._solve_for_center_of_mass_relative_rotation(
            com_measured_x_1,
            com_measured_y_1,
            com_normalized_x_1,
            com_normalized_y_1,
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            plot_center_of_mass=False,
            maximize_divergence=maximize_divergence,
            force_com_rotation=np.rad2deg(self._rotation_best_rad),
            force_com_transpose=self._rotation_best_transpose,
            **kwargs,
        )

        (
            amplitudes_1,
            mean_diffraction_intensity_1,
        ) = self._normalize_diffraction_intensities(
            intensities_1, com_fitted_x_1, com_fitted_y_1, crop_patterns
        )

        # explicitly delete namescapes
        del (
            intensities_1,
            com_measured_x_1,
            com_measured_y_1,
            com_fitted_x_1,
            com_fitted_y_1,
            com_normalized_x_1,
            com_normalized_y_1,
            _com_x_1,
            _com_y_1,
            com_x_1,
            com_y_1,
        )

        # Optionally, 3rd measurement
        if self._num_sim_measurements == 3:
            (
                measurement_2,
                _,
                _,
                force_com_shifts[2],
            ) = self._preprocess_datacube_and_vacuum_probe(
                self._datacube[2],
                diffraction_intensities_shape=self._diffraction_intensities_shape,
                reshaping_method=self._reshaping_method,
                probe_roi_shape=self._probe_roi_shape,
                vacuum_probe_intensity=None,
                dp_mask=None,
                com_shifts=force_com_shifts[2],
            )

            intensities_2 = self._extract_intensities_and_calibrations_from_datacube(
                measurement_2,
                require_calibrations=True,
                force_scan_sampling=force_scan_sampling,
                force_angular_sampling=force_angular_sampling,
                force_reciprocal_sampling=force_reciprocal_sampling,
            )

            (
                com_measured_x_2,
                com_measured_y_2,
                com_fitted_x_2,
                com_fitted_y_2,
                com_normalized_x_2,
                com_normalized_y_2,
            ) = self._calculate_intensities_center_of_mass(
                intensities_2,
                dp_mask=self._dp_mask,
                fit_function=fit_function,
                com_shifts=force_com_shifts[2],
            )

            (
                _,
                _,
                _com_x_2,
                _com_y_2,
                com_x_2,
                com_y_2,
            ) = self._solve_for_center_of_mass_relative_rotation(
                com_measured_x_2,
                com_measured_y_2,
                com_normalized_x_2,
                com_normalized_y_2,
                rotation_angles_deg=rotation_angles_deg,
                plot_rotation=plot_rotation,
                plot_center_of_mass=False,
                maximize_divergence=maximize_divergence,
                force_com_rotation=np.rad2deg(self._rotation_best_rad),
                force_com_transpose=self._rotation_best_transpose,
                **kwargs,
            )

            (
                amplitudes_2,
                mean_diffraction_intensity_2,
            ) = self._normalize_diffraction_intensities(
                intensities_2, com_fitted_x_2, com_fitted_y_2, crop_patterns
            )

            # explicitly delete namescapes
            del (
                intensities_2,
                com_measured_x_2,
                com_measured_y_2,
                com_fitted_x_2,
                com_fitted_y_2,
                com_normalized_x_2,
                com_normalized_y_2,
                _com_x_2,
                _com_y_2,
                com_x_2,
                com_y_2,
            )

            self._amplitudes = (amplitudes_0, amplitudes_1, amplitudes_2)
            self._mean_diffraction_intensity = (
                mean_diffraction_intensity_0
                + mean_diffraction_intensity_1
                + mean_diffraction_intensity_2
            ) / 3

            del amplitudes_0, amplitudes_1, amplitudes_2

        else:
            self._amplitudes = (amplitudes_0, amplitudes_1)
            self._mean_diffraction_intensity = (
                mean_diffraction_intensity_0 + mean_diffraction_intensity_1
            ) / 2

            del amplitudes_0, amplitudes_1

        # explicitly delete namespace
        self._num_diffraction_patterns = self._amplitudes[0].shape[0]
        self._region_of_interest_shape = np.array(self._amplitudes[0].shape[-2:])

        self._positions_px = self._calculate_scan_positions_in_pixels(
            self._scan_positions
        )

        # handle semiangle specified in pixels
        if self._semiangle_cutoff_pixels:
            self._semiangle_cutoff = (
                self._semiangle_cutoff_pixels * self._angular_sampling[0]
            )

        # Object Initialization
        if self._object is None:
            pad_x = self._object_padding_px[0][1]
            pad_y = self._object_padding_px[1][1]
            p, q = np.round(np.max(self._positions_px, axis=0))
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                "int"
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                "int"
            )
            if self._object_type == "potential":
                object_e = xp.zeros((p, q), dtype=xp.float32)
            elif self._object_type == "complex":
                object_e = xp.ones((p, q), dtype=xp.complex64)
            object_m = xp.zeros((p, q), dtype=xp.float32)
        else:
            if self._object_type == "potential":
                object_e = xp.asarray(self._object[0], dtype=xp.float32)
            elif self._object_type == "complex":
                object_e = xp.asarray(self._object[0], dtype=xp.complex64)
            object_m = xp.asarray(self._object[1], dtype=xp.float32)

        self._object = (object_e, object_m)
        self._object_initial = (object_e.copy(), object_m.copy())
        self._object_type_initial = self._object_type
        self._object_shape = self._object[0].shape

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

        # Vectorized Patches
        (
            self._vectorized_patch_indices_row,
            self._vectorized_patch_indices_col,
        ) = self._extract_vectorized_patch_indices()

        # Probe Initialization
        if self._probe is None:
            if self._vacuum_probe_intensity is not None:
                self._semiangle_cutoff = np.inf
                self._vacuum_probe_intensity = xp.asarray(
                    self._vacuum_probe_intensity, dtype=xp.float32
                )
                probe_x0, probe_y0 = get_CoM(
                    self._vacuum_probe_intensity, device=self._device
                )
                self._vacuum_probe_intensity = get_shifted_ar(
                    self._vacuum_probe_intensity,
                    -probe_x0,
                    -probe_y0,
                    bilinear=True,
                    device=self._device,
                )
                if crop_patterns:
                    self._vacuum_probe_intensity = self._vacuum_probe_intensity[
                        self._crop_mask
                    ].reshape(self._region_of_interest_shape)

            self._probe = (
                ComplexProbe(
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    energy=self._energy,
                    semiangle_cutoff=self._semiangle_cutoff,
                    rolloff=self._rolloff,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    parameters=self._polar_parameters,
                    device=self._device,
                )
                .build()
                ._array
            )

            # Normalize probe to match mean diffraction intensity
            probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probe)) ** 2)
            self._probe *= xp.sqrt(self._mean_diffraction_intensity / probe_intensity)

        else:
            if isinstance(self._probe, ComplexProbe):
                if self._probe._gpts != self._region_of_interest_shape:
                    raise ValueError()
                if hasattr(self._probe, "_array"):
                    self._probe = self._probe._array
                else:
                    self._probe._xp = xp
                    self._probe = self._probe.build()._array

                # Normalize probe to match mean diffraction intensity
                probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probe)) ** 2)
                self._probe *= xp.sqrt(
                    self._mean_diffraction_intensity / probe_intensity
                )
            else:
                self._probe = xp.asarray(self._probe, dtype=xp.complex64)

        self._probe_initial = self._probe.copy()
        self._probe_initial_aperture = xp.abs(xp.fft.fft2(self._probe))

        self._known_aberrations_array = ComplexProbe(
            energy=self._energy,
            gpts=self._region_of_interest_shape,
            sampling=self.sampling,
            parameters=self._polar_parameters,
            device=self._device,
        )._evaluate_ctf()

        # overlaps
        shifted_probes = fft_shift(self._probe, self._positions_px_fractional, xp)
        probe_intensities = xp.abs(shifted_probes) ** 2
        probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)
        probe_overlap = self._gaussian_filter(probe_overlap, 1.0)

        if object_fov_mask is None:
            self._object_fov_mask = asnumpy(probe_overlap > 0.25 * probe_overlap.max())
        else:
            self._object_fov_mask = np.asarray(object_fov_mask)
        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        if plot_probe_overlaps:
            figsize = kwargs.pop("figsize", (9, 4))
            chroma_boost = kwargs.pop("chroma_boost", 1)

            # initial probe
            complex_probe_rgb = Complex2RGB(
                self.probe_centered,
                power=2,
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
            add_colorbar_arg(
                cax1,
                chroma_boost=chroma_boost,
            )
            ax1.set_ylabel("x [A]")
            ax1.set_xlabel("y [A]")
            ax1.set_title("Initial probe intensity")

            ax2.imshow(
                asnumpy(probe_overlap),
                extent=extent,
                cmap="Greys_r",
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

    def _warmup_overlap_projection(self, current_object, current_probe):
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

        shifted_probes = fft_shift(current_probe, self._positions_px_fractional, xp)

        electrostatic_obj, _ = current_object

        if self._object_type == "potential":
            complex_object = xp.exp(1j * electrostatic_obj)
        else:
            complex_object = electrostatic_obj

        electrostatic_obj_patches = complex_object[
            self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]

        object_patches = (electrostatic_obj_patches, None)
        overlap = (shifted_probes * electrostatic_obj_patches, None)

        return shifted_probes, object_patches, overlap

    def _overlap_projection(self, current_object, current_probe):
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

        shifted_probes = fft_shift(current_probe, self._positions_px_fractional, xp)

        electrostatic_obj, magnetic_obj = current_object

        if self._object_type == "potential":
            complex_object_e = xp.exp(1j * electrostatic_obj)
        else:
            complex_object_e = electrostatic_obj

        complex_object_m = xp.exp(1j * magnetic_obj)

        electrostatic_obj_patches = complex_object_e[
            self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]
        magnetic_obj_patches = complex_object_m[
            self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]

        object_patches = (electrostatic_obj_patches, magnetic_obj_patches)

        if self._sim_recon_mode == 0:
            overlap_reverse = (
                shifted_probes
                * electrostatic_obj_patches
                * xp.conj(magnetic_obj_patches)
            )
            overlap_forward = (
                shifted_probes * electrostatic_obj_patches * magnetic_obj_patches
            )
            overlap = (overlap_reverse, overlap_forward)
        elif self._sim_recon_mode == 1:
            overlap_reverse = (
                shifted_probes
                * electrostatic_obj_patches
                * xp.conj(magnetic_obj_patches)
            )
            overlap_neutral = shifted_probes * electrostatic_obj_patches
            overlap_forward = (
                shifted_probes * electrostatic_obj_patches * magnetic_obj_patches
            )
            overlap = (overlap_reverse, overlap_neutral, overlap_forward)
        else:
            overlap_neutral = shifted_probes * electrostatic_obj_patches
            overlap_forward = (
                shifted_probes * electrostatic_obj_patches * magnetic_obj_patches
            )
            overlap = (overlap_neutral, overlap_forward)

        return shifted_probes, object_patches, overlap

    def _warmup_gradient_descent_fourier_projection(self, amplitudes, overlap):
        """
        Ptychographic fourier projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap

        Returns
        --------
        exit_waves:np.ndarray
            Difference between modified and estimated exit waves
        error: float
            Reconstruction error
        """

        xp = self._xp

        fourier_overlap = xp.fft.fft2(overlap[0])
        error = xp.sum(xp.abs(amplitudes[0] - xp.abs(fourier_overlap)) ** 2)

        fourier_modified_overlap = amplitudes[0] * xp.exp(
            1j * xp.angle(fourier_overlap)
        )
        modified_overlap = xp.fft.ifft2(fourier_modified_overlap)

        exit_waves = (modified_overlap - overlap[0],) + (None,) * (
            self._num_sim_measurements - 1
        )

        return exit_waves, error

    def _gradient_descent_fourier_projection(self, amplitudes, overlap):
        """
        Ptychographic fourier projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        overlap: np.ndarray
            object * probe overlap

        Returns
        --------
        exit_waves:np.ndarray
            Difference between modified and estimated exit waves
        error: float
            Reconstruction error
        """

        xp = self._xp

        error = 0.0
        exit_waves = []
        for amp, overl in zip(amplitudes, overlap):
            fourier_overl = xp.fft.fft2(overl)
            error += xp.sum(xp.abs(amp - xp.abs(fourier_overl)) ** 2)

            fourier_modified_overl = amp * xp.exp(1j * xp.angle(fourier_overl))
            modified_overl = xp.fft.ifft2(fourier_modified_overl)

            exit_waves.append(modified_overl - overl)

        error /= len(exit_waves)
        exit_waves = tuple(exit_waves)

        return exit_waves, error

    def _warmup_projection_sets_fourier_projection(
        self, amplitudes, overlap, exit_waves, projection_a, projection_b, projection_c
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

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        exit_wave = exit_waves[0]

        if exit_wave is None:
            exit_wave = overlap[0].copy()

        fourier_overlap = xp.fft.fft2(overlap[0])
        error = xp.sum(xp.abs(amplitudes[0] - xp.abs(fourier_overlap)) ** 2)

        factor_to_be_projected = projection_c * overlap[0] + projection_y * exit_wave
        fourier_projected_factor = xp.fft.fft2(factor_to_be_projected)

        fourier_projected_factor = amplitudes[0] * xp.exp(
            1j * xp.angle(fourier_projected_factor)
        )
        projected_factor = xp.fft.ifft2(fourier_projected_factor)

        exit_wave = (
            projection_x * exit_wave
            + projection_a * overlap[0]
            + projection_b * projected_factor
        )

        exit_waves = (exit_wave,) + (None,) * (self._num_sim_measurements - 1)

        return exit_waves, error

    def _projection_sets_fourier_projection(
        self, amplitudes, overlap, exit_waves, projection_a, projection_b, projection_c
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

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        error = 0.0
        _exit_waves = []
        for amp, overl, exit_wave in zip(amplitudes, overlap, exit_waves):
            if exit_wave is None:
                exit_wave = overl.copy()

            fourier_overl = xp.fft.fft2(overl)
            error += xp.sum(xp.abs(amp - xp.abs(fourier_overl)) ** 2)

            factor_to_be_projected = projection_c * overl + projection_y * exit_wave
            fourier_projected_factor = xp.fft.fft2(factor_to_be_projected)

            fourier_projected_factor = amp * xp.exp(
                1j * xp.angle(fourier_projected_factor)
            )
            projected_factor = xp.fft.ifft2(fourier_projected_factor)

            _exit_waves.append(
                projection_x * exit_wave
                + projection_a * overl
                + projection_b * projected_factor
            )

        error /= len(_exit_waves)
        exit_waves = tuple(_exit_waves)

        return exit_waves, error

    def _forward(
        self,
        current_object,
        current_probe,
        amplitudes,
        exit_waves,
        warmup_iteration,
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
        if warmup_iteration:
            shifted_probes, object_patches, overlap = self._warmup_overlap_projection(
                current_object, current_probe
            )
            if use_projection_scheme:
                exit_waves, error = self._warmup_projection_sets_fourier_projection(
                    amplitudes,
                    overlap,
                    exit_waves,
                    projection_a,
                    projection_b,
                    projection_c,
                )

            else:
                exit_waves, error = self._warmup_gradient_descent_fourier_projection(
                    amplitudes, overlap
                )

        else:
            shifted_probes, object_patches, overlap = self._overlap_projection(
                current_object, current_probe
            )
            if use_projection_scheme:
                exit_waves, error = self._projection_sets_fourier_projection(
                    amplitudes,
                    overlap,
                    exit_waves,
                    projection_a,
                    projection_b,
                    projection_c,
                )

            else:
                exit_waves, error = self._gradient_descent_fourier_projection(
                    amplitudes, overlap
                )

        return shifted_probes, object_patches, overlap, exit_waves, error

    def _warmup_gradient_descent_adjoint(
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

        electrostatic_obj, _ = current_object
        electrostatic_obj_patches, _ = object_patches

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(shifted_probes) ** 2
        )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        if self._object_type == "potential":
            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    xp.real(
                        -1j
                        * xp.conj(electrostatic_obj_patches)
                        * xp.conj(shifted_probes)
                        * exit_waves[0]
                    )
                )
                * probe_normalization
            )
        elif self._object_type == "complex":
            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    xp.conj(shifted_probes) * exit_waves[0]
                )
                * probe_normalization
            )

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(electrostatic_obj_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe += step_size * (
                xp.sum(
                    xp.conj(electrostatic_obj_patches) * exit_waves[0],
                    axis=0,
                )
                * object_normalization
            )

        return (electrostatic_obj, None), current_probe

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

        electrostatic_obj, magnetic_obj = current_object
        probe_conj = xp.conj(shifted_probes)

        electrostatic_obj_patches, magnetic_obj_patches = object_patches
        electrostatic_conj = xp.conj(electrostatic_obj_patches)
        magnetic_conj = xp.conj(magnetic_obj_patches)

        probe_electrostatic_abs = xp.abs(shifted_probes * electrostatic_obj_patches)
        probe_magnetic_abs = xp.abs(shifted_probes * magnetic_obj_patches)

        probe_electrostatic_normalization = self._sum_overlapping_patches_bincounts(
            probe_electrostatic_abs**2
        )
        probe_electrostatic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_electrostatic_normalization) ** 2
            + (normalization_min * xp.max(probe_electrostatic_normalization)) ** 2
        )

        probe_magnetic_normalization = self._sum_overlapping_patches_bincounts(
            probe_magnetic_abs**2
        )
        probe_magnetic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_magnetic_normalization) ** 2
            + (normalization_min * xp.max(probe_magnetic_normalization)) ** 2
        )

        if self._sim_recon_mode > 0:
            probe_abs = xp.abs(shifted_probes)
            probe_normalization = self._sum_overlapping_patches_bincounts(
                probe_abs**2
            )
            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

        if self._sim_recon_mode == 0:
            exit_waves_reverse, exit_waves_forward = exit_waves

            if self._object_type == "potential":
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * magnetic_obj_patches
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_reverse
                            )
                        )
                        * probe_magnetic_normalization
                    )
                    / 2
                )
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * magnetic_conj
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_forward
                            )
                        )
                        * probe_magnetic_normalization
                    )
                    / 2
                )

            elif self._object_type == "complex":
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            probe_conj * magnetic_obj_patches * exit_waves_reverse
                        )
                        * probe_magnetic_normalization
                    )
                    / 2
                )
                electrostatic_obj += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        probe_conj * magnetic_conj * exit_waves_forward
                    )
                    * probe_magnetic_normalization
                    / 2
                )

            magnetic_obj += (
                step_size
                * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(
                            1j
                            * magnetic_obj_patches
                            * electrostatic_conj
                            * xp.conj(shifted_probes)
                            * exit_waves_reverse
                        )
                    )
                    * probe_electrostatic_normalization
                )
                / 2
            )
            magnetic_obj += (
                step_size
                * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * magnetic_conj
                            * electrostatic_conj
                            * xp.conj(shifted_probes)
                            * exit_waves_forward
                        )
                    )
                    * probe_electrostatic_normalization
                )
                / 2
            )

        elif self._sim_recon_mode == 1:
            exit_waves_reverse, exit_waves_neutral, exit_waves_forward = exit_waves

            if self._object_type == "potential":
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * magnetic_obj_patches
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_reverse
                            )
                        )
                        * probe_magnetic_normalization
                    )
                    / 3
                )
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_neutral
                            )
                        )
                        * probe_normalization
                    )
                    / 3
                )
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * magnetic_conj
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_forward
                            )
                        )
                        * probe_magnetic_normalization
                    )
                    / 3
                )

            elif self._object_type == "complex":
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            probe_conj * magnetic_obj_patches * exit_waves_reverse
                        )
                        * probe_magnetic_normalization
                    )
                    / 3
                )
                electrostatic_obj += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        probe_conj * exit_waves_neutral
                    )
                    * probe_normalization
                    / 3
                )
                electrostatic_obj += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        probe_conj * magnetic_conj * exit_waves_forward
                    )
                    * probe_magnetic_normalization
                    / 3
                )

            magnetic_obj += (
                step_size
                * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(
                            1j
                            * magnetic_obj_patches
                            * electrostatic_conj
                            * xp.conj(shifted_probes)
                            * exit_waves_reverse
                        )
                    )
                    * probe_electrostatic_normalization
                )
                / 2
            )
            magnetic_obj += (
                step_size
                * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * magnetic_conj
                            * electrostatic_conj
                            * xp.conj(shifted_probes)
                            * exit_waves_forward
                        )
                    )
                    * probe_electrostatic_normalization
                )
                / 2
            )

        else:
            exit_waves_neutral, exit_waves_forward = exit_waves

            if self._object_type == "potential":
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_neutral
                            )
                        )
                        * probe_normalization
                    )
                    / 2
                )
                electrostatic_obj += (
                    step_size
                    * (
                        self._sum_overlapping_patches_bincounts(
                            xp.real(
                                -1j
                                * magnetic_conj
                                * electrostatic_conj
                                * xp.conj(shifted_probes)
                                * exit_waves_forward
                            )
                        )
                        * probe_magnetic_normalization
                    )
                    / 2
                )

            elif self._object_type == "complex":
                electrostatic_obj += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        probe_conj * exit_waves_neutral
                    )
                    * probe_normalization
                    / 2
                )
                electrostatic_obj += step_size * (
                    self._sum_overlapping_patches_bincounts(
                        probe_conj * magnetic_conj * exit_waves_forward
                    )
                    * probe_magnetic_normalization
                    / 2
                )

            magnetic_obj += (
                step_size
                * (
                    self._sum_overlapping_patches_bincounts(
                        xp.real(
                            -1j
                            * magnetic_conj
                            * electrostatic_conj
                            * xp.conj(shifted_probes)
                            * exit_waves_forward
                        )
                    )
                    * probe_electrostatic_normalization
                )
                / 3
            )

        if not fix_probe:
            electrostatic_magnetic_abs = xp.abs(
                electrostatic_obj_patches * magnetic_obj_patches
            )
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

            if self._sim_recon_mode > 0:
                electrostatic_abs = xp.abs(electrostatic_obj_patches)
                electrostatic_normalization = xp.sum(
                    electrostatic_abs**2,
                    axis=0,
                )
                electrostatic_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * electrostatic_normalization) ** 2
                    + (normalization_min * xp.max(electrostatic_normalization)) ** 2
                )

            if self._sim_recon_mode == 0:
                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * magnetic_obj_patches * exit_waves_reverse,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

            elif self._sim_recon_mode == 1:
                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * magnetic_obj_patches * exit_waves_reverse,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 3
                )

                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * exit_waves_neutral,
                        axis=0,
                    )
                    * electrostatic_normalization
                    / 3
                )

                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 3
                )
            else:
                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * exit_waves_neutral,
                        axis=0,
                    )
                    * electrostatic_normalization
                    / 2
                )

                current_probe += step_size * (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

        current_object = (electrostatic_obj, magnetic_obj)

        return current_object, current_probe

    def _warmup_projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
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

        electrostatic_obj, _ = current_object
        electrostatic_obj_patches, _ = object_patches

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(shifted_probes) ** 2
        )
        probe_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_normalization) ** 2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        electrostatic_obj = (
            self._sum_overlapping_patches_bincounts(
                xp.conj(shifted_probes) * exit_waves[0]
            )
            * probe_normalization
        )

        if not fix_probe:
            object_normalization = xp.sum(
                (xp.abs(electrostatic_obj_patches) ** 2),
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * object_normalization) ** 2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            current_probe = (
                xp.sum(
                    xp.conj(electrostatic_obj_patches) * exit_waves[0],
                    axis=0,
                )
                * object_normalization
            )

        return (electrostatic_obj, None), current_probe

    def _projection_sets_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
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

        electrostatic_obj, magnetic_obj = current_object
        probe_conj = xp.conj(shifted_probes)

        electrostatic_obj_patches, magnetic_obj_patches = object_patches
        electrostatic_conj = xp.conj(electrostatic_obj_patches)
        magnetic_conj = xp.conj(magnetic_obj_patches)

        probe_electrostatic_abs = xp.abs(shifted_probes * electrostatic_obj_patches)
        probe_magnetic_abs = xp.abs(shifted_probes * magnetic_obj_patches)

        probe_electrostatic_normalization = self._sum_overlapping_patches_bincounts(
            probe_electrostatic_abs**2
        )
        probe_electrostatic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_electrostatic_normalization) ** 2
            + (normalization_min * xp.max(probe_electrostatic_normalization)) ** 2
        )

        probe_magnetic_normalization = self._sum_overlapping_patches_bincounts(
            probe_magnetic_abs**2
        )
        probe_magnetic_normalization = 1 / xp.sqrt(
            1e-16
            + ((1 - normalization_min) * probe_magnetic_normalization) ** 2
            + (normalization_min * xp.max(probe_magnetic_normalization)) ** 2
        )

        if self._sim_recon_mode > 0:
            probe_abs = xp.abs(shifted_probes)

            probe_normalization = self._sum_overlapping_patches_bincounts(
                probe_abs**2
            )
            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

        if self._sim_recon_mode == 0:
            exit_waves_reverse, exit_waves_forward = exit_waves

            electrostatic_obj = (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_obj_patches * exit_waves_reverse
                )
                * probe_magnetic_normalization
                / 2
            )

            electrostatic_obj += (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_conj * exit_waves_forward
                )
                * probe_magnetic_normalization
                / 2
            )

            magnetic_obj = xp.conj(
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_reverse
                )
                * probe_electrostatic_normalization
                / 2
            )

            magnetic_obj += (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
                / 2
            )

        elif self._sim_recon_mode == 1:
            exit_waves_reverse, exit_waves_neutral, exit_waves_forward = exit_waves

            electrostatic_obj = (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_obj_patches * exit_waves_reverse
                )
                * probe_magnetic_normalization
                / 3
            )

            electrostatic_obj += (
                self._sum_overlapping_patches_bincounts(probe_conj * exit_waves_neutral)
                * probe_normalization
                / 3
            )

            electrostatic_obj += (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_conj * exit_waves_forward
                )
                * probe_magnetic_normalization
                / 3
            )

            magnetic_obj = xp.conj(
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_reverse
                )
                * probe_electrostatic_normalization
                / 2
            )

            magnetic_obj += (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
                / 2
            )

        else:
            raise NotImplementedError()

        if not fix_probe:
            electrostatic_magnetic_abs = xp.abs(
                electrostatic_obj_patches * magnetic_obj_patches
            )

            electrostatic_magnetic_normalization = xp.sum(
                (electrostatic_magnetic_abs**2),
                axis=0,
            )
            electrostatic_magnetic_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * electrostatic_magnetic_normalization) ** 2
                + (normalization_min * xp.max(electrostatic_magnetic_normalization))
                ** 2
            )

            if self._sim_recon_mode > 0:
                electrostatic_abs = xp.abs(electrostatic_obj_patches)
                electrostatic_normalization = xp.sum(
                    (electrostatic_abs**2),
                    axis=0,
                )
                electrostatic_normalization = 1 / xp.sqrt(
                    1e-16
                    + ((1 - normalization_min) * electrostatic_normalization) ** 2
                    + (normalization_min * xp.max(electrostatic_normalization)) ** 2
                )

            if self._sim_recon_mode == 0:
                current_probe = (
                    xp.sum(
                        electrostatic_conj * magnetic_obj_patches * exit_waves_reverse,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

                current_probe += (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

            elif self._sim_recon_mode == 1:
                current_probe = (
                    xp.sum(
                        electrostatic_conj * magnetic_obj_patches * exit_waves_reverse,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 3
                )

                current_probe += (
                    xp.sum(
                        electrostatic_conj * exit_waves_neutral,
                        axis=0,
                    )
                    * electrostatic_normalization
                    / 3
                )

                current_probe += (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 3
                )
            else:
                current_probe = (
                    xp.sum(
                        electrostatic_conj * exit_waves_neutral,
                        axis=0,
                    )
                    * electrostatic_normalization
                    / 2
                )

                current_probe += (
                    xp.sum(
                        electrostatic_conj * magnetic_conj * exit_waves_forward,
                        axis=0,
                    )
                    * electrostatic_magnetic_normalization
                    / 2
                )

        current_object = (electrostatic_obj, magnetic_obj)

        return current_object, current_probe

    def _adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        shifted_probes,
        exit_waves,
        warmup_iteration: bool,
        use_projection_scheme: bool,
        step_size: float,
        normalization_min: float,
        fix_probe: bool,
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

        if warmup_iteration:
            if use_projection_scheme:
                current_object, current_probe = self._warmup_projection_sets_adjoint(
                    current_object,
                    current_probe,
                    object_patches,
                    shifted_probes,
                    exit_waves,
                    normalization_min,
                    fix_probe,
                )
            else:
                current_object, current_probe = self._warmup_gradient_descent_adjoint(
                    current_object,
                    current_probe,
                    object_patches,
                    shifted_probes,
                    exit_waves,
                    step_size,
                    normalization_min,
                    fix_probe,
                )

        else:
            if use_projection_scheme:
                current_object, current_probe = self._projection_sets_adjoint(
                    current_object,
                    current_probe,
                    object_patches,
                    shifted_probes,
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
                    exit_waves,
                    step_size,
                    normalization_min,
                    fix_probe,
                )

        return current_object, current_probe

    def _constraints(
        self,
        current_object,
        current_probe,
        current_positions,
        pure_phase_object,
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
        tv_denoise,
        tv_denoise_weight,
        tv_denoise_inner_iter,
        warmup_iteration,
        object_positivity,
        shrinkage_rad,
        object_mask,
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
        pure_phase_object: bool
            If True, object amplitude is set to unity
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
        probe_gaussian_filter: bool
            If True, applies reciprocal-space gaussian filtering on residual aberrations
        probe_gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A^-1
        probe_gaussian_filter_fix_amplitude: bool
            If True, only the probe phase is smoothed
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
        tv_denoise: bool
            If True, applies TV denoising on object
        tv_denoise_weight: float
            Denoising weight. The greater `weight`, the more denoising.
        tv_denoise_inner_iter: float
            Number of iterations to run in inner loop of TV denoising
        warmup_iteration: bool
            If True, constraints electrostatic object only
        object_positivity: bool
            If True, clips negative potential values
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        object_mask: np.ndarray (boolean)
            If not None, used to calculate additional shrinkage using masked-mean of object

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        constrained_probe: np.ndarray
            Constrained probe estimate
        constrained_positions: np.ndarray
            Constrained positions estimate
        """

        electrostatic_obj, magnetic_obj = current_object

        if gaussian_filter:
            electrostatic_obj = self._object_gaussian_constraint(
                electrostatic_obj, gaussian_filter_sigma_e, pure_phase_object
            )
            if not warmup_iteration:
                magnetic_obj = self._object_gaussian_constraint(
                    magnetic_obj,
                    gaussian_filter_sigma_m,
                    pure_phase_object,
                )

        if butterworth_filter:
            electrostatic_obj = self._object_butterworth_constraint(
                electrostatic_obj,
                q_lowpass_e,
                q_highpass_e,
                butterworth_order,
            )
            if not warmup_iteration:
                magnetic_obj = self._object_butterworth_constraint(
                    magnetic_obj,
                    q_lowpass_m,
                    q_highpass_m,
                    butterworth_order,
                )

                if self._object_type == "complex":
                    magnetic_obj = magnetic_obj.real
        if tv_denoise:
            electrostatic_obj = self._object_denoise_tv_pylops(
                electrostatic_obj, tv_denoise_weight, tv_denoise_inner_iter
            )

            if not warmup_iteration:
                magnetic_obj = self._object_denoise_tv_pylops(
                    magnetic_obj, tv_denoise_weight, tv_denoise_inner_iter
                )

        if shrinkage_rad > 0.0 or object_mask is not None:
            electrostatic_obj = self._object_shrinkage_constraint(
                electrostatic_obj,
                shrinkage_rad,
                object_mask,
            )

        if self._object_type == "complex":
            electrostatic_obj = self._object_threshold_constraint(
                electrostatic_obj, pure_phase_object
            )
        elif object_positivity:
            electrostatic_obj = self._object_positivity_constraint(electrostatic_obj)

        current_object = (electrostatic_obj, magnetic_obj)

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
        pure_phase_object_iter: int = 0,
        fix_com: bool = True,
        fix_probe_iter: int = 0,
        warmup_iter: int = 0,
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
        tv_denoise_iter: int = np.inf,
        tv_denoise_weight: float = None,
        tv_denoise_inner_iter: float = 40,
        object_positivity: bool = True,
        shrinkage_rad: float = 0.0,
        fix_potential_baseline: bool = True,
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
        constrain_position_distance: float
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

        if use_projection_scheme and self._sim_recon_mode == 2:
            raise NotImplementedError(
                "simultaneous_measurements_mode == '0+' and projection set algorithms are currently incompatible."
            )

        if self._verbose:
            if switch_object_iter > max_iter:
                first_line = f"Performing {max_iter} iterations using a {self._object_type} object type, "
            else:
                switch_object_type = (
                    "complex" if self._object_type == "potential" else "potential"
                )
                first_line = (
                    f"Performing {switch_object_iter} iterations using a {self._object_type} object type and "
                    f"{max_iter - switch_object_iter} iterations using a {switch_object_type} object type, "
                )
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
                            first_line + f"with the {reconstruction_method} algorithm, "
                            f"with normalization_min: {normalization_min} and step _size: {step_size}, "
                            f"in batches of max {max_batch_size} measurements."
                        )
                    )

            else:
                if reconstruction_parameter is not None:
                    if np.array(reconstruction_parameter).shape == (3,):
                        print(
                            (
                                first_line
                                + f"with the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and (a,b,c): {reconstruction_parameter}."
                            )
                        )
                    else:
                        print(
                            (
                                first_line
                                + f"with the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and α: {reconstruction_parameter}."
                            )
                        )
                else:
                    if step_size is not None:
                        print(
                            (
                                first_line
                                + f"with the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min}."
                            )
                        )
                    else:
                        print(
                            (
                                first_line
                                + f"with the {reconstruction_method} algorithm, "
                                f"with normalization_min: {normalization_min} and step _size: {step_size}."
                            )
                        )

        # Batching
        shuffled_indices = np.arange(self._num_diffraction_patterns)
        unshuffled_indices = np.zeros_like(shuffled_indices)

        if max_batch_size is not None:
            xp.random.seed(seed_random)
        else:
            max_batch_size = self._num_diffraction_patterns

        # initialization
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []

        if reset:
            self._object = (
                self._object_initial[0].copy(),
                self._object_initial[1].copy(),
            )
            self._probe = self._probe_initial.copy()
            self.error_iterations = []
            self._positions_px = self._positions_px_initial.copy()
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            (
                self._vectorized_patch_indices_row,
                self._vectorized_patch_indices_col,
            ) = self._extract_vectorized_patch_indices()
            self._exit_waves = (None,) * self._num_sim_measurements
            self._object_type = self._object_type_initial
            if hasattr(self, "_tf"):
                del self._tf
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
                self._exit_waves = (None,) * self._num_sim_measurements

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
                    self._object = (xp.exp(1j * self._object[0]), self._object[1])
                elif self._object_type == "complex":
                    self._object_type = "potential"
                    self._object = (xp.angle(self._object[0]), self._object[1])

            if a0 == warmup_iter:
                self._object = (self._object[0], self._object_initial[1].copy())

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

                amps = []
                for amplitudes in self._amplitudes:
                    amps.append(amplitudes[shuffled_indices[start:end]])
                amplitudes = tuple(amps)

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
                    warmup_iteration=a0 < warmup_iter,
                    use_projection_scheme=use_projection_scheme,
                    projection_a=projection_a,
                    projection_b=projection_b,
                    projection_c=projection_c,
                )

                # adjoint operator
                self._object, self._probe = self._adjoint(
                    self._object,
                    self._probe,
                    object_patches,
                    shifted_probes,
                    self._exit_waves,
                    warmup_iteration=a0 < warmup_iter,
                    use_projection_scheme=use_projection_scheme,
                    step_size=step_size,
                    normalization_min=normalization_min,
                    fix_probe=a0 < fix_probe_iter,
                )

                # position correction
                if a0 >= fix_positions_iter:
                    positions_px[start:end] = self._position_correction(
                        self._object[0],
                        shifted_probes,
                        overlap[0],
                        amplitudes[0],
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
                warmup_iteration=a0 < warmup_iter,
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
                if fix_potential_baseline and self._object_fov_mask_inverse.sum() > 0
                else None,
                pure_phase_object=a0 < pure_phase_object_iter
                and self._object_type == "complex",
            )

            self.error_iterations.append(error.item())
            if store_iterations:
                if a0 < warmup_iter:
                    self.object_iterations.append(
                        (asnumpy(self._object[0].copy()), None)
                    )
                else:
                    self.object_iterations.append(
                        (
                            asnumpy(self._object[0].copy()),
                            asnumpy(self._object[1].copy()),
                        )
                    )
                self.probe_iterations.append(self.probe_centered)

        # store result
        if a0 < warmup_iter:
            self.object = (asnumpy(self._object[0]), None)
        else:
            self.object = (asnumpy(self._object[0]), asnumpy(self._object[1]))
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

        if plot_fourier_probe:
            chroma_boost = kwargs.pop("chroma_boost", 2)
        else:
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
                probe_array = Complex2RGB(
                    self.probe_fourier,
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
                cbar=cbar,
                padding=padding,
                **kwargs,
            )

        return self
