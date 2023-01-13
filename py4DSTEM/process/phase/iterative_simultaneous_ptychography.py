"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC and ptychography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

try:
    import cupy as cp
except ImportError:
    cp = None

from py4DSTEM.io import DataCube
from py4DSTEM.process.phase.iterative_base_class import PhaseReconstruction
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    estimate_global_transformation_ransac,
    fft_shift,
    generate_batches,
    polar_aliases,
    polar_symbols,
)
from py4DSTEM.process.utils import get_CoM, get_shifted_ar
from py4DSTEM.utils.tqdmnd import tqdmnd

warnings.simplefilter(action="always", category=UserWarning)


class SimultaneousPtychographicReconstruction(PhaseReconstruction):
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
        Set of input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    simultaneous_measurements_mode: str, optional
        One of '-+', '-0+', '0+', where -/0/+ refer to the sign of the magnetic potential
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial probe guess
    rolloff: float, optional
        Semiangle rolloff for the initial probe guess
    vacuum_probe_intensity: np.ndarray, optional
        Vacuum probe to use as intensity aperture for initial probe guess
    polar_parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in Å and angles should be given in radians.
    diffraction_intensities_shape: Tuple[int,int], optional
        Pixel dimensions (Qx',Qy') of the resampled diffraction intensities
        If None, no resampling of diffraction intenstities is performed
    reshaping_method: str, optional
        Method to use for reshaping, either 'bin', 'bilinear', or 'fourier' (default)
    probe_roi_shape, (int,int), optional
            Padded diffraction intensities shape.
            If None, no padding is performed
    object_padding_px: Tuple[int,int], optional
        Pixel dimensions to pad objects with
        If None, the padding is set to half the probe ROI dimensions
    dp_mask: ndarray, optional
        Mask for datacube intensities (Qx,Qy)
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (Px,Py)
        If None, initialized to 1.0j
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy). If None,
        initialized to ComplexProbe with semiangle_cutoff, energy, and aberrations
    scan_positions: np.ndarray, optional
        Probe positions in Å for each diffraction intensity
        If None, initialized to a grid scan
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    def __init__(
        self,
        datacube: Sequence[DataCube],
        energy: float,
        simultaneous_measurements_mode: str = "-+",
        semiangle_cutoff: float = None,
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
        diffraction_intensities_shape: Tuple[int, int] = None,
        reshaping_method: str = "fourier",
        probe_roi_shape: Tuple[int, int] = None,
        object_padding_px: Tuple[int, int] = None,
        dp_mask: np.ndarray = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        scan_positions: np.ndarray = None,
        verbose: bool = True,
        device: str = "cpu",
        **kwargs,
    ):

        if device == "cpu":
            self._xp = np
            self._asnumpy = np.asarray
            from scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
        elif device == "gpu":
            self._xp = cp
            self._asnumpy = cp.asnumpy
            from cupyx.scipy.ndimage import gaussian_filter

            self._gaussian_filter = gaussian_filter
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

        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._diffraction_intensities_shape = diffraction_intensities_shape
        self._reshaping_method = reshaping_method
        self._probe_roi_shape = probe_roi_shape
        self._object = initial_object_guess
        self._probe = initial_probe_guess
        self._scan_positions = scan_positions
        self._datacube = datacube
        self._dp_mask = dp_mask
        self._verbose = verbose
        self._object_padding_px = object_padding_px
        self._preprocessed = False

        if simultaneous_measurements_mode == "-+":
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
        elif simultaneous_measurements_mode == "-0+":
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
        elif simultaneous_measurements_mode == "0+":
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
                f"simultaneous_measurements_mode must be either '-+', '-0+', or '0+', not {simultaneous_measurements_mode}"
            )

    def preprocess(
        self,
        fit_function: str = "plane",
        plot_center_of_mass: str = "default",
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = np.arange(-89.0, 90.0, 1.0),
        plot_probe_overlaps: bool = True,
        force_com_rotation: float = None,
        force_com_transpose: float = None,
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

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        xp = self._xp
        asnumpy = self._asnumpy

        # Note: a lot of the methods below modify state. The only two property we mind this for are
        # self._amplitudes and self._mean_diffraction_intensity, so we simply proceed serially.

        # 1st measurement sets rotation angle and transposition
        (
            measurement_0,
            self._vacuum_probe_intensity,
            self._dp_mask,
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube[0],
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            probe_roi_shape=self._probe_roi_shape,
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            dp_mask=self._dp_mask,
        )

        self._extract_intensities_and_calibrations_from_datacube(
            measurement_0, require_calibrations=True, dp_mask=self._dp_mask
        )

        self._calculate_intensities_center_of_mass(
            self._intensities,
            fit_function=fit_function,
        )

        self._solve_for_center_of_mass_relative_rotation(
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            plot_center_of_mass=plot_center_of_mass,
            maximize_divergence=maximize_divergence,
            force_com_rotation=force_com_rotation,
            force_com_transpose=force_com_transpose,
            **kwargs,
        )

        (
            amplitudes_0,
            mean_diffraction_intensity_0,
        ) = self._normalize_diffraction_intensities(
            self._intensities,
            self._com_fitted_x,
            self._com_fitted_y,
        )

        # 2nd measurement
        (measurement_1, _, _,) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube[1],
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            reshaping_method=self._reshaping_method,
            probe_roi_shape=self._probe_roi_shape,
            vacuum_probe_intensity=None,
            dp_mask=None,
        )

        self._extract_intensities_and_calibrations_from_datacube(
            measurement_1, require_calibrations=True, dp_mask=self._dp_mask
        )

        self._calculate_intensities_center_of_mass(
            self._intensities,
            fit_function=fit_function,
        )

        self._solve_for_center_of_mass_relative_rotation(
            rotation_angles_deg=rotation_angles_deg,
            plot_rotation=plot_rotation,
            plot_center_of_mass=plot_center_of_mass,
            maximize_divergence=maximize_divergence,
            force_com_rotation=np.rad2deg(self._rotation_best_rad),
            force_com_transpose=self._rotation_best_transpose,
            **kwargs,
        )

        (
            amplitudes_1,
            mean_diffraction_intensity_1,
        ) = self._normalize_diffraction_intensities(
            self._intensities,
            self._com_fitted_x,
            self._com_fitted_y,
        )

        # Optionally, 3rd measurement
        if self._num_sim_measurements == 3:
            (measurement_2, _, _,) = self._preprocess_datacube_and_vacuum_probe(
                self._datacube[2],
                diffraction_intensities_shape=self._diffraction_intensities_shape,
                reshaping_method=self._reshaping_method,
                probe_roi_shape=self._probe_roi_shape,
                vacuum_probe_intensity=None,
                dp_mask=None,
            )

            self._extract_intensities_and_calibrations_from_datacube(
                measurement_2, require_calibrations=True, dp_mask=self._dp_mask
            )

            self._calculate_intensities_center_of_mass(
                self._intensities,
                fit_function=fit_function,
            )

            self._solve_for_center_of_mass_relative_rotation(
                rotation_angles_deg=rotation_angles_deg,
                plot_rotation=plot_rotation,
                plot_center_of_mass=plot_center_of_mass,
                maximize_divergence=maximize_divergence,
                force_com_rotation=np.rad2deg(self._rotation_best_rad),
                force_com_transpose=self._rotation_best_transpose,
                **kwargs,
            )

            (
                amplitudes_2,
                mean_diffraction_intensity_2,
            ) = self._normalize_diffraction_intensities(
                self._intensities,
                self._com_fitted_x,
                self._com_fitted_y,
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
        del self._intensities

        self._positions_px = self._calculate_scan_positions_in_pixels(
            self._scan_positions
        )

        # Object Initialization
        if self._object is None:
            pad_x, pad_y = self._object_padding_px
            p, q = np.max(self._positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._object = xp.ones((p, q), dtype=xp.complex64)
            self._object = (self._object,) * 2
        else:
            self._object = xp.asarray(self._object, dtype=xp.complex64)

        self._object_initial = (self._object[0].copy(), self._object[1].copy())
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
        self._set_vectorized_patch_indices()

        # Probe Initialization
        if self._probe is None:
            if self._vacuum_probe_intensity is not None:
                self._semiangle_cutoff = np.inf
                self._vacuum_probe_intensity = asnumpy(self._vacuum_probe_intensity)
                probe_x0, probe_y0 = get_CoM(self._vacuum_probe_intensity)
                shift_x = self._region_of_interest_shape[0] / 2 - probe_x0
                shift_y = self._region_of_interest_shape[1] / 2 - probe_y0
                self._vacuum_probe_intensity = xp.asarray(
                    get_shifted_ar(
                        self._vacuum_probe_intensity, shift_x, shift_y, bilinear=True
                    )
                )

            self._probe = (
                ComplexProbe(
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    energy=self._energy,
                    semiangle_cutoff=self._semiangle_cutoff,
                    rolloff=self._rolloff,
                    vacuum_probe_intensity=self._vacuum_probe_intensity,
                    parameters=self._polar_parameters,
                    device="cpu" if xp is np else "gpu",
                )
                .build()
                ._array
            )

        else:
            if isinstance(self._probe, ComplexProbe):
                if self._probe._gpts != self._region_of_interest_shape:
                    raise ValueError()
                if hasattr(self._probe, "_array"):
                    self._probe = self._probe._array
                else:
                    self._probe._xp = xp
                    self._probe = self._probe.build()._array
            else:
                self._probe = xp.asarray(self._probe, dtype=xp.complex64)

        # Normalize probe to match mean diffraction intensity
        # if self._vacuum_probe_intensity is None:
        probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probe)) ** 2)
        self._probe *= np.sqrt(self._mean_diffraction_intensity / probe_intensity)

        self._probe_initial = self._probe.copy()
        self._probe_initial_fft_amplitude = xp.abs(xp.fft.fft2(self._probe_initial))

        if plot_probe_overlaps:

            shifted_probes = fft_shift(self._probe, self._positions_px_fractional, xp)
            probe_intensities = xp.abs(shifted_probes) ** 2
            probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)

            figsize = kwargs.get("figsize", (8, 4))
            cmap = kwargs.get("cmap", "Greys_r")
            kwargs.pop("figsize", None)
            kwargs.pop("cmap", None)

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
                asnumpy(xp.abs(self._probe) ** 2),
                extent=probe_extent,
                cmap=cmap,
                **kwargs,
            )
            ax1.set_xlabel("x [A]")
            ax1.set_ylabel("y [A]")
            ax1.set_title("Initial Probe Intensity")

            ax2.imshow(
                asnumpy(probe_overlap),
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            ax2.scatter(
                self.positions[:, 1],
                self.positions[:, 0],
                s=2.5,
                color=(1, 0, 0, 1),
            )
            ax2.set_xlabel("x [A]")
            ax2.set_ylabel("y [A]")
            ax2.set_xlim((extent[0], extent[1]))
            ax2.set_ylim((extent[2], extent[3]))
            ax2.set_title("Object Field of View")

            fig.tight_layout()

        self._preprocessed = True

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

        electrostatic_obj_patches = electrostatic_obj[
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

        electrostatic_obj_patches = electrostatic_obj[
            self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]
        magnetic_obj_patches = magnetic_obj[
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
        error = (
            xp.mean(xp.abs(amplitudes[0] - xp.abs(fourier_overlap)) ** 2)
            / self._mean_diffraction_intensity
        )

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
            error += (
                xp.mean(xp.abs(amp - xp.abs(fourier_overl)) ** 2)
                / self._mean_diffraction_intensity
            )

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
            DM:   a = -1, b = 1, c = 1
            AP:   a =  0, b = 1, c = 1
            RAAR: a = 1-2\beta, b = \beta, c = 2

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
        error = (
            xp.mean(xp.abs(amplitudes[0] - xp.abs(fourier_overlap)) ** 2)
            / self._mean_diffraction_intensity
        )

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
            DM:   a = -1, b = 1, c = 1
            AP:   a =  0, b = 1, c = 1
            RAAR: a = 1-2\beta, b = \beta, c = 2

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
            error += (
                xp.mean(xp.abs(amp - xp.abs(fourier_overl)) ** 2)
                / self._mean_diffraction_intensity
            )

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

            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_obj_patches * exit_waves_reverse
                )
                * probe_magnetic_normalization
                / 2
            )

            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_conj * exit_waves_forward
                )
                * probe_magnetic_normalization
                / 2
            )

            magnetic_obj -= step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_reverse
                )
                * probe_electrostatic_normalization
                / 2
            )

            magnetic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
                / 2
            )

        elif self._sim_recon_mode == 1:
            exit_waves_reverse, exit_waves_neutral, exit_waves_forward = exit_waves

            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_obj_patches * exit_waves_reverse
                )
                * probe_magnetic_normalization
                / 3
            )

            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(probe_conj * exit_waves_neutral)
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

            magnetic_obj -= step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_reverse
                )
                * probe_electrostatic_normalization
                / 2
            )

            magnetic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
                / 2
            )

        else:
            exit_waves_neutral, exit_waves_forward = exit_waves

            electrostatic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(probe_conj * exit_waves_neutral)
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

            magnetic_obj += step_size * (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
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

            magnetic_obj = (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
                / 2
            )

            magnetic_obj -= (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_reverse
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

            magnetic_obj = (
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
            exit_waves_neutral, exit_waves_forward = exit_waves

            electrostatic_obj = (
                self._sum_overlapping_patches_bincounts(probe_conj * exit_waves_neutral)
                * probe_normalization
                / 2
            )

            electrostatic_obj += (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * magnetic_conj * exit_waves_forward
                )
                * probe_magnetic_normalization
                / 2
            )

            magnetic_obj = (
                self._sum_overlapping_patches_bincounts(
                    probe_conj * electrostatic_conj * exit_waves_forward
                )
                * probe_electrostatic_normalization
            )

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

    def _position_correction(
        self,
        current_object,
        shifted_probes,
        overlap,
        amplitudes,
        current_positions,
        positions_step_size,
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

        Returns
        --------
        updated_positions: np.ndarray
            Updated positions estimate
        """

        xp = self._xp

        electrostatic_obj, _ = current_object

        obj_rolled_x_patches = electrostatic_obj[
            (self._vectorized_patch_indices_row + 1) % self._object_shape[0],
            self._vectorized_patch_indices_col,
        ]
        obj_rolled_y_patches = electrostatic_obj[
            self._vectorized_patch_indices_row,
            (self._vectorized_patch_indices_col + 1) % self._object_shape[1],
        ]

        overlap_fft = xp.fft.fft2(overlap[0])

        exit_waves_dx_fft = overlap_fft - xp.fft.fft2(
            obj_rolled_x_patches * shifted_probes
        )
        exit_waves_dy_fft = overlap_fft - xp.fft.fft2(
            obj_rolled_y_patches * shifted_probes
        )

        overlap_fft_conj = xp.conj(overlap_fft)
        estimated_intensity = xp.abs(overlap_fft) ** 2
        measured_intensity = amplitudes[0] ** 2

        flat_shape = (overlap[0].shape[0], -1)
        difference_intensity = (measured_intensity - estimated_intensity).reshape(
            flat_shape
        )

        partial_intensity_dx = 2 * xp.real(
            exit_waves_dx_fft * overlap_fft_conj
        ).reshape(flat_shape)
        partial_intensity_dy = 2 * xp.real(
            exit_waves_dy_fft * overlap_fft_conj
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

        current_positions -= positions_step_size * positions_update[..., 0]

        return current_positions

    def _update(
        self,
    ):
        """
        Dummy, absorbed into adjoint operator.
        """
        pass

    def _warmup_object_threshold_constraint(self, current_object, pure_phase_object):
        """
        Ptychographic threshold constraint.
        Used for avoiding the scaling ambiguity between probe and object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        pure_phase_object: bool
            If True, object amplitude is set to unity

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        electrostatic_obj, _ = current_object

        phase_e = xp.exp(1.0j * xp.angle(electrostatic_obj))

        if pure_phase_object:
            amplitude_e = 1.0
        else:
            amplitude_e = xp.minimum(xp.abs(electrostatic_obj), 1.0)

        electrostatic_obj = amplitude_e * phase_e

        current_object = (electrostatic_obj, None)

        return current_object

    def _object_threshold_constraint(self, current_object, pure_phase_object):
        """
        Ptychographic threshold constraint.
        Used for avoiding the scaling ambiguity between probe and object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        pure_phase_object: bool
            If True, object amplitude is set to unity

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        electrostatic_obj, magnetic_obj = current_object

        phase_e = xp.exp(1.0j * xp.angle(electrostatic_obj))
        phase_m = xp.exp(1.0j * xp.angle(magnetic_obj))

        if pure_phase_object:
            amplitude_e = 1.0
            amplitude_m = 1.0
        else:
            amplitude_e = xp.minimum(xp.abs(electrostatic_obj), 1.0)
            amplitude_m = xp.minimum(xp.abs(magnetic_obj), 1.0)

        electrostatic_obj = amplitude_e * phase_e
        magnetic_obj = amplitude_m * phase_m

        current_object = (electrostatic_obj, magnetic_obj)

        return current_object

    def _warmup_object_smoothness_constraint(
        self, current_object, gaussian_blur_sigma, pure_phase_object
    ):
        """
        Ptychographic smoothness constraint.
        Used for blurring object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_blur_sigma: float
            Standard deviation of gaussian kernel
        pure_phase_object: bool
            If True, gaussian blur performed on phase only

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        gaussian_filter = self._gaussian_filter

        electrostatic_obj, _ = current_object

        if pure_phase_object:
            phase_e = xp.angle(electrostatic_obj)
            phase_e = gaussian_filter(phase_e, gaussian_blur_sigma)
            electrostatic_obj = xp.exp(1.0j * phase_e)
        else:
            electrostatic_obj = gaussian_filter(electrostatic_obj, gaussian_blur_sigma)

        current_object = (electrostatic_obj, None)

        return current_object

    def _object_smoothness_constraint(
        self, current_object, gaussian_blur_sigma, pure_phase_object
    ):
        """
        Ptychographic smoothness constraint.
        Used for blurring object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_blur_sigma: float
            Standard deviation of gaussian kernel
        pure_phase_object: bool
            If True, gaussian blur performed on phase only

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        gaussian_filter = self._gaussian_filter

        electrostatic_obj, magnetic_obj = current_object

        if pure_phase_object:
            phase_e = xp.angle(electrostatic_obj)
            phase_m = xp.angle(magnetic_obj)

            phase_e = gaussian_filter(phase_e, gaussian_blur_sigma)
            phase_m = gaussian_filter(phase_m, gaussian_blur_sigma)

            electrostatic_obj = xp.exp(1.0j * phase_e)
            magnetic_obj = xp.exp(1.0j * phase_m)
        else:
            electrostatic_obj = gaussian_filter(electrostatic_obj, gaussian_blur_sigma)
            magnetic_obj = gaussian_filter(magnetic_obj, gaussian_blur_sigma)

        current_object = (electrostatic_obj, magnetic_obj)

        return current_object

    def _probe_center_of_mass_constraint(self, current_probe):
        """
        Ptychographic threshold constraint.
        Used for avoiding the scaling ambiguity between probe and object.

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """
        xp = self._xp
        asnumpy = self._asnumpy

        probe_center = xp.array(self._region_of_interest_shape) / 2
        probe_intensity = asnumpy(xp.abs(current_probe) ** 2)

        probe_x0, probe_y0 = get_CoM(probe_intensity)
        shifted_probe = fft_shift(
            current_probe, probe_center - xp.array([probe_x0, probe_y0]), xp
        )

        return shifted_probe

    def _probe_fourier_amplitude_constraint(self, current_probe):
        """
        Ptychographic probe Fourier-amplitude constraint.
        Used for fixing the probe's amplitude in Fourier space.

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Fourier-amplitude constrained probe estimate
        """
        xp = self._xp

        current_probe_fft = xp.fft.fft2(current_probe)
        current_probe_fft_phase = xp.angle(current_probe_fft)

        constrained_probe_fft = self._probe_initial_fft_amplitude * xp.exp(
            1j * current_probe_fft_phase
        )
        constrained_probe = xp.fft.ifft2(constrained_probe_fft)

        return constrained_probe

    def _probe_finite_support_constraint(self, current_probe):
        """
        Ptychographic probe support constraint.
        Used for penalizing focused probes to replicate sample periodicity.

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Finite-support constrained probe estimate
        """

        return current_probe * self._probe_support_mask

    def _positions_center_of_mass_constraint(self, current_positions):
        """
        Ptychographic position center of mass constraint.
        Additionally updates vectorized indices used in _overlap_projection.

        Parameters
        ----------
        current_positions: np.ndarray
            Current positions estimate

        Returns
        --------
        constrained_positions: np.ndarray
            CoM constrained positions estimate
        """
        xp = self._xp

        current_positions -= xp.mean(current_positions, axis=0) - self._positions_px_com
        self._positions_px_fractional = current_positions - xp.round(current_positions)

        self._set_vectorized_patch_indices()

        return current_positions

    def _positions_affine_transformation_constraint(
        self, initial_positions, current_positions
    ):
        """
        Constrains the updated positions to be an affine transformation of the initial scan positions,
        composing of two scale factors, a shear, and a rotation angle.

        Uses RANSAC to estimate the global transformation robustly.
        Stores the AffineTransformation in self._tf.

        Parameters
        ----------
        initial_positions: np.ndarray
            Initial scan positions
        current_positions: np.ndarray
            Current positions estimate

        Returns
        -------
        constrained_positions: np.ndarray
            Affine-transform constrained positions estimate
        """

        xp = self._xp

        tf, _ = estimate_global_transformation_ransac(
            positions0=initial_positions,
            positions1=current_positions,
            origin=self._positions_px_com,
            translation_allowed=True,
            min_sample=self._num_diffraction_patterns // 10,
            xp=xp,
        )

        self._tf = tf
        current_positions = tf(initial_positions, origin=self._positions_px_com, xp=xp)

        return current_positions

    def _constraints(
        self,
        current_object,
        current_probe,
        current_positions,
        pure_phase_object,
        gaussian_blur_sigma,
        fix_com,
        fix_probe_fourier_amplitude,
        fix_positions,
        global_affine_transformation,
        warmup_iteration,
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
        pure_phase_object: bool
            If True, object amplitude is set to unity
        gaussian_blur_sigma: float
            Standard deviation of gaussian kernel
        fix_com: bool
            If True, probe CoM is fixed to the center
        fix_probe_fourier_amplitude: bool
            If True, probe fourier amplitude is set to initial probe
        fix_positions: bool
            If True, positions are not updated

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        constrained_probe: np.ndarray
            Constrained probe estimate
        constrained_positions: np.ndarray
            Constrained positions estimate
        """

        if gaussian_blur_sigma is not None:
            if warmup_iteration:
                current_object = self._warmup_object_smoothness_constraint(
                    current_object, gaussian_blur_sigma, pure_phase_object
                )
            else:
                current_object = self._object_smoothness_constraint(
                    current_object, gaussian_blur_sigma, pure_phase_object
                )

        if warmup_iteration:
            current_object = self._warmup_object_threshold_constraint(
                current_object, pure_phase_object
            )

        else:
            current_object = self._object_threshold_constraint(
                current_object, pure_phase_object
            )

        if fix_probe_fourier_amplitude:
            current_probe = self._probe_fourier_amplitude_constraint(current_probe)

        current_probe = self._probe_finite_support_constraint(current_probe)

        if fix_com:
            current_probe = self._probe_center_of_mass_constraint(current_probe)

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
        max_batch_size: int = None,
        seed_random: int = None,
        step_size: float = 0.9,
        normalization_min: float = 1e-3,
        positions_step_size: float = 0.9,
        pure_phase_object_iter: int = 0,
        fix_com: bool = True,
        fix_probe_iter: int = 0,
        warmup_iter: int = 0,
        fix_probe_fourier_amplitude_iter: int = 0,
        fix_positions_iter: int = np.inf,
        global_affine_transformation: bool = True,
        probe_support_relative_radius: float = 1.0,
        probe_support_supergaussian_degree: float = 10.0,
        gaussian_blur_sigma: float = None,
        gaussian_blur_iter: int = np.inf,
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
            "DM_AP" (or "difference-map_alternating-projections"),
            "RAAR" (or "relaxed-averaged-alternating-reflections"),
            "GD" (or "gradient_descent")
        reconstruction_parameter: float, optional
            Tuning parameter to interpolate b/w DM-AP and DM-RAAR
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
        fix_com: bool, optional
            If True, fixes center of mass of probe
        fix_probe_iter: int, optional
            Number of iterations to run with a fixed probe before updating probe estimate
        fix_probe_amplitude: int, optional
            Number of iterations to run with a fixed probe amplitude
        fix_positions_iter: int, optional
            Number of iterations to run with fixed positions before updating positions estimate
        global_affine_transformation: bool, optional
            If True, positions are assumed to be a global affine transform from initial scan
        probe_support_relative_radius: float, optional
            Radius of probe supergaussian support in scaled pixel units, between (0,1]
        probe_support_supergaussian_degree: float, optional
            Degree supergaussian support is raised to, higher is sharper cutoff
        gaussian_blur_sigma: float, optional
            Standard deviation of gaussian kernel
        gaussian_blur_iter: int, optional
            Number of iterations to run before applying object smoothness constraint
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
        if reconstruction_parameter < 0.0 or reconstruction_parameter > 1.0:
            raise ValueError("reconstruction_parameter must be between 0-1.")

        if (
            reconstruction_method == "DM_AP"
            or reconstruction_method == "difference-map_alternating-projections"
        ):
            use_projection_scheme = True
            projection_a = -reconstruction_parameter
            projection_b = 1
            projection_c = 1 + reconstruction_parameter
            step_size = None
        elif (
            reconstruction_method == "RAAR"
            or reconstruction_method == "relaxed-averaged-alternating-reflections"
        ):
            use_projection_scheme = True
            projection_a = 1 - 2 * reconstruction_parameter
            projection_b = reconstruction_parameter
            projection_c = 2
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
                    "reconstruction_method must be one of 'DM_AP' (or 'difference-map_alternating-projections'), "
                    "'RAAR' (or 'relaxed-averaged-alternating-reflections'), "
                    f"or 'GD' (or 'gradient-descent'), not  {reconstruction_method}."
                )
            )

        if self._verbose:
            if max_batch_size is not None:
                print(
                    (
                        f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                        f"in batches of max {max_batch_size} measurements."
                    )
                )
            else:
                if reconstruction_parameter is not None:
                    print(
                        (
                            f"Performing {max_iter} iterations using the {reconstruction_method} algorithm, "
                            f"with α: {reconstruction_parameter}."
                        )
                    )
                else:
                    print(
                        f"Performing {max_iter} iterations using the {reconstruction_method} algorithm."
                    )

        # Batching
        shuffled_indices = np.arange(self._num_diffraction_patterns)
        unshuffled_indices = np.zeros_like(shuffled_indices)

        if max_batch_size is not None:
            xp.random.seed(seed_random)
            if use_projection_scheme:
                raise ValueError(
                    (
                        "Stochastic object/probe updating is inconsistent with 'DM_AP' and 'RAAR'. "
                        "Use reconstruction_method='GD' or set max_batch_size=None."
                    )
                )

        else:
            max_batch_size = self._num_diffraction_patterns

        # initialization
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []
            self.error_iterations = []

        if reset:
            self._object = (
                self._object_initial[0].copy(),
                self._object_initial[1].copy(),
            )
            self._probe = self._probe_initial.copy()
            self._positions_px = self._positions_px_initial.copy()
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            self._set_vectorized_patch_indices()
            self._exit_waves = (None,) * self._num_sim_measurements
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
                self._exit_waves = (None,) * self._num_sim_measurements

        # Probe support mask initialization
        x = xp.linspace(-1, 1, self._region_of_interest_shape[0], endpoint=False)
        y = xp.linspace(-1, 1, self._region_of_interest_shape[1], endpoint=False)
        xx, yy = xp.meshgrid(x, y, indexing="ij")
        self._probe_support_mask = xp.exp(
            -(
                (
                    (xx / probe_support_relative_radius) ** 2
                    + (yy / probe_support_relative_radius) ** 2
                )
                ** probe_support_supergaussian_degree
            )
        )

        # main loop
        for a0 in tqdmnd(
            max_iter,
            desc="Reconstructing object and probe",
            unit=" iter",
            disable=not progress_bar,
        ):

            error = 0.0

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
                self._set_vectorized_patch_indices()

                amps = []
                for amplitudes in self._amplitudes:
                    amps.append(amplitudes[shuffled_indices][start:end])
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
                        self._object,
                        shifted_probes,
                        overlap,
                        amplitudes,
                        self._positions_px,
                        positions_step_size,
                    )

                error += batch_error

            # constraints
            gaussian_sigma = gaussian_blur_sigma if a0 < gaussian_blur_iter else None

            self._positions_px = positions_px.copy()[unshuffled_indices]
            self._object, self._probe, self._positions_px = self._constraints(
                self._object,
                self._probe,
                self._positions_px,
                pure_phase_object=a0 < pure_phase_object_iter,
                gaussian_blur_sigma=gaussian_sigma,
                fix_com=fix_com and a0 >= fix_probe_iter,
                fix_probe_fourier_amplitude=a0 < fix_probe_fourier_amplitude_iter,
                fix_positions=a0 < fix_positions_iter,
                global_affine_transformation=global_affine_transformation,
                warmup_iteration=a0 < warmup_iter,
            )

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
                self.probe_iterations.append(asnumpy(self._probe.copy()))
                self.error_iterations.append(error.item())

        # store result
        if a0 < warmup_iter:
            self.object = (asnumpy(self._object[0]), None)
        else:
            self.object = (asnumpy(self._object[0]), asnumpy(self._object[1]))
        self.probe = asnumpy(self._probe)
        self.error = error.item()

        return self

    def _visualize_last_iteration(
        self,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        object_mode: str,
        padding: int,
        **kwargs,
    ):
        """
        Displays last reconstructed object and probe iterations.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        object_mode: str
            Specifies the attribute of the object to plot.
            One of 'phase', 'amplitude', 'intensity'

        """
        figsize = kwargs.get("figsize", (8, 5))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

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

        rotated_electrostatic = self._crop_rotate_object_fov(
            self.object[0], padding=padding
        )
        rotated_magnetic = self._crop_rotate_object_fov(self.object[1], padding=padding)

        if plot_convergence:
            if plot_probe:
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
            if plot_probe:
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
                spec = GridSpec(ncols=1, nrows=1)

        fig = plt.figure(figsize=figsize)

        if plot_probe:

            # Electrostatic Object
            ax = fig.add_subplot(spec[0, 0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(rotated_electrostatic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(rotated_electrostatic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(rotated_electrostatic) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Electrostatic object {object_mode}")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Magnetic Object
            ax = fig.add_subplot(spec[0, 1])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(rotated_magnetic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(rotated_magnetic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(rotated_magnetic) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Magnetic object {object_mode}")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Probe
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            ax = fig.add_subplot(spec[0, 2])
            im = ax.imshow(
                np.abs(self.probe) ** 2,
                extent=probe_extent,
                cmap="Greys_r",
                **kwargs,
            )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title("Reconstructed probe intensity")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        else:
            # Electrostatic Object
            ax = fig.add_subplot(spec[0, 0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(rotated_electrostatic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(rotated_electrostatic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(rotated_electrostatic) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Electrostatic object {object_mode}")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Magnetic Object
            ax = fig.add_subplot(spec[0, 1])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(rotated_magnetic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(rotated_magnetic),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(rotated_magnetic) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Magnetic object {object_mode}")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "error_iterations"):
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            errors = self.error_iterations
            if plot_probe:
                ax = fig.add_subplot(spec[1, :])
            else:
                ax = fig.add_subplot(spec[1])

            ax.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Log RMS error")
            ax.yaxis.tick_right()

        fig.suptitle(f"RMS error: {self.error:.3e}")
        spec.tight_layout(fig)

    def _visualize_all_iterations(
        self,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        iterations_grid: Tuple[int, int],
        object_mode: str,
        padding: int,
        **kwargs,
    ):
        """
        Displays all reconstructed object and probe iterations.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        object_mode: str
            Specifies the attribute of the object to plot.
            One of 'phase', 'amplitude', 'intensity'

        """
        raise NotImplementedError()

    def visualize(
        self,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        plot_probe: bool = True,
        object_mode: str = "phase",
        cbar: bool = False,
        padding: int = 0,
        **kwargs,
    ):
        """
        Displays reconstructed object and probe.

        Parameters
        --------
        plot_convergence: bool, optional
            If true, the RMS error plot is displayed
        iterations_grid: Tuple[int,int]
            Grid dimensions to plot reconstruction iterations
        cbar: bool, optional
            If true, displays a colorbar
        plot_probe: bool
            If true, the reconstructed probe intensity is also displayed
        object_mode: str
            Specifies the attribute of the object to plot.
            One of 'phase', 'amplitude', 'intensity'

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """

        if (
            object_mode != "phase"
            and object_mode != "amplitude"
            and object_mode != "intensity"
        ):
            raise ValueError(
                (
                    "object_mode needs to be one of 'phase', 'amplitude', "
                    f"or 'intensity', not {object_mode}"
                )
            )

        if iterations_grid is None:
            self._visualize_last_iteration(
                plot_convergence=plot_convergence,
                plot_probe=plot_probe,
                object_mode=object_mode,
                cbar=cbar,
                padding=padding,
                **kwargs,
            )
        else:
            self._visualize_all_iterations(
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                plot_probe=plot_probe,
                object_mode=object_mode,
                cbar=cbar,
                padding=padding,
                **kwargs,
            )

        return self
