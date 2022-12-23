"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely DPC and ptychography.
"""

import warnings
from typing import Mapping, Tuple

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
    fft_shift,
    polar_aliases,
    polar_symbols,
)
from py4DSTEM.process.utils import get_CoM, get_shifted_ar, fourier_resample
from py4DSTEM.utils.tqdmnd import tqdmnd

warnings.simplefilter(action="always", category=UserWarning)


class PtychographicReconstruction(PhaseReconstruction):
    """
    Iterative Ptychographic Reconstruction Class.

    Diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed probe dimensions      : (Sx,Sy)
    Reconstructed object dimensions     : (Px,Py)

    such that (Sx,Sy) is the region-of-interest (ROI) size of our probe
    and (Px,Py) is the padded-object size we position our ROI around in.

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    diffraction_intensities_shape: Tuple[int,int], optional
        Pixel dimensions (Qx',Qy') of the resampled diffraction intensities
        If None, no resampling of diffraction intenstities is performed
    resampling_method: str, optional
        Method to use for resampling, either 'bilinear' or 'fourier' (default)
    region_of_interest_shape: Tuple[int,int], optional
        Pixel dimensions (Sx,Sy) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the intensity dimensions (Qx,Qy)
    object_padding_px: Tuple[int,int], optional
        Pixel dimensions to pad object with
        If None, the padding is set to half the probe ROI dimensions
    initial_object_guess: np.ndarray, optional
        Initial guess for complex-valued object of dimensions (Px,Py)
        If None, initialized to 1.0j
    initial_probe_guess: np.ndarray, optional
        Initial guess for complex-valued probe of dimensions (Sx,Sy). If None,
        initialized to ComplexProbe with semiangle_cutoff, energy, and aberrations
    scan_positions: np.ndarray, optional
        Probe positions in Å for each diffraction intensity
        If None, initialized to a grid scan
    dp_mask: ndarray, optional
        Mask for datacube intensities (Qx,Qy)
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial probe guess
    vacuum_probe_intensity: np.ndarray, optional
        Vacuum probe to use as intensity aperture for initial probe guess
    polar_parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in Å and angles should be given in radians.
    kwargs:
        Provide the aberration coefficients as keyword arguments.

    Assigns
    --------
    self._xp: Callable
        Array computing module
    self._intensities: (Rx,Ry,Qx,Qy) xp.ndarray
        Raw intensities array stored on device, with dtype xp.float32
    self._preprocessed: bool
        Flag to signal object has not yet been preprocessed
    """

    def __init__(
        self,
        datacube: DataCube,
        energy: float,
        diffraction_intensities_shape: Tuple[int, int] = None,
        resampling_method: str = 'fourier',
        probe_roi_shape: Tuple[int, int] = None,
        object_padding_px: Tuple[int, int] = None,
        initial_object_guess: np.ndarray = None,
        initial_probe_guess: np.ndarray = None,
        scan_positions: np.ndarray = None,
        vacuum_probe_intensity: np.ndarray = None,
        semiangle_cutoff: float = None,
        polar_parameters: Mapping[str, float] = None,
        dp_mask: np.ndarray = None,
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
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._diffraction_intensities_shape = diffraction_intensities_shape
        self._resampling_method = resampling_method
        self._probe_roi_shape = probe_roi_shape
        self._object = initial_object_guess
        self._probe = initial_probe_guess
        self._scan_positions = scan_positions
        self._datacube = datacube
        self._dp_mask = dp_mask
        self._verbose = verbose
        self._object_padding_px = object_padding_px
        self._preprocessed = False
        
    def _preprocess_datacube_and_vacuum_probe(
        self,
        datacube, 
        diffraction_intensities_shape=None,
        resampling_method = 'fourier',
        probe_roi_shape = None,
        vacuum_probe_intensity=None,
    ):
        """
        """

        if diffraction_intensities_shape is not None:
            
            Qx,Qy = datacube.shape[-2:]
            Sx,Sy = diffraction_intensities_shape

            resampling_factor_x = Sx/Qx
            resampling_factor_y = Sy/Qy

            if resampling_factor_x != resampling_factor_y:
                raise ValueError("Datacube calibration can only handle uniform Q-sampling.")

            Q_pixel_size = datacube.calibration.get_Q_pixel_size()
            datacube = datacube.resample_Q(N=resampling_factor_x, method=resampling_method)
            datacube.calibration.set_Q_pixel_size(Q_pixel_size/resampling_factor_x)

            if vacuum_probe_intensity is not None:
                vacuum_probe_intensity = fourier_resample(vacuum_probe_intensity, output_size = diffraction_intensities_shape, force_nonnegative=True)

        if probe_roi_shape is not None:
            
            Qx,Qy = datacube.shape[-2:]
            Sx,Sy = probe_roi_shape
            datacube = datacube.pad_Q(output_size=probe_roi_shape)
            
            if vacuum_probe_intensity is not None:
                
                pad_kx = Sx - Qx
                pad_kx = (pad_kx//2, pad_kx//2 + pad_kx %2)

                pad_ky = Sy - Qy
                pad_ky = (pad_ky//2, pad_ky//2 + pad_ky %2)

                vacuum_probe_intensity = np.pad(vacuum_probe_intensity,pad_width=(pad_kx,pad_ky),mode='constant')
     
        return datacube, vacuum_probe_intensity

    def preprocess(
        self,
        fit_function: str = "plane",
        plot_center_of_mass: str = 'default',
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

        Assigns
        --------
        self._preprocessed: bool
            Flag to signal object has been preprocessed

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        xp = self._xp
        asnumpy = self._asnumpy
        
        self._datacube, self._vacuum_probe_intensity = self._preprocess_datacube_and_vacuum_probe(
            self._datacube, 
            diffraction_intensities_shape=self._diffraction_intensities_shape,
            resampling_method = self._resampling_method,
            probe_roi_shape = self._probe_roi_shape,
            vacuum_probe_intensity = self._vacuum_probe_intensity,
        )

        self._extract_intensities_and_calibrations_from_datacube(
            self._datacube, require_calibrations=True, dp_mask=self._dp_mask
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
            self._amplitudes,
            self._mean_diffraction_intensity,
        ) = self._normalize_diffraction_intensities(
            self._intensities,
            self._com_fitted_x,
            self._com_fitted_y,
        )

        # explicitly delete namespace
        self._num_diffraction_patterns = self._amplitudes.shape[0]
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
        else:
            self._object = xp.asarray(self._object, dtype=xp.complex64)

        self._object_initial = self._object.copy()

        self._positions_px = xp.asarray(self._positions_px, dtype=xp.float32)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)
        self._positions_px_fractional = self._positions_px - xp.round(
            self._positions_px
        )

        self._positions_px_initial = self._positions_px.copy()
        self._positions_initial = self._positions_px_initial.copy()
        self._positions_initial[:, 0] *= self.sampling[0]
        self._positions_initial[:, 1] *= self.sampling[1]

        # Vectorized Patches
        self._object_shape = self._object.shape
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
        fourier_exit_waves:np.ndarray
            object * probe overlaps in reciprocal space
        """

        xp = self._xp

        self._shifted_probes = fft_shift(
            current_probe, self._positions_px_fractional, xp
        )
        exit_waves = (
            current_object[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        return xp.fft.fft2(exit_waves)

    def _fourier_projection(self, amplitudes, fourier_exit_waves):
        """
        Ptychographic fourier projection method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        fourier_exit_waves: np.ndarray
            object * probe overlaps in reciprocal space

        Returns
        --------
        difference_gradient_fourier:np.ndarray
            Difference between measured and estimated exit waves in reciprocal space
        error: float
            Reconstruction error
        """

        xp = self._xp
        abs_fourier_exit_waves = xp.abs(fourier_exit_waves)
        error = (
            xp.mean(xp.abs(amplitudes - abs_fourier_exit_waves) ** 2)
            / self._mean_diffraction_intensity
        )

        difference_gradient_fourier = (amplitudes - abs_fourier_exit_waves) * xp.exp(
            1j * xp.angle(fourier_exit_waves)
        )

        return difference_gradient_fourier, error

    def _forward(self, current_object, current_probe, amplitudes):
        """
        Ptychographic forward operator.
        Calls _overlap_projection() and _fourier_projection().

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        amplitudes: np.ndarray
            Normalized measured amplitudes

        Returns
        --------
        difference_gradient_fourier:np.ndarray
            Difference between measured and estimated exit waves in reciprocal space
        error: float
            Reconstruction error
        """

        self._fourier_exit_waves = self._overlap_projection(
            current_object, current_probe
        )
        difference_gradient_fourier, error = self._fourier_projection(
            amplitudes, self._fourier_exit_waves
        )

        return difference_gradient_fourier, error

    def _position_correction(
        self,
        current_object,
        current_probe,
        difference_gradient,
    ):
        """
        Ptychographic parallel position correction.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        difference_gradient: np.ndarray
            Difference between measured and estimated exit waves

        Returns
        --------
        positions_update: np.ndarray
            Negative positions gradient. If fix_positions is True returns None
        """

        xp = self._xp
        dx, dy = self._scan_sampling

        obj_dx = (
            xp.roll(current_object, -1, axis=0) - xp.roll(current_object, 1, axis=0)
        ) / (2 * dx)
        obj_dy = (
            xp.roll(current_object, -1, axis=1) - xp.roll(current_object, 1, axis=1)
        ) / (2 * dy)

        exit_waves_dx = (
            obj_dx[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        exit_waves_dy = (
            obj_dy[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        displacements_x = xp.sum(
            xp.real(xp.conj(exit_waves_dx) * difference_gradient), axis=(-2, -1)
        ) / xp.sum(xp.abs(exit_waves_dx) ** 2, axis=(-2, -1))

        displacements_y = xp.sum(
            xp.real(xp.conj(exit_waves_dy) * difference_gradient), axis=(-2, -1)
        ) / xp.sum(xp.abs(exit_waves_dy) ** 2, axis=(-2, -1))

        positions_update = xp.column_stack((displacements_x, displacements_y))

        return positions_update

    def _position_correction_alternative(
        self, current_object, current_probe, amplitudes
    ):
        """
        Alternative ptychographic parallel position correction.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        current_probe: np.ndarray
            Current probe estimate
        amplitudes: np.ndarray
            Measured amplitudes

        Returns
        --------
        positions_update: np.ndarray
            Negative positions gradient. If fix_positions is True returns None
        """

        xp = self._xp
        # dx, dy = self._scan_sampling
        dx, dy = (1.0, 1.0)

        obj_dx = (
            xp.roll(current_object, -1, axis=0) - xp.roll(current_object, 1, axis=0)
        ) / (2 * dx)
        obj_dy = (
            xp.roll(current_object, -1, axis=1) - xp.roll(current_object, 1, axis=1)
        ) / (2 * dy)

        exit_waves_dx_fft = xp.fft.fft2(
            obj_dx[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        exit_waves_dy_fft = xp.fft.fft2(
            obj_dy[
                self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
            ]
            * self._shifted_probes
        )

        fourier_exit_waves_conj = xp.conj(self._fourier_exit_waves)
        estimated_intensity = xp.abs(self._fourier_exit_waves) ** 2
        measured_intensity = amplitudes**2

        flat_shape = (self._num_diffraction_patterns, -1)
        difference_intensity = (measured_intensity - estimated_intensity).reshape(
            flat_shape
        )

        partial_intensity_dx = 2 * xp.real(
            exit_waves_dx_fft * fourier_exit_waves_conj
        ).reshape(flat_shape)
        partial_intensity_dy = 2 * xp.real(
            exit_waves_dy_fft * fourier_exit_waves_conj
        ).reshape(flat_shape)

        coefficients_matrix = xp.dstack((partial_intensity_dx, partial_intensity_dy))

        positions_update = xp.einsum(
            "idk,ik->id", xp.linalg.pinv(coefficients_matrix), difference_intensity
        )

        return positions_update

    def _adjoint(
        self,
        current_object,
        current_probe,
        difference_gradient_fourier,
        fix_probe: bool,
        fix_positions: bool,
        alternative_position_correction: bool,
        normalization_min: float,
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
        difference_gradient_fourier:np.ndarray
            Difference between measured and estimated exit waves in reciprocal space
        fix_probe: bool, optional
            If True, probe will not be updated
        fix_positions: bool, optional
            If True, positions will not be updated
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity

        Returns
        --------
        object_update: np.ndarray
            Negative object gradient
        probe_update: np.ndarray
            Negative probe gradient. If fix_probe is True returns None
        positions_update: np.ndarray
            Negative positions gradient. If fix_positions is True returns None
        """

        xp = self._xp

        difference_gradient = xp.fft.ifft2(difference_gradient_fourier)

        probe_normalization = self._sum_overlapping_patches_bincounts(
            xp.abs(self._shifted_probes) ** 2
        )
        probe_normalization = 1 / xp.sqrt(
            probe_normalization**2
            + (normalization_min * xp.max(probe_normalization)) ** 2
        )

        object_update = (
            self._sum_overlapping_patches_bincounts(
                xp.conj(self._shifted_probes) * difference_gradient
            )
            * probe_normalization
        )

        if fix_probe:
            probe_update = None
        else:
            object_normalization = xp.sum(
                (xp.abs(current_object) ** 2)[
                    self._vectorized_patch_indices_row,
                    self._vectorized_patch_indices_col,
                ],
                axis=0,
            )
            object_normalization = 1 / xp.sqrt(
                object_normalization**2
                + (normalization_min * xp.max(object_normalization)) ** 2
            )

            probe_update = (
                xp.sum(
                    xp.conj(current_object)[
                        self._vectorized_patch_indices_row,
                        self._vectorized_patch_indices_col,
                    ]
                    * difference_gradient,
                    axis=0,
                )
                * object_normalization
            )

        if fix_positions:
            positions_update = None
        elif alternative_position_correction:
            positions_update = self._position_correction_alternative(
                current_object,
                current_probe,
                self._amplitudes,
            )

        else:
            positions_update = self._position_correction(
                current_object,
                current_probe,
                difference_gradient,
            )

        return object_update, probe_update, positions_update

    def _update(
        self,
        current_object,
        object_update,
        current_probe,
        probe_update,
        current_positions,
        positions_update,
        step_size: float = 0.9,
        positions_step_size: float = 0.9,
    ):
        """
        Ptychographic update operator.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        object_update: np.ndarray
            Negative object gradient
        current_probe: np.ndarray
            Current probe estimate
        probe_update: np.ndarray
            Negative probe gradient
        current_positions: np.ndarray
            Current positions estimate
        positions_update: np.ndarray
            Negative positions gradient
        step_size: float, optional
            Update step size
        positions_step_size: float, optional
            Positions update step size

        Returns
        --------
        updated_object: np.ndarray
            Updated object estimate
        updated_probe: np.ndarray
            Updated probe estimate
        updated_positions: np.ndarray
            Updated positions estimate
        """
        current_object += step_size * object_update
        if probe_update is not None:
            current_probe += step_size * probe_update
        if positions_update is not None:
            current_positions += positions_step_size * positions_update
        return current_object, current_probe, current_positions

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
        phase = xp.exp(1.0j * xp.angle(current_object))
        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(current_object), 1.0)
        return amplitude * phase

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

        if pure_phase_object:
            amplitude = xp.abs(current_object)
            phase = xp.angle(current_object)
            phase = gaussian_filter(phase, gaussian_blur_sigma)
            current_object = amplitude * xp.exp(1.0j * phase)
        else:
            current_object = gaussian_filter(current_object, gaussian_blur_sigma)

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
        --------
        current_positions: np.ndarray
            Current positions estimate

        Assigns
        --------
        self._positions_px_fractional: np.ndarray
            Updated fractional positions used to shift probes

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
            current_object = self._object_smoothness_constraint(
                current_object, gaussian_blur_sigma, pure_phase_object
            )

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

        return current_object, current_probe, current_positions

    def reconstruct(
        self,
        reset: bool = None,
        max_iter: int = 64,
        step_size: float = 0.9,
        positions_step_size: float = 0.9,
        normalization_min: float = 1e-2,
        fix_com: bool = True,
        fix_probe_iter: int = 0,
        fix_positions_iter: int = np.inf,
        alternative_position_correction: bool = False,
        probe_support_relative_radius: float = 1.0,
        probe_support_supergaussian_degree: float = 10.0,
        fix_probe_fourier_amplitude_iter: int = 0,
        pure_phase_object_iter: int = 0,
        gaussian_blur_sigma: float = None,
        gaussian_blur_iter: int = np.inf,
        progress_bar: bool = True,
        store_iterations: bool = False,
    ):
        """
        Ptychographic reconstruction main method.

        Parameters
        --------
        reset: bool, optional
            If True, previous reconstructions are ignored
        max_iter: int, optional
            Maximum number of iterations to run
        step_size: float, optional
            Update step size
        positions_step_size: float, optional
            Positions update step size
        normalization_min: float, optional
            Probe normalization minimum as a fraction of the maximum overlap intensity
        fix_probe_iter: int, optional
            Number of iterations to run with a fixed probe before updating probe estimate
        fix_positions_iter: int, optional
            Number of iterations to run with fixed positions before updating positions estimate
        fix_com: bool, optional
            If True, fixes center of mass of probe
        gaussian_blur_sigma: float, optional
            Standard deviation of gaussian kernel
        gaussian_blur_iter: int, optional
            Number of iterations to run before applying object smoothness constraint
        probe_support_relative_radius: float, optional
            Radius of probe supergaussian support in scaled pixel units, between (0,1]
        probe_support_supergaussian_degree: float, optional
            Degree supergaussian support is raised to, higher is sharper cutoff
        fix_probe_amplitude: int, optional
            Number of iterations to run with a fixed probe amplitude
        pure_phase_object: bool, optional
            If True, object amplitude is set to unity
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        store_iterations: bool, optional
            If True, reconstructed objects and probes are stored at each iteration

        Returns
        --------
        self: PtychographicReconstruction
            Self to accommodate chaining
        """
        asnumpy = self._asnumpy
        xp = self._xp

        # initialization
        if store_iterations and (not hasattr(self, "object_iterations") or reset):
            self.object_iterations = []
            self.probe_iterations = []
            self.error_iterations = []

        if reset:
            self._object = self._object_initial.copy()
            self._probe = self._probe_initial.copy()
            self._positions_px = self._positions_px_initial.copy()
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            self._set_vectorized_patch_indices()
        elif reset is None and hasattr(self, "error"):
            warnings.warn(
                (
                    "Continuing reconstruction from previous result. "
                    "Use reset=True for a fresh start."
                ),
                UserWarning,
            )

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

            # forward operator
            difference_gradient_fourier, error = self._forward(
                self._object, self._probe, self._amplitudes
            )

            # adjoint operator
            object_update, probe_update, positions_update = self._adjoint(
                self._object,
                self._probe,
                difference_gradient_fourier,
                fix_probe=a0 < fix_probe_iter,
                fix_positions=a0 < fix_positions_iter,
                normalization_min=normalization_min,
                alternative_position_correction=alternative_position_correction,
            )

            # update
            self._object, self._probe, self._positions_px = self._update(
                self._object,
                object_update,
                self._probe,
                probe_update,
                self._positions_px,
                positions_update,
                step_size=step_size,
                positions_step_size=positions_step_size,
            )

            # constraints
            gaussian_sigma = gaussian_blur_sigma if a0 < gaussian_blur_iter else None

            self._object, self._probe, self._positions_px = self._constraints(
                self._object,
                self._probe,
                self._positions_px,
                pure_phase_object=a0 < pure_phase_object_iter,
                gaussian_blur_sigma=gaussian_sigma,
                fix_com=fix_com and a0 >= fix_probe_iter,
                fix_probe_fourier_amplitude=a0 < fix_probe_fourier_amplitude_iter,
                fix_positions=a0 < fix_positions_iter,
            )

            if store_iterations:
                self.object_iterations.append(asnumpy(self._object.copy()))
                self.probe_iterations.append(asnumpy(self._probe.copy()))
                self.error_iterations.append(error.item())

        # store result
        self.object = asnumpy(self._object)
        self.probe = asnumpy(self._probe)
        self.error = error.item()

        return self

    def _visualize_last_iteration(
        self,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        object_mode: str,
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

        if plot_convergence:
            if plot_probe:
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
            if plot_probe:
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

        fig = plt.figure(figsize=figsize)

        if plot_probe:

            # Object
            ax = fig.add_subplot(spec[0, 0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(self.object),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(self.object),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(self.object) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Reconstructed object {object_mode}")

            if cbar:
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

            # Probe
            ax = fig.add_subplot(spec[0, 1])
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
            ax = fig.add_subplot(spec[0])
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(self.object),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(self.object),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            else:
                im = ax.imshow(
                    np.abs(self.object) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title(f"Reconstructed object {object_mode}")

            if cbar:

                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)

        if plot_convergence and hasattr(self, "error_iterations"):
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
        if iterations_grid == "auto":
            iterations_grid = (2, 4)
        else:
            if plot_probe and iterations_grid[0] != 2:
                raise ValueError()

        figsize = kwargs.get("figsize", (12, 7))
        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("figsize", None)
        kwargs.pop("cmap", None)

        errors = self.error_iterations
        objects = self.object_iterations

        if plot_probe:
            total_grids = (np.prod(iterations_grid) / 2).astype("int")
            probes = self.probe_iterations
        else:
            total_grids = np.prod(iterations_grid)
        max_iter = len(objects) - 1
        grid_range = range(0, max_iter + 1, max_iter // (total_grids - 1))

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

        if plot_convergence:
            if plot_probe:
                spec = GridSpec(ncols=1, nrows=3, height_ratios=[4, 4, 1], hspace=0)
            else:
                spec = GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0)
        else:
            if plot_probe:
                spec = GridSpec(ncols=1, nrows=2)
            else:
                spec = GridSpec(ncols=1, nrows=1)

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
            if object_mode == "phase":
                im = ax.imshow(
                    np.angle(objects[grid_range[n]]),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
                ax.set_title(f"Iter: {grid_range[n]} Phase")
            elif object_mode == "amplitude":
                im = ax.imshow(
                    np.abs(objects[grid_range[n]]),
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
                ax.set_title(f"Iter: {grid_range[n]} Amplitude")
            else:
                im = ax.imshow(
                    np.abs(objects[grid_range[n]]) ** 2,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )
                ax.set_title(f"Iter: {grid_range[n]} Intensity")

            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            if cbar:
                grid.cbar_axes[n].colorbar(im)

        if plot_probe:
            grid = ImageGrid(
                fig,
                spec[1],
                nrows_ncols=(1, iterations_grid[1]),
                axes_pad=(0.75, 0.5) if cbar else 0.5,
                cbar_mode="each" if cbar else None,
                cbar_pad="2.5%" if cbar else None,
            )

            for n, ax in enumerate(grid):
                im = ax.imshow(
                    np.abs(probes[grid_range[n]]) ** 2,
                    extent=probe_extent,
                    cmap="Greys_r",
                    **kwargs,
                )
                ax.set_title(f"Iter: {grid_range[n]} Probe")

                ax.set_xlabel("x [A]")
                ax.set_ylabel("y [A]")

                if cbar:
                    grid.cbar_axes[n].colorbar(im)

        if plot_convergence:
            if plot_probe:
                ax2 = fig.add_subplot(spec[2])
            else:
                ax2 = fig.add_subplot(spec[1])
            ax2.semilogy(np.arange(len(errors)), errors, **kwargs)
            ax2.set_xlabel("Iteration Number")
            ax2.set_ylabel("Log RMS error")
            ax2.yaxis.tick_right()

        spec.tight_layout(fig)

    def visualize(
        self,
        iterations_grid: Tuple[int, int] = None,
        plot_convergence: bool = True,
        plot_probe: bool = True,
        object_mode: str = "phase",
        cbar: bool = False,
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
                **kwargs,
            )
        else:
            self._visualize_all_iterations(
                plot_convergence=plot_convergence,
                iterations_grid=iterations_grid,
                plot_probe=plot_probe,
                object_mode=object_mode,
                cbar=cbar,
                **kwargs,
            )

        return self

    @property
    def positions(self):
        """Probe positions [A]"""

        if self.angular_sampling is None:
            return None

        asnumpy = self._asnumpy

        positions = self._positions_px.copy()
        positions[:, 0] *= self.sampling[0]
        positions[:, 1] *= self.sampling[1]

        return asnumpy(positions)
