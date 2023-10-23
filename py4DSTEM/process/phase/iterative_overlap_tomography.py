"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely overlap tomography.
"""

import warnings
from typing import Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pylops
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from py4DSTEM.visualize import show
from py4DSTEM.visualize.vis_special import Complex2RGB, add_colorbar_arg
from scipy.ndimage import rotate as rotate_np

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
    spatial_frequencies,
)
from py4DSTEM.process.utils import electron_wavelength_angstrom, get_CoM, get_shifted_ar

warnings.simplefilter(action="always", category=UserWarning)


class OverlapTomographicReconstruction(PtychographicReconstruction):
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
    name: str, optional
        Class name
    kwargs:
        Provide the aberration coefficients as keyword arguments.
    """

    # Class-specific Metadata
    _class_specific_metadata = ("_num_slices", "_tilt_orientation_matrices")
    _swap_zxy_to_xyz = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

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
        self._num_slices = num_slices
        self._tilt_orientation_matrices = tuple(tilt_orientation_matrices)
        self._num_tilts = num_tilts

    def _precompute_propagator_arrays(
        self,
        gpts: Tuple[int, int],
        sampling: Tuple[float, float],
        energy: float,
        slice_thicknesses: Sequence[float],
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
            propagators[i] *= xp.exp(
                1.0j * (-(ky**2)[None] * np.pi * wavelength * dz)
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

    def _project_sliced_object(self, array: np.ndarray, output_z):
        """
        Expands supersliced object or projects voxel-sliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to expand/project
        output_z: int
            Output_dimension to expand/project array to.
            If output_z > array.shape[0] array is expanded, else it's projected

        Returns
        -------
        expanded_or_projected_array: np.ndarray
            expanded or projected array
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
        Expands supersliced object or projects voxel-sliced object.

        Parameters
        ----------
        array: np.ndarray
            3D array to expand/project
        output_z: int
            Output_dimension to expand/project array to.
            If output_z > array.shape[0] array is expanded, else it's projected

        Returns
        -------
        expanded_or_projected_array: np.ndarray
            expanded or projected array
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
    ):
        """ """

        xp = self._xp
        affine_transform = self._affine_transform
        swap_zxy_to_xyz = self._swap_zxy_to_xyz

        volume = volume_array.copy()
        volume_shape = xp.asarray(volume.shape)
        tf = xp.asarray(swap_zxy_to_xyz.T @ rot_matrix.T @ swap_zxy_to_xyz)

        in_center = (volume_shape - 1) / 2
        out_center = tf @ in_center
        offset = in_center - out_center

        volume = affine_transform(volume, tf, offset=offset, order=3)

        return volume

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

        # Prepopulate various arrays
        num_probes_per_tilt = [0]
        for dc in self._datacube:
            rx, ry = dc.Rshape
            num_probes_per_tilt.append(rx * ry)

        self._num_diffraction_patterns = sum(num_probes_per_tilt)
        self._cum_probes_per_tilt = np.cumsum(np.array(num_probes_per_tilt))

        self._mean_diffraction_intensity = []
        self._positions_px_all = np.empty((self._num_diffraction_patterns, 2))

        self._rotation_best_rad = np.deg2rad(diffraction_patterns_rotate_degrees)
        self._rotation_best_transpose = diffraction_patterns_transpose

        if force_com_shifts is None:
            force_com_shifts = [None] * self._num_tilts

        for tilt_index in tqdmnd(
            self._num_tilts,
            desc="Preprocessing data",
            unit="tilt",
            disable=not progress_bar,
        ):
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

                self._amplitudes = xp.empty(
                    (self._num_diffraction_patterns,) + self._datacube[0].Qshape
                )
                self._region_of_interest_shape = np.array(
                    self._amplitudes[0].shape[-2:]
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

            intensities = self._extract_intensities_and_calibrations_from_datacube(
                self._datacube[tilt_index],
                require_calibrations=True,
                force_scan_sampling=force_scan_sampling,
                force_angular_sampling=force_angular_sampling,
                force_reciprocal_sampling=force_reciprocal_sampling,
            )

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

            (
                self._amplitudes[
                    self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                        tilt_index + 1
                    ]
                ],
                mean_diffraction_intensity_temp,
            ) = self._normalize_diffraction_intensities(
                intensities, com_fitted_x, com_fitted_y, crop_patterns
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

            self._positions_px_all[
                self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                    tilt_index + 1
                ]
            ] = self._calculate_scan_positions_in_pixels(
                self._scan_positions[tilt_index]
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
            p, q = np.round(np.max(self._positions_px_all, axis=0))
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                "int"
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                "int"
            )
            self._object = xp.zeros((q, p, q), dtype=xp.float32)
        else:
            self._object = xp.asarray(self._object, dtype=xp.float32)

        self._object_initial = self._object.copy()
        self._object_type_initial = self._object_type
        self._object_shape = self._object.shape[-2:]
        self._num_voxels = self._object.shape[0]

        # Center Probes
        self._positions_px_all = xp.asarray(self._positions_px_all, dtype=xp.float32)

        for tilt_index in range(self._num_tilts):
            self._positions_px = self._positions_px_all[
                self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                    tilt_index + 1
                ]
            ]
            self._positions_px_com = xp.mean(self._positions_px, axis=0)
            self._positions_px -= (
                self._positions_px_com - xp.array(self._object_shape) / 2
            )

            self._positions_px_all[
                self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                    tilt_index + 1
                ]
            ] = self._positions_px.copy()

        self._positions_px_initial_all = self._positions_px_all.copy()
        self._positions_initial_all = self._positions_px_initial_all.copy()
        self._positions_initial_all[:, 0] *= self.sampling[0]
        self._positions_initial_all[:, 1] *= self.sampling[1]

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
            self._probe *= xp.sqrt(
                sum(self._mean_diffraction_intensity)
                / self._num_tilts
                / probe_intensity
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

                # Normalize probe to match mean diffraction intensity
                probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probe)) ** 2)
                self._probe *= xp.sqrt(
                    sum(self._mean_diffraction_intensity)
                    / self._num_tilts
                    / probe_intensity
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

            for tilt_index in np.arange(self._num_tilts):
                rot_matrix = self._tilt_orientation_matrices[tilt_index]

                probe_overlap_3D = self._rotate_zxy_volume(
                    probe_overlap_3D,
                    rot_matrix @ old_rot_matrix.T,
                )

                self._positions_px = self._positions_px_all[
                    self._cum_probes_per_tilt[tilt_index] : self._cum_probes_per_tilt[
                        tilt_index + 1
                    ]
                ]
                self._positions_px_fractional = self._positions_px - xp.round(
                    self._positions_px
                )
                shifted_probes = fft_shift(
                    self._probe, self._positions_px_fractional, xp
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

            probe_overlap_3D = self._gaussian_filter(probe_overlap_3D, 1.0)
            self._object_fov_mask = asnumpy(
                probe_overlap_3D > 0.25 * probe_overlap_3D.max()
            )
        else:
            self._object_fov_mask = np.asarray(object_fov_mask)
            self._positions_px = self._positions_px_all[: self._cum_probes_per_tilt[1]]
            self._positions_px_fractional = self._positions_px - xp.round(
                self._positions_px
            )
            shifted_probes = fft_shift(self._probe, self._positions_px_fractional, xp)
            probe_intensities = xp.abs(shifted_probes) ** 2
            probe_overlap = self._sum_overlapping_patches_bincounts(probe_intensities)
            probe_overlap = self._gaussian_filter(probe_overlap, 1.0)

        self._object_fov_mask_inverse = np.invert(self._object_fov_mask)

        if plot_probe_overlaps:
            figsize = kwargs.pop("figsize", (13, 4))
            chroma_boost = kwargs.pop("chroma_boost", 1)

            # initial probe
            complex_probe_rgb = Complex2RGB(
                self.probe_centered,
                power=2,
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
        propagated_probes: np.ndarray
            Shifted probes at each layer
        object_patches: np.ndarray
            Patched object view
        transmitted_probes: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions
        """

        xp = self._xp

        complex_object = xp.exp(1j * current_object)
        object_patches = complex_object[
            :, self._vectorized_patch_indices_row, self._vectorized_patch_indices_col
        ]

        propagated_probes = xp.empty_like(object_patches)
        propagated_probes[0] = fft_shift(
            current_probe, self._positions_px_fractional, xp
        )

        for s in range(self._num_slices):
            # transmit
            transmitted_probes = object_patches[s] * propagated_probes[s]

            # propagate
            if s + 1 < self._num_slices:
                propagated_probes[s + 1] = self._propagate_array(
                    transmitted_probes, self._propagator_arrays[s]
                )

        return propagated_probes, object_patches, transmitted_probes

    def _gradient_descent_fourier_projection(self, amplitudes, transmitted_probes):
        """
        Ptychographic fourier projection method for GD method.

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        transmitted_probes: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions

        Returns
        --------
        exit_waves:np.ndarray
            Updated exit wave difference
        error: float
            Reconstruction error
        """

        xp = self._xp
        fourier_exit_waves = xp.fft.fft2(transmitted_probes)

        error = xp.sum(xp.abs(amplitudes - xp.abs(fourier_exit_waves)) ** 2)

        modified_exit_wave = xp.fft.ifft2(
            amplitudes * xp.exp(1j * xp.angle(fourier_exit_waves))
        )

        exit_waves = modified_exit_wave - transmitted_probes

        return exit_waves, error

    def _projection_sets_fourier_projection(
        self,
        amplitudes,
        transmitted_probes,
        exit_waves,
        projection_a,
        projection_b,
        projection_c,
    ):
        """
        Ptychographic fourier projection method for DM_AP and RAAR methods.
        Generalized projection using three parameters: a,b,c

            DM_AP(\\alpha)   :   a =  -\\alpha, b = 1, c = 1 + \\alpha
              DM: DM_AP(1.0), AP: DM_AP(0.0)

            RAAR(\\beta)     :   a = 1-2\\beta, b = \beta, c = 2
              DM : RAAR(1.0)

            RRR(\\gamma)     :   a = -\\gamma, b = \\gamma, c = 2
              DM: RRR(1.0)

            SUPERFLIP       :   a = 0, b = 1, c = 2

        Parameters
        --------
        amplitudes: np.ndarray
            Normalized measured amplitudes
        transmitted_probes: np.ndarray
            Transmitted probes after N-1 propagations and N transmissions
        exit_waves: np.ndarray
            previously estimated exit waves
        projection_a: float
        projection_b: float
        projection_c: float

        Returns
        --------
        exit_waves:np.ndarray
            Updated exit wave difference
        error: float
            Reconstruction error
        """

        xp = self._xp
        projection_x = 1 - projection_a - projection_b
        projection_y = 1 - projection_c

        if exit_waves is None:
            exit_waves = transmitted_probes.copy()

        fourier_exit_waves = xp.fft.fft2(transmitted_probes)
        error = xp.sum(xp.abs(amplitudes - xp.abs(fourier_exit_waves)) ** 2)

        factor_to_be_projected = (
            projection_c * transmitted_probes + projection_y * exit_waves
        )
        fourier_projected_factor = xp.fft.fft2(factor_to_be_projected)

        fourier_projected_factor = amplitudes * xp.exp(
            1j * xp.angle(fourier_projected_factor)
        )
        projected_factor = xp.fft.ifft2(fourier_projected_factor)

        exit_waves = (
            projection_x * exit_waves
            + projection_a * transmitted_probes
            + projection_b * projected_factor
        )

        return exit_waves, error

    def _forward(
        self,
        current_object,
        current_probe,
        amplitudes,
        exit_waves,
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
        propagated_probes:np.ndarray
            Prop[object^n*probe^n]
        object_patches: np.ndarray
            Patched object view
        transmitted_probes: np.ndarray
            Transmitted probes at each layer
        exit_waves:np.ndarray
            Updated exit_waves
        error: float
            Reconstruction error
        """

        (
            propagated_probes,
            object_patches,
            transmitted_probes,
        ) = self._overlap_projection(current_object, current_probe)

        if use_projection_scheme:
            (
                exit_waves[self._active_tilt_index],
                error,
            ) = self._projection_sets_fourier_projection(
                amplitudes,
                transmitted_probes,
                exit_waves[self._active_tilt_index],
                projection_a,
                projection_b,
                projection_c,
            )

        else:
            exit_waves, error = self._gradient_descent_fourier_projection(
                amplitudes, transmitted_probes
            )

        return propagated_probes, object_patches, transmitted_probes, exit_waves, error

    def _gradient_descent_adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        propagated_probes,
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
            probe = propagated_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = self._sum_overlapping_patches_bincounts(
                xp.abs(probe) ** 2
            )
            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            current_object[s] += step_size * (
                self._sum_overlapping_patches_bincounts(
                    xp.real(-1j * xp.conj(obj) * xp.conj(probe) * exit_waves)
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
        propagated_probes,
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
            probe = propagated_probes[s]
            obj = object_patches[s]

            # object-update
            probe_normalization = self._sum_overlapping_patches_bincounts(
                xp.abs(probe) ** 2
            )
            probe_normalization = 1 / xp.sqrt(
                1e-16
                + ((1 - normalization_min) * probe_normalization) ** 2
                + (normalization_min * xp.max(probe_normalization)) ** 2
            )

            current_object[s] = (
                self._sum_overlapping_patches_bincounts(
                    xp.real(-1j * xp.conj(obj) * xp.conj(probe) * exit_waves_copy)
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

    def _adjoint(
        self,
        current_object,
        current_probe,
        object_patches,
        propagated_probes,
        exit_waves,
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
        transmitted_probes: np.ndarray
            Transmitted probes at each layer
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
                propagated_probes,
                exit_waves[self._active_tilt_index],
                normalization_min,
                fix_probe,
            )
        else:
            current_object, current_probe = self._gradient_descent_adjoint(
                current_object,
                current_probe,
                object_patches,
                propagated_probes,
                exit_waves,
                step_size,
                normalization_min,
                fix_probe,
            )

        return current_object, current_probe

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

        complex_object = xp.exp(1j * current_object)

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

    def _object_gaussian_constraint(self, current_object, gaussian_filter_sigma):
        """
        Ptychographic smoothness constraint.
        Used for blurring object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        gaussian_filter = self._gaussian_filter

        gaussian_filter_sigma /= self.sampling[0]
        current_object = gaussian_filter(current_object, gaussian_filter_sigma)

        return current_object

    def _object_butterworth_constraint(
        self, current_object, q_lowpass, q_highpass, butterworth_order
    ):
        """
        Butterworth filter

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter
        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        qz = xp.fft.fftfreq(current_object.shape[0], self.sampling[1])
        qx = xp.fft.fftfreq(current_object.shape[1], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[2], self.sampling[1])
        qza, qxa, qya = xp.meshgrid(qz, qx, qy, indexing="ij")
        qra = xp.sqrt(qza**2 + qxa**2 + qya**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object)
        current_object -= current_object_mean
        current_object = xp.fft.ifftn(xp.fft.fftn(current_object) * env)
        current_object += current_object_mean
        return xp.real(current_object)

    def _object_denoise_tv_pylops(self, current_object, weights, iterations):
        """
        Performs second order TV denoising along x and y

        Parameters
        ----------
        current_object: np.ndarray
            Current object estimate
        weights : [float, float]
            Denoising weights[z weight, r weight]. The greater `weight`,
            the more denoising.
        iterations: float
            Number of iterations to run in denoising algorithm.
            `niter_out` in pylops

        Returns
        -------
        constrained_object: np.ndarray
            Constrained object estimate

        """
        xp = self._xp

        if xp.iscomplexobj(current_object):
            current_object_tv = current_object
            warnings.warn(
                ("TV denoising is currently only supported for potential objects."),
                UserWarning,
            )

        else:
            # zero pad at top and bottom slice
            pad_width = ((1, 1), (0, 0), (0, 0))
            current_object = xp.pad(
                current_object, pad_width=pad_width, mode="constant"
            )

            # run tv denoising
            nz, nx, ny = current_object.shape
            niter_out = iterations
            niter_in = 1
            Iop = pylops.Identity(nx * ny * nz)

            if weights[0] == 0:
                xy_laplacian = pylops.Laplacian(
                    (nz, nx, ny), axes=(1, 2), edge=False, kind="backward"
                )
                l1_regs = [xy_laplacian]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=[weights[1]],
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            elif weights[1] == 0:
                z_gradient = pylops.FirstDerivative(
                    (nz, nx, ny), axis=0, edge=False, kind="backward"
                )
                l1_regs = [z_gradient]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=[weights[0]],
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            else:
                z_gradient = pylops.FirstDerivative(
                    (nz, nx, ny), axis=0, edge=False, kind="backward"
                )
                xy_laplacian = pylops.Laplacian(
                    (nz, nx, ny), axes=(1, 2), edge=False, kind="backward"
                )
                l1_regs = [z_gradient, xy_laplacian]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=weights,
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            # remove padding
            current_object_tv = current_object_tv.reshape(current_object.shape)[1:-1]

        return current_object_tv

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
                current_object, gaussian_filter_sigma
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
        constrain_position_distance: float, optional
            Distance to constrain position correction within original
            field of view in A
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
        else:
            max_batch_size = self._num_diffraction_patterns

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
                        propagated_probes,
                        object_patches,
                        transmitted_probes,
                        self._exit_waves,
                        batch_error,
                    ) = self._forward(
                        object_sliced,
                        self._probe,
                        amplitudes,
                        self._exit_waves,
                        use_projection_scheme,
                        projection_a,
                        projection_b,
                        projection_c,
                    )

                    # adjoint operator
                    object_sliced, self._probe = self._adjoint(
                        object_sliced,
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
                            object_sliced,
                            self._probe,
                            transmitted_probes,
                            amplitudes,
                            self._positions_px,
                            positions_step_size,
                            constrain_position_distance,
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

    def _crop_rotate_object_manually(
        self,
        array,
        angle,
        x_lims,
        y_lims,
    ):
        """
        Crops and rotates rotates object manually.

        Parameters
        ----------
        array: np.ndarray
            Object array to crop and rotate. Only operates on numpy arrays for comptatibility.
        angle: float
            In-plane angle in degrees to rotate by
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices

        Returns
        -------
        cropped_rotated_array: np.ndarray
            Cropped and rotated object array
        """

        asnumpy = self._asnumpy
        min_x, max_x = x_lims
        min_y, max_y = y_lims

        if angle is not None:
            rotated_array = rotate_np(
                asnumpy(array), angle, reshape=False, axes=(-2, -1)
            )
        else:
            rotated_array = asnumpy(array)

        return rotated_array[..., min_x:max_x, min_y:max_y]

    def _visualize_last_iteration_figax(
        self,
        fig,
        object_ax,
        convergence_ax,
        cbar: bool,
        projection_angle_deg: float,
        projection_axes: Tuple[int, int],
        x_lims: Tuple[int, int],
        y_lims: Tuple[int, int],
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
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """

        cmap = kwargs.pop("cmap", "magma")

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
            convergence_ax.semilogy(np.arange(errors.shape[0]), errors, **kwargs)

    def _visualize_last_iteration(
        self,
        fig,
        cbar: bool,
        plot_convergence: bool,
        plot_probe: bool,
        plot_fourier_probe: bool,
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
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        figsize = kwargs.pop("figsize", (8, 5))
        cmap = kwargs.pop("cmap", "magma")

        if plot_fourier_probe:
            chroma_boost = kwargs.pop("chroma_boost", 2)
        else:
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
                probe_array = Complex2RGB(
                    self.probe_fourier,
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

        if plot_fourier_probe:
            chroma_boost = kwargs.pop("chroma_boost", 2)
        else:
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
                    probe_array = Complex2RGB(
                        asnumpy(
                            self._return_fourier_probe_from_centered_probe(
                                probes[grid_range[n]]
                            )
                        ),
                        chroma_boost=chroma_boost,
                    )
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
                cbar=cbar,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
                **kwargs,
            )

        return self

    def _return_object_fft(
        self,
        obj=None,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims: Tuple[int, int] = (None, None),
        y_lims: Tuple[int, int] = (None, None),
    ):
        """
        Returns obj fft shifted to center of array

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if obj is None:
            obj = self._object
        else:
            obj = xp.asarray(obj, dtype=xp.float32)

        if projection_angle_deg is not None:
            rotated_3d_obj = self._rotate(
                obj,
                projection_angle_deg,
                axes=projection_axes,
                reshape=False,
                order=2,
            )
            rotated_3d_obj = asnumpy(rotated_3d_obj)
        else:
            rotated_3d_obj = asnumpy(obj)

        rotated_object = self._crop_rotate_object_manually(
            rotated_3d_obj.sum(0), angle=None, x_lims=x_lims, y_lims=y_lims
        )

        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rotated_object))))

    def show_object_fft(
        self,
        obj=None,
        projection_angle_deg: float = None,
        projection_axes: Tuple[int, int] = (0, 2),
        x_lims: Tuple[int, int] = (None, None),
        y_lims: Tuple[int, int] = (None, None),
        **kwargs,
    ):
        """
        Plot FFT of reconstructed object

        Parameters
        ----------
        obj: array, optional
            if None is specified, uses self._object
        projection_angle_deg: float
            Angle in degrees to rotate 3D array around prior to projection
        projection_axes: tuple(int,int)
            Axes defining projection plane
        x_lims: tuple(float,float)
            min/max x indices
        y_lims: tuple(float,float)
            min/max y indices
        """
        if obj is None:
            object_fft = self._return_object_fft(
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
            )
        else:
            object_fft = self._return_object_fft(
                obj,
                projection_angle_deg=projection_angle_deg,
                projection_axes=projection_axes,
                x_lims=x_lims,
                y_lims=y_lims,
            )

        figsize = kwargs.pop("figsize", (6, 6))
        cmap = kwargs.pop("cmap", "magma")
        vmin = kwargs.pop("vmin", 0)
        vmax = kwargs.pop("vmax", 1)
        power = kwargs.pop("power", 0.2)

        pixelsize = 1 / (object_fft.shape[1] * self.sampling[1])
        show(
            object_fft,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            scalebar=True,
            pixelsize=pixelsize,
            ticks=False,
            pixelunits=r"$\AA^{-1}$",
            power=power,
            **kwargs,
        )

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
