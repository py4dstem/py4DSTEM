"""
Module for reconstructing phase objects from 4DSTEM datasets using iterative methods,
namely (single-slice) ptychography.
"""

from typing import Mapping, Sequence, Union

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
except (ImportError, ModuleNotFoundError):
    cp = np

from emdfile import Array, Custom, Metadata, _read_metadata, tqdmnd
from py4DSTEM.data import Calibration
from py4DSTEM.datacube import DataCube
from py4DSTEM.process.phase.phase_base_class import PhaseReconstruction
from py4DSTEM.process.phase.utils import (
    ComplexProbe,
    copy_to_device,
    mask_array_using_virtual_detectors,
    polar_aliases,
    polar_symbols,
    spatial_frequencies,
)
from py4DSTEM.process.utils import electron_wavelength_angstrom


class DirectPtychography(
    PhaseReconstruction,
):
    """
    Direct Ptychographic Reconstruction Class (SSB-like).

    Diffraction intensities dimensions  : (Rx,Ry,Qx,Qy)
    Reconstructed object dimensions     : (Rx,Ry)

    Parameters
    ----------
    energy: float
        The electron energy of the wave functions in eV
    datacube: DataCube
        Input 4D diffraction pattern intensities
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial probe guess in mrad
    rolloff: float, optional
        Semiangle rolloff for the initial probe guess
    vacuum_probe_intensity: np.ndarray, optional
        Vacuum probe to use as intensity aperture for initial probe guess
    polar_parameters: dict, optional
        Mapping from aberration symbols to their corresponding values. All aberration
        magnitudes should be given in Å and angles should be given in radians.
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
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
        rolloff: float = 2.0,
        vacuum_probe_intensity: np.ndarray = None,
        polar_parameters: Mapping[str, float] = None,
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

        self.set_save_defaults()

        # Data
        self._datacube = datacube

        # Common Metadata
        self._vacuum_probe_intensity = vacuum_probe_intensity
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._rolloff = rolloff
        self._verbose = verbose
        self._preprocessed = False

    def to_h5(self, group):
        """
        Wraps datasets and metadata to write in emdfile classes,
        notably: the object array.
        """

        asnumpy = self._asnumpy

        # instantiation metadata
        vacuum_probe_intensity = (
            asnumpy(self._vacuum_probe_intensity)
            if self._vacuum_probe_intensity is not None
            else None
        )

        self.metadata = Metadata(
            name="instantiation_metadata",
            data={
                "energy": self._energy,
                "semiangle_cutoff": self._semiangle_cutoff,
                "vacuum_probe_intensity": vacuum_probe_intensity,
                "rolloff": self._rolloff,
                "verbose": self._verbose,
                "device": self._device,
                "storage": self._storage,
                "clear_fft_cache": self._clear_fft_cache,
                "name": self.name,
            },
        )

        self.metadata = Metadata(
            name="aberrations_metadata",
            data=self._polar_parameters,
        )

        # preprocessing metadata
        self.metadata = Metadata(
            name="preprocess_metadata",
            data={
                "rotation_angle_rad": self._rotation_best_rad,
                "data_transpose": self._rotation_best_transpose,
                "sampling": self.sampling,
                "scan_sampling": self.scan_sampling,
                "angular_sampling": self.angular_sampling,
                "intensities_shape": self._intensities_shape,
                "grid_scan_shape": self._grid_scan_shape,
                "scan_units": self._scan_units,
            },
        )

        # object
        self._object_emd = Array(
            name="reconstruction_object",
            data=asnumpy(self._object),
        )

        self._probe_emd = Array(
            name="initialized_probe",
            data=asnumpy(self._probe),
        )

        # datacube
        if self._save_datacube:
            self.metadata = self._datacube.calibration
            Custom.to_h5(self, group)
        else:
            dc = self._datacube
            self._datacube = None
            Custom.to_h5(self, group)
            self._datacube = dc

    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of arguments/values to pass
        to the class' __init__ function
        """
        # Get data
        dict_data = cls._get_emd_attr_data(cls, group)

        # Get metadata dictionaries
        instance_md = _read_metadata(group, "instantiation_metadata")
        aberrations_md = _read_metadata(group, "aberrations_metadata")

        # Fix calibrations bug
        if "_datacube" in dict_data:
            calibrations_dict = _read_metadata(group, "calibration")._params
            cal = Calibration()
            cal._params.update(calibrations_dict)
            dc = dict_data["_datacube"]
            dc.calibration = cal
        else:
            dc = None

        # Populate args and return
        kwargs = {
            "datacube": dc,
            "energy": instance_md["energy"],
            "name": instance_md["name"],
            "semiangle_cutoff": instance_md["semiangle_cutoff"],
            "rolloff": instance_md["rolloff"],
            "vacuum_probe_intensity": instance_md["vacuum_probe_intensity"],
            "polar_parameters": aberrations_md._params,
            "verbose": True,  # for compatibility
            "device": "cpu",  # for compatibility
            "storage": "cpu",  # for compatibility
            "clear_fft_cache": True,  # for compatibility
        }

        return kwargs

    def _populate_instance(self, group):
        """
        Sets post-initialization properties, notably some preprocessing meta
        optional; during read, this method is run after object instantiation.
        """
        # Preprocess metadata
        preprocess_md = _read_metadata(group, "preprocess_metadata")
        self._rotation_best_rad = preprocess_md["rotation_angle_rad"]
        self._rotation_best_transpose = preprocess_md["data_transpose"]
        self._angular_sampling = preprocess_md["angular_sampling"]
        self._scan_sampling = preprocess_md["scan_sampling"]
        self._grid_scan_shape = preprocess_md["grid_scan_shape"]
        self._scan_units = preprocess_md["scan_units"]
        self._intensities_shape = preprocess_md["intensities_shape"]
        self._preprocessed = False

        # Data
        dict_data = Custom._get_emd_attr_data(Custom, group)
        obj = dict_data["_object_emd"].data
        probe = dict_data["_probe_emd"].data

        self._object = obj
        self.object = obj
        self._probe = probe

    def _set_polar_parameters(self, parameters: dict):
        """
        Set the probe aberrations dictionary.

        Parameters
        ----------
        parameters: dict
            Mapping from aberration symbols to their corresponding values.
        """

        for symbol, value in parameters.items():
            if symbol in self._polar_parameters.keys():
                self._polar_parameters[symbol] = value

            elif symbol == "defocus":
                self._polar_parameters[polar_aliases[symbol]] = -value

            elif symbol in polar_aliases.keys():
                self._polar_parameters[polar_aliases[symbol]] = value

            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

    def preprocess(
        self,
        dp_mask: np.ndarray = None,
        in_place_datacube_modification: bool = False,
        fit_function: str = "plane",
        plot_center_of_mass: str = "default",
        plot_rotation: bool = True,
        maximize_divergence: bool = False,
        rotation_angles_deg: np.ndarray = None,
        plot_overlap_trotters: bool = True,
        force_com_rotation: float = None,
        force_com_transpose: float = None,
        force_com_shifts: Union[Sequence[np.ndarray], Sequence[float]] = None,
        force_com_measured: Sequence[np.ndarray] = None,
        vectorized_com_calculation: bool = True,
        force_scan_sampling: float = None,
        force_angular_sampling: float = None,
        force_reciprocal_sampling: float = None,
        crop_patterns: bool = False,
        device: str = None,
        clear_fft_cache: bool = None,
        **kwargs,
    ):
        """
        Ptychographic preprocessing step.

        Parameters
        ----------
        dp_mask: ndarray, optional
            Mask for datacube intensities (Qx,Qy)
        in_place_datacube_modification: bool, optional
            If True, the datacube will be preprocessed in-place. Note this is not possible
            when either crop_patterns or positions_mask are used.
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
        plot_overlap_trotters: bool, optional
            If True, low and high frequency example trotters will be plotted
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
        crop_patterns: bool
            if True, crop patterns to avoid wrap around of patterns when centering
        device: str, optional
            if not none, overwrites self._device to set device preprocess will be perfomed on.
        clear_fft_cache: bool, optional
            if true, and device = 'gpu', clears the cached fft plan at the end of function calls

        Returns
        --------
        self: DirectPtychography
            Self to accommodate chaining
        """

        # handle device/storage
        self.set_device(device, clear_fft_cache)

        device = self._device
        storage = self._storage
        xp_storage = self._xp_storage
        asnumpy = self._asnumpy

        # set additional metadata
        self._dp_mask = dp_mask

        if self._datacube is None:
            raise ValueError(
                (
                    "The preprocess() method requires a DataCube. "
                    "Please run ssb.attach_datacube(DataCube) first."
                )
            )

        # preprocess datacube
        (
            self._datacube,
            self._vacuum_probe_intensity,
            self._dp_mask,
            force_com_shifts,
            force_com_measured,
        ) = self._preprocess_datacube_and_vacuum_probe(
            self._datacube,
            vacuum_probe_intensity=self._vacuum_probe_intensity,
            dp_mask=self._dp_mask,
            com_shifts=force_com_shifts,
            com_measured=force_com_measured,
            diffraction_intensities_shape=None,
            padded_diffraction_intensities_shape=None,
        )

        # calibrations
        _intensities = self._extract_intensities_and_calibrations_from_datacube(
            self._datacube,
            require_calibrations=True,
            force_scan_sampling=force_scan_sampling,
            force_angular_sampling=force_angular_sampling,
            force_reciprocal_sampling=force_reciprocal_sampling,
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

        # correct relative-rotation for non-symmetric aberrations
        self._polar_parameters_relative = self._polar_parameters.copy()
        for k, v in self._polar_parameters_relative.items():
            if k[:3] == "phi":
                theta = v - self._rotation_best_rad
                if self._rotation_best_transpose:
                    theta = np.pi / 2 - theta
                self._polar_parameters_relative[k] = theta

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

        # corner-center intensities
        (
            self._intensities,
            self._mean_diffraction_intensity,
            self._crop_mask,
            self._crop_mask_shape,
        ) = self._normalize_diffraction_intensities(
            _intensities,
            self._com_fitted_x,
            self._com_fitted_y,
            None,
            crop_patterns,
            in_place_datacube_modification,
            return_intensities_instead=True,
        )

        self._intensities = self._intensities.reshape(_intensities.shape)

        # explicitly transfer arrays to storage
        if not in_place_datacube_modification:
            del _intensities

        self._intensities = copy_to_device(self._intensities, storage)
        self._intensities_shape = np.array(self._intensities.shape[-2:])

        # take FFT wrt real-space
        self._intensities_FFT = xp_storage.fft.fft2(self._intensities, axes=(0, 1))

        # initialize probe
        self._probe = ComplexProbe(
            energy=self._energy,
            gpts=self._intensities_shape,
            sampling=self.sampling,
            semiangle_cutoff=self._semiangle_cutoff,
            rolloff=self._rolloff,
            parameters=self._polar_parameters_relative,
            device=device,
        )._evaluate_ctf()

        # initialize frequencies
        self._spatial_frequencies = spatial_frequencies(
            self._intensities_shape, self.sampling
        )

        fx, fy = spatial_frequencies(self._grid_scan_shape, self.scan_sampling)
        fx, fy = np.meshgrid(fx, fy, indexing="ij")

        # rotation
        ct = np.cos(-self._rotation_best_rad)
        st = np.sin(-self._rotation_best_rad)
        fx, fy = fx * ct - fy * st, fy * ct + fx * st
        # transpose
        if self._rotation_best_transpose:
            fx, fy = fy, fx
        self._scan_frequencies = fx, fy

        # plot probe overlaps
        if plot_overlap_trotters:

            trotter_intensities = (
                (self._intensities_FFT * self._intensities_FFT.conj())
                .sum((-1, -2))
                .real
            )
            trotter_intensities[0, 0] = 0.0
            trotter_inds = xp_storage.unravel_index(
                xp_storage.argsort(-trotter_intensities.ravel()),
                trotter_intensities.shape,
            )

            f = fx**2 + fy**2
            q_probe = (
                self._reciprocal_sampling[0]
                * self._semiangle_cutoff
                / self.angular_sampling[0]
            )

            bf_inds = f[*trotter_inds] < q_probe
            low_ind_x = trotter_inds[0][bf_inds][0]
            low_ind_y = trotter_inds[1][bf_inds][0]
            high_ind_x = trotter_inds[0][~bf_inds][0]
            high_ind_y = trotter_inds[1][~bf_inds][0]

            figsize = kwargs.pop("figsize", (13, 4))
            chroma_boost = kwargs.pop("chroma_boost", 1)
            power = kwargs.pop("power", 2)

            reciprocal_extent = [
                -self.angular_sampling[1] * self._intensities_shape[1] / 2,
                self.angular_sampling[1] * self._intensities_shape[1] / 2,
                self.angular_sampling[0] * self._intensities_shape[0] / 2,
                -self.angular_sampling[0] * self._intensities_shape[0] / 2,
            ]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

            # BF probe
            complex_probe_rgb = Complex2RGB(
                self.probe_centered,
                power=power,
                chroma_boost=chroma_boost,
                vmin=0,
                vmax=1,
            )

            ax1.imshow(
                complex_probe_rgb,
                extent=reciprocal_extent,
            )

            # low frequency trotter
            low_trotter_rgb = Complex2RGB(
                asnumpy(
                    self._return_centered_probe(
                        self._intensities_FFT[low_ind_x, low_ind_y]
                    )
                ),
                power=power,
                chroma_boost=chroma_boost,
                vmin=0,
                vmax=1,
            )

            ax2.imshow(
                low_trotter_rgb,
                extent=reciprocal_extent,
            )

            # high frequency trotter
            high_trotter_rgb = Complex2RGB(
                asnumpy(
                    self._return_centered_probe(
                        self._intensities_FFT[high_ind_x, high_ind_y]
                    )
                ),
                power=power,
                chroma_boost=chroma_boost,
                vmin=0,
                vmax=1,
            )

            ax3.imshow(
                high_trotter_rgb,
                extent=reciprocal_extent,
            )

            for ax, title in zip(
                [ax1, ax2, ax3],
                [
                    "Bright-field complex probe",
                    "Low frequency trotter example",
                    "High frequency trotter example",
                ],
            ):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="2.5%")
                add_colorbar_arg(cax, chroma_boost=chroma_boost)
                ax.set_ylabel("kx [mrad]")
                ax.set_xlabel("ky [mrad]")
                ax.set_title(title)

            fig.tight_layout()

        self._preprocessed = True
        self.clear_device_mem(self._device, self._clear_fft_cache)

        return self

    def reconstruct(
        self,
        virtual_detector_masks: Sequence[np.ndarray] = None,
        progress_bar: bool = True,
        device: str = None,
        clear_fft_cache: bool = None,
    ):
        """
        Ptychographic reconstruction main method.

        Parameters
        --------
        virtual_detector_masks: np.ndarray
            List of corner-centered boolean masks for binning forward model trotters,
            to allow comparison with arbitrary geometry detector datasets. TO-DO
        progress_bar: bool, optional
            If True, reconstruction progress is displayed
        device: str, optional
            If not none, overwrites self._device to set device preprocess will be perfomed on.
        clear_fft_cache: bool, optional
            If true, and device = 'gpu', clears the cached fft plan at the end of function calls

        Returns
        --------
        self: DirectPtychography
            Self to accommodate chaining
        """
        # handle device/storage
        self.set_device(device, clear_fft_cache)

        if device is not None:
            attrs = [
                "_probe",
            ]
            self.copy_attributes_to_device(attrs, device)

        xp = self._xp
        device = self._device
        asnumpy = self._asnumpy

        sx, sy = self._grid_scan_shape
        psi = xp.empty((sx, sy), dtype=xp.complex64)
        probe_conj = xp.conj(self._probe)
        threshold = 1e-3

        Kx, Ky = self._spatial_frequencies
        Qx, Qy = self._scan_frequencies

        if virtual_detector_masks is not None:
            virtual_detector_masks = xp.asarray(virtual_detector_masks).astype(xp.bool_)

        # main loop
        for ind_x, ind_y in tqdmnd(
            sx,
            sy,
            desc="Reconstructing object",
            unit=" spatial frequency",
            disable=not progress_bar,
        ):
            G = xp.asarray(self._intensities_FFT[ind_x, ind_y])
            if ind_x == 0 and ind_y == 0:
                psi[ind_x, ind_y] = xp.abs(G).sum()
            else:
                Kx_plus_Qx = Kx + Qx[ind_x, ind_y]
                Ky_plus_Qy = Ky + Qy[ind_x, ind_y]

                probe_plus = ComplexProbe(
                    energy=self._energy,
                    gpts=self._intensities_shape,
                    sampling=self.sampling,
                    semiangle_cutoff=self._semiangle_cutoff,
                    rolloff=self._rolloff,
                    parameters=self._polar_parameters_relative,
                    device=self._device,
                    force_spatial_frequencies=(Kx_plus_Qx, Ky_plus_Qy),
                )._evaluate_ctf()

                Kx_minus_Qx = Kx - Qx[ind_x, ind_y]
                Ky_minus_Qy = Ky - Qy[ind_x, ind_y]

                probe_minus = ComplexProbe(
                    energy=self._energy,
                    gpts=self._intensities_shape,
                    sampling=self.sampling,
                    semiangle_cutoff=self._semiangle_cutoff,
                    rolloff=self._rolloff,
                    parameters=self._polar_parameters_relative,
                    device=self._device,
                    force_spatial_frequencies=(Kx_minus_Qx, Ky_minus_Qy),
                )._evaluate_ctf()

                gamma = probe_conj * probe_minus - self._probe * probe_plus.conj()

                if virtual_detector_masks is not None:
                    gamma = mask_array_using_virtual_detectors(
                        gamma, virtual_detector_masks, in_place=True
                    )

                gamma_abs = np.abs(gamma)
                gamma_ind = gamma_abs > threshold
                psi[ind_x, ind_y] = (
                    G[gamma_ind] * xp.conj(gamma[gamma_ind]) / gamma_abs[gamma_ind]
                ).sum()

        self._object = xp.fft.ifft2(psi) / self._mean_diffraction_intensity

        # store result
        self.object = asnumpy(self._object)
        self.probe = self.probe_centered

        self.clear_device_mem(self._device, self._clear_fft_cache)

        return self

    def visualize(self, fig=None, cbar: bool = True, **kwargs):
        """
        Displays reconstructed amplitude and phase of object.

        Parameters
        --------
        fig, optional
            Matplotlib figure to draw Gridspec on
        cbar: bool, optional
            If true, displays a colorbar

        Returns
        --------
        self: DirectPtychography
            Self to accommodate chaining
        """

        figsize = kwargs.pop("figsize", (9, 4))
        cmap = kwargs.pop("cmap", "magma")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        spec = GridSpec(ncols=2, nrows=1)

        if fig is None:
            fig = plt.figure(figsize=figsize)

        extent = [
            0,
            self.scan_sampling[1] * self._grid_scan_shape[1],
            self.scan_sampling[0] * self._grid_scan_shape[0],
            0,
        ]

        ax1 = fig.add_subplot(spec[0])
        amp, _vmin, _vmax = return_scaled_histogram_ordering(
            np.abs(self.object), vmin, vmax
        )
        im1 = ax1.imshow(
            amp, extent=extent, cmap="gray", vmin=_vmin, vmax=_vmax, **kwargs
        )

        ax2 = fig.add_subplot(spec[1])
        phase, _vmin, _vmax = return_scaled_histogram_ordering(
            np.angle(self.object), vmin, vmax
        )
        im2 = ax2.imshow(
            phase, extent=extent, cmap=cmap, vmin=_vmin, vmax=_vmax, **kwargs
        )

        if cbar:
            for ax, im, attr in zip([ax1, ax2], [im1, im2], ["amplitude", "phase"]):
                divider = make_axes_locatable(ax)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                fig.add_axes(ax_cb)
                fig.colorbar(im, cax=ax_cb)
                ax.set_ylabel(f"x [{self._scan_units[0]}]")
                ax.set_xlabel(f"y [{self._scan_units[1]}]")
                ax.set_title(f"Reconstructed object {attr}")

        spec.tight_layout(fig)

        return self

    def _return_centered_probe(
        self,
        probe=None,
    ):
        """
        Returns complex probe centered in middle of the array.

        Parameters
        ----------
        probe: complex array, optional
            if None is specified, uses self._probe

        Returns
        -------
        centered_probe: np.ndarray
            Center-shifted probe.
        """
        xp = self._xp

        if probe is None:
            probe = self._probe
        else:
            probe = xp.asarray(probe, dtype=xp.complex64)

        return xp.fft.fftshift(probe, axes=(-2, -1))

    @property
    def probe_centered(self):
        """Center-shifted probe"""
        if not hasattr(self, "_probe"):
            return None

        asnumpy = self._asnumpy
        return asnumpy(self._return_centered_probe(self._probe))

    @property
    def scan_sampling(self):
        """Scan sampling [Å]"""
        return getattr(self, "_scan_sampling", None)

    @property
    def angular_sampling(self):
        """Angular sampling [mrad]"""
        return getattr(self, "_angular_sampling", None)

    @property
    def sampling(self):
        """Sampling [Å]"""

        if self.angular_sampling is None:
            return None

        return tuple(
            electron_wavelength_angstrom(self._energy) * 1e3 / dk / n
            for dk, n in zip(self.angular_sampling, self._intensities_shape)
        )
