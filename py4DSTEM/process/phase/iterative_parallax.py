"""
Module for reconstructing virtual parallax (also known as tilted-shifted bright field)
images by aligning each virtual BF image.
"""

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from emdfile import Array, Custom, Metadata, _read_metadata, tqdmnd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from py4DSTEM import Calibration, DataCube
from py4DSTEM.preprocess.utils import get_shifted_ar
from py4DSTEM.process.phase.iterative_base_class import PhaseReconstruction
from py4DSTEM.process.phase.utils import AffineTransform
from py4DSTEM.process.utils.cross_correlate import align_images_fourier
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from py4DSTEM.visualize import show
from scipy.linalg import polar
from scipy.optimize import minimize
from scipy.special import comb

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np

warnings.simplefilter(action="always", category=UserWarning)

_aberration_names = {
    (1, 0): "C1        ",
    (1, 2): "stig      ",
    (2, 1): "coma      ",
    (2, 3): "trefoil   ",
    (3, 0): "C3        ",
    (3, 2): "stig2     ",
    (3, 4): "quadfoil  ",
    (4, 1): "coma2     ",
    (4, 3): "trefoil2  ",
    (4, 5): "pentafoil ",
    (5, 0): "C5        ",
    (5, 2): "stig3     ",
    (5, 4): "quadfoil2 ",
    (5, 6): "hexafoil  ",
}


class ParallaxReconstruction(PhaseReconstruction):
    """
    Iterative parallax reconstruction class.

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    object_padding_px: Tuple[int,int], optional
        Pixel dimensions to pad object with
        If None, the padding is set to half the probe ROI dimensions
    """

    def __init__(
        self,
        energy: float,
        datacube: DataCube = None,
        verbose: bool = False,
        object_padding_px: Tuple[int, int] = (32, 32),
        device: str = "cpu",
        name: str = "parallax_reconstruction",
    ):
        Custom.__init__(self, name=name)

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

        self.set_save_defaults()

        # Data
        self._datacube = datacube

        # Metadata
        self._energy = energy
        self._verbose = verbose
        self._device = device
        self._object_padding_px = object_padding_px
        self._preprocessed = False

    def to_h5(self, group):
        """
        Wraps datasets and metadata to write in emdfile classes,
        notably the (subpixel-)aligned BF.
        """
        # instantiation metadata
        self.metadata = Metadata(
            name="instantiation_metadata",
            data={
                "energy": self._energy,
                "verbose": self._verbose,
                "device": self._device,
                "object_padding_px": self._object_padding_px,
                "name": self.name,
            },
        )

        # preprocessing metadata
        self.metadata = Metadata(
            name="preprocess_metadata",
            data={
                "scan_sampling": self._scan_sampling,
                "wavelength": self._wavelength,
            },
        )

        # reconstruction metadata
        recon_metadata = {"reconstruction_error": float(self._recon_error)}

        if hasattr(self, "aberration_C1"):
            recon_metadata |= {
                "aberration_rotation_QR": self.rotation_Q_to_R_rads,
                "aberration_transpose": self.transpose_detected,
                "aberration_C1": self.aberration_C1,
                "aberration_A1x": self.aberration_A1x,
                "aberration_A1y": self.aberration_A1y,
            }

        if hasattr(self, "_kde_upsample_factor"):
            recon_metadata |= {
                "kde_upsample_factor": self._kde_upsample_factor,
            }
            self._subpixel_aligned_BF_emd = Array(
                name="subpixel_aligned_BF",
                data=self._asnumpy(self._recon_BF_subpixel_aligned),
            )

        if hasattr(self, "aberration_dict"):
            self.metadata = Metadata(
                name="aberrations_metadata",
                data={
                    v["aberration name"]: v["value [Ang]"]
                    for k, v in self.aberration_dict.items()
                },
            )

        self.metadata = Metadata(
            name="reconstruction_metadata",
            data=recon_metadata,
        )

        self._aligned_BF_emd = Array(
            name="aligned_BF",
            data=self._asnumpy(self._recon_BF),
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
            "object_padding_px": instance_md["object_padding_px"],
            "name": instance_md["name"],
            "verbose": True,  # for compatibility
            "device": "cpu",  # for compatibility
        }

        return kwargs

    def _populate_instance(self, group):
        """
        Sets post-initialization properties, notably some preprocessing meta
        optional; during read, this method is run after object instantiation.
        """

        xp = self._xp

        # Preprocess metadata
        preprocess_md = _read_metadata(group, "preprocess_metadata")
        self._scan_sampling = preprocess_md["scan_sampling"]
        self._wavelength = preprocess_md["wavelength"]

        # Reconstruction metadata
        reconstruction_md = _read_metadata(group, "reconstruction_metadata")
        self._recon_error = reconstruction_md["reconstruction_error"]

        # Data
        dict_data = Custom._get_emd_attr_data(Custom, group)

        if "aberration_C1" in reconstruction_md.keys:
            self.rotation_Q_to_R_rads = reconstruction_md["aberration_rotation_QR"]
            self.transpose_detected = reconstruction_md["aberration_transpose"]
            self.aberration_C1 = reconstruction_md["aberration_C1"]
            self.aberration_A1x = reconstruction_md["aberration_A1x"]
            self.aberration_A1y = reconstruction_md["aberration_A1y"]

        if "kde_upsample_factor" in reconstruction_md.keys:
            self._kde_upsample_factor = reconstruction_md["kde_upsample_factor"]
            self._recon_BF_subpixel_aligned = xp.asarray(
                dict_data["_subpixel_aligned_BF_emd"].data, dtype=xp.float32
            )

        self._recon_BF = xp.asarray(dict_data["_aligned_BF_emd"].data, dtype=xp.float32)

    def preprocess(
        self,
        edge_blend: int = 16,
        threshold_intensity: float = 0.8,
        normalize_images: bool = True,
        normalize_order=0,
        descan_correct: bool = True,
        defocus_guess: float = None,
        rotation_guess: float = None,
        plot_average_bf: bool = True,
        **kwargs,
    ):
        """
        Iterative parallax reconstruction preprocessing method.

        Parameters
        ----------
        edge_blend: int, optional
            Pixels to blend image at the border
        threshold: float, optional
            Fraction of max of dp_mean for bright-field pixels
        normalize_images: bool, optional
            If True, bright images normalized to have a mean of 1
        normalize_order: integer, optional
            Polynomial order for normalization. 0 means constant, 1 means linear, etc.
            Higher orders not yet implemented.
        defocus_guess: float, optional
            Initial guess of defocus value (defocus dF) in A
            If None, first iteration is assumed to be in-focus
        descan_correct: float, optional
            If True, aligns bright field stack based on measured descan
        rotation_guess: float, optional
            Initial guess of defocus value in degrees
            If None, first iteration assumed to be 0
        plot_average_bf: bool, optional
            If True, plots the average bright field image, using defocus_guess

        Returns
        --------
        self: ParallaxReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if self._datacube is None:
            raise ValueError(
                (
                    "The preprocess() method requires a DataCube. "
                    "Please run parallax.attach_datacube(DataCube) first."
                )
            )

        # get mean diffraction pattern
        try:
            self._dp_mean = xp.asarray(
                self._datacube.tree("dp_mean").data, dtype=xp.float32
            )
        except AssertionError:
            self._dp_mean = xp.asarray(
                self._datacube.get_dp_mean().data, dtype=xp.float32
            )

        # extract calibrations
        self._intensities = self._extract_intensities_and_calibrations_from_datacube(
            self._datacube,
            require_calibrations=True,
        )

        self._region_of_interest_shape = np.array(self._intensities.shape[-2:])
        self._scan_shape = np.array(self._intensities.shape[:2])

        # make sure mean diffraction pattern is shaped correctly
        if (self._dp_mean.shape[0] != self._intensities.shape[2]) or (
            self._dp_mean.shape[1] != self._intensities.shape[3]
        ):
            raise ValueError(
                "dp_mean must match the datacube shape. Try setting dp_mean = None."
            )

        # descan correct
        if descan_correct:
            (
                _,
                _,
                com_fitted_x,
                com_fitted_y,
                _,
                _,
            ) = self._calculate_intensities_center_of_mass(
                self._intensities,
                dp_mask=None,
                fit_function="plane",
                com_shifts=None,
                com_measured=None,
            )

            com_fitted_x = asnumpy(com_fitted_x)
            com_fitted_y = asnumpy(com_fitted_y)
            intensities = asnumpy(self._intensities)
            intensities_shifted = np.zeros_like(intensities)

            center_x, center_y = self._region_of_interest_shape / 2

            for rx in range(intensities_shifted.shape[0]):
                for ry in range(intensities_shifted.shape[1]):
                    intensity_shifted = get_shifted_ar(
                        intensities[rx, ry],
                        -com_fitted_x[rx, ry] + center_x,
                        -com_fitted_y[rx, ry] + center_y,
                        bilinear=True,
                        device="cpu",
                    )

                    intensities_shifted[rx, ry] = intensity_shifted

            self._intensities = xp.asarray(intensities_shifted, xp.float32)
            self._dp_mean = self._intensities.mean((0, 1))

        # select virtual detector pixels
        self._dp_mask = self._dp_mean >= (xp.max(self._dp_mean) * threshold_intensity)
        self._num_bf_images = int(xp.count_nonzero(self._dp_mask))
        self._wavelength = electron_wavelength_angstrom(self._energy)

        # diffraction space coordinates
        self._xy_inds = np.argwhere(self._dp_mask)
        self._kxy = xp.asarray(
            (self._xy_inds - xp.mean(self._xy_inds, axis=0)[None])
            * xp.array(self._reciprocal_sampling)[None],
            dtype=xp.float32,
        )
        self._probe_angles = self._kxy * self._wavelength
        self._kr = xp.sqrt(xp.sum(self._kxy**2, axis=1))

        # Window function
        x = xp.linspace(-1, 1, self._grid_scan_shape[0] + 1, dtype=xp.float32)[1:]
        x -= (x[1] - x[0]) / 2
        wx = (
            xp.sin(
                xp.clip(
                    (1 - xp.abs(x)) * self._grid_scan_shape[0] / edge_blend / 2, 0, 1
                )
                * (xp.pi / 2)
            )
            ** 2
        )
        y = xp.linspace(-1, 1, self._grid_scan_shape[1] + 1, dtype=xp.float32)[1:]
        y -= (y[1] - y[0]) / 2
        wy = (
            xp.sin(
                xp.clip(
                    (1 - xp.abs(y)) * self._grid_scan_shape[1] / edge_blend / 2, 0, 1
                )
                * (xp.pi / 2)
            )
            ** 2
        )
        self._window_edge = wx[:, None] * wy[None, :]
        self._window_inv = 1 - self._window_edge
        self._window_pad = xp.zeros(
            (
                self._grid_scan_shape[0] + self._object_padding_px[0],
                self._grid_scan_shape[1] + self._object_padding_px[1],
            ),
            dtype=xp.float32,
        )
        self._window_pad[
            self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
            + self._object_padding_px[0] // 2,
            self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
            + self._object_padding_px[1] // 2,
        ] = self._window_edge

        # Collect BF images
        all_bfs = xp.moveaxis(
            self._intensities[:, :, self._xy_inds[:, 0], self._xy_inds[:, 1]],
            (0, 1, 2),
            (1, 2, 0),
        )

        # initalize
        stack_shape = (
            self._num_bf_images,
            self._grid_scan_shape[0] + self._object_padding_px[0],
            self._grid_scan_shape[1] + self._object_padding_px[1],
        )
        if normalize_images:
            self._stack_BF = xp.ones(stack_shape, dtype=xp.float32)
            self._stack_BF_no_window = xp.ones(stack_shape, xp.float32)

            if normalize_order == 0:
                all_bfs /= xp.mean(all_bfs, axis=(1, 2))[:, None, None]
                self._stack_BF[
                    :,
                    self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                    + self._object_padding_px[0] // 2,
                    self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                    + self._object_padding_px[1] // 2,
                ] = (
                    self._window_inv[None] + self._window_edge[None] * all_bfs
                )

                self._stack_BF_no_window[
                    :,
                    self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                    + self._object_padding_px[0] // 2,
                    self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                    + self._object_padding_px[1] // 2,
                ] = all_bfs

            elif normalize_order == 1:
                x = xp.linspace(-0.5, 0.5, all_bfs.shape[1], xp.float32)
                y = xp.linspace(-0.5, 0.5, all_bfs.shape[2], xp.float32)
                ya, xa = xp.meshgrid(y, x)
                basis = np.vstack(
                    (
                        xp.ones_like(xa),
                        xa.ravel(),
                        ya.ravel(),
                    )
                ).T
                for a0 in range(all_bfs.shape[0]):
                    coefs = np.linalg.lstsq(basis, all_bfs[a0].ravel(), rcond=None)

                    self._stack_BF[
                        a0,
                        self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                        + self._object_padding_px[0] // 2,
                        self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                        + self._object_padding_px[1] // 2,
                    ] = self._window_inv[None] + self._window_edge[None] * all_bfs[
                        a0
                    ] / xp.reshape(
                        basis @ coefs[0], all_bfs.shape[1:3]
                    )

                    self._stack_BF_no_window[
                        a0,
                        self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                        + self._object_padding_px[0] // 2,
                        self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                        + self._object_padding_px[1] // 2,
                    ] = all_bfs[a0] / xp.reshape(basis @ coefs[0], all_bfs.shape[1:3])

        else:
            all_means = xp.mean(all_bfs, axis=(1, 2))
            self._stack_BF = xp.full(stack_shape, all_means[:, None, None])
            self._stack_BF_no_window = xp.full(stack_shape, all_means[:, None, None])
            self._stack_BF[
                :,
                self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                + self._object_padding_px[0] // 2,
                self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                + self._object_padding_px[1] // 2,
            ] = (
                self._window_inv[None] * all_means[:, None, None]
                + self._window_edge[None] * all_bfs
            )

            self._stack_BF_no_window[
                :,
                self._object_padding_px[0] // 2 : self._grid_scan_shape[0]
                + self._object_padding_px[0] // 2,
                self._object_padding_px[1] // 2 : self._grid_scan_shape[1]
                + self._object_padding_px[1] // 2,
            ] = all_bfs

        # Fourier space operators for image shifts
        qx = xp.fft.fftfreq(self._stack_BF.shape[1], d=1)
        qx = xp.asarray(qx, dtype=xp.float32)

        qy = xp.fft.fftfreq(self._stack_BF.shape[2], d=1)
        qy = xp.asarray(qy, dtype=xp.float32)

        qxa, qya = xp.meshgrid(qx, qy, indexing="ij")
        self._qx_shift = -2j * xp.pi * qxa
        self._qy_shift = -2j * xp.pi * qya

        # Initialization utilities
        self._stack_mask = xp.tile(self._window_pad[None], (self._num_bf_images, 1, 1))
        if defocus_guess is not None:
            Gs = xp.fft.fft2(self._stack_BF)

            self._xy_shifts = (
                -self._probe_angles * defocus_guess / xp.array(self._scan_sampling)
            )

            if rotation_guess:
                angle = xp.deg2rad(rotation_guess)
                rotation_matrix = xp.array(
                    [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
                )
                self._xy_shifts = xp.dot(self._xy_shifts, rotation_matrix)

            dx = self._xy_shifts[:, 0]
            dy = self._xy_shifts[:, 1]

            shift_op = xp.exp(
                self._qx_shift[None] * dx[:, None, None]
                + self._qy_shift[None] * dy[:, None, None]
            )
            self._stack_BF = xp.real(xp.fft.ifft2(Gs * shift_op))
            self._stack_mask = xp.real(
                xp.fft.ifft2(xp.fft.fft2(self._stack_mask) * shift_op)
            )

            del Gs
        else:
            self._xy_shifts = xp.zeros((self._num_bf_images, 2), dtype=xp.float32)

        self._stack_mean = xp.mean(self._stack_BF)
        self._mask_sum = xp.sum(self._window_edge) * self._num_bf_images
        self._recon_mask = xp.sum(self._stack_mask, axis=0)

        mask_inv = 1 - xp.clip(self._recon_mask, 0, 1)

        self._recon_BF = (
            self._stack_mean * mask_inv
            + xp.sum(self._stack_BF * self._stack_mask, axis=0)
        ) / (self._recon_mask + mask_inv)

        self._recon_error = (
            xp.atleast_1d(
                xp.sum(xp.abs(self._stack_BF - self._recon_BF[None]) * self._stack_mask)
            )
            / self._mask_sum
        )

        self._recon_BF_initial = self._recon_BF.copy()
        self._stack_BF_initial = self._stack_BF.copy()
        self._stack_mask_initial = self._stack_mask.copy()
        self._recon_mask_initial = self._recon_mask.copy()
        self._xy_shifts_initial = self._xy_shifts.copy()

        self.recon_BF = asnumpy(self._recon_BF)

        if plot_average_bf:
            figsize = kwargs.pop("figsize", (6, 6))

            fig, ax = plt.subplots(figsize=figsize)

            self._visualize_figax(fig, ax, **kwargs)

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Average Bright Field Image")

        self._preprocessed = True

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def tune_angle_and_defocus(
        self,
        angle_guess=None,
        defocus_guess=None,
        angle_step_size=5,
        defocus_step_size=100,
        num_angle_values=5,
        num_defocus_values=5,
        return_values=False,
        plot_reconstructions=True,
        plot_convergence=True,
        **kwargs,
    ):
        """
        Run parallax reconstruction over a parameters space of pre-determined angles
        and defocus

        Parameters
        ----------
        angle_guess: float (degrees), optional
            initial starting guess for rotation angle between real and reciprocal space
            if None, uses 0
        defocus_guess: float (A), optional
            initial starting guess for defocus (defocus dF)
            if None, uses 0
        angle_step_size: float (degrees), optional
            size of change of rotation angle between real and reciprocal space for
            each step in parameter space
        defocus_step_size: float (A), optional
            size of change of defocus for each step in parameter space
        num_angle_values: int, optional
            number of values of angle to test, must be >= 1.
        num_defocus_values: int,optional
            number of values of defocus to test, must be >= 1
        plot_reconstructions: bool, optional
            if True, plot phase of reconstructed objects
        plot_convergence: bool, optional
            if True, makes 2D plot of error metrix
        return_values: bool, optional
            if True, returns objects, convergence

        Returns
        -------
        objects: list
            reconstructed objects
        convergence: np.ndarray
            array of convergence values from reconstructions
        """
        asnumpy = self._asnumpy

        if angle_guess is None:
            angle_guess = 0
        if defocus_guess is None:
            defocus_guess = 0

        if num_angle_values == 1:
            angle_step_size = 0

        if num_defocus_values == 1:
            defocus_step_size = 0

        angles = np.linspace(
            angle_guess - angle_step_size * (num_angle_values - 1) / 2,
            angle_guess + angle_step_size * (num_angle_values - 1) / 2,
            num_angle_values,
        )

        defocus_values = np.linspace(
            defocus_guess - defocus_step_size * (num_defocus_values - 1) / 2,
            defocus_guess + defocus_step_size * (num_defocus_values - 1) / 2,
            num_defocus_values,
        )
        if return_values or plot_convergence:
            recon_BF = []
            convergence = []

        if plot_reconstructions:
            spec = GridSpec(
                ncols=num_defocus_values,
                nrows=num_angle_values,
                hspace=0.15,
                wspace=0.35,
            )
            figsize = kwargs.get(
                "figsize", (4 * num_defocus_values, 4 * num_angle_values)
            )

            fig = plt.figure(figsize=figsize)

        # run loop and plot along the way
        self._verbose = False
        for flat_index, (angle, defocus) in enumerate(
            tqdmnd(angles, defocus_values, desc="Tuning angle and defocus")
        ):
            self.preprocess(
                defocus_guess=defocus,
                rotation_guess=angle,
                plot_average_bf=False,
                **kwargs,
            )

            if plot_reconstructions:
                row_index, col_index = np.unravel_index(
                    flat_index, (num_angle_values, num_defocus_values)
                )
                object_ax = fig.add_subplot(spec[row_index, col_index])
                self._visualize_figax(
                    fig,
                    ax=object_ax,
                )

                object_ax.set_title(
                    f" angle = {angle:.1f} °, defocus = {defocus:.1f} A \n error = {self._recon_error[0]:.3e}"
                )
                object_ax.set_xticks([])
                object_ax.set_yticks([])

            if return_values:
                recon_BF.append(self.recon_BF)
            if return_values or plot_convergence:
                convergence.append(asnumpy(self._recon_error[0]))

        if plot_convergence:
            fig, ax = plt.subplots()
            ax.set_title("convergence")
            im = ax.imshow(
                np.array(convergence).reshape(angles.shape[0], defocus_values.shape[0]),
                cmap="magma",
            )

            if angles.shape[0] > 1:
                ax.set_ylabel("angles")
                ax.set_yticks(np.arange(angles.shape[0]))
                ax.set_yticklabels([f"{angle:.1f} °" for angle in angles])
            else:
                ax.set_yticks([])
                ax.set_ylabel(f"angle {angles[0]:.1f}")

            if defocus_values.shape[0] > 1:
                ax.set_xlabel("defocus values")
                ax.set_xticks(np.arange(defocus_values.shape[0]))
                ax.set_xticklabels([f"{df:.1f}" for df in defocus_values])
            else:
                ax.set_xticks([])
                ax.set_xlabel(f"defocus value: {defocus_values[0]:.1f}")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            fig.tight_layout()

        if return_values:
            convergence = np.array(convergence).reshape(
                angles.shape[0], defocus_values.shape[0]
            )
            return recon_BF, convergence

    def reconstruct(
        self,
        max_alignment_bin: int = None,
        min_alignment_bin: int = 1,
        max_iter_at_min_bin: int = 2,
        cross_correlation_upsample_factor: int = 8,
        regularizer_matrix_size: Tuple[int, int] = (1, 1),
        regularize_shifts: bool = True,
        running_average: bool = True,
        progress_bar: bool = True,
        plot_aligned_bf: bool = True,
        plot_convergence: bool = True,
        reset: bool = None,
        **kwargs,
    ):
        """
        Iterative Parallax Reconstruction main reconstruction method.

        Parameters
        ----------
        max_alignment_bin: int, optional
            Maximum bin size for bright field alignment
            If None, the bright field disk radius is used
        min_alignment_bin: int, optional
            Minimum bin size for bright field alignment
        max_iter_at_min_bin: int, optional
            Number of iterations to run at the smallest bin size
        cross_correlation_upsample_factor: int, optional
            DFT upsample factor for subpixel alignment
        regularizer_matrix_size: Tuple[int,int], optional
            Bernstein basis degree used for regularizing shifts
        regularize_shifts: bool, optional
            If True, the cross-correlated shifts are constrained to a spline interpolation
        running_average: bool, optional
            If True, the bright field reference image is updated in a spiral from the origin
        progress_bar: bool, optional
            If True, progress bar is displayed
        plot_aligned_bf: bool, optional
            If True, the aligned bright field image is plotted at each bin level
        plot_convergence: bool, optional
            If True, the convergence error is also plotted
        reset: bool, optional
            If True, the reconstruction is reset

        Returns
        --------
        self: BFReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if reset:
            self._recon_BF = self._recon_BF_initial.copy()
            self._stack_BF = self._stack_BF_initial.copy()
            self._stack_mask = self._stack_mask_initial.copy()
            self._recon_mask = self._recon_mask_initial.copy()
            self._xy_shifts = self._xy_shifts_initial.copy()
        elif reset is None:
            if hasattr(self, "_basis"):
                warnings.warn(
                    (
                        "Continuing reconstruction from previous result. "
                        "Use reset=True for a fresh start."
                    ),
                    UserWarning,
                )

        if not regularize_shifts:
            self._basis = self._kxy
        else:
            kr_max = xp.max(self._kr)
            u = self._kxy[:, 0] * 0.5 / kr_max + 0.5
            v = self._kxy[:, 1] * 0.5 / kr_max + 0.5

            self._basis = xp.zeros(
                (
                    self._num_bf_images,
                    (regularizer_matrix_size[0] + 1) * (regularizer_matrix_size[1] + 1),
                ),
                dtype=xp.float32,
            )
            for ii in np.arange(regularizer_matrix_size[0] + 1):
                Bi = (
                    comb(regularizer_matrix_size[0], ii)
                    * (u**ii)
                    * ((1 - u) ** (regularizer_matrix_size[0] - ii))
                )

                for jj in np.arange(regularizer_matrix_size[1] + 1):
                    Bj = (
                        comb(regularizer_matrix_size[1], jj)
                        * (v**jj)
                        * ((1 - v) ** (regularizer_matrix_size[1] - jj))
                    )

                    ind = ii * (regularizer_matrix_size[1] + 1) + jj
                    self._basis[:, ind] = Bi * Bj

        # Iterative binning for more robust alignment
        diameter_pixels = int(
            xp.maximum(
                xp.max(self._xy_inds[:, 0]) - xp.min(self._xy_inds[:, 0]),
                xp.max(self._xy_inds[:, 1]) - xp.min(self._xy_inds[:, 1]),
            )
            + 1
        )

        if max_alignment_bin is not None:
            max_alignment_bin = np.minimum(diameter_pixels, max_alignment_bin)
        else:
            max_alignment_bin = diameter_pixels

        bin_min = np.ceil(np.log(min_alignment_bin) / np.log(2))
        bin_max = np.ceil(np.log(max_alignment_bin) / np.log(2))
        bin_vals = 2 ** np.arange(bin_min, bin_max)[::-1]

        if max_iter_at_min_bin > 1:
            bin_vals = np.hstack(
                (bin_vals, np.repeat(bin_vals[-1], max_iter_at_min_bin - 1))
            )

        if plot_aligned_bf:
            num_plots = bin_vals.shape[0]
            nrows = int(np.sqrt(num_plots))
            ncols = int(np.ceil(num_plots / nrows))

            if plot_convergence:
                errors = []
                spec = GridSpec(
                    ncols=ncols,
                    nrows=nrows + 1,
                    hspace=0.15,
                    wspace=0.15,
                    height_ratios=[1] * nrows + [1 / 4],
                )

                figsize = kwargs.get("figsize", (4 * ncols, 4 * nrows + 1))
            else:
                spec = GridSpec(
                    ncols=ncols,
                    nrows=nrows,
                    hspace=0.15,
                    wspace=0.15,
                )

                figsize = kwargs.get("figsize", (4 * ncols, 4 * nrows))

            kwargs.pop("figsize", None)
            fig = plt.figure(figsize=figsize)

        xy_center = (self._xy_inds - xp.median(self._xy_inds, axis=0)).astype("float")

        # Loop over all binning values
        for a0 in range(bin_vals.shape[0]):
            G_ref = xp.fft.fft2(self._recon_BF)

            # Segment the virtual images with current binning values
            xy_inds = xp.round(xy_center / bin_vals[a0] + 0.5).astype("int")
            xy_vals = np.unique(
                asnumpy(xy_inds), axis=0
            )  # axis is not yet supported in cupy
            # Sort by radial order, from center to outer edge
            inds_order = xp.argsort(xp.sum(xy_vals**2, axis=1))

            shifts_update = xp.zeros((self._num_bf_images, 2), dtype=xp.float32)

            for a1 in tqdmnd(
                xy_vals.shape[0],
                desc="Alignment at bin " + str(bin_vals[a0].astype("int")),
                unit=" image subsets",
                disable=not progress_bar,
            ):
                ind_align = inds_order[a1]

                # Generate mean image for alignment
                sub = xp.logical_and(
                    xy_inds[:, 0] == xy_vals[ind_align, 0],
                    xy_inds[:, 1] == xy_vals[ind_align, 1],
                )

                G = xp.fft.fft2(xp.mean(self._stack_BF[sub], axis=0))

                # Get best fit alignment
                xy_shift = align_images_fourier(
                    G_ref,
                    G,
                    upsample_factor=cross_correlation_upsample_factor,
                    device=self._device,
                )

                dx = (
                    xp.mod(
                        xy_shift[0] + self._stack_BF.shape[1] / 2,
                        self._stack_BF.shape[1],
                    )
                    - self._stack_BF.shape[1] / 2
                )
                dy = (
                    xp.mod(
                        xy_shift[1] + self._stack_BF.shape[2] / 2,
                        self._stack_BF.shape[2],
                    )
                    - self._stack_BF.shape[2] / 2
                )

                # output shifts
                shifts_update[sub, 0] = dx
                shifts_update[sub, 1] = dy

                # update running estimate of reference image
                shift_op = xp.exp(self._qx_shift * dx + self._qy_shift * dy)

                if running_average:
                    G_ref = G_ref * a1 / (a1 + 1) + (G * shift_op) / (a1 + 1)

            # regularize the shifts
            xy_shifts_new = self._xy_shifts + shifts_update
            coefs = xp.linalg.lstsq(self._basis, xy_shifts_new, rcond=None)[0]
            xy_shifts_fit = self._basis @ coefs
            shifts_update = xy_shifts_fit - self._xy_shifts

            # apply shifts
            Gs = xp.fft.fft2(self._stack_BF)

            dx = shifts_update[:, 0]
            dy = shifts_update[:, 1]
            self._xy_shifts[:, 0] += dx
            self._xy_shifts[:, 1] += dy

            shift_op = xp.exp(
                self._qx_shift[None] * dx[:, None, None]
                + self._qy_shift[None] * dy[:, None, None]
            )

            self._stack_BF = xp.real(xp.fft.ifft2(Gs * shift_op))
            self._stack_mask = xp.real(
                xp.fft.ifft2(xp.fft.fft2(self._stack_mask) * shift_op)
            )

            self._stack_BF = xp.asarray(
                self._stack_BF, dtype=xp.float32
            )  # numpy fft upcasts?
            self._stack_mask = xp.asarray(
                self._stack_mask, dtype=xp.float32
            )  # numpy fft upcasts?

            del Gs

            # Center the shifts
            xy_shifts_median = xp.round(xp.median(self._xy_shifts, axis=0)).astype(int)
            self._xy_shifts -= xy_shifts_median[None, :]
            self._stack_BF = xp.roll(self._stack_BF, -xy_shifts_median, axis=(1, 2))
            self._stack_mask = xp.roll(self._stack_mask, -xy_shifts_median, axis=(1, 2))

            # Generate new estimate
            self._recon_mask = xp.sum(self._stack_mask, axis=0)

            mask_inv = 1 - np.clip(self._recon_mask, 0, 1)
            self._recon_BF = (
                self._stack_mean * mask_inv
                + xp.sum(self._stack_BF * self._stack_mask, axis=0)
            ) / (self._recon_mask + mask_inv)

            self._recon_error = (
                xp.atleast_1d(
                    xp.sum(
                        xp.abs(self._stack_BF - self._recon_BF[None]) * self._stack_mask
                    )
                )
                / self._mask_sum
            )

            if plot_aligned_bf:
                row_index, col_index = np.unravel_index(a0, (nrows, ncols))

                ax = fig.add_subplot(spec[row_index, col_index])
                self._visualize_figax(fig, ax, **kwargs)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Aligned BF at bin {int(bin_vals[a0])}")

                if plot_convergence:
                    errors.append(float(self._recon_error))

        if plot_aligned_bf:
            if plot_convergence:
                ax = fig.add_subplot(spec[-1, :])
                ax.plot(np.arange(num_plots), errors)
                ax.set_xticks(np.arange(num_plots))
                ax.set_ylabel("Error")
            spec.tight_layout(fig)

        self.recon_BF = asnumpy(self._recon_BF)

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return self

    def subpixel_alignment(
        self,
        kde_upsample_factor=None,
        kde_sigma=0.125,
        plot_upsampled_BF_comparison: bool = True,
        plot_upsampled_FFT_comparison: bool = False,
        **kwargs,
    ):
        """
        Upsample and subpixel-align BFs using the measured image shifts.
        Uses kernel density estimation (KDE) to align upsampled BFs.

        Parameters
        ----------
        kde_upsample_factor: int, optional
            Real-space upsampling factor
        kde_sigma: float, optional
            KDE gaussian kernel bandwidth
        plot_upsampled_BF_comparison: bool, optional
            If True, the pre/post alignment BF images are plotted for comparison
        plot_upsampled_FFT_comparison: bool, optional
            If True, the pre/post alignment BF FFTs are plotted for comparison

        """
        xp = self._xp
        asnumpy = self._asnumpy
        gaussian_filter = self._gaussian_filter

        xy_shifts = self._xy_shifts
        BF_size = np.array(self._stack_BF_no_window.shape[-2:])

        self._DF_upsample_limit = np.max(
            self._region_of_interest_shape / self._scan_shape
        )
        self._BF_upsample_limit = (
            2 * self._kr.max() / self._reciprocal_sampling[0]
        ) / self._scan_shape.max()
        if self._device == "gpu":
            self._BF_upsample_limit = self._BF_upsample_limit.item()

        if kde_upsample_factor is None:
            kde_upsample_factor = np.minimum(
                self._BF_upsample_limit * 3 / 2, self._DF_upsample_limit
            )

            warnings.warn(
                (
                    f"Upsampling factor set to {kde_upsample_factor:.2f} (1.5 times the "
                    f"bright-field upsampling limit of {self._BF_upsample_limit:.2f})."
                ),
                UserWarning,
            )

        if kde_upsample_factor < 1:
            raise ValueError("kde_upsample_factor must be larger than 1")

        if kde_upsample_factor > self._DF_upsample_limit:
            warnings.warn(
                (
                    "Requested upsampling factor exceeds "
                    f"dark-field upsampling limit of {self._DF_upsample_limit:.2f}."
                ),
                UserWarning,
            )

        self._kde_upsample_factor = kde_upsample_factor
        pixel_output = np.round(BF_size * self._kde_upsample_factor).astype("int")
        pixel_size = pixel_output.prod()

        # shifted coordinates
        x = xp.arange(BF_size[0])
        y = xp.arange(BF_size[1])

        xa, ya = xp.meshgrid(x, y, indexing="ij")
        xa = ((xa + xy_shifts[:, 0, None, None]) * self._kde_upsample_factor).ravel()
        ya = ((ya + xy_shifts[:, 1, None, None]) * self._kde_upsample_factor).ravel()

        # bilinear sampling
        xF = xp.floor(xa).astype("int")
        yF = xp.floor(ya).astype("int")
        dx = xa - xF
        dy = ya - yF

        # resampling
        inds_1D = xp.ravel_multi_index(
            xp.hstack(
                [
                    [xF, yF],
                    [xF + 1, yF],
                    [xF, yF + 1],
                    [xF + 1, yF + 1],
                ]
            ),
            pixel_output,
            mode=["wrap", "wrap"],
        )

        weights = xp.hstack(
            (
                (1 - dx) * (1 - dy),
                (dx) * (1 - dy),
                (1 - dx) * (dy),
                (dx) * (dy),
            )
        )

        pix_count = xp.reshape(
            xp.bincount(inds_1D, weights=weights, minlength=pixel_size), pixel_output
        )
        pix_output = xp.reshape(
            xp.bincount(
                inds_1D,
                weights=weights * xp.tile(self._stack_BF_no_window.ravel(), 4),
                minlength=pixel_size,
            ),
            pixel_output,
        )

        # kernel density estimate
        sigma = kde_sigma * self._kde_upsample_factor
        pix_count = gaussian_filter(pix_count, sigma)
        pix_count[pix_output == 0.0] = np.inf
        pix_output = gaussian_filter(pix_output, sigma)
        pix_output /= pix_count

        self._recon_BF_subpixel_aligned = pix_output
        self.recon_BF_subpixel_aligned = asnumpy(self._recon_BF_subpixel_aligned)

        # plotting
        if plot_upsampled_BF_comparison:
            if plot_upsampled_FFT_comparison:
                figsize = kwargs.pop("figsize", (8, 8))
                fig, axs = plt.subplots(2, 2, figsize=figsize)
            else:
                figsize = kwargs.pop("figsize", (8, 4))
                fig, axs = plt.subplots(1, 2, figsize=figsize)

            axs = axs.flat
            cmap = kwargs.pop("cmap", "magma")

            cropped_object = self._crop_padded_object(self._recon_BF)
            cropped_object_aligned = self._crop_padded_object(
                self._recon_BF_subpixel_aligned, upsampled=True
            )

            extent = [
                0,
                self._scan_sampling[1] * cropped_object.shape[1],
                self._scan_sampling[0] * cropped_object.shape[0],
                0,
            ]

            axs[0].imshow(
                cropped_object,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            axs[0].set_title("Aligned Bright Field")

            axs[1].imshow(
                cropped_object_aligned,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )
            axs[1].set_title("Upsampled Bright Field")

            for ax in axs[:2]:
                ax.set_ylabel("x [A]")
                ax.set_xlabel("y [A]")

            if plot_upsampled_FFT_comparison:
                recon_fft = xp.fft.fftshift(xp.abs(xp.fft.fft2(self._recon_BF)))
                pad_x = np.round(
                    BF_size[0] * (self._kde_upsample_factor - 1) / 2
                ).astype("int")
                pad_y = np.round(
                    BF_size[1] * (self._kde_upsample_factor - 1) / 2
                ).astype("int")
                pad_recon_fft = asnumpy(
                    xp.pad(recon_fft, ((pad_x, pad_x), (pad_y, pad_y)))
                )

                upsampled_fft = asnumpy(
                    xp.fft.fftshift(
                        xp.abs(xp.fft.fft2(self._recon_BF_subpixel_aligned))
                    )
                )

                reciprocal_extent = [
                    -0.5 / (self._scan_sampling[1] / self._kde_upsample_factor),
                    0.5 / (self._scan_sampling[1] / self._kde_upsample_factor),
                    0.5 / (self._scan_sampling[0] / self._kde_upsample_factor),
                    -0.5 / (self._scan_sampling[0] / self._kde_upsample_factor),
                ]

                show(
                    pad_recon_fft,
                    figax=(fig, axs[2]),
                    extent=reciprocal_extent,
                    cmap="gray",
                    title="Aligned Bright Field FFT",
                    **kwargs,
                )

                show(
                    upsampled_fft,
                    figax=(fig, axs[3]),
                    extent=reciprocal_extent,
                    cmap="gray",
                    title="Upsampled Bright Field FFT",
                    **kwargs,
                )

                for ax in axs[2:]:
                    ax.set_ylabel(r"$k_x$ [$A^{-1}$]")
                    ax.set_xlabel(r"$k_y$ [$A^{-1}$]")
                    ax.xaxis.set_ticks_position("bottom")

            fig.tight_layout()

    def aberration_fit(
        self,
        fit_BF_shifts: bool = False,
        fit_CTF_FFT: bool = False,
        fit_aberrations_max_radial_order: int = 3,
        fit_aberrations_max_angular_order: int = 4,
        fit_aberrations_min_radial_order: int = 2,
        fit_aberrations_min_angular_order: int = 0,
        fit_max_thon_rings: int = 6,
        fit_power_alpha: float = 2.0,
        plot_CTF_comparison: bool = None,
        plot_BF_shifts_comparison: bool = None,
        upsampled: bool = True,
        force_transpose: bool = None,
    ):
        """
        Fit aberrations to the measured image shifts.

        Parameters
        ----------
        fit_BF_shifts: bool
            Set to True to fit aberrations to the measured BF shifts directly.
        fit_CTF_FFT: bool
            Set to True to fit aberrations in the FFT of the (upsampled) BF
            image. Note that this method relies on visible zero crossings in the FFT.
        fit_aberrations_max_radial_order: int
            Max radial order for fitting of aberrations.
        fit_aberrations_max_angular_order: int
            Max angular order for fitting of aberrations.
        fit_aberrations_min_radial_order: int
            Min radial order for fitting of aberrations.
        fit_aberrations_min_angular_order: int
            Min angular order for fitting of aberrations.
        fit_max_thon_rings: int
            Max number of Thon rings to search for during CTF FFT fitting.
        fit_power_alpha: int
            Power to raise FFT alpha weighting during CTF FFT fitting.
        plot_CTF_comparison: bool, optional
            If True, the fitted CTF is plotted against the reconstructed frequencies.
        plot_BF_shifts_comparison: bool, optional
            If True, the measured vs fitted BF shifts are plotted.
        upsampled: bool
            If True, and upsampled BF is available, uses that for CTF FFT fitting.
        force_transpose: bool
            If True, and fit_BF_shifts is True, flips the measured x and y shifts

        """
        xp = self._xp
        asnumpy = self._asnumpy

        ### First pass

        # Convert real space shifts to Angstroms

        if force_transpose is None:
            self.transpose_detected = False
        else:
            self.transpose_detected = force_transpose

        if force_transpose is True:
            self._xy_shifts_Ang = xp.flip(self._xy_shifts, axis=1) * xp.array(
                self._scan_sampling
            )
        else:
            self._xy_shifts_Ang = self._xy_shifts * xp.array(self._scan_sampling)

        # Solve affine transformation
        m = asnumpy(
            xp.linalg.lstsq(self._probe_angles, self._xy_shifts_Ang, rcond=None)[0]
        )
        m_rotation, m_aberration = polar(m, side="right")

        # Convert into rotation and aberration coefficients
        self.rotation_Q_to_R_rads = -1 * np.arctan2(m_rotation[1, 0], m_rotation[0, 0])
        if np.abs(np.mod(self.rotation_Q_to_R_rads + np.pi, 2.0 * np.pi) - np.pi) > (
            np.pi * 0.5
        ):
            self.rotation_Q_to_R_rads = (
                np.mod(self.rotation_Q_to_R_rads, 2.0 * np.pi) - np.pi
            )
            m_aberration = -1.0 * m_aberration
        self.aberration_C1 = (m_aberration[0, 0] + m_aberration[1, 1]) / 2.0
        self.aberration_A1x = (m_aberration[0, 0] - m_aberration[1, 1]) / 2.0
        self.aberration_A1y = (m_aberration[1, 0] + m_aberration[0, 1]) / 2.0

        ### Second pass

        # Aberration coefs
        mn = []

        for m in range(
            fit_aberrations_min_radial_order - 1, fit_aberrations_max_radial_order
        ):
            n_max = np.minimum(fit_aberrations_max_angular_order, m + 1)
            for n in range(fit_aberrations_min_angular_order, n_max + 1):
                if (m + n) % 2:
                    mn.append([m, n, 0])
                    if n > 0:
                        mn.append([m, n, 1])

        self._aberrations_mn = np.array(mn)
        self._aberrations_mn = self._aberrations_mn[
            np.argsort(self._aberrations_mn[:, 1]), :
        ]

        sub = self._aberrations_mn[:, 1] > 0
        self._aberrations_mn[sub, :] = self._aberrations_mn[sub, :][
            np.argsort(self._aberrations_mn[sub, 0]), :
        ]
        self._aberrations_mn[~sub, :] = self._aberrations_mn[~sub, :][
            np.argsort(self._aberrations_mn[~sub, 0]), :
        ]
        self._aberrations_num = self._aberrations_mn.shape[0]

        if plot_CTF_comparison is None:
            if fit_CTF_FFT:
                plot_CTF_comparison = True

        if plot_BF_shifts_comparison is None:
            if fit_BF_shifts:
                plot_BF_shifts_comparison = True

        # Thon Rings Fitting
        if fit_CTF_FFT or plot_CTF_comparison:
            if upsampled and hasattr(self, "_kde_upsample_factor"):
                im_FFT = xp.abs(xp.fft.fft2(self._recon_BF_subpixel_aligned))
                sx = self._scan_sampling[0] / self._kde_upsample_factor
                sy = self._scan_sampling[1] / self._kde_upsample_factor

            else:
                im_FFT = xp.abs(xp.fft.fft2(self._recon_BF))
                sx = self._scan_sampling[0]
                sy = self._scan_sampling[1]
                upsampled = False

            # FFT coordinates
            qx = xp.fft.fftfreq(im_FFT.shape[0], sx)
            qy = xp.fft.fftfreq(im_FFT.shape[1], sy)
            qr2 = qx[:, None] ** 2 + qy[None, :] ** 2

            alpha_FFT = xp.sqrt(qr2) * self._wavelength
            theta_FFT = xp.arctan2(qy[None, :], qx[:, None])

            # Aberration basis
            self._aberrations_basis_FFT = xp.zeros(
                (alpha_FFT.size, self._aberrations_num)
            )
            for a0 in range(self._aberrations_num):
                m, n, a = self._aberrations_mn[a0]
                if n == 0:
                    # Radially symmetric basis
                    self._aberrations_basis_FFT[:, a0] = (
                        alpha_FFT ** (m + 1) / (m + 1)
                    ).ravel()

                elif a == 0:
                    # cos coef
                    self._aberrations_basis_FFT[:, a0] = (
                        alpha_FFT ** (m + 1) * xp.cos(n * theta_FFT) / (m + 1)
                    ).ravel()
                else:
                    # sin coef
                    self._aberrations_basis_FFT[:, a0] = (
                        alpha_FFT ** (m + 1) * xp.sin(n * theta_FFT) / (m + 1)
                    ).ravel()

            # global scaling
            self._aberrations_basis_FFT *= 2 * np.pi / self._wavelength
            self._aberrations_surface_shape_FFT = alpha_FFT.shape
            plot_mask = qr2 > np.pi**2 / 4 / np.abs(self.aberration_C1)
            angular_mask = np.cos(8.0 * theta_FFT) ** 2 < 0.25

            # CTF function
            def calculate_CTF_FFT(alpha_shape, *coefs):
                chi = xp.zeros_like(self._aberrations_basis_FFT[:, 0])
                for a0 in range(len(coefs)):
                    chi += coefs[a0] * self._aberrations_basis_FFT[:, a0]
                return xp.reshape(chi, alpha_shape)

        # Direct Shifts Fitting
        if fit_BF_shifts:
            # FFT coordinates
            sx = 1 / (self._reciprocal_sampling[0] * self._region_of_interest_shape[0])
            sy = 1 / (self._reciprocal_sampling[1] * self._region_of_interest_shape[1])
            qx = xp.fft.fftfreq(self._region_of_interest_shape[0], sx)
            qy = xp.fft.fftfreq(self._region_of_interest_shape[1], sy)
            qr2 = qx[:, None] ** 2 + qy[None, :] ** 2

            u = qx[:, None] * self._wavelength
            v = qy[None, :] * self._wavelength
            alpha = xp.sqrt(qr2) * self._wavelength
            theta = xp.arctan2(qy[None, :], qx[:, None])

            # Aberration basis
            self._aberrations_basis = xp.zeros((alpha.size, self._aberrations_num))
            self._aberrations_basis_du = xp.zeros((alpha.size, self._aberrations_num))
            self._aberrations_basis_dv = xp.zeros((alpha.size, self._aberrations_num))
            for a0 in range(self._aberrations_num):
                m, n, a = self._aberrations_mn[a0]

                if n == 0:
                    # Radially symmetric basis
                    self._aberrations_basis[:, a0] = (
                        alpha ** (m + 1) / (m + 1)
                    ).ravel()
                    self._aberrations_basis_du[:, a0] = (u * alpha ** (m - 1)).ravel()
                    self._aberrations_basis_dv[:, a0] = (v * alpha ** (m - 1)).ravel()

                elif a == 0:
                    # cos coef
                    self._aberrations_basis[:, a0] = (
                        alpha ** (m + 1) * xp.cos(n * theta) / (m + 1)
                    ).ravel()
                    self._aberrations_basis_du[:, a0] = (
                        alpha ** (m - 1)
                        * ((m + 1) * u * xp.cos(n * theta) + n * v * xp.sin(n * theta))
                        / (m + 1)
                    ).ravel()
                    self._aberrations_basis_dv[:, a0] = (
                        alpha ** (m - 1)
                        * ((m + 1) * v * xp.cos(n * theta) - n * u * xp.sin(n * theta))
                        / (m + 1)
                    ).ravel()

                else:
                    # sin coef
                    self._aberrations_basis[:, a0] = (
                        alpha ** (m + 1) * xp.sin(n * theta) / (m + 1)
                    ).ravel()
                    self._aberrations_basis_du[:, a0] = (
                        alpha ** (m - 1)
                        * ((m + 1) * u * xp.sin(n * theta) - n * v * xp.cos(n * theta))
                        / (m + 1)
                    ).ravel()
                    self._aberrations_basis_dv[:, a0] = (
                        alpha ** (m - 1)
                        * ((m + 1) * v * xp.sin(n * theta) + n * u * xp.cos(n * theta))
                        / (m + 1)
                    ).ravel()

            # global scaling
            self._aberrations_basis *= 2 * np.pi / self._wavelength
            self._aberrations_surface_shape = alpha.shape

            # CTF function
            def calculate_CTF(alpha_shape, *coefs):
                chi = xp.zeros_like(self._aberrations_basis[:, 0])
                for a0 in range(len(coefs)):
                    chi += coefs[a0] * self._aberrations_basis[:, a0]
                return xp.reshape(chi, alpha_shape)

        # initial coefficients and plotting intensity range mask
        self._aberrations_coefs = np.zeros(self._aberrations_num)
        ind = np.argmin(
            np.abs(self._aberrations_mn[:, 0] - 1.0) + self._aberrations_mn[:, 1]
        )
        self._aberrations_coefs[ind] = self.aberration_C1

        # Refinement using CTF fitting / Thon rings
        if fit_CTF_FFT:
            # scoring function to minimize - mean value of zero crossing regions of FFT
            def score_CTF(coefs):
                im_CTF = xp.abs(
                    calculate_CTF_FFT(self._aberrations_surface_shape_FFT, *coefs)
                )
                mask = xp.logical_and(
                    im_CTF > 0.5 * np.pi,
                    im_CTF < (max_num_rings + 0.5) * np.pi,
                )
                if np.any(mask):
                    weights = xp.cos(im_CTF[mask]) ** 4
                    return asnumpy(
                        xp.sum(
                            weights * im_FFT[mask] * alpha_FFT[mask] ** fit_power_alpha
                        )
                        / xp.sum(weights)
                    )
                else:
                    return np.inf

            for max_num_rings in range(1, fit_max_thon_rings + 1):
                # minimization
                res = minimize(
                    score_CTF,
                    self._aberrations_coefs,
                    # method = 'Nelder-Mead',
                    # method = 'CG',
                    method="BFGS",
                    tol=1e-8,
                )
                self._aberrations_coefs = res.x

        # Refinement using CTF fitting / Thon rings
        elif fit_BF_shifts:
            # Gradient basis
            corner_indices = self._xy_inds - xp.asarray(
                self._region_of_interest_shape // 2
            )
            raveled_indices = np.ravel_multi_index(
                corner_indices.T, self._region_of_interest_shape, mode="wrap"
            )
            gradients = xp.vstack(
                (
                    self._aberrations_basis_du[raveled_indices, :],
                    self._aberrations_basis_dv[raveled_indices, :],
                )
            )

            # (Relative) untransposed fit
            tf = AffineTransform(angle=self.rotation_Q_to_R_rads)
            rotated_shifts = tf(self._xy_shifts_Ang, xp=xp).T.ravel()
            aberrations_coefs, res = xp.linalg.lstsq(
                gradients, rotated_shifts, rcond=None
            )[:2]

            if force_transpose is None:
                # (Relative) transposed fit
                transposed_shifts = xp.flip(self._xy_shifts_Ang, axis=1)
                m_T = asnumpy(
                    xp.linalg.lstsq(self._probe_angles, transposed_shifts, rcond=None)[
                        0
                    ]
                )
                m_rotation_T, _ = polar(m_T, side="right")
                rotation_Q_to_R_rads_T = -1 * np.arctan2(
                    m_rotation_T[1, 0], m_rotation_T[0, 0]
                )
                if np.abs(
                    np.mod(rotation_Q_to_R_rads_T + np.pi, 2.0 * np.pi) - np.pi
                ) > (np.pi * 0.5):
                    rotation_Q_to_R_rads_T = (
                        np.mod(rotation_Q_to_R_rads_T, 2.0 * np.pi) - np.pi
                    )

                tf_T = AffineTransform(angle=rotation_Q_to_R_rads_T)
                rotated_shifts_T = tf_T(transposed_shifts, xp=xp).T.ravel()
                aberrations_coefs_T, res_T = xp.linalg.lstsq(
                    gradients, rotated_shifts_T, rcond=None
                )[:2]

                # Compare fits
                if res_T.sum() < res.sum():
                    self.rotation_Q_to_R_rads = rotation_Q_to_R_rads_T
                    self.transpose_detected = not self.transpose_detected
                    self._aberrations_coefs = asnumpy(aberrations_coefs_T)
                    self._rotated_shifts = rotated_shifts_T

                    warnings.warn(
                        (
                            "Data transpose detected. "
                            f"Overwriting rotation value to {np.rad2deg(rotation_Q_to_R_rads_T):.3f}"
                        ),
                        UserWarning,
                    )
            else:
                self._aberrations_coefs = asnumpy(aberrations_coefs)
                self._rotated_shifts = rotated_shifts

        # Plot the CTF comparison between experiment and fit
        if plot_CTF_comparison:
            # Generate FFT plotting image
            im_scale = asnumpy(im_FFT * alpha_FFT**fit_power_alpha)
            int_vals = np.sort(im_scale.ravel())
            int_range = (
                int_vals[np.round(0.02 * im_scale.size).astype("int")],
                int_vals[np.round(0.98 * im_scale.size).astype("int")],
            )
            int_range = (
                int_range[0],
                (int_range[1] - int_range[0]) * 1.0 + int_range[0],
            )
            im_scale = np.clip(
                (np.fft.fftshift(im_scale) - int_range[0])
                / (int_range[1] - int_range[0]),
                0,
                1,
            )
            im_plot = np.tile(im_scale[:, :, None], (1, 1, 3))

            # Add CTF zero crossings
            im_CTF = calculate_CTF_FFT(
                self._aberrations_surface_shape_FFT, *self._aberrations_coefs
            )
            im_CTF_cos = xp.cos(xp.abs(im_CTF)) ** 4
            im_CTF[xp.abs(im_CTF) > (fit_max_thon_rings + 0.5) * np.pi] = np.pi / 2
            im_CTF = xp.abs(xp.sin(im_CTF)) < 0.15
            im_CTF[xp.logical_not(plot_mask)] = 0

            im_CTF = np.fft.fftshift(asnumpy(im_CTF * angular_mask))
            im_plot[:, :, 0] += im_CTF
            im_plot[:, :, 1] -= im_CTF
            im_plot[:, :, 2] -= im_CTF
            im_plot = np.clip(im_plot, 0, 1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(im_plot, vmin=int_range[0], vmax=int_range[1])

            ax2.imshow(np.fft.fftshift(asnumpy(im_CTF_cos)), cmap="gray")

            fig.tight_layout()

        # Plot the measured/fitted shifts comparison
        if plot_BF_shifts_comparison:
            if not fit_BF_shifts:
                raise ValueError()

            measured_shifts_sx = xp.zeros(
                self._region_of_interest_shape, dtype=xp.float32
            )
            measured_shifts_sx[
                self._xy_inds[:, 0], self._xy_inds[:, 1]
            ] = self._rotated_shifts[: self._xy_inds.shape[0]]

            measured_shifts_sy = xp.zeros(
                self._region_of_interest_shape, dtype=xp.float32
            )
            measured_shifts_sy[
                self._xy_inds[:, 0], self._xy_inds[:, 1]
            ] = self._rotated_shifts[self._xy_inds.shape[0] :]

            fitted_shifts = xp.tensordot(
                gradients, xp.array(self._aberrations_coefs), axes=1
            )

            fitted_shifts_sx = xp.zeros(
                self._region_of_interest_shape, dtype=xp.float32
            )
            fitted_shifts_sx[self._xy_inds[:, 0], self._xy_inds[:, 1]] = fitted_shifts[
                : self._xy_inds.shape[0]
            ]

            fitted_shifts_sy = xp.zeros(
                self._region_of_interest_shape, dtype=xp.float32
            )
            fitted_shifts_sy[self._xy_inds[:, 0], self._xy_inds[:, 1]] = fitted_shifts[
                self._xy_inds.shape[0] :
            ]

            max_shift = xp.max(
                xp.array(
                    [
                        xp.abs(measured_shifts_sx).max(),
                        xp.abs(measured_shifts_sy).max(),
                        xp.abs(fitted_shifts_sx).max(),
                        xp.abs(fitted_shifts_sy).max(),
                    ]
                )
            )

            show(
                [
                    [asnumpy(measured_shifts_sx), asnumpy(measured_shifts_sy)],
                    [asnumpy(fitted_shifts_sx), asnumpy(fitted_shifts_sy)],
                ],
                cmap="PiYG",
                vmin=-max_shift,
                vmax=max_shift,
                intensity_range="absolute",
                axsize=(4, 4),
                ticks=False,
                title=[
                    "Measured Vertical Shifts",
                    "Measured Horizontal Shifts",
                    "Fitted Vertical Shifts",
                    "Fitted Horizontal Shifts",
                ],
            )

        self.aberration_dict = {
            tuple(self._aberrations_mn[a0]): {
                "aberration name": _aberration_names.get(
                    tuple(self._aberrations_mn[a0, :2]), "-"
                ).strip(),
                "value [Ang]": self._aberrations_coefs[a0],
            }
            for a0 in range(self._aberrations_num)
        }

        # Print results
        if self._verbose:
            if fit_CTF_FFT or fit_BF_shifts:
                print("Initial Aberration coefficients")
                print("-------------------------------")
            print(
                (
                    "Rotation of Q w.r.t. R = "
                    f"{np.rad2deg(self.rotation_Q_to_R_rads):.3f} deg"
                )
            )
            print(
                (
                    "Astigmatism (A1x,A1y)  = ("
                    f"{self.aberration_A1x:.0f},"
                    f"{self.aberration_A1y:.0f}) Ang"
                )
            )
            print(f"Aberration C1          =  {self.aberration_C1:.0f} Ang")
            print(f"Defocus dF             = {-1*self.aberration_C1:.0f} Ang")

            if fit_CTF_FFT or fit_BF_shifts:
                print()
                print("Refined Aberration coefficients")
                print("-------------------------------")
                print("aberration    radial   angular   dir.   coefs")
                print("name          order    order             Ang ")
                print("----------   -------   -------   ----   -----")

                for a0 in range(self._aberrations_mn.shape[0]):
                    m, n, a = self._aberrations_mn[a0]
                    name = _aberration_names.get((m, n), "    --    ")
                    if n == 0:
                        print(
                            name
                            + "      "
                            + str(m + 1)
                            + "        0         -      "
                            + str(np.round(self._aberrations_coefs[a0]).astype("int"))
                        )
                    elif a == 0:
                        print(
                            name
                            + "      "
                            + str(m + 1)
                            + "        "
                            + str(n)
                            + "         x      "
                            + str(np.round(self._aberrations_coefs[a0]).astype("int"))
                        )
                    else:
                        print(
                            name
                            + "      "
                            + str(m + 1)
                            + "        "
                            + str(n)
                            + "         y      "
                            + str(np.round(self._aberrations_coefs[a0]).astype("int"))
                        )

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

    def _calculate_CTF(self, alpha_shape, sampling, *coefs):
        xp = self._xp

        # FFT coordinates
        sx, sy = sampling
        qx = xp.fft.fftfreq(alpha_shape[0], sx)
        qy = xp.fft.fftfreq(alpha_shape[1], sy)
        qr2 = qx[:, None] ** 2 + qy[None, :] ** 2

        alpha = xp.sqrt(qr2) * self._wavelength
        theta = xp.arctan2(qy[None, :], qx[:, None])

        # Aberration basis
        aberrations_basis = xp.zeros((alpha.size, self._aberrations_num))
        for a0 in range(self._aberrations_num):
            m, n, a = self._aberrations_mn[a0]
            if n == 0:
                # Radially symmetric basis
                aberrations_basis[:, a0] = (alpha ** (m + 1) / (m + 1)).ravel()

            elif a == 0:
                # cos coef
                aberrations_basis[:, a0] = (
                    alpha ** (m + 1) * xp.cos(n * theta) / (m + 1)
                ).ravel()
            else:
                # sin coef
                aberrations_basis[:, a0] = (
                    alpha ** (m + 1) * xp.sin(n * theta) / (m + 1)
                ).ravel()

        # global scaling
        aberrations_basis *= 2 * np.pi / self._wavelength

        chi = xp.zeros_like(aberrations_basis[:, 0])

        for a0 in range(len(coefs)):
            chi += coefs[a0] * aberrations_basis[:, a0]

        return xp.reshape(chi, alpha_shape)

    def aberration_correct(
        self,
        use_CTF_fit=None,
        plot_corrected_phase: bool = True,
        k_info_limit: float = None,
        k_info_power: float = 1.0,
        Wiener_filter=False,
        Wiener_signal_noise_ratio: float = 1.0,
        Wiener_filter_low_only: bool = False,
        upsampled: bool = True,
        **kwargs,
    ):
        """
        CTF correction of the phase image using the measured defocus aberration.

        Parameters
        ----------
        use_FFT_fit: bool
            Use the CTF fitted to the zero crossings of the FFT.
            Default is True
        plot_corrected_phase: bool, optional
            If True, the CTF-corrected phase is plotted
        k_info_limit: float, optional
            maximum allowed frequency in butterworth filter
        k_info_power: float, optional
            power of butterworth filter
        Wiener_filter: bool, optional
            Use Wiener filtering instead of CTF sign correction.
        Wiener_signal_noise_ratio: float, optional
            Signal to noise radio at k = 0 for Wiener filter
        Wiener_filter_low_only: bool, optional
            Apply Wiener filtering only to the CTF portions before the 1st CTF maxima.
        """

        xp = self._xp
        asnumpy = self._asnumpy

        if not hasattr(self, "aberration_C1"):
            raise ValueError(
                (
                    "CTF correction is meant to be ran after alignment and aberration fitting. "
                    "Please run the `reconstruct()` and `aberration_fit()` functions first."
                )
            )

        if upsampled and hasattr(self, "_kde_upsample_factor"):
            im = self._recon_BF_subpixel_aligned
            sx = self._scan_sampling[0] / self._kde_upsample_factor
            sy = self._scan_sampling[1] / self._kde_upsample_factor
        else:
            upsampled = False
            im = self._recon_BF
            sx = self._scan_sampling[0]
            sy = self._scan_sampling[1]

        # Fourier coordinates
        kx = xp.fft.fftfreq(im.shape[0], sx)
        ky = xp.fft.fftfreq(im.shape[1], sy)
        kra2 = (kx[:, None]) ** 2 + (ky[None, :]) ** 2

        if use_CTF_fit is None:
            if hasattr(self, "_aberrations_surface_shape"):
                use_CTF_fit = True

        if use_CTF_fit:
            sin_chi = np.sin(
                self._calculate_CTF(im.shape, (sx, sy), *self._aberrations_coefs)
            )

            CTF_corr = xp.sign(sin_chi)
            CTF_corr[0, 0] = 0

            # apply correction to mean reconstructed BF image
            im_fft_corr = xp.fft.fft2(im) * CTF_corr

            # if needed, add low pass filter output image
            if k_info_limit is not None:
                im_fft_corr /= 1 + (kra2**k_info_power) / (
                    (k_info_limit) ** (2 * k_info_power)
                )
        else:
            # CTF
            sin_chi = xp.sin((xp.pi * self._wavelength * self.aberration_C1) * kra2)

            if Wiener_filter:
                SNR_inv = (
                    xp.sqrt(
                        1
                        + (kra2**k_info_power)
                        / ((k_info_limit) ** (2 * k_info_power))
                    )
                    / Wiener_signal_noise_ratio
                )
                CTF_corr = xp.sign(sin_chi) / (sin_chi**2 + SNR_inv)
                if Wiener_filter_low_only:
                    # limit Wiener filter to only the part of the CTF before 1st maxima
                    k_thresh = 1 / xp.sqrt(
                        2.0 * self._wavelength * xp.abs(self.aberration_C1)
                    )
                    k_mask = kra2 >= k_thresh**2
                    CTF_corr[k_mask] = xp.sign(sin_chi[k_mask])

                # apply correction to mean reconstructed BF image
                im_fft_corr = xp.fft.fft2(im) * CTF_corr

            else:
                # CTF without tilt correction (beyond the parallax operator)
                CTF_corr = xp.sign(sin_chi)
                CTF_corr[0, 0] = 0

                # apply correction to mean reconstructed BF image
                im_fft_corr = xp.fft.fft2(im) * CTF_corr

                # if needed, add low pass filter output image
                if k_info_limit is not None:
                    im_fft_corr /= 1 + (kra2**k_info_power) / (
                        (k_info_limit) ** (2 * k_info_power)
                    )

        # Output phase image
        self._recon_phase_corrected = xp.real(xp.fft.ifft2(im_fft_corr))
        self.recon_phase_corrected = asnumpy(self._recon_phase_corrected)

        if self._device == "gpu":
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        # plotting
        if plot_corrected_phase:
            figsize = kwargs.pop("figsize", (6, 6))
            cmap = kwargs.pop("cmap", "magma")

            fig, ax = plt.subplots(figsize=figsize)

            cropped_object = self._crop_padded_object(
                self._recon_phase_corrected, upsampled=upsampled
            )

            extent = [
                0,
                sy * cropped_object.shape[1],
                sx * cropped_object.shape[0],
                0,
            ]

            ax.imshow(
                cropped_object,
                extent=extent,
                cmap=cmap,
                **kwargs,
            )

            ax.set_ylabel("x [A]")
            ax.set_xlabel("y [A]")
            ax.set_title("Parallax-Corrected Phase Image")

    def depth_section(
        self,
        depth_angstroms=np.arange(-250, 260, 100),
        plot_depth_sections=True,
        k_info_limit: float = None,
        k_info_power: float = 1.0,
        progress_bar=True,
        **kwargs,
    ):
        """
        CTF correction of the BF image using the measured defocus aberration.

        Parameters
        ----------
        depth_angstroms: np.array
            Specify the depths
        k_info_limit: float, optional
            maximum allowed frequency in butterworth filter
        k_info_power: float, optional
            power of butterworth filter


        Returns
        -------
        stack_depth: np.array
            stack of phase images at different depths with shape [depth Nx Ny]

        """

        xp = self._xp
        asnumpy = self._asnumpy
        depth_angstroms = xp.atleast_1d(depth_angstroms)

        if not hasattr(self, "aberration_C1"):
            raise ValueError(
                (
                    "Depth sectioning is meant to be ran after alignment and aberration fitting. "
                    "Please run the `reconstruct()` and `aberration_fit()` functions first."
                )
            )

        # Fourier coordinates
        kx = xp.fft.fftfreq(self._recon_BF.shape[0], self._scan_sampling[0])
        ky = xp.fft.fftfreq(self._recon_BF.shape[1], self._scan_sampling[1])
        kra2 = (kx[:, None]) ** 2 + (ky[None, :]) ** 2

        # information limit
        if k_info_limit is not None:
            k_filt = 1 / (
                1 + (kra2**k_info_power) / ((k_info_limit) ** (2 * k_info_power))
            )

        # init
        stack_depth = xp.zeros(
            (depth_angstroms.shape[0], self._recon_BF.shape[0], self._recon_BF.shape[1])
        )

        # plotting
        if plot_depth_sections:
            num_plots = depth_angstroms.shape[0]
            nrows = int(np.sqrt(num_plots))
            ncols = int(np.ceil(num_plots / nrows))

            spec = GridSpec(
                ncols=ncols,
                nrows=nrows,
                hspace=0.15,
                wspace=0.15,
            )

            figsize = kwargs.pop("figsize", (4 * ncols, 4 * nrows))
            cmap = kwargs.pop("cmap", "magma")

            fig = plt.figure(figsize=figsize)

        # main loop
        for a0 in tqdmnd(
            depth_angstroms.shape[0],
            desc="Depth sectioning ",
            unit="plane",
            disable=not progress_bar,
        ):
            dz = depth_angstroms[a0]

            # Parallax
            im_depth = xp.zeros_like(self._recon_BF, dtype="complex")
            for a1 in range(self._stack_BF.shape[0]):
                dx = self._probe_angles[a1, 0] * dz
                dy = self._probe_angles[a1, 1] * dz
                im_depth += xp.fft.fft2(self._stack_BF[a1]) * xp.exp(
                    self._qx_shift * dx + self._qy_shift * dy
                )

            # CTF correction
            sin_chi = xp.sin(
                (xp.pi * self._wavelength * (self.aberration_C1 + dz)) * kra2
            )
            CTF_corr = xp.sign(sin_chi)
            CTF_corr[0, 0] = 0
            if k_info_limit is not None:
                CTF_corr *= k_filt

            # apply correction to mean reconstructed BF image
            stack_depth[a0] = (
                xp.real(xp.fft.ifft2(im_depth * CTF_corr)) / self._stack_BF.shape[0]
            )

            if plot_depth_sections:
                row_index, col_index = np.unravel_index(a0, (nrows, ncols))
                ax = fig.add_subplot(spec[row_index, col_index])

                cropped_object = self._crop_padded_object(asnumpy(stack_depth[a0]))

                extent = [
                    0,
                    self._scan_sampling[1] * cropped_object.shape[1],
                    self._scan_sampling[0] * cropped_object.shape[0],
                    0,
                ]

                ax.imshow(
                    cropped_object,
                    extent=extent,
                    cmap=cmap,
                    **kwargs,
                )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Depth section: {dz}A")

        if self._device == "gpu":
            xp = self._xp
            xp._default_memory_pool.free_all_blocks()
            xp.clear_memo()

        return stack_depth

    def _crop_padded_object(
        self,
        padded_object: np.ndarray,
        remaining_padding: int = 0,
        upsampled: bool = False,
    ):
        """
        Utility function to crop padded object

        Parameters
        ----------
        padded_object: np.ndarray
            Padded object to be cropped
        remaining_padding: int, optional
            Padding to leave uncropped

        Returns
        -------
        cropped_object: np.ndarray
            Cropped object

        """

        asnumpy = self._asnumpy

        if upsampled:
            pad_x = np.round(
                self._object_padding_px[0] / 2 * self._kde_upsample_factor
            ).astype("int")
            pad_y = np.round(
                self._object_padding_px[1] / 2 * self._kde_upsample_factor
            ).astype("int")
        else:
            pad_x = self._object_padding_px[0] // 2
            pad_y = self._object_padding_px[1] // 2

        pad_x -= remaining_padding
        pad_y -= remaining_padding

        return asnumpy(padded_object[pad_x:-pad_x, pad_y:-pad_y])

    def _visualize_figax(
        self,
        fig,
        ax,
        remaining_padding: int = 0,
        upsampled: bool = False,
        **kwargs,
    ):
        """
        Utility function to visualize bright field average on given fig/ax

        Parameters
        ----------
        fig: Figure
            Matplotlib figure ax lives in
        ax: Axes
            Matplotlib axes to plot bright field average in
        remaining_padding: int, optional
            Padding to leave uncropped

        """

        cmap = kwargs.pop("cmap", "magma")

        if upsampled:
            cropped_object = self._crop_padded_object(
                self._recon_BF_subpixel_aligned, remaining_padding, upsampled
            )

            extent = [
                0,
                self._scan_sampling[1]
                * cropped_object.shape[1]
                / self._kde_upsample_factor,
                self._scan_sampling[0]
                * cropped_object.shape[0]
                / self._kde_upsample_factor,
                0,
            ]

        else:
            cropped_object = self._crop_padded_object(self._recon_BF, remaining_padding)

            extent = [
                0,
                self._scan_sampling[1] * cropped_object.shape[1],
                self._scan_sampling[0] * cropped_object.shape[0],
                0,
            ]

        ax.imshow(
            cropped_object,
            extent=extent,
            cmap=cmap,
            **kwargs,
        )

    def show_shifts(
        self,
        scale_arrows=1,
        plot_arrow_freq=1,
        **kwargs,
    ):
        """
        Utility function to visualize bright field disk pixel shifts

        Parameters
        ----------
        scale_arrows: float, optional
            Scale to multiply shifts by
        plot_arrow_freq: int, optional
            Frequency of shifts to plot in quiver plot
        """

        xp = self._xp
        asnumpy = self._asnumpy

        figsize = kwargs.pop("figsize", (6, 6))
        color = kwargs.pop("color", (1, 0, 0, 1))

        fig, ax = plt.subplots(figsize=figsize)

        dp_mask_ind = xp.nonzero(self._dp_mask)
        yy, xx = xp.meshgrid(
            xp.arange(self._dp_mean.shape[1]), xp.arange(self._dp_mean.shape[0])
        )
        freq_mask = xp.logical_and(xx % plot_arrow_freq == 0, yy % plot_arrow_freq == 0)
        masked_ind = xp.logical_and(freq_mask, self._dp_mask)
        plot_ind = masked_ind[dp_mask_ind]

        ax.quiver(
            asnumpy(self._kxy[plot_ind, 1]),
            asnumpy(self._kxy[plot_ind, 0]),
            asnumpy(
                self._xy_shifts[plot_ind, 1]
                * scale_arrows
                * self._reciprocal_sampling[0]
            ),
            asnumpy(
                self._xy_shifts[plot_ind, 0]
                * scale_arrows
                * self._reciprocal_sampling[1]
            ),
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            **kwargs,
        )

        kr_max = xp.max(self._kr)
        ax.set_xlim([-1.2 * kr_max, 1.2 * kr_max])
        ax.set_ylim([-1.2 * kr_max, 1.2 * kr_max])

    def visualize(
        self,
        **kwargs,
    ):
        """
        Visualization function for bright field average

        Returns
        --------
        self: BFReconstruction
            Self to accommodate chaining
        """

        figsize = kwargs.pop("figsize", (6, 6))

        fig, ax = plt.subplots(figsize=figsize)

        self._visualize_figax(fig, ax, **kwargs)

        ax.set_ylabel("x [A]")
        ax.set_xlabel("y [A]")
        ax.set_title("Reconstructed Bright Field Image")

        return self
