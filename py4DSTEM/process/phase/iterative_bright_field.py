"""
Module for reconstructing virtual bright field images by aligning each virtual BF image.
"""

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from py4DSTEM.io import DataCube
from py4DSTEM.process.phase.iterative_base_class import PhaseReconstruction
from py4DSTEM.process.utils.cross_correlate import align_images_fourier
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from py4DSTEM.utils.tqdmnd import tqdmnd
from scipy.linalg import polar
from scipy.optimize import curve_fit
from scipy.special import comb

try:
    import cupy as cp
except ImportError:
    cp = None

warnings.simplefilter(action="always", category=UserWarning)


class BFReconstruction(PhaseReconstruction):
    """
    Iterative BrightField Reconstruction Class.

    Parameters
    ----------
    datacube: DataCube
        Input 4D diffraction pattern intensities
    energy: float
        The electron energy of the wave functions in eV
    dp_mean: ndarray, optional
        Mean diffraction pattern
        If None, get_dp_mean() is used
    verbose: bool, optional
        If True, class methods will inherit this and print additional information
    device: str, optional
        Calculation device will be perfomed on. Must be 'cpu' or 'gpu'
    """

    def __init__(
        self,
        datacube: DataCube,
        energy: float,
        dp_mean: np.ndarray = None,
        verbose: bool = False,
        device: str = "cpu",
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

        self._energy = energy
        self._datacube = datacube
        self._dp_mean = dp_mean
        self._verbose = verbose
        self._preprocessed = False

    def preprocess(
        self,
        object_padding_px: Tuple[int, int] = (32, 32),
        edge_blend: int = 16,
        threshold_intensity: float = 0.8,
        normalize_images: bool = True,
        defocus_guess: float = None,
        plot_average_bf: bool = True,
        **kwargs,
    ):
        """
        Iterative BrightField Reconstruction preprocessing method.

        Parameters
        ----------
        object_padding_px: Tuple[int,int], optional
            Pixel dimensions to pad object with
            If None, the padding is set to half the probe ROI dimensions
        edge_blend: int, optional
            Pixels to blend image at the border
        threshold: float, optional
            Fraction of max of dp_mean for bright-field pixels
        normalize_images: bool, optional
            If True, bright images normalized to have a mean of 1
        defocus_guess: float, optional
            Initial guess of defocus value
            If None, first iteration is assumed to be in-focus
        plot_average_bf: bool, optional
            If True, plots the average bright field image, using defocus_guess

        Returns
        --------
        self: BFReconstruction
            Self to accommodate chaining
        """

        xp = self._xp
        asnumpy = self._asnumpy
        self._object_padding_px = object_padding_px

        # get mean diffraction pattern
        if self._dp_mean is not None:
            self._dp_mean = xp.asarray(self._dp_mean, dtype=xp.float32)
        if "dp_mean" in self._datacube.tree.keys():
            self._dp_mean = xp.asarray(
                self._datacube.tree["dp_mean"].data, dtype=xp.float32
            )
        else:
            self._dp_mean = xp.asarray(
                self._datacube.get_dp_mean().data, dtype=xp.float32
            )

        # extract calibrations
        self._extract_intensities_and_calibrations_from_datacube(
            self._datacube,
            require_calibrations=True,
        )

        # make sure mean diffraction pattern is shaped correctly
        if (self._dp_mean.shape[0] != self._intensities_shape[2]) or (
            self._dp_mean.shape[1] != self._intensities_shape[3]
        ):
            raise ValueError(
                "dp_mean must match the datacube shape. Try setting dp_mean = None."
            )

        # select virtual detector pixels
        self._dp_mask = self._dp_mean >= (xp.max(self._dp_mean) * threshold_intensity)
        self._num_bf_images = int(xp.count_nonzero(self._dp_mask))
        self._wavelength = electron_wavelength_angstrom(self._energy)

        # diffraction space coordinates
        self._xy_inds = np.argwhere(self._dp_mask)
        self._kxy = (self._xy_inds - xp.mean(self._xy_inds, axis=0)[None]) * xp.array(
            self._reciprocal_sampling
        )[None]
        self._probe_angles = self._kxy * self._wavelength
        self._kr = xp.sqrt(xp.sum(self._kxy**2, axis=1))

        # Window function
        x = xp.linspace(-1, 1, self._intensities_shape[0] + 1)[1:]
        x -= (x[1] - x[0]) / 2
        wx = (
            xp.sin(
                xp.clip(
                    (1 - xp.abs(x)) * self._intensities_shape[0] / edge_blend / 2, 0, 1
                )
                * (xp.pi / 2)
            )
            ** 2
        )

        y = xp.linspace(-1, 1, self._intensities_shape[1] + 1)[1:]
        y -= (y[1] - y[0]) / 2
        wy = (
            xp.sin(
                xp.clip(
                    (1 - xp.abs(y)) * self._intensities_shape[1] / edge_blend / 2, 0, 1
                )
                * (xp.pi / 2)
            )
            ** 2
        )

        self._window_edge = wx[:, None] * wy[None, :]
        self._window_inv = 1 - self._window_edge
        self._window_pad = xp.zeros(
            (
                self._intensities_shape[0] + object_padding_px[0],
                self._intensities_shape[1] + object_padding_px[1],
            )
        )
        self._window_pad[
            object_padding_px[0] // 2 : self._intensities_shape[0]
            + object_padding_px[0] // 2,
            object_padding_px[1] // 2 : self._intensities_shape[1]
            + object_padding_px[1] // 2,
        ] = self._window_edge

        # Collect BF images
        all_bfs = self._intensities[:, :, self._xy_inds[:, 0], self._xy_inds[:, 1]]
        if normalize_images:
            all_bfs /= xp.mean(all_bfs, axis=(0, 1))
            all_means = xp.ones(self._num_bf_images)
        else:
            all_means = xp.mean(all_bfs, axis=(0, 1))

        stack_shape = (
            self._intensities_shape[0] + object_padding_px[0],
            self._intensities_shape[1] + object_padding_px[1],
            self._num_bf_images,
        )

        self._stack_BF = xp.full(stack_shape, all_means[None, None])

        self._stack_BF[
            object_padding_px[0] // 2 : self._intensities_shape[0]
            + object_padding_px[0] // 2,
            object_padding_px[1] // 2 : self._intensities_shape[1]
            + object_padding_px[1] // 2,
        ] = (
            self._window_inv[:, :, None] * all_means[None, None]
            + self._window_edge[:, :, None] * all_bfs
        )
        self._stack_BF = xp.moveaxis(self._stack_BF, [0, 1, 2], [1, 2, 0])

        # Fourier space operators for image shifts
        qx = xp.fft.fftfreq(self._stack_BF.shape[1], d=1)
        qy = xp.fft.fftfreq(self._stack_BF.shape[2], d=1)
        qxa, qya = xp.meshgrid(qx, qy, indexing="ij")
        self._qx_shift = -2j * xp.pi * qxa
        self._qy_shift = -2j * xp.pi * qya

        # Initialization utilities
        self._stack_mask = xp.tile(self._window_pad[None], (self._num_bf_images, 1, 1))
        if defocus_guess is not None:
            Gs = xp.fft.fft2(self._stack_BF)

            self._xy_shifts = (
                self._probe_angles * defocus_guess / xp.array(self._scan_sampling)
            )
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
            self._xy_shifts = xp.zeros((self._num_bf_images, 2))

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
            figsize = kwargs.get("figsize", (6, 6))
            kwargs.pop("figsize", None)

            fig, ax = plt.subplots(figsize=figsize)

            self._visualize_figax(fig, ax, **kwargs)

            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title("Average Bright Field Image")

        self._preprocessed = True
        return self

    def reconstruct(
        self,
        max_alignment_bin: int = None,
        min_alignment_bin: int = 1,
        max_iter_at_min_bin: int = 2,
        upsample_factor: int = 8,
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
        Iterative BrightField Reconstruction main reconstruction method.

        Parameters
        ----------
        max_alignment_bin: int, optional
            Maximum bin size for bright field alignment
            If None, the bright field disk radius is used
        min_alignment_bin: int, optional
            Minimum bin size for bright field alignment
        max_iter_at_min_bin: int, optional
            Number of iterations to run at the smallest bin size
        upsample_factor: int, optional
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
                )
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

            shifts_update = xp.zeros((self._num_bf_images, 2))

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
                    upsample_factor=upsample_factor,
                    device="cpu" if xp is np else "gpu",
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

        return self

    def aberration_fit(
        self,
        plot_CTF_compare: bool = False,
        plot_dk: float = 0.005,
        plot_k_sigma: float = 0.02,
    ):
        """
        Fit aberrations to the measured image shifts.

        Parameters
        ----------
        plot_CTF_compare: bool, optional
            If True, the fitted CTF is plotted against the reconstructed frequencies
        plot_dk: float, optional
            Reciprocal bin-size for polar-averaged FFT
        plot_k_sigma: float, optional
            sigma to gaussian blur polar-averaged FFT by

        """
        xp = self._xp
        asnumpy = self._asnumpy
        gaussian_filter = self._gaussian_filter

        # Convert real space shifts to Angstroms
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
        self.aberration_A1x = (
            m_aberration[0, 0] - m_aberration[1, 1]
        ) / 2.0  # factor /2 for A1 astigmatism? /4?
        self.aberration_A1y = (m_aberration[1, 0] + m_aberration[0, 1]) / 2.0

        # Print results
        if self._verbose:
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
            print(f"Defocus C1             = {self.aberration_C1:.0f} Ang")

        # Plot the CTF comparison between experiment and fit
        if plot_CTF_compare:

            # Get polar mean from FFT of BF reconstruction
            im_fft = xp.abs(xp.fft.fft2(self._recon_BF))

            # coordinates
            kx = xp.fft.fftfreq(self._recon_BF.shape[0], self._scan_sampling[0])
            ky = xp.fft.fftfreq(self._recon_BF.shape[1], self._scan_sampling[1])
            kra = xp.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            k_max = xp.max(kra) / np.sqrt(2.0)
            k_num_bins = int(xp.ceil(k_max / plot_dk))
            k_bins = xp.arange(k_num_bins + 1) * plot_dk

            # histogram
            k_ind = kra / plot_dk
            kf = np.floor(k_ind).astype("int")
            dk = k_ind - kf
            sub = kf <= k_num_bins
            hist_exp = xp.bincount(
                kf[sub], weights=im_fft[sub] * (1 - dk[sub]), minlength=k_num_bins
            )
            hist_norm = xp.bincount(
                kf[sub], weights=(1 - dk[sub]), minlength=k_num_bins
            )
            sub = kf <= k_num_bins - 1

            hist_exp += xp.bincount(
                kf[sub] + 1, weights=im_fft[sub] * (dk[sub]), minlength=k_num_bins
            )
            hist_norm += xp.bincount(
                kf[sub] + 1, weights=(dk[sub]), minlength=k_num_bins
            )

            # KDE and normalizing
            k_sigma = plot_dk / plot_k_sigma
            hist_exp[0] = 0.0
            hist_exp = gaussian_filter(hist_exp, sigma=k_sigma, mode="nearest")
            hist_norm = gaussian_filter(hist_norm, sigma=k_sigma, mode="nearest")
            hist_exp /= hist_norm

            # CTF comparison
            CTF_fit = xp.sin(
                (-np.pi * self._wavelength * self.aberration_C1) * k_bins**2
            )

            # plotting input - log scale
            hist_plot = xp.log(hist_exp)
            hist_plot -= xp.min(hist_plot)
            hist_plot /= xp.max(hist_plot)

            hist_plot = asnumpy(hist_plot)
            k_bins = asnumpy(k_bins)
            CTF_fit = asnumpy(CTF_fit)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.fill_between(
                k_bins,
                hist_plot,
                color=(0.7, 0.7, 0.7, 1),
            )

            ax.plot(
                k_bins,
                np.clip(CTF_fit, 0.0, np.inf),
                color=(1, 0, 0, 1),
                linewidth=2,
            )
            ax.plot(
                k_bins,
                np.clip(-CTF_fit, 0.0, np.inf),
                color=(0, 0.5, 1, 1),
                linewidth=2,
            )
            ax.set_xlim([0, k_bins[-1]])
            ax.set_ylim([0, 1.05])

    def aberration_correct(
        self,
        plot_corrected_bf: bool = True,
        k_info_limit: float = None,
        k_info_power: float = 2.0,
        LASSO_filter: bool = False,
        LASSO_scale: float = 1.0,
        **kwargs,
    ):
        """
        CTF correction of the BF image using the measured defocus aberration.

        Parameters
        ----------
        plot_corrected_bf: bool, optional
            If True, the CTF-corrected bright field average image is plotted
        k_info_limit: float, optional
            maximum allowed frequency in butterworth filter
        k_info_power: float, optional
            power of butterworth filter
        LASSO_filter: bool, optional
            If True, the measured CTF is fitted to a LASSO-type curve_fit
        LASSO_scale: float, optional
            scale of LASSO filter
        """

        xp = self._xp
        asnumpy = self._asnumpy

        # Fourier coordinates
        kx = xp.fft.fftfreq(self._recon_BF.shape[0], self._scan_sampling[0])
        ky = xp.fft.fftfreq(self._recon_BF.shape[1], self._scan_sampling[1])
        kra2 = (kx[:, None]) ** 2 + (ky[None, :]) ** 2
        sin_chi = xp.sin((np.pi * self._wavelength * self.aberration_C1) * kra2)

        # CTF without tilt correction (beyond the parallax operator)
        CTF_corr = xp.sign(sin_chi)
        CTF_corr[0, 0] = 0

        # apply correction to mean reconstructed BF image
        im_fft_corr = xp.fft.fft2(self._recon_BF) * CTF_corr

        # if needed, Fourier filter output image
        if LASSO_filter:

            def CTF_fit(kra2_CTFmag, I0, I1, I2, I3, sigma1, sigma2, sigma3):
                kra2, CTF_mag = kra2_CTFmag
                int_fit = (
                    I0
                    + I1 * np.exp(kra2 / (-2 * sigma1**2))
                    + I2 * np.exp(kra2 / (-2 * sigma2**2))
                    + I3 * np.exp(kra2 / (-2 * sigma3**2)) * CTF_mag
                )
                return int_fit.ravel()

            sin_chi = asnumpy(sin_chi)
            im_fft_corr = asnumpy(im_fft_corr)

            CTF_mag = np.abs(sin_chi)
            sig = np.abs(im_fft_corr)
            sig_mean = np.mean(sig)
            sig_min = np.min(sig)
            sig_max = np.max(sig)
            k_max = np.max(asnumpy(kx))
            coefs = (
                sig_min,
                sig_max,
                sig_mean,
                sig_mean,
                k_max / 16.0,
                k_max / 4.0,
                k_max / 1.0,
            )
            lb = (
                0.0,
                0.0,
                0.0,
                0.0,
                k_max / 100.0,
                k_max / 10.0,
                k_max / 10.0,
            )
            ub = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

            # curve_fit the background image
            coefs, pcov = curve_fit(
                CTF_fit,
                (kra2, CTF_mag),
                sig.ravel(),
                p0=coefs,
                bounds=(lb, ub),
                maxfev=1000,
            )
            coefs_bg = coefs.copy()
            coefs_bg[3] = 0
            sig_bg = np.reshape(CTF_fit((kra2, CTF_mag), *coefs_bg), sig.shape)

            # apply LASSO filter
            im_fft_corr = np.clip(
                np.abs(im_fft_corr) - sig_bg * LASSO_scale, 0, np.inf
            ) * np.exp(1j * np.angle(im_fft_corr))

            im_fft_corr = xp.asarray(im_fft_corr)

        # if needed, add low pass filter output image
        if k_info_limit is not None:
            im_fft_corr /= 1 + (kra2**k_info_power) / (
                (k_info_limit) ** (2 * k_info_power)
            )

        # Output image
        self._recon_BF_corrected = xp.real(xp.fft.ifft2(im_fft_corr))
        self.recon_BF_corrected = asnumpy(self._recon_BF_corrected)

        # plotting
        if plot_corrected_bf:

            figsize = kwargs.get("figsize", (6, 6))
            cmap = kwargs.get("cmap", "magma")
            kwargs.pop("figsize", None)
            kwargs.pop("cmap", None)

            fig, ax = plt.subplots(figsize=figsize)

            cropped_object = self._crop_padded_object(self._recon_BF_corrected)

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

            ax.set_xlabel("x [A]")
            ax.set_ylabel("y [A]")
            ax.set_title("Corrected Bright Field Image")

    def _crop_padded_object(
        self,
        padded_object: np.ndarray,
        remaining_padding: int = 0,
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

        pad_x = self._object_padding_px[0] // 2 - remaining_padding
        pad_y = self._object_padding_px[1] // 2 - remaining_padding

        return asnumpy(padded_object[pad_x:-pad_x, pad_y:-pad_y])

    def _visualize_figax(
        self,
        fig,
        ax,
        remaining_padding: int = 0,
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

        cmap = kwargs.get("cmap", "magma")
        kwargs.pop("cmap", None)

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

    def _visualize_shifts(
        self,
        scale_arrows=0.002,
        **kwargs,
    ):
        """
        Utility function to visualize bright field disk pixel shifts

        Parameters
        ----------
        scale_arrows: float, optional
            Scale to multiply shifts by

        """

        xp = self._xp
        asnumpy = self._asnumpy

        figsize = kwargs.get("figsize", (6, 6))
        color = kwargs.get("color", (1, 0, 0, 1))
        kwargs.pop("figsize", None)
        kwargs.pop("color", None)

        fig, ax = plt.subplots(figsize=figsize)

        ax.quiver(
            asnumpy(self._kxy[:, 1]),
            asnumpy(self._kxy[:, 0]),
            asnumpy(self._xy_shifts[:, 1] * scale_arrows),
            asnumpy(self._xy_shifts[:, 0] * scale_arrows),
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
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

        figsize = kwargs.get("figsize", (6, 6))
        kwargs.pop("figsize", None)

        fig, ax = plt.subplots(figsize=figsize)

        self._visualize_figax(fig, ax, **kwargs)

        ax.set_xlabel("x [A]")
        ax.set_ylabel("y [A]")
        ax.set_title("Reconstructed Bright Field Image")

        return self
