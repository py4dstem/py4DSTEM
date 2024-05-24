import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import peak_prominences
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit, leastsq
import warnings

# from emdfile import tqdmnd, PointList, PointListArray
from py4DSTEM import tqdmnd, PointList, PointListArray
from py4DSTEM.process.fit import (
    polar_twofold_gaussian_2D,
    polar_twofold_gaussian_2D_background,
)
from py4DSTEM.visualize import show


def find_peaks_single_pattern(
    self,
    x,
    y,
    mask=None,
    bragg_peaks=None,
    bragg_mask_radius=None,
    sigma_annular_deg=10.0,
    sigma_radial_px=3.0,
    sigma_annular_deg_max=None,
    radial_background_subtract=True,
    radial_background_thresh=0.25,
    num_peaks_max=100,
    threshold_abs=1.0,
    threshold_prom_annular=None,
    threshold_prom_radial=None,
    remove_masked_peaks=False,
    scale_sigma_annular=0.5,
    scale_sigma_radial=0.25,
    return_background=False,
    plot_result=True,
    plot_power_scale=1.0,
    plot_scale_size=10.0,
    figsize=(12, 6),
    returnfig=False,
    **kwargs
):
    """
    Peak detection function for polar transformations.

    Parameters
    --------
    x: int
        x index of diffraction pattern
    y: int
        y index of diffraction pattern
    mask: np.array
        Boolean mask in Cartesian space, to filter detected peaks.
    bragg_peaks: py4DSTEM.BraggVectors
        Set of Bragg peaks used to generated a mask in Cartesian space, to filter detected peaks
    sigma_annular_deg: float
        smoothing along the annular direction in degrees, periodic
    sigma_radial_px: float
        smoothing along the radial direction in pixels, not periodic
    sigma_annular_deg_max: float
        Specify this value for the max annular sigma.  Peaks larger than this will be split
        into multiple peaks, depending on the ratio.
    radial_background_subtract: bool
        If true, subtract radial background estimate
    radial_background_thresh: float
        Relative order of sorted values to use as background estimate.
        Setting to 0.5 is equivalent to median, 0.0 is min value.
    num_peaks_max = 100
        Max number of peaks to return.
    threshold_abs: float
        Absolute image intensity threshold for peaks.
    threshold_prom_annular: float
        Threshold for prominance, along annular direction.
    threshold_prom_radial: float
        Threshold for prominance, along radial direction.
    remove_masked_peaks: bool
        Delete peaks that are in the region masked by "mask"
    scale_sigma_annular: float
        Scaling of the estimated annular standard deviation.
    scale_sigma_radial: float
        Scaling of the estimated radial standard deviation.
    return_background: bool
        Return the background signal.
    plot_result:
        Plot the detector peaks
    plot_power_scale: float
        Image intensity power law scaling.
    plot_scale_size: float
        Marker scaling in the plot.
    figsize: 2-tuple
        Size of the result plotting figure.
    returnfig: bool
        Return the figure and axes handles.

    Returns
    --------

    peaks_polar : pointlist
        The detected peaks
    fig, ax : (optional)
        Figure and axes handles

    """

    # if needed, generate mask from Bragg peaks
    if bragg_peaks is not None:
        mask_bragg = self._datacube.get_braggmask(
            bragg_peaks,
            x,
            y,
            radius=bragg_mask_radius,
        )
        if mask is None:
            mask = mask_bragg
        else:
            mask = np.logical_or(mask, mask_bragg)

    # Convert sigma values into units of bins
    sigma_annular = np.deg2rad(sigma_annular_deg) / self.annular_step
    sigma_radial = sigma_radial_px / self.qstep

    # Get transformed image and normalization array
    im_polar, im_polar_norm, norm_array, mask_bool = self.transform(
        self._datacube.data[x, y],
        mask=mask,
        returnval="all_zeros",
        origin=(self.calibration.qx0[x, y], self.calibration.qy0[x, y]),
    )
    # Change sign convention of mask
    mask_bool = np.logical_not(mask_bool)

    # Background subtraction
    if radial_background_subtract:
        sig_bg = np.zeros(im_polar.shape[1])
        for a0 in range(im_polar.shape[1]):
            if np.any(mask_bool[:, a0]):
                vals = np.sort(im_polar[mask_bool[:, a0], a0])
                ind = np.round(radial_background_thresh * (vals.shape[0] - 1)).astype(
                    "int"
                )
                sig_bg[a0] = vals[ind]
        sig_bg_mask = np.sum(mask_bool, axis=0) >= (im_polar.shape[0] // 2)
        im_polar = np.maximum(im_polar - sig_bg[None, :], 0)

    # apply smoothing and normalization
    im_polar_sm = gaussian_filter(
        im_polar * norm_array,
        sigma=(sigma_annular, sigma_radial),
        mode=("wrap", "nearest"),
    )
    im_mask = gaussian_filter(
        norm_array,
        sigma=(sigma_annular, sigma_radial),
        mode=("wrap", "nearest"),
    )
    sub = im_mask > 0.001 * np.max(im_mask)
    im_polar_sm[sub] /= im_mask[sub]

    # Find local maxima
    peaks = peak_local_max(
        im_polar_sm,
        num_peaks=num_peaks_max,
        threshold_abs=threshold_abs,
    )

    # check if peaks should be removed from the polar transformation mask
    if remove_masked_peaks:
        peaks = np.delete(
            peaks,
            mask_bool[peaks[:, 0], peaks[:, 1]] == False,  # noqa: E712
            axis=0,
        )

    # peak intensity
    peaks_int = im_polar_sm[peaks[:, 0], peaks[:, 1]]

    # Estimate prominance of peaks, and their size in units of pixels
    peaks_prom = np.zeros((peaks.shape[0], 4))
    annular_ind_center = np.atleast_1d(
        np.array(im_polar_sm.shape[0] // 2).astype("int")
    )
    for a0 in range(peaks.shape[0]):
        # annular
        trace_annular = np.roll(
            np.squeeze(im_polar_sm[:, peaks[a0, 1]]), annular_ind_center - peaks[a0, 0]
        )
        p_annular = peak_prominences(
            trace_annular,
            annular_ind_center,
        )
        sigma_annular = scale_sigma_annular * np.minimum(
            annular_ind_center - p_annular[1], p_annular[2] - annular_ind_center
        )

        # radial
        trace_radial = im_polar_sm[peaks[a0, 0], :]
        p_radial = peak_prominences(
            trace_radial,
            np.atleast_1d(peaks[a0, 1]),
        )
        sigma_radial = scale_sigma_radial * np.minimum(
            peaks[a0, 1] - p_radial[1], p_radial[2] - peaks[a0, 1]
        )

        # output
        peaks_prom[a0, 0] = p_annular[0]
        peaks_prom[a0, 1] = sigma_annular[0]
        peaks_prom[a0, 2] = p_radial[0]
        peaks_prom[a0, 3] = sigma_radial[0]

    # if needed, remove peaks using prominance criteria
    if threshold_prom_annular is not None:
        remove = peaks_prom[:, 0] < threshold_prom_annular
        peaks = np.delete(
            peaks,
            remove,
            axis=0,
        )
        peaks_int = np.delete(
            peaks_int,
            remove,
        )
        peaks_prom = np.delete(
            peaks_prom,
            remove,
            axis=0,
        )
    if threshold_prom_radial is not None:
        remove = peaks_prom[:, 2] < threshold_prom_radial
        peaks = np.delete(
            peaks,
            remove,
            axis=0,
        )
        peaks_int = np.delete(
            peaks_int,
            remove,
        )
        peaks_prom = np.delete(
            peaks_prom,
            remove,
            axis=0,
        )

    # combine peaks into one array
    peaks_all = np.column_stack((peaks, peaks_int, peaks_prom))

    # Split peaks into multiple peaks if they have sigma values larger than user-specified threshold
    if sigma_annular_deg_max is not None:
        peaks_new = np.zeros((0, peaks_all.shape[1]))
        for a0 in range(peaks_all.shape[0]):
            if peaks_all[a0, 4] >= (1.5 * sigma_annular_deg_max):
                num = np.round(peaks_all[a0, 4] / sigma_annular_deg_max)
                sigma_annular_new = peaks_all[a0, 4] / num

                v = np.arange(num)
                v -= np.mean(v)
                t_new = np.mod(
                    peaks_all[a0, 0] + 2 * v * sigma_annular_new, self._n_annular
                )

                for a1 in range(num.astype("int")):
                    peaks_new = np.vstack(
                        (
                            peaks_new,
                            np.array(
                                (
                                    t_new[a1],
                                    peaks_all[a0, 1],
                                    peaks_all[a0, 2],
                                    peaks_all[a0, 3],
                                    sigma_annular_new,
                                    peaks_all[a0, 5],
                                    peaks_all[a0, 6],
                                )
                            ),
                        )
                    )
            else:
                peaks_new = np.vstack((peaks_new, peaks_all[a0, :]))
        peaks_all = peaks_new

    # Output data as a pointlist
    peaks_polar = PointList(
        peaks_all.ravel().view(
            [
                ("qt", float),
                ("qr", float),
                ("intensity", float),
                ("prom_annular", float),
                ("sigma_annular", float),
                ("prom_radial", float),
                ("sigma_radial", float),
            ]
        ),
        name="peaks_polar",
    )

    if plot_result:
        # init
        im_plot = im_polar.copy()
        im_plot = np.maximum(im_plot, 0) ** plot_power_scale

        t = np.linspace(0, 2 * np.pi, 180 + 1)
        ct = np.cos(t)
        st = np.sin(t)

        fig, ax = plt.subplots(figsize=figsize)

        cmap = kwargs.pop("cmap", "gray")
        vmax = kwargs.pop("vmax", 1)
        vmin = kwargs.pop("vmin", 0)
        show(im_plot, figax=(fig, ax), cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)

        # peaks
        ax.scatter(
            peaks_polar["qr"],
            peaks_polar["qt"],
            s=peaks_polar["intensity"] * plot_scale_size,
            marker="o",
            color=(1, 0, 0),
        )
        for a0 in range(peaks_polar.data.shape[0]):
            ax.plot(
                peaks_polar["qr"][a0] + st * peaks_polar["sigma_radial"][a0],
                peaks_polar["qt"][a0] + ct * peaks_polar["sigma_annular"][a0],
                linewidth=1,
                color="r",
            )
            if peaks_polar["qt"][a0] - peaks_polar["sigma_annular"][a0] < 0:
                ax.plot(
                    peaks_polar["qr"][a0] + st * peaks_polar["sigma_radial"][a0],
                    peaks_polar["qt"][a0]
                    + ct * peaks_polar["sigma_annular"][a0]
                    + im_plot.shape[0],
                    linewidth=1,
                    color="r",
                )
            if (
                peaks_polar["qt"][a0] + peaks_polar["sigma_annular"][a0]
                > im_plot.shape[0]
            ):
                ax.plot(
                    peaks_polar["qr"][a0] + st * peaks_polar["sigma_radial"][a0],
                    peaks_polar["qt"][a0]
                    + ct * peaks_polar["sigma_annular"][a0]
                    - im_plot.shape[0],
                    linewidth=1,
                    color="r",
                )

        # plot appearance
        ax.set_xlim((0, im_plot.shape[1] - 1))
        ax.set_ylim((im_plot.shape[0] - 1, 0))

    if returnfig and plot_result:
        if return_background:
            return peaks_polar, sig_bg, sig_bg_mask, fig, ax
        else:
            return peaks_polar, fig, ax
    else:
        if return_background:
            return peaks_polar, sig_bg, sig_bg_mask
        else:
            return peaks_polar


def find_peaks(
    self,
    mask=None,
    bragg_peaks=None,
    bragg_mask_radius=None,
    sigma_annular_deg=10.0,
    sigma_radial_px=3.0,
    sigma_annular_deg_max=None,
    radial_background_subtract=True,
    radial_background_thresh=0.25,
    num_peaks_max=100,
    threshold_abs=1.0,
    threshold_prom_annular=None,
    threshold_prom_radial=None,
    remove_masked_peaks=False,
    scale_sigma_annular=0.5,
    scale_sigma_radial=0.25,
    progress_bar=True,
):
    """
    Peak detection function for polar transformations. Loop through all probe positions,
    find peaks.  Store the peak positions and background signals.

    Parameters
    --------
    sigma_annular_deg: float
        smoothing along the annular direction in degrees, periodic
    sigma_radial_px: float
        smoothing along the radial direction in pixels, not periodic

    Returns
    --------

    """

    # init
    self.bragg_peaks = bragg_peaks
    self.bragg_mask_radius = bragg_mask_radius
    self.peaks = PointListArray(
        dtype=[
            ("qt", "<f8"),
            ("qr", "<f8"),
            ("intensity", "<f8"),
            ("prom_annular", "<f8"),
            ("sigma_annular", "<f8"),
            ("prom_radial", "<f8"),
            ("sigma_radial", "<f8"),
        ],
        shape=self._datacube.Rshape,
        name="peaks_polardata",
    )
    self.background_radial = np.zeros(
        (
            self._datacube.Rshape[0],
            self._datacube.Rshape[1],
            self.radial_bins.shape[0],
        )
    )
    self.background_radial_mask = np.zeros(
        (
            self._datacube.Rshape[0],
            self._datacube.Rshape[1],
            self.radial_bins.shape[0],
        ),
        dtype="bool",
    )

    # Loop over probe positions
    for rx, ry in tqdmnd(
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        desc="Finding peaks ",
        unit=" images",
        disable=not progress_bar,
    ):
        polar_peaks, sig_bg, sig_bg_mask = self.find_peaks_single_pattern(
            rx,
            ry,
            mask=mask,
            bragg_peaks=bragg_peaks,
            bragg_mask_radius=bragg_mask_radius,
            sigma_annular_deg=sigma_annular_deg,
            sigma_radial_px=sigma_radial_px,
            sigma_annular_deg_max=sigma_annular_deg_max,
            radial_background_subtract=radial_background_subtract,
            radial_background_thresh=radial_background_thresh,
            num_peaks_max=num_peaks_max,
            threshold_abs=threshold_abs,
            threshold_prom_annular=threshold_prom_annular,
            threshold_prom_radial=threshold_prom_radial,
            remove_masked_peaks=remove_masked_peaks,
            scale_sigma_annular=scale_sigma_annular,
            scale_sigma_radial=scale_sigma_radial,
            return_background=True,
            plot_result=False,
        )

        self.peaks[rx, ry] = polar_peaks
        self.background_radial[rx, ry] = sig_bg
        self.background_radial_mask[rx, ry] = sig_bg_mask


def refine_peaks_local(
    self,
    mask=None,
    radial_background_subtract=False,
    reset_fits_to_init_positions=False,
    fit_range_sigma_annular=2.0,
    fit_range_sigma_radial=2.0,
    min_num_pixels_fit=10,
    progress_bar=True,
):
    """
    Use local 2D elliptic gaussian fitting to refine the peak locations.
    Optionally include background offset of the peaks.

    Parameters
    --------
    mask: np.array
        Mask image to apply to all images
    radial_background_subtract: bool
        Subtract radial background before fitting
    reset_init_positions: bool
        Use the initial peak parameters for fitting
    fit_range_sigma_annular: float
        Fit range in annular direction, in terms of the current sigma_annular
    fit_range_sigma_radial: float
        Fit range in radial direction, in terms of the current sigma_radial
    min_num_pixels_fit: int
        Minimum number of pixels to perform fitting
    progress_bar: bool
        Enable progress bar

    Returns
    --------

    """

    # See if the intial detected peaks have been saved yet, copy them if needed
    if not hasattr(self, "peaks_init"):
        self.peaks_init = self.peaks.copy()

    # coordinate scaling
    t_step = self._annular_step
    q_step = self._radial_step

    # Coordinate array for the fit
    qq, tt = np.meshgrid(
        self.qq,
        self.tt,
    )
    tq = np.vstack(
        [
            tt.ravel(),
            qq.ravel(),
        ]
    )

    # Loop over probe positions
    for rx, ry in tqdmnd(
        self._datacube.Rshape[0],
        self._datacube.Rshape[1],
        desc="Refining peaks ",
        unit=" images",
        disable=not progress_bar,
    ):
        # mask
        # if needed, generate mask from Bragg peaks
        if self.bragg_peaks is not None:
            mask_bragg = self._datacube.get_braggmask(
                self.bragg_peaks,
                rx,
                ry,
                radius=self.bragg_mask_radius,
            )
            if mask is None:
                mask_fit = mask_bragg
            else:
                mask_fit = np.logical_or(mask, mask_bragg)

        # Get polar image
        im_polar, im_polar_norm, norm_array, mask_bool = self.transform(
            self._datacube.data[rx, ry],
            mask=mask_fit,
            returnval="all_zeros",
        )
        # Change sign convention of mask
        mask_bool = np.logical_not(mask_bool)
        # Background subtraction
        if radial_background_subtract:
            sig_bg = self.background_radial[rx, ry]
            im_polar = np.maximum(im_polar - sig_bg[None, :], 0)

        # initial peak positions
        if reset_fits_to_init_positions:
            p = self.peaks_init[rx, ry]
        else:
            p = self.peaks[rx, ry]

        # loop over peaks
        for a0 in range(p.data.shape[0]):
            if radial_background_subtract:
                # initial parameters
                p0 = [
                    p["intensity"][a0],
                    p["qt"][a0] * t_step,
                    p["qr"][a0] * q_step,
                    p["sigma_annular"][a0] * t_step,
                    p["sigma_radial"][a0] * q_step,
                ]

                # Mask around peak for fitting
                dt = np.mod(tt - p0[1] + np.pi / 2, np.pi) - np.pi / 2
                mask_peak = np.logical_and(
                    mask_bool,
                    dt**2 / (fit_range_sigma_annular * p0[3]) ** 2
                    + (qq - p0[2]) ** 2 / (fit_range_sigma_radial * p0[4]) ** 2
                    <= 1,
                )

                if np.sum(mask_peak) > min_num_pixels_fit:
                    try:
                        # perform fitting
                        p0, pcov = curve_fit(
                            polar_twofold_gaussian_2D,
                            tq[:, mask_peak.ravel()],
                            im_polar[mask_peak],
                            p0=p0,
                            # bounds = bounds,
                        )

                        # Output parameters
                        self.peaks[rx, ry]["intensity"][a0] = p0[0]
                        self.peaks[rx, ry]["qt"][a0] = p0[1] / t_step
                        self.peaks[rx, ry]["qr"][a0] = p0[2] / q_step
                        self.peaks[rx, ry]["sigma_annular"][a0] = p0[3] / t_step
                        self.peaks[rx, ry]["sigma_radial"][a0] = p0[4] / q_step

                    except:
                        pass

            else:
                # initial parameters
                p0 = [
                    p["intensity"][a0],
                    p["qt"][a0] * t_step,
                    p["qr"][a0] * q_step,
                    p["sigma_annular"][a0] * t_step,
                    p["sigma_radial"][a0] * q_step,
                    0,
                ]

                # Mask around peak for fitting
                dt = np.mod(tt - p0[1] + np.pi / 2, np.pi) - np.pi / 2
                mask_peak = np.logical_and(
                    mask_bool,
                    dt**2 / (fit_range_sigma_annular * p0[3]) ** 2
                    + (qq - p0[2]) ** 2 / (fit_range_sigma_radial * p0[4]) ** 2
                    <= 1,
                )

                if np.sum(mask_peak) > min_num_pixels_fit:
                    try:
                        # perform fitting
                        p0, pcov = curve_fit(
                            polar_twofold_gaussian_2D_background,
                            tq[:, mask_peak.ravel()],
                            im_polar[mask_peak],
                            p0=p0,
                            # bounds = bounds,
                        )

                        # Output parameters
                        self.peaks[rx, ry]["intensity"][a0] = p0[0]
                        self.peaks[rx, ry]["qt"][a0] = p0[1] / t_step
                        self.peaks[rx, ry]["qr"][a0] = p0[2] / q_step
                        self.peaks[rx, ry]["sigma_annular"][a0] = p0[3] / t_step
                        self.peaks[rx, ry]["sigma_radial"][a0] = p0[4] / q_step

                    except:
                        pass


def plot_radial_peaks(
    self,
    q_pixel_units=False,
    qmin=None,
    qmax=None,
    qstep=None,
    label_y_axis=False,
    figsize=(8, 4),
    returnfig=False,
):
    """
    Calculate and plot the total peak signal as a function of the radial coordinate.

    """

    # Get all peak data
    vects = np.concatenate(
        [
            self.peaks[i, j].data
            for i in range(self._datacube.Rshape[0])
            for j in range(self._datacube.Rshape[1])
        ]
    )
    if q_pixel_units:
        qr = vects["qr"]
    else:
        qr = (vects["qr"] + self.qmin) * self._radial_step
    intensity = vects["intensity"]

    # bins
    if qmin is None:
        qmin = self.qq[0]
    if qmax is None:
        qmax = self.qq[-1]
    if qstep is None:
        qstep = self.qq[1] - self.qq[0]
    q_bins = np.arange(qmin, qmax, qstep)
    q_num = q_bins.shape[0]
    if q_pixel_units:
        q_bins /= self._radial_step

    # histogram
    q_ind = (qr - q_bins[0]) / (q_bins[1] - q_bins[0])
    qf = np.floor(q_ind).astype("int")
    dq = q_ind - qf

    sub = np.logical_and(qf >= 0, qf < q_num)
    int_peaks = np.bincount(
        np.floor(q_ind[sub]).astype("int"),
        weights=(1 - dq[sub]) * intensity[sub],
        minlength=q_num,
    )
    sub = np.logical_and(q_ind >= -1, q_ind < q_num - 1)
    int_peaks += np.bincount(
        np.floor(q_ind[sub] + 1).astype("int"),
        weights=dq[sub] * intensity[sub],
        minlength=q_num,
    )

    # storing arrays for further plotting
    self.q_bins = q_bins
    self.int_peaks = int_peaks

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        q_bins,
        int_peaks,
        color="r",
        linewidth=2,
    )
    ax.set_xlim((q_bins[0], q_bins[-1]))
    if q_pixel_units:
        ax.set_xlabel(
            "Scattering Angle (pixels)",
            fontsize=14,
        )
    else:
        ax.set_xlabel(
            "Scattering Angle (" + self.calibration.get_Q_pixel_units() + ")",
            fontsize=14,
        )
    ax.set_ylabel(
        "Total Peak Signal",
        fontsize=14,
    )
    if not label_y_axis:
        ax.tick_params(left=False, labelleft=False)

    if returnfig:
        return fig, ax


def model_radial_background(
    self,
    ring_position=None,
    ring_sigma=None,
    ring_int=None,
    refine_model=True,
    plot_result=True,
    figsize=(8, 4),
):
    """
    User provided radial background model, of the form:

    int = int_const
        + int_0 * exp( - q**2 / (2*s0**2) )
        + int_1 * exp( - (q - q_1)**2 / (2*s1**2) )
        + ...
        + int_n * exp( - (q - q_n)**2 / (2*sn**2) )

    where n is the number of amorphous halos / rings included in the fit.

    """

    # Get mean radial background and mask
    self.background_radial_mean = np.sum(
        self.background_radial * self.background_radial_mask, axis=(0, 1)
    )
    background_radial_mean_norm = np.sum(self.background_radial_mask, axis=(0, 1))
    self.background_mask = background_radial_mean_norm > (
        np.max(background_radial_mean_norm) * 0.05
    )
    self.background_radial_mean[self.background_mask] /= background_radial_mean_norm[
        self.background_mask
    ]
    self.background_radial_mean[np.logical_not(self.background_mask)] = 0

    # init
    if ring_position is not None:
        ring_position = np.atleast_1d(np.array(ring_position))
        num_rings = ring_position.shape[0]
    else:
        num_rings = 0
    self.background_coefs = np.zeros(3 + 3 * num_rings)

    if ring_sigma is None:
        ring_sigma = (
            np.atleast_1d(np.ones(num_rings))
            * self.polar_shape[1]
            * 0.05
            * self._radial_step
        )
    else:
        ring_sigma = np.atleast_1d(np.array(ring_sigma))

    # Background model initial parameters
    int_const = np.min(self.background_radial_mean)
    int_0 = np.max(self.background_radial_mean) - int_const
    sigma_0 = self.polar_shape[1] * 0.25 * self._radial_step
    self.background_coefs[0] = int_const
    self.background_coefs[1] = int_0
    self.background_coefs[2] = sigma_0

    # Additional Gaussians
    if ring_int is None:
        # Estimate peak intensities
        sig_0 = int_const + int_0 * np.exp(self.qq**2 / (-2 * sigma_0**2))
        sig_peaks = np.maximum(self.background_radial_mean - sig_0, 0.0)

        ring_int = np.atleast_1d(np.zeros(num_rings))
        for a0 in range(num_rings):
            ind = np.argmin(np.abs(self.qq - ring_position[a0]))
            ring_int[a0] = sig_peaks[ind]

    else:
        ring_int = np.atleast_1d(np.array(ring_int))
    for a0 in range(num_rings):
        self.background_coefs[3 * a0 + 3] = ring_int[a0]
        self.background_coefs[3 * a0 + 4] = ring_sigma[a0]
        self.background_coefs[3 * a0 + 5] = ring_position[a0]
    lb = np.zeros_like(self.background_coefs)
    ub = np.ones_like(self.background_coefs) * np.inf

    # Create background model
    def background_model(q, *coefs):
        coefs = np.squeeze(np.array(coefs))
        num_rings = np.round((coefs.shape[0] - 3) / 3).astype("int")

        sig = np.ones(q.shape[0]) * coefs[0]
        sig += coefs[1] * np.exp(q**2 / (-2 * coefs[2] ** 2))

        for a0 in range(num_rings):
            sig += coefs[3 * a0 + 3] * np.exp(
                (q - coefs[3 * a0 + 5]) ** 2 / (-2 * coefs[3 * a0 + 4] ** 2)
            )

        return sig

    self.background_model = background_model

    # Refine background model coefficients
    if refine_model:
        self.background_coefs = curve_fit(
            self.background_model,
            self.qq[self.background_mask],
            self.background_radial_mean[self.background_mask],
            p0=self.background_coefs,
            xtol=1e-12,
            bounds=(lb, ub),
        )[0]

    # plotting
    if plot_result:
        self.plot_radial_background(
            q_pixel_units=False,
            plot_background_model=True,
            figsize=figsize,
        )


def refine_peaks(
    self,
    mask=None,
    # reset_fits_to_init_positions = False,
    scale_sigma_estimate=0.5,
    min_num_pixels_fit=10,
    maxfev=None,
    progress_bar=True,
):
    """
    Use global fitting model for all images. Requires an background model
    specified with self.model_radial_background().

    TODO:   add fitting reset
            add min number pixels condition
            track any failed fitting points, output as a boolean array

    Parameters
    --------
    mask: np.array
        Mask image to apply to all images
    radial_background_subtract: bool
        Subtract radial background before fitting
    reset_fits_to_init_positions: bool
        Use the initial peak parameters for fitting
    scale_sigma_estimate: float
        Factor to reduce sigma of peaks by, to prevent fit from running away.
    min_num_pixels_fit: int
        Minimum number of pixels to perform fitting
    maxfev: int
        Maximum number of iterations in fit.  Set to a low number for a fast fit.
    progress_bar: bool
        Enable progress bar

    Returns
    --------

    """

    # coordinate scaling
    t_step = self._annular_step
    q_step = self._radial_step

    # Background model params
    num_rings = np.round((self.background_coefs.shape[0] - 3) / 3).astype("int")

    # basis
    qq, tt = np.meshgrid(
        self.qq,
        self.tt,
    )
    basis = np.zeros((qq.size, 3))
    basis[:, 0] = tt.ravel()
    basis[:, 1] = qq.ravel()
    basis[:, 2] = num_rings

    # init
    self.peaks_refine = PointListArray(
        dtype=[
            ("qt", "float"),
            ("qr", "float"),
            ("intensity", "float"),
            ("sigma_annular", "float"),
            ("sigma_radial", "float"),
        ],
        shape=self._datacube.Rshape,
        name="peaks_polardata_refined",
    )
    self.background_refine = np.zeros(
        (
            self._datacube.Rshape[0],
            self._datacube.Rshape[1],
            np.round(3 * num_rings + 3).astype("int"),
        )
    )

    # Main loop over probe positions
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Refining peaks ",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        # Get transformed image and normalization array
        im_polar, im_polar_norm, norm_array, mask_bool = self.transform(
            self._datacube.data[rx, ry],
            mask=mask,
            returnval="all_zeros",
        )
        # Change sign convention of mask
        mask_bool = np.logical_not(mask_bool)

        # Get initial peaks, in dimensioned units
        p = self.peaks[rx, ry]
        qt = p.data["qt"] * t_step
        qr = (p.data["qr"] + self.qmin) * q_step
        int_peaks = p.data["intensity"]
        s_annular = p.data["sigma_annular"] * t_step
        s_radial = p.data["sigma_radial"] * q_step
        num_peaks = p["qt"].shape[0]

        # unified coefficients
        # Note we sharpen sigma estimate for refinement
        coefs_all = np.hstack(
            (
                self.background_coefs,
                qt,
                qr,
                int_peaks,
                s_annular * scale_sigma_estimate,
                s_radial * scale_sigma_estimate,
            )
        )

        # bounds
        lb = np.zeros_like(coefs_all)
        ub = np.ones_like(coefs_all) * np.inf

        # Construct fitting model
        def fit_image(basis, *coefs):
            coefs = np.squeeze(np.array(coefs))

            num_rings = np.round(basis[0, 2]).astype("int")
            num_peaks = np.round((coefs.shape[0] - (3 * num_rings + 3)) / 5).astype(
                "int"
            )

            coefs_bg = coefs[: (3 * num_rings + 3)]
            coefs_peaks = coefs[(3 * num_rings + 3) :]

            # Background
            sig = self.background_model(basis[:, 1], coefs_bg)

            # add peaks
            for a0 in range(num_peaks):
                dt = (
                    np.mod(
                        basis[:, 0] - coefs_peaks[num_peaks * 0 + a0] + np.pi / 2, np.pi
                    )
                    - np.pi / 2
                )
                dq = basis[:, 1] - coefs_peaks[num_peaks * 1 + a0]

                sig += coefs_peaks[num_peaks * 2 + a0] * np.exp(
                    dt**2 / (-2 * coefs_peaks[num_peaks * 3 + a0] ** 2)
                    + dq**2 / (-2 * coefs_peaks[num_peaks * 4 + a0] ** 2)
                )

            return sig

        # refine fitting model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if maxfev is None:
                    coefs_all = curve_fit(
                        fit_image,
                        basis[mask_bool.ravel(), :],
                        im_polar[mask_bool],
                        p0=coefs_all,
                        xtol=1e-12,
                        bounds=(lb, ub),
                    )[0]
                else:
                    coefs_all = curve_fit(
                        fit_image,
                        basis[mask_bool.ravel(), :],
                        im_polar[mask_bool],
                        p0=coefs_all,
                        xtol=1e-12,
                        maxfev=maxfev,
                        bounds=(lb, ub),
                    )[0]

            # Output refined peak parameters
            coefs_peaks = np.reshape(coefs_all[(3 * num_rings + 3) :], (5, num_peaks)).T
            self.peaks_refine[rx, ry] = PointList(
                coefs_peaks.ravel().view(
                    [
                        ("qt", float),
                        ("qr", float),
                        ("intensity", float),
                        ("sigma_annular", float),
                        ("sigma_radial", float),
                    ]
                ),
                name="peaks_polar",
            )
        except:
            # if fitting has failed, we will still output the last iteration
            # TODO - add a flag for unconverged fits
            coefs_peaks = np.reshape(coefs_all[(3 * num_rings + 3) :], (5, num_peaks)).T
            self.peaks_refine[rx, ry] = PointList(
                coefs_peaks.ravel().view(
                    [
                        ("qt", float),
                        ("qr", float),
                        ("intensity", float),
                        ("sigma_annular", float),
                        ("sigma_radial", float),
                    ]
                ),
                name="peaks_polar",
            )

            #  mean background signal,
            # # but none of the peaks.
            # pass

        # Output refined parameters for background
        coefs_bg = coefs_all[: (3 * num_rings + 3)]
        self.background_refine[rx, ry] = coefs_bg

    # # Testing
    # im_fit = np.reshape(
    #     fit_image(basis,coefs_all),
    #     self.polar_shape)

    # fig,ax = plt.subplots(figsize=(8,6))
    # ax.imshow(
    #     np.vstack((
    #         im_polar,
    #         im_fit,
    #     )),
    #     cmap = 'turbo',
    #     )


def plot_radial_background(
    self,
    q_pixel_units=False,
    label_y_axis=False,
    plot_background_model=False,
    figsize=(8, 4),
    returnfig=False,
):
    """
    Calculate and plot the mean background signal, background standard deviation.

    """

    # mean
    self.background_radial_mean = np.sum(
        self.background_radial * self.background_radial_mask, axis=(0, 1)
    )
    background_radial_mean_norm = np.sum(self.background_radial_mask, axis=(0, 1))
    self.background_mask = background_radial_mean_norm > (
        np.max(background_radial_mean_norm) * 0.05
    )
    self.background_radial_mean[self.background_mask] /= background_radial_mean_norm[
        self.background_mask
    ]
    self.background_radial_mean[np.logical_not(self.background_mask)] = 0

    # variance and standard deviation
    self.background_radial_var = np.sum(
        (self.background_radial - self.background_radial_mean[None, None, :]) ** 2
        * self.background_radial_mask,
        axis=(0, 1),
    )
    self.background_radial_var[self.background_mask] /= self.background_radial_var[
        self.background_mask
    ]
    self.background_radial_var[np.logical_not(self.background_mask)] = 0
    self.background_radial_std = np.sqrt(self.background_radial_var)

    if q_pixel_units:
        q_axis = np.arange(self.qq.shape[0])
    else:
        q_axis = self.qq[self.background_mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        q_axis,
        self.background_radial_mean[self.background_mask]
        - self.background_radial_std[self.background_mask],
        self.background_radial_mean[self.background_mask]
        + self.background_radial_std[self.background_mask],
        color="r",
        alpha=0.2,
    )
    ax.plot(
        q_axis,
        self.background_radial_mean[self.background_mask],
        color="r",
        linewidth=2,
    )

    # overlay fitting model
    if plot_background_model:
        sig = self.background_model(
            self.qq,
            self.background_coefs,
        )
        ax.plot(q_axis, sig, color="k", linewidth=2, linestyle="--")

    # plot appearance
    ax.set_xlim((q_axis[0], q_axis[-1]))
    if q_pixel_units:
        ax.set_xlabel(
            "Scattering Angle (pixels)",
            fontsize=14,
        )
    else:
        ax.set_xlabel(
            "Scattering Angle (" + self.calibration.get_Q_pixel_units() + ")",
            fontsize=14,
        )
    ax.set_ylabel(
        "Background Signal",
        fontsize=14,
    )
    if not label_y_axis:
        ax.tick_params(left=False, labelleft=False)

    if returnfig:
        return fig, ax


def make_orientation_histogram(
    self,
    radial_ranges: np.ndarray = None,
    orientation_flip_sign: bool = False,
    orientation_offset_degrees: float = 0.0,
    orientation_separate_bins: bool = False,
    upsample_factor: float = 4.0,
    use_refined_peaks=True,
    use_peak_sigma=False,
    peak_sigma_samples=6,
    theta_step_deg: float = None,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    sigma_theta: float = 3.0,
    normalize_intensity_image: bool = False,
    normalize_intensity_stack: bool = True,
    progress_bar: bool = True,
):
    """
    Make an orientation histogram, in order to use flowline visualization of orientation maps.
    Use peaks attached to polardatacube.

    NOTE - currently assumes two fold rotation symmetry
    TODO - add support for non two fold symmetry polardatacube

    Args:
        radial_ranges (np array):           Size (N x 2) array for N radial bins, or (2,) for a single bin.
        orientation_flip_sign (bool):       Flip the direction of theta
        orientation_offset_degrees (float): Offset for orientation angles
        orientation_separate_bins (bool):   whether to place multiple angles into multiple radial bins.
        upsample_factor (float):            Upsample factor
        use_refined_peaks (float):          Use refined peak positions
        use_peak_sigma (float):             Spread signal along annular direction using measured std.
        theta_step_deg (float):             Step size along annular direction in degrees
        sigma_x (float):                    Smoothing in x direction before upsample
        sigma_y (float):                    Smoothing in x direction before upsample
        sigma_theta (float):                Smoothing in annular direction (units of bins, periodic)
        normalize_intensity_image (bool):   Normalize to max peak intensity = 1, per image
        normalize_intensity_stack (bool):   Normalize to max peak intensity = 1, all images
        progress_bar (bool):                Enable progress bar

    Returns:
        orient_hist (array):                4D array containing Bragg peak intensity histogram
                                            [radial_bin x_probe y_probe theta]
    """

    # coordinates
    if theta_step_deg is None:
        # Get angles from polardatacube
        theta = self.tt
    else:
        theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
    dtheta = theta[1] - theta[0]
    dtheta_deg = dtheta * 180 / np.pi
    num_theta_bins = np.size(theta)

    # Input bins
    radial_ranges = np.array(radial_ranges)
    if radial_ranges.ndim == 1:
        radial_ranges = radial_ranges[None, :]
    radial_ranges_2 = radial_ranges**2
    num_radii = radial_ranges.shape[0]
    size_input = self._datacube.shape[0:2]

    # Output size
    size_output = np.round(
        np.array(size_input).astype("float") * upsample_factor
    ).astype("int")

    # output init
    orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])

    if use_peak_sigma:
        v_sigma = np.linspace(-2, 2, 2 * peak_sigma_samples + 1)
        w_sigma = np.exp(-(v_sigma**2) / 2)

    if use_refined_peaks is False:
        warnings.warn("Orientation histogram is using non-refined peak positions")

    # Loop over all probe positions
    for a0 in range(num_radii):
        t = "Generating histogram " + str(a0)

        for rx, ry in tqdmnd(
            *size_input, desc=t, unit=" probe positions", disable=not progress_bar
        ):
            x = (rx + 0.5) * upsample_factor - 0.5
            y = (ry + 0.5) * upsample_factor - 0.5
            x = np.clip(x, 0, size_output[0] - 2)
            y = np.clip(y, 0, size_output[1] - 2)

            xF = np.floor(x).astype("int")
            yF = np.floor(y).astype("int")
            dx = x - xF
            dy = y - yF

            add_data = False
            if use_refined_peaks:
                q = self.peaks_refine[rx, ry]["qr"]
            else:
                q = (self.peaks[rx, ry]["qr"] + self.qmin) * self._radial_step
            r2 = q**2
            sub = np.logical_and(
                r2 >= radial_ranges_2[a0, 0], r2 < radial_ranges_2[a0, 1]
            )

            if np.any(sub):
                add_data = True
                intensity = self.peaks[rx, ry]["intensity"][sub]

                # Angles of all peaks
                if use_refined_peaks:
                    theta = self.peaks_refine[rx, ry]["qt"][sub]
                else:
                    theta = self.peaks[rx, ry]["qt"][sub] * self._annular_step
                if orientation_flip_sign:
                    theta *= -1
                theta += orientation_offset_degrees

                t = theta / dtheta

                # If needed, expand signal using peak sigma to write into multiple bins
                if use_peak_sigma:
                    if use_refined_peaks:
                        theta_std = (
                            self.peaks_refine[rx, ry]["sigma_annular"][sub] / dtheta
                        )
                    else:
                        theta_std = self.peaks[rx, ry]["sigma_annular"][sub] / dtheta
                    t = (t[:, None] + theta_std[:, None] * v_sigma[None, :]).ravel()
                    intensity = (intensity[:, None] * w_sigma[None, :]).ravel()

            if add_data:
                tF = np.floor(t).astype("int")
                dt = t - tF

                orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(1 - dx) * (1 - dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(1 - dx) * (1 - dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF + 1, yF, :] = orient_hist[
                    a0, xF + 1, yF, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(dx) * (1 - dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF + 1, yF, :] = orient_hist[
                    a0, xF + 1, yF, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(dx) * (1 - dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF, yF + 1, :] = orient_hist[
                    a0, xF, yF + 1, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(1 - dx) * (dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF, yF + 1, :] = orient_hist[
                    a0, xF, yF + 1, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(1 - dx) * (dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                    a0, xF + 1, yF + 1, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(dx) * (dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                    a0, xF + 1, yF + 1, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(dx) * (dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

    # smoothing / interpolation
    if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
        if num_radii > 1:
            print("Interpolating orientation matrices ...", end="")
        else:
            print("Interpolating orientation matrix ...", end="")
        if sigma_x is not None and sigma_x > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,
                sigma_x * upsample_factor,
                mode="nearest",
                axis=1,
                truncate=3.0,
            )
        if sigma_y is not None and sigma_y > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,
                sigma_y * upsample_factor,
                mode="nearest",
                axis=2,
                truncate=3.0,
            )
        if sigma_theta is not None and sigma_theta > 0:
            orient_hist = gaussian_filter1d(
                orient_hist, sigma_theta / dtheta_deg, mode="wrap", axis=3, truncate=2.0
            )
        print(" done.")

    # normalization
    if normalize_intensity_stack is True:
        orient_hist = orient_hist / np.max(orient_hist)
    elif normalize_intensity_image is True:
        for a0 in range(num_radii):
            orient_hist[a0, :, :, :] = orient_hist[a0, :, :, :] / np.max(
                orient_hist[a0, :, :, :]
            )

    return orient_hist
