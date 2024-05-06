# Analysis scripts for amorphous 4D-STEM data using polar transformations.

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

from emdfile import tqdmnd


def calculate_radial_statistics(
    self,
    mask_realspace=None,
    plot_results_mean=False,
    plot_results_var=False,
    figsize=(8, 4),
    returnval=False,
    returnfig=False,
    progress_bar=True,
):
    """
    Calculate the radial statistics used in fluctuation electron microscopy (FEM)
    and as an initial step in radial distribution function (RDF) calculation.
    The computed quantities are the radial mean, variance, and normalized variance.

    There are several ways the means and variances can be computed.  Here we first
    compute the mean and standard deviation pattern by pattern, i.e. for
    diffraction signal d(x,y; q,theta) we take

        d_mean_all(x,y; q) = \int_{0}^{2\pi} d(x,y; q,\theta) d\theta
        d_var_all(x,y; q) = \int_{0}^{2\pi}
            \( d(x,y; q,\theta) - d_mean_all(x,y; q,\theta) \)^2 d\theta

    Then we find the mean and variance profiles by taking the means of these
    quantities over all scan positions:

        d_mean(q) = \sum_{x,y} d_mean_all(x,y; q)
        d_var(q) = \sum_{x,y} d_var_all(x,y; q)

    and the normalized variance is d_var/d_mean.

    This follows the methods described in [@cophus TODO ADD CITATION].


    Parameters
    --------
    plot_results_mean: bool
        Toggles plotting the computed radial means
    plot_results_var: bool
        Toggles plotting the computed radial variances
    figsize: 2-tuple
        Size of output figures
    returnval: bool
        Toggles returning the answer. Answers are always stored internally.
    returnfig: bool
        Toggles returning figures that have been plotted.  Only figures for
        which `plot_results_*` is True are returned.

    Returns
    --------
    radial_avg: np.array
        Optional - returned iff returnval is True. The average radial intensity.
    radial_var: np.array
        Optional - returned iff returnval is True. The radial variance.
    fig_means: 2-tuple (fig,ax)
        Optional - returned iff returnfig is True. Plot of the radial means.
    fig_var: 2-tuple (fig,ax)
        Optional - returned iff returnfig is True. Plot of the radial variances.
    """

    # init radial data arrays
    self.radial_all = np.zeros(
        (
            self._datacube.shape[0],
            self._datacube.shape[1],
            self.polar_shape[1],
        )
    )
    self.radial_all_std = np.zeros(
        (
            self._datacube.shape[0],
            self._datacube.shape[1],
            self.polar_shape[1],
        )
    )

    # Compute the radial mean and standard deviation for each probe position
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Radial statistics",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        if mask_realspace is None or mask_realspace[rx, ry]:
            self.radial_all[rx, ry] = np.mean(self.data[rx, ry], axis=0)
            self.radial_all_std[rx, ry] = np.sqrt(
                np.mean(
                    (self.data[rx, ry] - self.radial_all[rx, ry][None]) ** 2, axis=0
                )
            )

    if mask_realspace is None:
        self.radial_mean = np.mean(self.radial_all, axis=(0, 1))
        self.radial_var = np.mean(
            (self.radial_all - self.radial_mean[None, None]) ** 2, axis=(0, 1)
        )

    else:
        self.radial_mean = np.sum(self.radial_all, axis=(0, 1)) / np.sum(mask_realspace)
        self.radial_var = np.zeros_like(self.radial_mean)
        for rx in range(self._datacube.shape[0]):
            for ry in range(self._datacube.shape[1]):
                if mask_realspace[rx, ry]:
                    self.radial_var += (self.radial_all[rx, ry] - self.radial_mean) ** 2
        self.radial_var /= np.sum(mask_realspace)

    # Compute normalized variance
    self.radial_var_norm = np.copy(self.radial_var)
    sub = self.radial_mean > 0.0
    self.radial_var_norm[sub] /= self.radial_mean[sub] ** 2

    # prepare answer
    statistics = self.radial_mean, self.radial_var, self.radial_var_norm
    if returnval:
        ans = statistics if not returnfig else [statistics]
    else:
        ans = None if not returnfig else []

    # plot results
    if plot_results_mean:
        fig, ax = plot_radial_mean(
            self,
            figsize=figsize,
            returnfig=True,
        )
        if returnfig:
            ans.append((fig, ax))
    if plot_results_var:
        fig, ax = plot_radial_var_norm(
            self,
            figsize=figsize,
            returnfig=True,
        )
        if returnfig:
            ans.append((fig, ax))

    # return
    return ans


def plot_radial_mean(
    self,
    log_x=False,
    log_y=False,
    figsize=(8, 4),
    returnfig=False,
):
    """
    Plot the radial means.

    Parameters
    ----------
    log_x : bool
        Toggle log scaling of the x-axis
    log_y : bool
        Toggle log scaling of the y-axis
    figsize : 2-tuple
        Size of the output figure
    returnfig : bool
        Toggle returning the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.qq,
        self.radial_mean,
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Scattering Vector (" + self.calibration.get_Q_pixel_units() + ")")
    ax.set_ylabel("Radial Mean")
    if log_x and self.qq[0] == 0.0:
        ax.set_xlim((self.qq[1], self.qq[-1]))
    else:
        ax.set_xlim((self.qq[0], self.qq[-1]))

    if returnfig:
        return fig, ax


def plot_radial_var_norm(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    Plot the radial variances.

    Parameters
    ----------
    figsize : 2-tuple
        Size of the output figure
    returnfig : bool
        Toggle returning the figure

    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.qq,
        self.radial_var_norm,
    )

    ax.set_xlabel("Scattering Vector (" + self.calibration.get_Q_pixel_units() + ")")
    ax.set_ylabel("Normalized Variance")
    ax.set_xlim((self.qq[0], self.qq[-1]))

    if returnfig:
        return fig, ax


def calculate_pair_dist_function(
    self,
    k_min=0.05,
    k_max=None,
    k_width=0.25,
    k_lowpass=None,
    k_highpass=None,
    r_min=0.0,
    r_max=20.0,
    r_step=0.02,
    damp_origin_fluctuations=True,
    enforce_positivity=True,
    density=None,
    plot_background_fits=False,
    plot_sf_estimate=False,
    plot_reduced_pdf=True,
    plot_pdf=False,
    figsize=(8, 4),
    maxfev=None,
    returnval=False,
    returnfig=False,
):
    """
    Calculate the pair distribution function (PDF).

    First a background is calculated using primarily the signal at the highest
    scattering vectors available, given by a sum of two exponentials ~exp(-k^2)
    and ~exp(-k^4) modelling the single atom scattering factor plus a constant
    offset. Next, the structure factor is computed as

        S(k) = (I(k) - bg(k)) * k / f(k)

    where k is the magnitude of the scattering vector, I(k) is the mean radial
    signal, f(k) is the single atom scattering factor, and bg(k) is the total
    background signal (i.e. f(k) plus a constant offset). S(k) is masked outside
    of the selected fitting region of k-values [k_min,k_max] and low/high pass
    filters are optionally applied. The structure factor is then inverted into
    the reduced pair distribution function g(r) using

        g(r) = \frac{2}{\pi) \int sin( 2\pi r k ) S(k) dk

    The value of the integral is (optionally) damped to zero at the origin to
    match the physical requirement that this condition holds. Finally, the
    full PDF G(r) is computed if a known density is provided, using

        G(r) = 1 + [ \frac{2}{\pi} * g(r) / ( 4\pi * D * r dr ) ]

    This follows the methods described in [@cophus TODO ADD CITATION].


    Parameters
    ----------
    k_min : number
        Minimum scattering vector to include in the calculation
    k_max : number or None
        Maximum scattering vector to include in the calculation. Note that
        this cutoff is used when calculating the structure factor - however it
        is *not* used when estimating the background / single atom scattering
        factor, which is best estimated from high scattering lengths.
    k_width : number
        The fitting window for the structure factor calculation [k_min,k_max]
        includes a damped region at its edges, i.e. the signal is smoothly dampled
        to zero in the regions [k_min, k_min+k_width] and [k_max-k_width,k_max]
    k_lowpass : number or None
        Lowpass filter, in units the scattering vector stepsize (i.e. self.qstep)
    k_highpass : number or None
        Highpass filter, in units the scattering vector stepsize (i.e. self.qstep)
    r_min,r_max,r_step : numbers
        Define the real space coordinates r that the PDF g(r) will be computed in.
        The coordinates will be np.arange(r_min,r_max,r_step), given in units
        inverse to the scattering vector units.
    damp_origin_fluctuations : bool
        The value of the PDF approaching the origin should be zero, however numerical
        instability may result in non-physical finite values there. This flag toggles
        damping the value of the PDF to zero near the origin.
    enforce_positivity:
        Force all pdf values to be >0.
    density : number or None
        The density of the sample, if known.  If this is not provided, only the
        reduced PDF is calculated.  If this value is provided, the PDF is also
        calculated.
    plot_background_fits : bool
    plot_sf_estimate : bool
    plot_reduced_pdf=True : bool
    plot_pdf : bool
    figsize : 2-tuple
    maxfev : integer or None
        Max number of iterations to use when fitting the background
    returnval: bool
        Toggles returning the answer. Answers are always stored internally.
    returnfig: bool
        Toggles returning figures that have been plotted.  Only figures for
        which `plot_*` is True are returned.
    """

    # set up coordinates and scaling
    k = self.qq
    dk = k[1] - k[0]
    k2 = k**2
    Ik = self.radial_mean
    int_mean = np.mean(Ik)
    sub_fit = k >= k_min

    # initial guesses for background coefs
    const_bg = np.min(self.radial_mean) / int_mean
    int0 = np.median(self.radial_mean) / int_mean - const_bg
    sigma0 = np.mean(k)
    coefs = [const_bg, int0, sigma0, int0, sigma0]
    lb = [0, 0, 0, 0, 0]
    ub = [np.inf, np.inf, np.inf, np.inf, np.inf]
    # Weight the fit towards high k values
    noise_est = k[-1] - k + dk

    # Estimate the mean atomic form factor + background
    if maxfev is None:
        coefs = curve_fit(
            scattering_model,
            k2[sub_fit],
            Ik[sub_fit] / int_mean,
            sigma=noise_est[sub_fit],
            p0=coefs,
            xtol=1e-8,
            bounds=(lb, ub),
        )[0]
    else:
        coefs = curve_fit(
            scattering_model,
            k2[sub_fit],
            Ik[sub_fit] / int_mean,
            sigma=noise_est[sub_fit],
            p0=coefs,
            xtol=1e-8,
            bounds=(lb, ub),
            maxfev=maxfev,
        )[0]

    coefs[0] *= int_mean
    coefs[1] *= int_mean
    coefs[3] *= int_mean

    # Calculate the mean atomic form factor without a constant offset
    # coefs_fk = (0.0, coefs[1], coefs[2], coefs[3], coefs[4])
    # fk = scattering_model(k2, coefs_fk)
    bg = scattering_model(k2, coefs)
    fk = bg - coefs[0]

    # mask for structure factor estimate
    if k_max is None:
        k_max = np.max(k)
    mask = np.clip(
        np.minimum(
            (k - 0.0) / k_width,
            (k_max - k) / k_width,
        ),
        0,
        1,
    )
    mask = np.sin(mask * (np.pi / 2))

    # Estimate the reduced structure factor S(k)
    Sk = (Ik - bg) * k / fk

    # Masking edges of S(k)
    mask_sum = np.sum(mask)
    Sk = (Sk - np.sum(Sk * mask) / mask_sum) * mask

    # Filtering of S(k)
    if k_lowpass is not None and k_lowpass > 0.0:
        Sk = gaussian_filter(Sk, sigma=k_lowpass / dk, mode="nearest")
    if k_highpass is not None and k_highpass > 0.0:
        Sk_lowpass = gaussian_filter(Sk, sigma=k_highpass / dk, mode="nearest")
        Sk -= Sk_lowpass

    # Calculate the PDF
    r = np.arange(r_min, r_max, r_step)
    ra, ka = np.meshgrid(r, k)
    pdf_reduced = (
        (2 / np.pi)
        * dk
        * np.sum(
            np.sin(2 * np.pi * ra * ka) * Sk[:, None],
            axis=0,
        )
    )

    # Damp the unphysical fluctuations at the PDF origin
    if damp_origin_fluctuations:
        ind_max = np.argmax(pdf_reduced)
        r_ind_max = r[ind_max]
        r_mask = np.minimum(r / r_ind_max, 1.0)
        r_mask = np.sin(r_mask * np.pi / 2) ** 2
        pdf_reduced *= r_mask

    # Store results
    self.pdf_r = r
    self.pdf_reduced = pdf_reduced

    self.Sk = Sk
    self.fk = fk
    self.bg = bg
    self.offset = coefs[0]
    self.Sk_mask = mask

    # if density is provided, we can estimate the full PDF
    if density is not None:
        pdf = pdf_reduced.copy()
        pdf[1:] /= 4 * np.pi * density * r[1:] * (r[1] - r[0])
        pdf += 1

        # damp and clip values below zero
        if damp_origin_fluctuations:
            pdf *= r_mask
        if enforce_positivity:
            pdf = np.maximum(pdf, 0.0)

        # store results
        self.pdf = pdf

    # prepare answer
    if density is None:
        return_values = self.pdf_r, self.pdf_reduced
    else:
        return_values = self.pdf_r, self.pdf_reduced, self.pdf
    if returnval:
        ans = return_values if not returnfig else [return_values]
    else:
        ans = None if not returnfig else []

    # Plots
    if plot_background_fits:
        fig, ax = self.plot_background_fits(figsize=figsize, returnfig=True)
        if returnfig:
            ans.append((fig, ax))

    if plot_sf_estimate:
        fig, ax = self.plot_sf_estimate(figsize=figsize, returnfig=True)
        if returnfig:
            ans.append((fig, ax))

    if plot_reduced_pdf:
        fig, ax = self.plot_reduced_pdf(figsize=figsize, returnfig=True)
        if returnfig:
            ans.append((fig, ax))

    if plot_pdf:
        fig, ax = self.plot_pdf(figsize=figsize, returnfig=True)
        if returnfig:
            ans.append((fig, ax))

    # return
    return ans


def plot_background_fits(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    TODO
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.qq,
        self.radial_mean,
        color="k",
    )
    ax.plot(
        self.qq,
        self.bg,
        color="r",
    )
    ax.set_xlabel("Scattering Vector (" + self.calibration.get_Q_pixel_units() + ")")
    ax.set_ylabel("Radial Mean")
    ax.set_xlim((self.qq[0], self.qq[-1]))
    ax.set_xlabel("Scattering Vector [A^-1]")
    ax.set_ylabel("I(k) and Background Fit Estimates")
    ax.set_ylim(
        (
            np.min(self.radial_mean[self.radial_mean > 0]) * 0.8,
            np.max(self.radial_mean * self.Sk_mask) * 1.25,
        )
    )
    ax.set_yscale("log")
    if returnfig:
        return fig, ax
    plt.show()


def plot_sf_estimate(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    TODO
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.qq,
        self.Sk,
        color="r",
    )
    yr = (np.min(self.Sk), np.max(self.Sk))
    ax.set_ylim(
        (
            yr[0] - 0.05 * (yr[1] - yr[0]),
            yr[1] + 0.05 * (yr[1] - yr[0]),
        )
    )
    ax.set_xlabel("Scattering Vector [A^-1]")
    ax.set_ylabel("Reduced Structure Factor")
    if returnfig:
        return fig, ax
    plt.show()


def plot_reduced_pdf(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    TODO
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.pdf_r,
        self.pdf_reduced,
        color="r",
    )
    ax.set_xlabel("Radius [A]")
    ax.set_ylabel("Reduced Pair Distribution Function")
    if returnfig:
        return fig, ax
    plt.show()


def plot_pdf(
    self,
    figsize=(8, 4),
    returnfig=False,
):
    """
    TODO
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.pdf_r,
        self.pdf,
        color="r",
    )
    ax.set_xlabel("Radius [A]")
    ax.set_ylabel("Pair Distribution Function")
    if returnfig:
        return fig, ax
    plt.show()

    # functions for inverting from reduced PDF back to S(k)

    # # invert
    # ind_max = np.argmax(pdf_reduced* np.sqrt(r))
    # r_ind_max = r[ind_max-1]
    # r_mask = np.minimum(r / (r_ind_max), 1.0)
    # r_mask = np.sin(r_mask*np.pi/2)**2

    # Sk_back_proj = (0.5*r_step)*np.sum(
    #     np.sin(
    #         2*np.pi*ra*ka
    #     ) * pdf_corr[None,:],# * r_mask[None,:],
    #     # ) * pdf_corr[None,:],# * r_mask[None,:],
    #     axis=1,
    # )


def calculate_FEM_local(
    self,
    use_median=False,
    plot_normalized_variance=True,
    figsize=(8, 4),
    return_values=False,
    returnfig=False,
    progress_bar=True,
):
    """
    Calculate fluctuation electron microscopy (FEM) statistics, including radial mean,
    variance, and normalized variance. This function computes the radial average and variance
    for each individual probe position, which can then be mapped over the field-of-view.

    Parameters
    --------
    self: PolarDatacube
        Polar datacube used for measuring FEM properties.
    use_median: Bool
        Use median instead of mean for statistics.

    Returns
    --------
    local_radial_mean: np.array
        Average radial intensity of each probe position
    local_radial_var: np.array
        Variance in the radial dimension of each probe position

    """

    # init radial data arrays
    self.local_radial_mean = np.zeros(
        (
            self._datacube.shape[0],
            self._datacube.shape[1],
            self.polar_shape[1],
        )
    )
    self.local_radial_var = np.zeros(
        (
            self._datacube.shape[0],
            self._datacube.shape[1],
            self.polar_shape[1],
        )
    )

    # Compute the radial mean and standard deviation for each probe position
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Radial statistics",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        im = self.data[rx, ry]

        if use_median:
            im_mean = np.ma.median(im, axis=0)
            im_var = np.ma.median((im - im_mean) ** 2, axis=0)
        else:
            im_mean = np.ma.mean(im, axis=0)
            im_var = np.ma.mean((im - im_mean) ** 2, axis=0)

        self.local_radial_mean[rx, ry] = im_mean
        self.local_radial_var[rx, ry] = im_var

    if plot_normalized_variance:
        fig, ax = plt.subplots(figsize=figsize)

        sig = self.local_radial_var / self.local_radial_mean**2
        if use_median:
            sig_plot = np.median(sig, axis=(0, 1))
        else:
            sig_plot = np.mean(sig, axis=(0, 1))

        ax.plot(
            self.qq,
            sig_plot,
        )
        ax.set_xlabel(
            "Scattering Vector (" + self.calibration.get_Q_pixel_units() + ")"
        )
        ax.set_ylabel("Normalized Variance")
        ax.set_xlim((self.qq[0], self.qq[-1]))

    if return_values:
        if returnfig:
            return self.local_radial_mean, self.local_radial_var, fig, ax
        else:
            return self.local_radial_mean, self.local_radial_var
    else:
        if returnfig:
            return fig, ax


def calculate_annular_symmetry(
    self,
    max_symmetry=12,
    mask_realspace=None,
    plot_result=False,
    figsize=(8, 4),
    return_symmetry_map=False,
    progress_bar=True,
):
    """
    This function calculates radial symmetry of diffraction patterns, typically applied
    to amorphous scattering, but it can also be used for crystalline Bragg diffraction.

    Parameters
    --------
    self: PolarDatacube
        Polar transformed datacube
    max_symmetry: int
        Symmetry orders will be computed from 1 to max_symmetry for n-fold symmetry orders.
    mask_realspace: np.array
        Boolean mask, symmetries will only be computed at probe positions where mask is True.
    plot_result: bool
        Plot the resulting array
    figsize: (float, float)
        Size of the plot.
    return_symmetry_map: bool
        Set to true to return the symmetry array.
    progress_bar: bool
        Show progress bar during calculation.

    Returns
    --------
    annular_symmetry: np.array
        Array with annular symmetry magnitudes, with shape [max_symmetry, num_radial_bins]

    """

    # Initialize outputs
    self.annular_symmetry_max = max_symmetry
    self.annular_symmetry = np.zeros(
        (
            self.data_raw.shape[0],
            self.data_raw.shape[1],
            max_symmetry,
            self.polar_shape[1],
        )
    )

    # Loop over all probe positions
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Annular symmetry",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        # Get polar transformed image
        im = self.transform(
            self.data_raw.data[rx, ry],
        )
        polar_im = np.ma.getdata(im)
        polar_mask = np.ma.getmask(im)
        polar_im[polar_mask] = 0
        polar_mask = np.logical_not(polar_mask)

        # Calculate normalized correlation of polar image along angular direction (axis = 0)
        polar_corr = np.real(
            np.fft.ifft(
                np.abs(
                    np.fft.fft(
                        polar_im,
                        axis=0,
                    )
                )
                ** 2,
                axis=0,
            ),
        )
        polar_corr_norm = (
            np.sum(
                polar_im,
                axis=0,
            )
            ** 2
        )
        sub = polar_corr_norm > 0
        polar_corr[:, sub] /= polar_corr_norm[
            sub
        ]  # gets rid of divide by 0 (False near center)
        polar_corr[:, sub] -= 1

        # Calculate normalized correlation of polar mask along angular direction (axis = 0)
        mask_corr = np.real(
            np.fft.ifft(
                np.abs(
                    np.fft.fft(
                        polar_mask.astype("float"),
                        axis=0,
                    )
                )
                ** 2,
                axis=0,
            ),
        )
        mask_corr_norm = (
            np.sum(
                polar_mask.astype("float"),
                axis=0,
            )
            ** 2
        )
        sub = mask_corr_norm > 0
        mask_corr[:, sub] /= mask_corr_norm[
            sub
        ]  # gets rid of divide by 0 (False near center)
        mask_corr[:, sub] -= 1

        # Normalize polar correlation by mask correlation (beam stop removal)
        sub = np.abs(mask_corr) > 0
        polar_corr[sub] -= mask_corr[sub]

        # Measure symmetry
        self.annular_symmetry[rx, ry, :, :] = np.abs(np.fft.fft(polar_corr, axis=0))[
            1 : max_symmetry + 1
        ]

    if plot_result:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            np.mean(self.annular_symmetry, axis=(0, 1)),
            aspect="auto",
            extent=[
                self.qq[0],
                self.qq[-1],
                max_symmetry,
                0,
            ],
        )
        ax.set_yticks(
            np.arange(max_symmetry) + 0.5,
            range(1, max_symmetry + 1),
        )
        ax.set_xlabel("Scattering angle (1/Å)")
        ax.set_ylabel("Symmetry Order")

    if return_symmetry_map:
        return self.annular_symmetry


def plot_annular_symmetry(
    self,
    symmetry_orders=None,
    plot_std=False,
    normalize_by_mean=False,
    cmap="turbo",
    vmin=0.01,
    vmax=0.99,
    figsize=(8, 4),
):
    """
    Plot the symmetry orders
    """

    if symmetry_orders is None:
        symmetry_orders = np.arange(1, self.annular_symmetry_max + 1)
    else:
        symmetry_orders = np.array(symmetry_orders)

    # plotting image
    if plot_std:
        im_plot = np.std(
            self.annular_symmetry,
            axis=(0, 1),
        )[symmetry_orders - 1, :]
    else:
        im_plot = np.mean(
            self.annular_symmetry,
            axis=(0, 1),
        )[symmetry_orders - 1, :]
    if normalize_by_mean:
        im_plot /= self.radial_mean[None, :]

    # plotting range
    int_vals = np.sort(im_plot.ravel())
    ind0 = np.clip(np.round(im_plot.size * vmin).astype("int"), 0, im_plot.size - 1)
    ind1 = np.clip(np.round(im_plot.size * vmax).astype("int"), 0, im_plot.size - 1)
    vmin = int_vals[ind0]
    vmax = int_vals[ind1]

    # plot image
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        im_plot,
        aspect="auto",
        extent=[
            self.qq[0],
            self.qq[-1],
            np.max(symmetry_orders),
            np.min(symmetry_orders) - 1,
        ],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yticks(
        symmetry_orders - 0.5,
        symmetry_orders,
    )
    ax.set_xlabel("Scattering angle (1/A)")
    ax.set_ylabel("Symmetry Order")


def scattering_model(k2, *coefs):
    """
    The scattering model used to fit the PDF background. The fit
    function is a constant plus two exponentials - one in k^2 and one
    in k^4:

        f(k; c,i0,s0,i1,s1) =
            c + i0*exp(k^2/-2*s0^2)  + i1*exp(k^4/-2*s1^4)

    Parameters
    ----------
    k2 : 1d array
        the scattering vector squared
    coefs : 5-tuple
        Initial guesses at the parameters (c,i0,s0,i1,s1)
    """
    coefs = np.squeeze(np.array(coefs))

    const_bg = coefs[0]
    int0 = coefs[1]
    sigma0 = coefs[2]
    int1 = coefs[3]
    sigma1 = coefs[4]

    int_model = (
        const_bg
        + int0 * np.exp(k2 / (-2 * sigma0**2))
        + int1 * np.exp(k2**2 / (-2 * sigma1**4))
    )

    # (int1*sigma1)/(k2 + sigma1**2)
    # int1*np.exp(k2/(-2*sigma1**2))
    # int1*np.exp(k2/(-2*sigma1**2))

    return int_model


def background_pca(
    self,
    pca_index: int = 0,
    n_components: int = None,
    intensity_range: Tuple[float, float] = (0, 1),
    normalize_mean: bool = True,
    normalize_std: bool = True,
    plot_result: bool = True,
    plot_coef: bool = False,
):
    """
    Generate PCA decompositions of the background signal.
    This function must be run after `calculate_radial_statistics`.

    Parameters
    --------
    pca_index: int
        index of PCA component and loadings to return
    intensity_range: tuple (float, float)
        intensity range for plotting
    normalize_mean: bool
        if True, normalize mean of radial data before PCA
    normalize_std: bool
        if True, normalize standard deviation of radial data before PCA
    plot_results: bool
        if True, plot results
    plot_coef: bool
        if True, plot radial PCA component selected

    Returns
    --------
    im_pca: np,array
        rgb image array
    coef_pca: np.array
        radial PCA component selected
    """

    # PCA decomposition
    shape = self.radial_all.shape
    A = np.reshape(self.radial_all, (shape[0] * shape[1], shape[2]))
    if normalize_mean:
        A -= np.mean(A, axis=0)
    if normalize_std:
        A /= np.std(A, axis=0)

    pca = PCA(n_components=np.maximum(pca_index + 1, 2))
    pca.fit(A)

    components = pca.components_
    loadings = pca.transform(A)

    # output image data
    sig_pca = np.reshape(loadings[:, pca_index], shape[0:2])
    sig_pca -= intensity_range[0]
    sig_pca /= intensity_range[1] - intensity_range[0]
    sig_pca = np.clip(sig_pca, 0, 1)
    im_pca = np.tile(sig_pca[:, :, None], (1, 1, 3))

    # output PCA coefficient
    coef_pca = np.vstack((self.radial_bins, components[pca_index, :])).T

    if plot_result:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            im_pca,
            vmin=0,
            vmax=5,
        )
    if plot_coef:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(coef_pca[:, 0], coef_pca[:, 1])

    return im_pca, coef_pca
