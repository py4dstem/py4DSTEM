# Analysis scripts for amorphous 4D-STEM data using polar transformations.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from emdfile import tqdmnd


def calculate_radial_statistics(
    self,
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
        self.radial_all[rx, ry] = np.mean(self.data[rx, ry], axis=0)
        self.radial_all_std[rx, ry] = np.sqrt(
            np.mean((self.data[rx, ry] - self.radial_all[rx, ry][None]) ** 2, axis=0)
        )

    self.radial_mean = np.mean(self.radial_all, axis=(0, 1))
    self.radial_var = np.mean(
        (self.radial_all - self.radial_mean[None, None]) ** 2, axis=(0, 1)
    )

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
    figsize=(8, 6),
    returnfig=False,
):
    """
    Calculate fluctuation electron microscopy (FEM) statistics, including radial mean,
    variance, and normalized variance. This function computes the radial average and variance
    for each individual probe position, which can then be mapped over the field-of-view.

    Parameters
    --------
    self: PolarDatacube
        Polar datacube used for measuring FEM properties.

    Returns
    --------
    radial_avg: np.array
        Average radial intensity
    radial_var: np.array
        Variance in the radial dimension


    """

    pass


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
