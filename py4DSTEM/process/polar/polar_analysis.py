# Analysis scripts for amorphous 4D-STEM data using polar transformations.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import comb, erf
from scipy.ndimage import gaussian_filter

from emdfile import tqdmnd


def calculate_radial_statistics(
    self,
    median_local = False,
    median_global = False,
    plot_results_mean = False,
    plot_results_var = False,
    figsize = (8,4),
    returnval = False,
    returnfig = False,
    progress_bar = True,
    ):
    """
    Calculate fluctuation electron microscopy (FEM) statistics, including radial mean,
    variance, and normalized variance. This function uses the original FEM definitions,
    where the signal is computed pattern-by-pattern.

    TODO - finish docstrings, add median statistics.

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

    # Get the dimensioned radial bins
    self.scattering_vector = (
        self.radial_bins * self.qstep * self.calibration.get_Q_pixel_size()
    )
    self.scattering_vector_units = self.calibration.get_Q_pixel_units()

    # init radial data arrays
    self.radial_all = np.zeros((
        self._datacube.shape[0],
        self._datacube.shape[1],
        self.polar_shape[1],
    ))
    self.radial_all_std = np.zeros((
        self._datacube.shape[0],
        self._datacube.shape[1],
        self.polar_shape[1],
    ))

    # Compute the radial mean for each probe position
    for rx, ry in tqdmnd(
        self._datacube.shape[0],
        self._datacube.shape[1],
        desc="Radial statistics",
        unit=" probe positions",
        disable=not progress_bar):
        
        self.radial_all[rx,ry] = np.mean(
            self.data[rx,ry],
            axis=0)
        self.radial_all_std[rx,ry] = np.sqrt(np.mean(
            (self.data[rx,ry] - self.radial_all[rx,ry][None])**2, 
            axis=0))

    self.radial_mean = np.mean(self.radial_all, axis=(0,1))
    self.radial_var = np.mean(
        (self.radial_all - self.radial_mean[None,None])**2,
        axis=(0,1))

    self.radial_var_norm = self.radial_var 
    sub = self.radial_mean > 0.0
    self.radial_var_norm[sub] /= self.radial_mean[sub]**2

    # plot results
    if plot_results_mean:
        if returnfig:
            fig,ax = plot_radial_mean(
                self,
                figsize=figsize,
                returnfig=True,
            )
        else:
            plot_radial_mean(
                self,
                figsize = figsize,
                )
    elif plot_results_var:
        if returnfig:
            fig,ax = plot_radial_var_norm(
                self,
                figsize = figsize,
                returnfig = True,
                )
        else:
            plot_radial_var_norm(
                self,
                figsize = figsize,
                )

    # Return values
    if returnval:
        if returnfig:
            return self.radial_mean, self.radial_var, fig, ax
        else:
            return self.radial_mean, self.radial_var
    else:
        if returnfig:
            return fig, ax
        else:
            pass


def plot_radial_mean(
    self,
    log_x = False,
    log_y = False,
    figsize = (8,4),
    returnfig = False,
    ):
    """
    Plot radial mean
    """
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.scattering_vector,
        self.radial_mean,
        )

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel('Scattering Vector (' + self.scattering_vector_units + ')')
    ax.set_ylabel('Radial Mean')
    if log_x and self.scattering_vector[0] == 0.0:
        ax.set_xlim((self.scattering_vector[1],self.scattering_vector[-1]))
    else:
        ax.set_xlim((self.scattering_vector[0],self.scattering_vector[-1]))

    if returnfig:
        return fig, ax


def plot_radial_var_norm(
    self,
    figsize = (8,4),
    returnfig = False,
    ):
    """
    Plotting function for the global FEM.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.scattering_vector,
        self.radial_var_norm,
    )

    ax.set_xlabel("Scattering Vector (" + self.scattering_vector_units + ")")
    ax.set_ylabel("Normalized Variance")
    ax.set_xlim((self.scattering_vector[0], self.scattering_vector[-1]))

    if returnfig:
        return fig, ax


def calculate_pair_dist_function(
    self,
    k_min = 0.05,
    k_max = None,
    k_width = 0.25,
    k_lowpass = None,
    k_highpass = None,
    # k_pad_max = 10.0,
    r_min = 0.0,
    r_max = 20.0,
    r_step = 0.02,
    damp_origin_fluctuations = False,
    # poly_background_order = 2,
    # iterative_pdf_refine = True,
    # num_iter = 10,
    dens = None,
    plot_fits = False,
    plot_sf_estimate = False,
    plot_reduced_pdf = True,
    plot_pdf = False,
    figsize = (8,4),
    maxfev = None,
    ):
    """
    Calculate the pair distribution function (PDF).

    """

    # init
    k = self.scattering_vector
    dk = k[1] - k[0]
    k2 = k**2
    Ik = self.radial_mean
    int_mean = np.mean(Ik)
    sub_fit = k >= k_min

    # initial coefs
    const_bg = np.min(self.radial_mean) / int_mean
    int0 = np.median(self.radial_mean) / int_mean - const_bg
    sigma0 = np.mean(k)
    coefs = [const_bg, int0, sigma0, int0, sigma0]
    lb = [0,0,0,0,0]
    ub = [np.inf, np.inf, np.inf, np.inf, np.inf]
    # Weight the fit towards high k values
    noise_est = k[-1] - k + dk

    # Estimate the mean atomic form factor + background
    if maxfev is None:
        coefs = curve_fit(
            scattering_model, 
            k2[sub_fit], 
            Ik[sub_fit] / int_mean, 
            sigma = noise_est[sub_fit],
            p0=coefs,
            xtol = 1e-8,
            bounds = (lb,ub),
        )[0]
    else:
        coefs = curve_fit(
            scattering_model, 
            k2[sub_fit], 
            Ik[sub_fit] / int_mean, 
            sigma = noise_est[sub_fit],
            p0=coefs,
            xtol = 1e-8,
            bounds = (lb,ub),
            maxfev = maxfev,
        )[0]

    coefs[0] *= int_mean
    coefs[1] *= int_mean
    coefs[3] *= int_mean


    # Calculate the mean atomic form factor wthout any background
    coefs_fk = (0.0, coefs[1], coefs[2], coefs[3], coefs[4])
    fk = scattering_model(k2, coefs_fk)
    bg = scattering_model(k2, coefs)

    # mask for structure factor estimate
    if k_max is None:
        k_max = np.max(k)
    # mask = np.clip(np.minimum(
    #     (k - k_min) / k_width,
    #     (k_max - k) / k_width,
    #     ),0,1)
    mask = np.clip(np.minimum(
        (k - 0.0) / k_width,
        (k_max - k) / k_width,
        ),0,1)
    mask = np.sin(mask*(np.pi/2))

    # Estimate the reduced structure factor S(k)
    Sk = (Ik - bg) * k / fk

    # Masking edges of S(k)
    mask_sum = np.sum(mask)
    Sk = (Sk - np.sum(Sk*mask)/mask_sum) * mask

    # Filtering of S(k)
    if k_lowpass is not None and k_lowpass > 0.0:
        Sk = gaussian_filter(
            Sk, 
            sigma=k_lowpass / dk,
            mode = 'nearest')
    if k_highpass is not None:
        Sk_lowpass = gaussian_filter(
            Sk, 
            sigma=k_highpass / dk,
            mode = 'nearest')
        Sk -= Sk_lowpass

    # Calculate the real space PDF
    r = np.arange(r_min, r_max, r_step)
    ra,ka = np.meshgrid(r,k)
    pdf_reduced = (2/np.pi)*dk*np.sum(
        np.sin(
            2*np.pi*ra*ka
        ) * Sk[:,None],
        axis=0,
    )

    # Damp the unphysical fluctuations at the PDF origin
    if damp_origin_fluctuations:
        ind_max = np.argmax(pdf_reduced)
        r_ind_max = r[ind_max]
        r_mask = np.minimum(r / r_ind_max, 1.0)
        r_mask = np.sin(r_mask*np.pi/2)**2
        pdf_reduced *= r_mask

    # Store results
    self.pdf_r = r
    self.pdf_reduced = pdf_reduced

    # if density is provided, we can estimate the full PDF
    if dens is not None:
        pdf = pdf_reduced.copy()
        pdf[1:] /= (4*np.pi*dens*r[1:]*(r[1]-r[0]))
        pdf *= (2/np.pi)
        pdf += 1

        if damp_origin_fluctuations:
            pdf *= r_mask

        pdf = np.maximum(pdf, 0.0)



    # Plots
    if plot_fits:
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(
            self.scattering_vector,
            self.radial_mean,
            color = 'k',
            )
        ax.plot(
            k,
            bg,
            color = 'r',
            )
        ax.set_xlabel('Scattering Vector (' + self.scattering_vector_units + ')')
        ax.set_ylabel('Radial Mean')
        ax.set_xlim((self.scattering_vector[0],self.scattering_vector[-1]))
        # ax.set_ylim((0,2e-5))
        ax.set_xlabel('Scattering Vector [A^-1]')
        ax.set_ylabel('I(k) and Fit Estimates')

        ax.set_ylim((np.min(self.radial_mean[self.radial_mean>0])*0.8,
            np.max(self.radial_mean*mask)*1.25))
        ax.set_yscale('log')

    if plot_sf_estimate:
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(
            k,
            Sk,
            color = 'r',
            )
        yr = (np.min(Sk),np.max(Sk))
        ax.set_ylim((
            yr[0]-0.05*(yr[1]-yr[0]),
            yr[1]+0.05*(yr[1]-yr[0]),
            ))
        ax.set_xlabel('Scattering Vector [A^-1]')
        ax.set_ylabel('Reduced Structure Factor')

    if plot_reduced_pdf:
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(
            r,
            pdf_reduced,
            color = 'r',
            )
        ax.set_xlabel('Radius [A]')
        ax.set_ylabel('Reduced Pair Distribution Function')

    if plot_pdf:
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(
            r,
            pdf,
            color = 'r',
            )
        ax.set_xlabel('Radius [A]')
        ax.set_ylabel('Pair Distribution Function')



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
    coefs = np.squeeze(np.array(coefs))

    const_bg = coefs[0]
    int0 = coefs[1]
    sigma0 = coefs[2]
    int1 = coefs[3]
    sigma1 = coefs[4]

    int_model = const_bg + \
        int0*np.exp(k2/(-2*sigma0**2)) + \
        int1*np.exp(k2**2/(-2*sigma1**4))

        # (int1*sigma1)/(k2 + sigma1**2)
        # int1*np.exp(k2/(-2*sigma1**2))
        # int1*np.exp(k2/(-2*sigma1**2))


    return int_model

