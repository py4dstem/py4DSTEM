# Analysis scripts for amorphous 4D-STEM data using polar transformations.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


from emdfile import tqdmnd


def calculate_radial_statistics(
    self,
    median_local = False,
    median_global = False,
    plot_results = False,
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
    self.scattering_vector = self.radial_bins * self.qstep * self.calibration.get_Q_pixel_size()
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
    if plot_results:
        if returnfig:
            fig,ax = plot_FEM_global(
                self,
                figsize = figsize,
                returnfig = True,
                )
        else:
            plot_FEM_global(
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
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(
        self.scattering_vector,
        self.radial_var_norm,
        )

    ax.set_xlabel('Scattering Vector (' + self.scattering_vector_units + ')')
    ax.set_ylabel('Normalized Variance')
    ax.set_xlim((self.scattering_vector[0],self.scattering_vector[-1]))

    if returnfig:
        return fig, ax


def calculate_pair_dist_function(
    self,
    k_min = 0.05,
    k_max = None,
    k_width = 0.25,
    # k_pad_max = 10.0,
    r_min = 0.0,
    r_max = 20.0,
    r_step = 0.02,
    # iterative_pdf_refine = True,
    num_iter = 10,
    plot_fits = False,
    plot_sf_estimate = False,
    plot_pdf = True,
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
    const_bg = np.min(self.radial_mean)
    int0 = np.median(self.radial_mean) - const_bg
    sigma0 = np.mean(k)
    coefs = [const_bg, int0, sigma0, int0, sigma0]
    lb = [0,0,0,0,0]
    ub = [np.inf, np.inf, np.inf, np.inf, np.inf]
    # noise_est = 1/k
    # noise_est = np.divide(1.0, k, out=np.zeros_like(k), where=k!=0)
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
    mask = np.clip(np.minimum(
        (k - k_min) / k_width,
        (k_max - k) / k_width,
        ),0,1)
    mask = np.sin(mask*(np.pi/2))**2

    # Estimate the reduced structure factor S(k)
    Sk = (Ik - bg) * k / fk
    mask_sum = np.sum(mask)
    Sk = (Sk - np.sum(Sk*mask)/mask_sum) * mask

    # # pad or crop S(k) to 0 and k_pad_max
    # k_pad = np.arange(0, k_pad_max, dk)
    # Sk_pad = np.zeros_like(k_pad)
    # ind_0 = np.argmin(np.abs(k_pad-k[0]))
    # ind_max = ind_0 + k.size
    # if ind_max > k_pad.size:
    #     Sk_pad[ind_0:] = Sk[ind_0:k_pad.size]
    # else:
    #     Sk_pad[ind_0:ind_max] = Sk

    # Calculate the real space PDF
    # dr = 1/(2*k_pad[-1])
    r = np.arange(r_min, r_max, r_step)
    ra,ka = np.meshgrid(r,k)
    pdf = (2/np.pi)*np.pi*dk*np.sum(
        np.sin(
            2*np.pi*ra*ka
        ) * Sk[:,None],
        axis=0,
    )

    # invert

    ind_max = np.argmax(pdf)
    r_ind_max = r[ind_max]
    r_mask = np.minimum(r / r_ind_max, 1.0)
    r_mask = np.sin(r_mask*np.pi/2)**2

    Sk_back_proj = (2*r_step)*np.sum(
        np.sin(
            2*np.pi*ra*ka
        ) * pdf[None,:] * r_mask[None,:],
        axis=1,
    )

    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(
        k,
        Sk,
        color = 'k',
        )
    ax.plot(
        k,
        Sk_back_proj,
        color = 'r',
        )



    # # iterative refinement of the PDF
    # if iterative_pdf_refine:
    #     # pdf = np.maximum(pdf + (r/r[-1]), 0.0)

    #     ind_max = np.argmax(pdf)
    #     r_ind_max = r[ind_max]
    #     r_mask = np.minimum(r / r_ind_max, 1.0)
    #     r_mask = np.sin(r_mask*np.pi/2)**2

    #     pdf = np.maximum(pdf * r_mask + (r/r[-1]), 0.0)
    #     r_weight = r_mask * (1 - r / r[-1])**2



    #     # basis = np.vstack((np.ones_like(r),r)).T
    #     # coefs_lin = np.linalg.lstsq(basis, pdf, rcond=None)[0]
    #     # pdf_lin = basis * coefs_lin
    #     # print(coefs_lin)


    #     for a0 in range(10):
    #         Sk_back_proj = (1*np.pi/r.size)*np.sum(
    #             np.sin(
    #                 2*np.pi*ra*ka
    #             ) * pdf[None,:],
    #             axis=1,
    #         )

    #         Sk_diff = Sk - Sk_back_proj
    #         Sk_diff = (Sk_diff - np.mean(Sk_diff*mask)/mask_sum) * mask

    #         pdf_update = 4*np.pi*dk*np.sum(
    #             np.sin(
    #                 8*np.pi*ra*ka
    #             ) * Sk_diff[:,None],
    #             axis=0,
    #         ) * r_weight

    #         pdf = np.maximum(pdf + 0.5*pdf_update, 0.0)

    #     fig,ax = plt.subplots(figsize=figsize)
    #     ax.plot(
    #         r,
    #         pdf,
    #         color = 'k',
    #         )
    #     # ax.plot(
    #     #     r,
    #     #     pdf_lin,
    #     #     color = 'r',
    #     #     )

    #     # ax.plot(
    #     #     r,
    #     #     pdf + pdf_update,
    #     #     color = 'r',
    #     #     )

    #     # ax.plot(
    #     #     k,
    #     #     Sk,
    #     #     color = 'k',
    #     #     )
    #     # ax.plot(
    #     #     k,
    #     #     Sk_back_proj,
    #     #     color = 'r',
    #     #     )
    #     # ax.plot(
    #     #     Sk_diff,
    #     #     color = 'r',
    #     #     )


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
            np.ones(k.size)*coefs[0],
            color = 'r',
            )
        ax.plot(
            k,
            fk + coefs[0],
            color = 'r',
            )
        ax.set_xlabel('Scattering Vector (' + self.scattering_vector_units + ')')
        ax.set_ylabel('Radial Mean')
        ax.set_xlim((self.scattering_vector[0],self.scattering_vector[-1]))
        ax.set_ylim((0,2e-5))
        ax.set_xlabel('Scattering Vector [A^-1]')
        ax.set_ylabel('I(k) and Fit Estimates')


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
        ax.set_ylabel('Structure Factor')

    if plot_pdf:
        fig,ax = plt.subplots(figsize=figsize)
        ax.plot(
            r,
            pdf,
            color = 'r',
            )
        ax.set_xlabel('Radius [A]')
        ax.set_ylabel('Pair Distribution Function')
        # r = (np.min(Sk),np.max(Sk))
        # ax.set_ylim((
        #     r[0]-0.05*(r[1]-r[0]),
        #     r[1]+0.05*(r[1]-r[0]),
        #     ))


        # ax.set_yscale('log')






def calculate_FEM_local(
    self,
    figsize = (8,6),
    returnfig = False,
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
        (int1*sigma1)**2/(k2 + sigma1**2)

    return int_model

