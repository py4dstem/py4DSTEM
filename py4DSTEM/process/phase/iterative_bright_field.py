"""
Module for reconstructing virtual bright field images by aligning each virtual BF image.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.special import comb
from scipy.ndimage import gaussian_filter

from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.process.utils import get_shifted_ar, get_shift
from py4DSTEM.process.utils.get_maxima_2D import get_maxima_2D
from py4DSTEM.process.utils.multicorr import upsampled_correlation
from py4DSTEM.process.utils.utils import electron_wavelength_angstrom
from py4DSTEM.visualize import show


class BFreconstruction():
    """
    A class for reconstructing aligned virtual bright field images.

    """

    def __init__(
        self,
        dataset,
        accelerating_voltage_kV = 300e3,
        threshold_intensity = 0.8,
        normalize_images = True,
        padding = (64,64),
        edge_blend = 16,
        initial_align = True,
        subpixel = 'multicorr',
        upsample_factor = 8,
        plot_recon = True,
        progress_bar = True,
        ):

        # store parameters
        self.threshold_intensity = threshold_intensity
        self.padding = padding
        self.edge_blend = edge_blend
        self.calibration = dataset.calibration

        # get mean diffraction pattern
        if 'dp_mean' not in dataset.tree.keys():
            self.dp_mean = dataset.get_dp_mean().data
        else:
            self.dp_mean = dataset.tree['dp_mean'].data

        # make sure mean diffraction pattern is shaped correctly
        if (self.dp_mean.shape[0] != dataset.shape[2]) \
            or (self.dp_mean.shape[1] != dataset.shape[2]):
            self.dp_mean = dataset.get_dp_mean().data

        # select virtual detector pixels
        self.dp_mask = self.dp_mean >= (np.max(self.dp_mean) * threshold_intensity)
        self.num_images = np.count_nonzero(self.dp_mask)

        # wavelength
        self.accelerating_voltage_kV = accelerating_voltage_kV
        self.wavelength = electron_wavelength_angstrom(self.accelerating_voltage_kV)

        # diffraction space coordinates
        self.xy_inds = np.argwhere(self.dp_mask)
        if self.calibration.get_Q_pixel_units() == 'A^-1':
            self.kxy = (self.xy_inds - np.mean(
                self.xy_inds,axis=0)[None,:]) * dataset.calibration.get_Q_pixel_size()
            self.probe_angles = self.kxy * self.wavelength

        elif self.calibration.get_Q_pixel_units() == 'mrad':
            self.probe_angles = (self.xy_inds - np.mean(
                self.xy_inds,axis=0)[None,:]) * dataset.calibration.get_Q_pixel_size() / 1000
            self.kxy = self.probe_angles / self.wavelength
        else:
            raise Exception('diffraction space pixel size must be A^-1 or mrad')

        self.kr = np.sqrt(np.sum(self.kxy**2,axis=1))

        # Window function
        x = np.linspace(-1,1,dataset.data.shape[0] + 1)
        x = x[1:]
        x -= (x[1] - x[0]) / 2
        wx = np.sin(np.clip((1 - np.abs(x)) * dataset.data.shape[0] / edge_blend / 2,0,1) * (np.pi/2))**2
        y = np.linspace(-1,1,dataset.data.shape[1] + 1)
        y = y[1:]
        y -= (y[1] - y[0]) / 2
        wy = np.sin(np.clip((1 - np.abs(y)) * dataset.data.shape[1] / edge_blend / 2,0,1) * (np.pi/2))**2
        self.window_edge = wx[:,None] * wy[None,:]
        self.window_inv = 1 - self.window_edge
        self.window_pad = np.zeros((
            dataset.data.shape[0] + padding[0],
            dataset.data.shape[1] + padding[1]))
        self.window_pad[
            padding[0]//2:dataset.data.shape[0] + padding[0]//2,
            padding[1]//2:dataset.data.shape[1] + padding[1]//2] = self.window_edge

        # init virtual image array and mask
        self.stack_BF = np.zeros((
            self.num_images,
            dataset.data.shape[0] + padding[0],
            dataset.data.shape[1] + padding[1],
        ), dtype='float')
        self.stack_mask = np.tile(self.window_pad[None,:,:],(self.num_images,1,1))

        # populate image array
        for a0 in tqdmnd(
            self.num_images,
            desc="Getting BF images",
            unit=" images",
            disable=not progress_bar,
            ):
            im = dataset.data[:,:,self.xy_inds[a0,0],self.xy_inds[a0,1]].copy().astype('float')

            if normalize_images:
                im /= np.mean(im)
                int_mean = 1.0
            else:
                int_mean = np.mean(im)

            self.stack_BF[a0,:padding[0]//2,:] = int_mean
            self.stack_BF[a0,padding[0]//2:,:padding[1]//2] = int_mean
            self.stack_BF[a0,padding[0]//2:,dataset.data.shape[1] + padding[1]//2:] = int_mean
            self.stack_BF[a0,
                dataset.data.shape[0] + padding[0]//2:,
                padding[1]//2:dataset.data.shape[1] + padding[1]//2] = int_mean
            self.stack_BF[
                a0,
                padding[0]//2:dataset.data.shape[0] + padding[0]//2,
                padding[1]//2:dataset.data.shape[1] + padding[1]//2,
                ] = self.window_inv * int_mean + self.window_edge * im

        # initial image shifts, mean BF image, and error
        self.xy_shifts = np.zeros((self.num_images,2))
        self.stack_mean = np.mean(self.stack_BF)
        self.mask_sum = np.sum(self.window_edge) * self.num_images
        # self.recon_BF = np.mean(self.stack_BF, axis=0)
        # self.recon_error = np.atleast_1d(np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:])))
        self.recon_mask = np.sum(self.stack_mask, axis=0)
        mask_inv = 1 - np.clip(self.recon_mask,0,1)
        self.recon_BF = (self.stack_mean * mask_inv \
            + np.sum(self.stack_BF * self.stack_mask, axis=0)) \
            / (self.recon_mask + mask_inv)
        self.recon_error = np.atleast_1d(
            np.sum(np.abs(self.stack_BF - self.recon_BF[None,:,:]) * self.stack_mask)) / self.mask_sum

        # Fourier space operators for image shifts
        qx = np.fft.fftfreq(self.stack_BF.shape[1], d=1)
        qy = np.fft.fftfreq(self.stack_BF.shape[2], d=1)
        qxa,qya = np.meshgrid(qx,qy,indexing='ij')
        self.qx_shift = -2j*np.pi*qxa
        self.qy_shift = -2j*np.pi*qya

        # initial image alignment
        if initial_align:
            inds_order = np.argsort(self.kr)
            G_ref = np.fft.fft2(self.stack_BF[inds_order[0]])

            # Loop over all images
            for a0 in tqdmnd(
                range(1,self.num_images),
                desc="Initial alignment",
                unit=" images",
                disable=not progress_bar,
                ):
                ind = inds_order[a0]
                G = np.fft.fft2(self.stack_BF[ind])
                
                # Get subpixel shifts
                xy_shift = align_images(
                    G_ref, 
                    G,
                    upsample_factor = upsample_factor)
                dx = np.mod(xy_shift[0] + self.stack_BF.shape[1]/2,
                    self.stack_BF.shape[1]) - self.stack_BF.shape[1]/2
                dy = np.mod(xy_shift[1] + self.stack_BF.shape[2]/2,
                    self.stack_BF.shape[2]) - self.stack_BF.shape[2]/2

                # apply shifts
                self.xy_shifts[ind,0] += dx
                self.xy_shifts[ind,1] += dy

                # shift image
                shift_op = np.exp(self.qx_shift * dx + self.qy_shift * dy)
                G_shift = G * shift_op
                self.stack_BF[ind] = np.real(np.fft.ifft2(G_shift))
                self.stack_mask[ind] = np.real(np.fft.ifft2(
                    np.fft.fft2(self.stack_mask[ind]) * shift_op))

                # running average for reference image
                G_ref = G_ref * (a0-1)/a0 + G_shift

            # Center the shifts
            xy_shifts_median = np.round(np.median(self.xy_shifts, axis = 0)).astype(int)
            self.xy_shifts -= xy_shifts_median[None,:]
            self.stack_BF = np.roll(self.stack_BF, -xy_shifts_median, axis=(1,2))
            self.stack_mask = np.roll(self.stack_mask, -xy_shifts_median, axis=(1,2))

            # if alignment was performed, update error
            # self.recon_BF = np.mean(self.stack_BF, axis=0)
            # self.recon_error = np.append(self.recon_error,
            #     np.atleast_1d(np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:]))))
            self.recon_mask = np.sum(self.stack_mask, axis=0)
            mask_inv = 1 - np.clip(self.recon_mask,0,1)
            self.recon_BF = (self.stack_mean * mask_inv \
                + np.sum(self.stack_BF * self.stack_mask, axis=0)) \
                / (self.recon_mask + mask_inv)
            self.recon_error = np.append(self.recon_error,
                np.sum(np.abs(self.stack_BF - self.recon_BF[None,:,:]) * self.stack_mask) / self.mask_sum)
    
    def align_image(
        self,
        num_iter = 1,
        subpixel = 'multicorr',
        upsample_factor = 8,
        regularize_shifts = True,
        regularize_size = (1,1),
        plot_stats = True,
        plot_recon = True,
        progress_bar = True,
        ):
        """
        Iterative alignment of the BF images.
        """

        # construct regularization basis if needed
        if regularize_shifts:
            if regularize_size[0] == 1 and regularize_size[1] == 1:
                basis = self.kxy
            else:
                kr_max = np.max(self.kr)
                u = self.kxy[:,0] * 0.5 / kr_max + 0.5
                v = self.kxy[:,1] * 0.5 / kr_max + 0.5
                u[0] = 0
                v[0] = 1

                basis = np.zeros((self.num_images,(regularize_size[0]+1)*(regularize_size[1]+1)))
                for ii in np.arange(regularize_size[0]+1):
                    Bi = comb(regularize_size[0],ii) * (u**ii) * ((1-u)**(regularize_size[0]-ii))

                    for jj in np.arange(regularize_size[1]+1):
                        Bj = comb(regularize_size[1],jj) * (v**jj) * ((1-v)**(regularize_size[1]-jj))

                        ind = ii*(regularize_size[1]+1) + jj
                        basis[:,ind] = Bi * Bj

        # Loop over iterations
        for a0 in tqdmnd(
            num_iter,
            desc="Aligning BF images",
            unit=" iterations",
            disable=not progress_bar,
            ):

            # Reference image
            G_ref = np.fft.fft2(self.recon_BF)

            # align images
            if regularize_shifts:
                shifts_update = np.zeros((self.num_images,2))

                for a1 in range(self.num_images):
                    G = np.fft.fft2(self.stack_BF[a1])

                    # Get subpixel shifts
                    xy_shift = align_images(
                        G_ref, 
                        G,
                        upsample_factor = upsample_factor)
                    dx = np.mod(xy_shift[0] + self.stack_BF.shape[1]/2,
                        self.stack_BF.shape[1]) - self.stack_BF.shape[1]/2
                    dy = np.mod(xy_shift[1] + self.stack_BF.shape[2]/2,
                        self.stack_BF.shape[2]) - self.stack_BF.shape[2]/2

                    # apply shifts
                    shifts_update[a1,0] = dx
                    shifts_update[a1,1] = dy

                # Calculate regularized shifts
                xy_shifts_new = self.xy_shifts + shifts_update
                coefs = np.linalg.lstsq(basis, xy_shifts_new, rcond=None)[0]
                xy_shifts_fit = basis @ coefs
                shifts_update = xy_shifts_fit - self.xy_shifts

                # Apply shifts
                for a1 in range(self.num_images):
                    G = np.fft.fft2(self.stack_BF[a1])

                    dx = shifts_update[a1,0]
                    dy = shifts_update[a1,1]
                    self.xy_shifts[a1,0] += dx
                    self.xy_shifts[a1,1] += dy

                    # shift image
                    shift_op = np.exp(self.qx_shift * dx + self.qy_shift * dy)
                    self.stack_BF[a1] = np.real(np.fft.ifft2(G * shift_op))
                    self.stack_mask[a1] = np.real(np.fft.ifft2(
                        np.fft.fft2(self.stack_mask[a1]) * shift_op))

            else:
                for a1 in range(self.num_images):
                    G = np.fft.fft2(self.stack_BF[a1])

                    # Get subpixel shifts
                    xy_shift = align_images(
                        G_ref, 
                        G,
                        upsample_factor = upsample_factor)
                    dx = np.mod(xy_shift[0] + self.stack_BF.shape[1]/2,
                        self.stack_BF.shape[1]) - self.stack_BF.shape[1]/2
                    dy = np.mod(xy_shift[1] + self.stack_BF.shape[2]/2,
                        self.stack_BF.shape[2]) - self.stack_BF.shape[2]/2

                    # apply shifts
                    self.xy_shifts[a1,0] += dx
                    self.xy_shifts[a1,1] += dy

                    # shift image
                    shift_op = np.exp(self.qx_shift * dx + self.qy_shift * dy)
                    self.stack_BF[a1] = np.real(np.fft.ifft2(G * shift_op))
                    self.stack_mask[a1] = np.real(np.fft.ifft2(
                        np.fft.fft2(self.stack_mask[a1]) * shift_op))

            # # Center the shifts
            # xy_shifts_median = np.round(np.median(self.xy_shifts, axis = 0)).astype(int)
            # self.xy_shifts -= xy_shifts_median[None,:]
            # self.stack_BF = np.roll(self.stack_BF, -xy_shifts_median, axis=(1,2))
            # self.stack_mask = np.roll(self.stack_mask, -xy_shifts_median, axis=(1,2))

            # update reconstruction and error
            # self.recon_BF = np.median(self.stack_BF, axis=0)
            # self.recon_error = np.append(self.recon_error,
            #     np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:])))
            self.recon_mask = np.sum(self.stack_mask, axis=0)
            mask_inv = 1 - np.clip(self.recon_mask,0,1)
            self.recon_BF = (self.stack_mean * mask_inv \
                + np.sum(self.stack_BF * self.stack_mask, axis=0)) \
                / (self.recon_mask + mask_inv)
            self.recon_error = np.append(self.recon_error,
                np.sum(np.abs(self.stack_BF - self.recon_BF[None,:,:]) * self.stack_mask) / self.mask_sum)

        # plot convergence
        if plot_stats:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(
                np.arange(self.recon_error.size),
                self.recon_error,
            )
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Error')

    def plot_recon(
        self,
        bound = None,
        figsize=(8,8),
        ):

        # Image boundaries
        if bound == 0:
            im = self.recon_BF
        else:
            if bound is None:
                bound = (self.padding[0]//2, self.padding[1]//2)
            else:
                if np.array(bound).ndim == 0:
                    bound = (bound, bound)
                bound = np.array(bound)
            im = self.recon_BF[bound[0]:-bound[0],bound[1]:-bound[1]]
        
        show(
            im,
            figsize=figsize,
        )

    def plot_shifts(
        self,
        scale_arrows = 0.002,
        figsize=(8,8),
        ):

        fig, ax = plt.subplots(figsize=figsize)

        ax.quiver(
            self.kxy[:,1],
            self.kxy[:,0],
            self.xy_shifts[:,1] * scale_arrows,
            self.xy_shifts[:,0] * scale_arrows,
            color = (1,0,0,1),
            angles='xy', 
            scale_units='xy', 
            scale=1,
        )

        kr_max = np.max(self.kr)
        ax.set_xlim([-1.2*kr_max, 1.2*kr_max])
        ax.set_ylim([-1.2*kr_max, 1.2*kr_max])

    def aberration_fit(
        self,
        print_result = True,
        plot_CTF_compare = False,
        plot_dk = 0.001,
        plot_k_sigma = 0.002,
        ):
        '''
        Fit aberrations to the measured image shifts.
        '''

        # Convert real space shifts to Angstroms
        self.xy_shifts_Ang = self.xy_shifts * self.calibration.get_R_pixel_size()

        # Solve affine transformation
        m = np.linalg.lstsq(self.probe_angles, self.xy_shifts_Ang, rcond=None)[0]        
        m_rotation, m_aberration = sp.linalg.polar(m, side='right')
        # m_test = m_rotation @ m_aberration

        # Convert into rotation and aberration coefficients
        self.rotation_Q_to_R_rads = -1*np.arctan2(m_rotation[1,0],m_rotation[0,0])
        self.aberration_C1  = (m_aberration[0,0] + m_aberration[1,1]) / 2.0 
        self.aberration_A1x = (m_aberration[0,0] - m_aberration[1,1]) / 2.0 # factor /2 for A1 astigmatism? /4?
        self.aberration_A1y = (m_aberration[1,0] + m_aberration[0,1]) / 2.0

        # Print results
        if print_result:
            print('Rotation of Q w.r.t. R = ' \
                + str(np.round(np.rad2deg(self.rotation_Q_to_R_rads),decimals=3)) \
                + ' deg')
            print('Astigmatism (A1x,A1y)  = (' \
                + str(np.round(self.aberration_A1x,decimals=0)) \
                + ','
                + str(np.round(self.aberration_A1y,decimals=0)) \
                + ') Ang')
            print('Defocus C1             = ' \
                + str(np.round(self.aberration_C1,decimals=0)) \
                + ' Ang')


        # Plot the CTF comparison between experiment and fit
        if plot_CTF_compare:
            # Get polar mean from FFT of BF reconstruction
            im_fft = np.abs(np.fft.fft2(self.recon_BF))

            # coordinates
            kx = np.fft.fftfreq(self.recon_BF.shape[0], self.calibration.get_R_pixel_size())
            ky = np.fft.fftfreq(self.recon_BF.shape[1], self.calibration.get_R_pixel_size())
            kra = np.sqrt(kx[:,None]**2 + ky[None,:]**2)
            k_max = np.max(kra) / np.sqrt(2.0)
            k_num_bins = np.ceil(k_max / plot_dk).astype('int')
            k_bins = np.arange(k_num_bins+1) * plot_dk

            # histogram
            k_ind = kra / plot_dk
            kf = np.floor(k_ind).astype('int')
            dk = k_ind - kf
            sub = kf <= k_num_bins
            hist_exp = np.bincount(
                kf[sub], 
                weights = im_fft[sub] * (1-dk[sub]),
                minlength = k_num_bins)
            hist_norm = np.bincount(
                kf[sub], 
                weights = (1-dk[sub]),
                minlength = k_num_bins)
            sub = kf <= k_num_bins - 1
            hist_exp += np.bincount(
                kf[sub] + 1, 
                weights = im_fft[sub] * ( dk[sub]),
                minlength = k_num_bins)
            hist_norm += np.bincount(
                kf[sub] + 1, 
                weights = (  dk[sub]),
                minlength = k_num_bins)

            # KDE and normalizing
            k_sigma = plot_dk / plot_k_sigma
            hist_exp[0] = 0.0
            hist_exp = gaussian_filter(hist_exp, sigma = k_sigma, mode='nearest')
            hist_norm = gaussian_filter(hist_norm, sigma = k_sigma, mode='nearest')
            hist_exp /= hist_norm

            # CTF comparison
            CTF_fit = np.sin((-np.pi * self.wavelength * self.aberration_C1) * k_bins**2)

            # plotting
            hist_plot = hist_exp * k_bins
            hist_plot /= np.max(hist_plot)
            # ind = np.argmax(hist_exp * k_bins**0.7)
            # hist_plot = hist_exp / hist_exp[ind]

            # CTF_plot = np.abs(CTF_fit) * k_bins
            # CTF_plot /= np.max(CTF_plot)


            fig,ax = plt.subplots(figsize = (8,4))
            ax.fill_between(
                k_bins,
                hist_plot,
                color=(0.7,0.7,0.7,1),
                )
            ax.plot(
                k_bins,
                np.clip(CTF_fit, 0.0, np.inf),
                color=(1,0,0,1),
                linewidth=2,
                )
            ax.plot(
                k_bins,
                np.clip(-CTF_fit, 0.0, np.inf),
                color=(0,0.5,1,1),
                linewidth=2,
                )
            ax.set_xlim([0, k_bins[-1]])
            ax.set_ylim([0, 1.05])



    def aberration_correct(
        self,
        k_info_limit = None,
        k_info_power = 2.0,
        plot_result = True,
        figsize = (8,8),
        plot_range = (-2,2),
        returnval = False,
        ):
        '''
        CTF correction of the BF image using the measured defocus aberration.
        '''

        # Fourier coordinates, aberration surface
        kx = np.fft.fftfreq(self.recon_BF.shape[0], self.calibration.get_R_pixel_size())
        ky = np.fft.fftfreq(self.recon_BF.shape[1], self.calibration.get_R_pixel_size())
        kra2 = kx[:,None]**2 + ky[None,:]**2
        CTF = np.sin((-np.pi*self.wavelength*self.aberration_C1) * kra2)
        CTF_corr = np.sign(CTF)

        # if needed, make low pass filter
        if k_info_limit is not None:
            k_filt = 1/(1 + (kra2**k_info_power)/((k_info_limit)**(2*k_info_power)))

        # apply correction to reconstructed BF image
        if k_info_limit is not None:
            self.recon_BF_corr = np.real(np.fft.ifft2(np.fft.fft2(self.recon_BF) * CTF_corr * k_filt))
        else:
            self.recon_BF_corr = np.real(np.fft.ifft2(np.fft.fft2(self.recon_BF) * CTF_corr))

        # sub = self.kxy[:,0] < -0.0


        # im_test = np.mean(self.stack_BF[sub,:,:],axis=0)


        # plotting
        if plot_result:
            im_plot = self.recon_BF_corr.copy()
            im_plot -= np.mean(im_plot)
            im_plot /= np.sqrt(np.mean(im_plot**2))

            fig,ax = plt.subplots(figsize = figsize)
            ax.imshow(
                im_plot[ \
                    self.padding[0]//2:self.recon_BF.shape[0]-self.padding[0]//2,
                    self.padding[1]//2:self.recon_BF.shape[1]-self.padding[1]//2],
                vmin=plot_range[0],
                vmax=plot_range[1],
                cmap = 'gray',
                )




        # np.abs(np.fft.fftshift(CTF)),


        # print(np.mean(sub))




def align_images(
    G1, 
    G2,
    upsample_factor,
    ):
    '''
    Alignment of two images using DFT upsampling of cross correlation.

    Returns: xy_shift [pixels]
    '''

    # cross correlation
    cc = G1 * np.conj(G2)
    cc_real = np.real(np.fft.ifft2(cc))

    # local max
    x0, y0 = np.unravel_index(cc_real.argmax(), cc.shape)

    # half pixel shifts
    x_inds = np.mod(x0 + np.arange(-1,2), cc.shape[0]).astype('int')
    y_inds = np.mod(y0 + np.arange(-1,2), cc.shape[0]).astype('int')
    vx = cc_real[x_inds,y0]
    vy = cc_real[x0,y_inds]
    dx = (vx[2]-vx[0])/(4*vx[1]-2*vx[2]-2*vx[0])
    dy = (vy[2]-vy[0])/(4*vy[1]-2*vy[2]-2*vy[0])
    x0 = np.round((x0 + dx)*2.0)/2.0
    y0 = np.round((y0 + dy)*2.0)/2.0

    # subpixel shifts
    xy_shift = upsampled_correlation(cc,upsample_factor,np.array((x0,y0)))

    return xy_shift
    
    
