"""
Module for reconstructing virtual bright field images by aligning each virtual BF image.
"""

import matplotlib.pyplot as plt
import numpy as np
from py4DSTEM import show
from py4DSTEM.utils.tqdmnd import tqdmnd
from py4DSTEM.process.utils import get_shifted_ar, get_shift
from py4DSTEM.process.utils.get_maxima_2D import get_maxima_2D
# from py4DSTEM.process.utils.cross_correlate import get_cross_correlation_FT
from py4DSTEM.process.utils.multicorr import upsampled_correlation


class BFreconstruction():
    """
    A class for reconstructing aligned virtual bright field images.

    """

    def __init__(
        self,
        dataset,
        threshold_intensity = 0.5,
        normalize_images = True,
        padding = (128,128),
        edge_blend = 16,
        initial_align = True,
        subpixel = 'multicorr',
        upsample_factor = 8,
        progress_bar = True,
        ):

        # store parameters
        self.threshold_intensity = threshold_intensity
        self.padding = padding
        self.edge_blend = edge_blend

        # Get mean diffraction pattern
        if 'dp_mean' not in dataset.tree.keys():
            self.dp_mean = dataset.get_dp_mean().data
        else:
            self.dp_mean = dataset.tree['dp_mean'].data

        # Select virtual detector pixels
        self.dp_mask = self.dp_mean >= (np.max(self.dp_mean) * threshold_intensity)
        self.num_images = np.count_nonzero(self.dp_mask)

        # coordinates
        self.xy_inds = np.argwhere(self.dp_mask)
        self.kxy = (self.xy_inds - np.mean(self.xy_inds,axis=0)[None,:]) * dataset.calibration.get_Q_pixel_size()
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

        # init and populate virtual image array
        self.stack_BF = np.zeros((
            self.num_images,
            dataset.data.shape[0] + padding[0],
            dataset.data.shape[1] + padding[1],
        ))
        for a0 in tqdmnd(
            self.num_images,
            desc="Getting BF images",
            unit=" images",
            disable=not progress_bar,
            ):
            im = dataset.data[:,:,self.xy_inds[a0,0],self.xy_inds[a0,1]].copy()

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
                ] = self.window_inv * int_mean + self.window_edge * im \

        # initial image shifts, mean BF image, and error
        self.xy_shifts = np.zeros((self.num_images,2))
        self.recon_BF = np.mean(self.stack_BF, axis=0)
        self.recon_error = np.atleast_1d(np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:])))

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
                G_shift = G * np.exp(self.qx_shift * dx + self.qy_shift * dy)
                self.stack_BF[ind] = np.real(np.fft.ifft2(G_shift))

                # running average for reference image
                G_ref = G_ref * (a0-1)/a0 + G_shift

            # Center the shifts
            xy_shifts_median = np.round(np.median(self.xy_shifts, axis = 0)).astype(int)
            self.xy_shifts -= xy_shifts_median[None,:]
            self.stack_BF = np.roll(self.stack_BF, -xy_shifts_median, axis=(1,2))

            # if alignment perform, update error
            self.recon_BF = np.mean(self.stack_BF, axis=0)
            self.recon_error = np.append(self.recon_error,
                np.atleast_1d(np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:]))))

    
    def align_image(
        self,
        num_iter = 1,
        subpixel = 'multicorr',
        upsample_factor = 8,
        # max_shift = 4,
        # step_size = 0.9,
        plot_stats = True,
        plot_recon = True,
        progress_bar = True,
        ):
        """
        Iterative alignment of the BF images.
        """

        # Loop over iterations
        # for ind_iter, ind_image in tqdmnd(
        #     num_iter,
        #     self.num_images,
        #     desc="Aligning BF images",
        #     disable=not progress_bar,
        #     ):
            
        #     xshift,yshift = get_shift(probe, curr_DP)
        #     curr_DP = get_shifted_ar(curr_DP, xshift, yshift)

            #             unit=" iteration",

        for a0 in tqdmnd(
            num_iter,
            desc="Aligning BF images",
            unit=" iterations",
            disable=not progress_bar,
            ):

            # Reference image
            G_ref = np.fft.fft2(self.recon_BF)

            # align images
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
                self.stack_BF[a1] = np.real(np.fft.ifft2(
                    G * np.exp(self.qx_shift * dx + self.qy_shift * dy)))


                # G = np.fft.fft2(self.stack_BF[a1])

                # # Compute cross correlation
                # cc = G_ref * np.conj(G)

                # # Get maxima
                # maxima = get_maxima_2D(
                #     np.real(np.fft.ifft2(cc)),
                #     subpixel = subpixel,
                #     upsample_factor = upsample_factor,
                #     maxNumPeaks = 1,
                #     _ar_FT = cc,
                # )[0]

                # # Get subpixel shifts
                # dx = np.mod(maxima[0] + self.stack_BF.shape[1]/2,
                #     self.stack_BF.shape[1]) - self.stack_BF.shape[0]/2
                # dy = np.mod(maxima[1] + self.stack_BF.shape[2]/2,
                #     self.stack_BF.shape[2]) - self.stack_BF.shape[2]/2

                # # apply shifts
                # self.xy_shifts[a1,0] += dx
                # self.xy_shifts[a1,1] += dy

                # # shift output image
                # self.stack_BF[a1] = np.real(np.fft.ifft2(G * np.exp(
                #     self.qx_shift * dx + self.qy_shift * dy)))

            # Center the shifts
            xy_shifts_median = np.round(np.median(self.xy_shifts, axis = 0)).astype(int)
            self.xy_shifts -= xy_shifts_median[None,:]
            self.stack_BF = np.roll(self.stack_BF, -xy_shifts_median, axis=(1,2))

            # update reconstruction and error
            self.recon_BF = np.median(self.stack_BF, axis=0)
            self.recon_error = np.append(self.recon_error,
                np.mean(np.abs(self.stack_BF - self.recon_BF[None,:,:])))

        # plot convergence
        if plot_stats:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(
                np.arange(self.recon_error.size),
                self.recon_error,
            )
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Error')


            # fig,ax = plt.subplots(figsize=(8,8))
            # ax.imshow(np.real(np.fft.fftshift(np.fft.ifft2(cc))))

                # xshift,yshift = get_shift(probe, curr_DP)
                


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
    
    
