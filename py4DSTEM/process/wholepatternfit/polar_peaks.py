"""
This sub-module contains functions for polar transform peak detection of amorphous / semicrystalline datasets.

"""

from py4DSTEM import tqdmnd
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from itertools import product
from typing import Optional
from matplotlib.colors import hsv_to_rgb

import time, sys
import numpy as np
import matplotlib.pyplot as plt



class PolarPeaks:
    """
    Primary class for polar transform peak detection.
    """


    def __init__(
        self,
        datacube,
        radial_min = 0.0,
        radial_max = None,
        radial_step = 1.0,
        num_annular_bins = 180,
        progress_bar = True,
        ):
        """
        Initialize class by performing an intensity-preserving polar transformation.

        Parameters
        --------
        datacube: py4DSTEM.io.DataCube
            4D-STEM dataset, requires origin calibration
        radial_min: float
            Minimum radius of polar transformation.
        radial_max: float
            Maximum radius of polar transformation.
        radial_step: float
            Width of radial bins of polar transformation.
        num_annular_bins: int
            Number of bins in annular direction. Note that we fold data over 
            180 degrees periodically, so setting this value to 60 gives bin 
            widths of 180/60 = 3.0 degrees.
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        

        """

        # radial bin coordinates
        if radial_max is None:
            radial_max = np.min(datacube.Qshape) / np.sqrt(2)
        self.radial_bins = np.arange(
            radial_min,
            radial_max,
            radial_step,
            )
        self.radial_step = np.array(radial_step)

        # annular bin coordinates
        self.annular_bins = np.linspace(
            0,
            np.pi,
            num_annular_bins,
            endpoint = False,
            )
        self.annular_step = self.annular_bins[1] - self.annular_bins[0]

        # init polar transformation array
        self.polar_shape = np.array((self.annular_bins.shape[0], self.radial_bins.shape[0]))
        self.polar_size = np.prod(self.polar_shape)
        self.data_polar = np.zeros((
            datacube.R_Nx,
            datacube.R_Ny,
            self.polar_shape[0],
            self.polar_shape[1],
            ))

        # init coordinates
        xa, ya = np.meshgrid(
            np.arange(datacube.Q_Nx),
            np.arange(datacube.Q_Ny),
            indexing = 'ij',
        )

        # polar transformation
        for rx, ry in tqdmnd(
            range(datacube.R_Nx),
            range(datacube.R_Ny),
            desc="polar transformation",
            unit=" images",
            disable=not progress_bar,
            ):

            # shifted coordinates
            x = xa - datacube.calibration.get_qx0(rx,ry)
            y = ya - datacube.calibration.get_qy0(rx,ry)

            # polar coordinate indices
            r_ind = (np.sqrt(x**2 + y**2) - self.radial_bins[0]) / self.radial_step
            t_ind = np.arctan2(y, x) / self.annular_step
            r_ind_floor = np.floor(r_ind).astype('int')
            t_ind_floor = np.floor(t_ind).astype('int')
            dr = r_ind - r_ind_floor
            dt = t_ind - t_ind_floor
            # t_ind_floor = np.mod(t_ind_floor, self.num_annular_bins)

            # polar transformation
            sub = np.logical_and(r_ind_floor >= 0, r_ind_floor < self.polar_shape[1])
            im = np.bincount(
                r_ind_floor[sub] + \
                np.mod(t_ind_floor[sub],self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (1 - dr[sub]) * (1 - dt[sub]),
                minlength = self.polar_size,
            )
            im += np.bincount(
                r_ind_floor[sub] + \
                np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (1 - dr[sub]) * (    dt[sub]),
                minlength = self.polar_size,
            )
            sub = np.logical_and(r_ind_floor >= -1, r_ind_floor < self.polar_shape[1]-1)
            im += np.bincount(
                r_ind_floor[sub] + 1 + \
                np.mod(t_ind_floor[sub],self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (    dr[sub]) * (1 - dt[sub]),
                minlength = self.polar_size,
            )
            im += np.bincount(
                r_ind_floor[sub] + 1 + \
                np.mod(t_ind_floor[sub] + 1,self.polar_shape[0]) * self.polar_shape[1],
                weights = datacube.data[rx,ry][sub] * (    dr[sub]) * (    dt[sub]),
                minlength = self.polar_size,
            )

            # output
            self.data_polar[rx,ry] = np.reshape(im, self.polar_shape)




    def fit_peaks(
        self,
        num_peaks_fit = 1,
        sigma_radial_pixels = 0.0,
        sigma_annular_degrees = 1.0,
        progress_bar = True,
        ):
        """
        Fit both background signal and peak positions and intensities for each radial bin.

        Parameters
        --------
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        

        """

        # sigma in pixels
        self._sigma_radial_px = sigma_radial_pixels / self.radial_step
        self._sigma_annular_px = np.deg2rad(sigma_annular_degrees) / self.annular_step

        # init
        self.radial_median = np.zeros((
            self.data_polar.shape[0],
            self.data_polar.shape[1],
            self.polar_shape[1],
            ))
        self.radial_peaks = np.zeros((
            self.data_polar.shape[0],
            self.data_polar.shape[1],
            self.polar_shape[1],
            num_peaks_fit,
            2,
            ))


        # loop over probe positions
        for rx, ry in tqdmnd(
            self.data_polar.shape[0],
            self.data_polar.shape[1],
            desc="polar transformation",
            unit=" positions",
            disable=not progress_bar,
            ):

            im = gaussian_filter(
                self.data_polar[rx,ry],
                sigma = (self._sigma_annular_px, self._sigma_radial_px),
                mode = ('wrap', 'nearest'),
                truncate = 3.0,
            )

            # background signal
            self.radial_median[rx,ry] = np.median(im,axis=0)

            # local maxima
            sub_peaks = np.logical_and(
                im > np.roll(im,-1,axis=0),
                im > np.roll(im, 1,axis=0),
            )

            for a0 in range(self.polar_shape[1]):
                inds = np.atleast_1d(np.squeeze(np.argwhere(sub_peaks[:,a0])))
                if inds.size > 0:
                    vals = im[inds,a0]

                    inds_sort = np.argsort(vals)[::-1]
                    inds_keep = inds_sort[:num_peaks_fit]

                    peaks_val = np.maximum(vals[inds_keep] - self.radial_median[rx,ry,a0], 0)
                    peaks_ind = inds[inds_keep]
                    peaks_angle = self.annular_bins[peaks_ind]

                    # TODO - add subpixel peak fitting?

                    # output
                    num_peaks = peaks_val.shape[0]
                    self.radial_peaks[rx,ry,a0,:num_peaks,0] = peaks_angle
                    self.radial_peaks[rx,ry,a0,:num_peaks,1] = peaks_val

    def plot_peak_signal(
        self,
        figsize = (8,4),
        ):
        """
        Plot the mean peak signal from all radial bins.

        """

        sig = np.mean(
            np.sum(
                self.radial_peaks[:,:,:,:,1], axis = 3,
                ), axis = (0,1),
            )

        # plot
        fig = plt.figure(figsize = figsize)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        ax1.plot(
            self.radial_bins,
            sig,
            )
        ax1.set_xlabel('Radial Bin [pixels]')
        ax1.set_ylabel('Mean Peak Signal [counts]')

        def tick_function(x):
            v = (x - self.radial_bins[0]) / self.radial_step
            return ["%.0f" % z for z in v]
        tick_locations = ax1.get_xticks()
        tick_locations += self.radial_bins[0]
        ax2.set_xticks(tick_locations)
        ax2.set_xticklabels(tick_function(tick_locations))
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xlabel('Radial Bin Index')
        ax2.minorticks_on()
        ax2.grid()

        plt.show()



        # fig,ax = plt.subplots(figsize = figsize)
        # ax.plot(
        #     self.radial_bins,
        #     sig,
        #     )

        


        #         fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax2 = ax1.twiny()

        # X = np.linspace(0,1,1000)
        # Y = np.cos(X*20)

        # ax1.plot(X,Y)
        # ax1.set_xlabel(r"Original x-axis: $X$")

        # new_tick_locations = np.array([.2, .5, .9])

        # def tick_function(X):
        #     V = 1/(1+X)
        #     return ["%.3f" % z for z in V]

        # ax2.set_xlim(ax1.get_xlim())
        # ax2.set_xticks(new_tick_locations)
        # ax2.set_xticklabels(tick_function(new_tick_locations))
        # ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
        # plt.show()




    def orientation_map(
        self,
        radial_index = 0,
        peak_index = 0,
        intensity_range = (0,1),
        plot_result = True,
        ):
        """
        Create an RGB orientation map from a given peak bin

        Parameters
        --------
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        im_orientation: np.array
            rgb image array

        """

        # intensity mask
        val = np.squeeze(self.radial_peaks[:,:,radial_index,peak_index,1]).copy()
        val -= intensity_range[0]
        val /= intensity_range[1] - intensity_range[0]
        val = np.clip(val,0,1)

        # orientation
        hue = np.squeeze(self.radial_peaks[:,:,radial_index,peak_index,0]).copy()
        hue = np.mod(2*hue,1)


        # generate image
        im_orientation = np.ones((
            self.data_polar.shape[0],
            self.data_polar.shape[1],
            3))
        im_orientation[:,:,0] = hue
        im_orientation[:,:,2] = val
        im_orientation = hsv_to_rgb(im_orientation)

        if plot_result:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(
                im_orientation,
                vmin = 0,
                vmax = 5,
            )
        # ax.plot(
        #     im[:,6]
        #     )

        return im_orientation


    def cluster_grains(
        self,
        radial_index = 0,
        threshold_add = 1.0,
        threshold_grow = 0.0,
        angle_tolerance_deg = 5.0,
        progress_bar = False,
        ):
        """
        Cluster grains from a specific radial bin

        Parameters
        --------
        radial_index: int
            Which radial bin to perform the clustering over.
        threshold_add: float
            Minimum signal required for a probe position to initialize a cluster.
        threshold_grow: float
            Minimum signal required for a probe position to be added to a cluster.
        angle_tolerance_deg: float
            Rotation rolerance for clustering grains.
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        

        """

        # Get data
        phi = np.squeeze(self.radial_peaks[:,:,radial_index,:,0]).copy()
        sig = np.squeeze(self.radial_peaks[:,:,radial_index,:,1]).copy()
        sig_init = sig.copy()
        mark = sig >= threshold_grow
        sig[np.logical_not(mark)] = 0
        
        # init
        self.cluster_sizes = np.array((), dtype='int')
        self.cluster_sig = np.array(())
        self.cluster_inds = []
        inds_all = np.zeros_like(sig, dtype='int')
        inds_all.ravel()[:] = np.arange(inds_all.size)

        # Tolerance
        tol = np.deg2rad(angle_tolerance_deg)

        # Main loop
        search = True
        while search is True:
            inds_grain = np.argmax(sig)
            val = sig.ravel()[inds_grain]

            if val < threshold_add:
                search = False
            
            else:
                # progressbar
                comp = 1 - np.mean(np.max(mark,axis = 2))
                update_progress(comp)

                # Start cluster
                x,y,z = np.unravel_index(inds_grain, sig.shape)
                mark[x,y,z] = False
                sig[x,y,z] = 0
                phi_cluster = phi[x,y,z]

                # Neighbors to search
                xr = np.clip(x + np.arange(-1,2,dtype='int'), 0, sig.shape[0] - 1)
                yr = np.clip(y + np.arange(-1,2,dtype='int'), 0, sig.shape[1] - 1)
                inds_cand = inds_all[xr[:,None],yr[None],:].ravel()
                inds_cand = np.delete(inds_cand, mark.ravel()[inds_cand] == False)

                if inds_cand.size == 0:
                    grow = False
                else:
                    grow = True

                # grow the cluster
                while grow is True:
                    inds_new = np.array((),dtype='int')

                    keep = np.zeros(inds_cand.size, dtype='bool')
                    for a0 in range(inds_cand.size):
                        xc,yc,zc = np.unravel_index(inds_cand[a0], sig.shape)

                        phi_test = phi[xc,yc,zc]
                        dphi = np.mod(phi_cluster - phi_test + np.pi/2.0, np.pi) - np.pi/2.0

                        if np.abs(dphi) < tol:
                            keep[a0] = True

                            sig[xc,yc,zc] = 0
                            mark[xc,yc,zc] = False 

                            xr = np.clip(xc + np.arange(-1,2,dtype='int'), 0, sig.shape[0] - 1)
                            yr = np.clip(yc + np.arange(-1,2,dtype='int'), 0, sig.shape[1] - 1)
                            inds_add = inds_all[xr[:,None],yr[None],:].ravel()
                            inds_new = np.append(inds_new, inds_add)


                    inds_grain = np.append(inds_grain, inds_cand[keep])
                    inds_cand = np.unique(np.delete(inds_new, mark.ravel()[inds_new] == False))

                    if inds_cand.size == 0:
                        grow = False

            # convert grain to x,y coordinates, add = list
            xg,yg,zg = np.unravel_index(inds_grain, sig.shape)
            xyg = np.unique(np.vstack((xg,yg)), axis = 1)
            sig_mean = np.mean(sig_init.ravel()[inds_grain])
            self.cluster_sizes = np.append(self.cluster_sizes, xyg.shape[1])
            self.cluster_sig = np.append(self.cluster_sig, sig_mean)
            self.cluster_inds.append(xyg)

        # finish progressbar
        update_progress(1)

    def cluster_plot_size(
        self,
        area_max = None,
        weight_intensity = False,
        pixel_area = 1.0,
        pixel_area_units = 'px^2',
        figsize = (8,6),
        returnfig = False,
        ):
        """
        Plot the cluster sizes

        Parameters
        --------
        area_max: int (optional)
            Max area bin in pixels
        weight_intensity: bool
            Weight histogram by the peak intensity.
        pixel_area: float
            Size of pixel area unit square
        pixel_area_units: string
            Units of the pixel area 
        figsize: tuple
            Size of the figure panel
        returnfig: bool
            Setting this to true returns the figure and axis handles
    
        Returns
        --------
        fig, ax (optional)
            Figure and axes handles

        """

        if area_max is None:
            area_max = np.max(self.cluster_sizes)
        area = np.arange(area_max)
        sub = self.cluster_sizes.astype('int') < area_max
        if weight_intensity:
            hist = np.bincount(
                self.cluster_sizes[sub],
                weights = self.cluster_sig[sub],
                minlength = area_max,
                )
        else:
            hist = np.bincount(
                self.cluster_sizes[sub],
                minlength = area_max,
                )


        # plotting
        fig,ax = plt.subplots(figsize = figsize)
        ax.bar(
            area * pixel_area,
            hist,
            width = 0.8 * pixel_area,
            )

        ax.set_xlabel('Grain Area [' + pixel_area_units + ']')
        if weight_intensity:
            ax.set_ylabel('Total Signal [arb. units]')
        else:
            ax.set_ylabel('Number of Grains')

        if returnfig:
            return fig,ax


    def background_pca(
        self,
        pca_index = 0,
        intensity_range = (0,1),
        normalize_mean = True,
        normalize_std = True,
        plot_result = True,
        plot_coef = False,
        ):
        """
        Generate PCA decompositions of the background signal

        Parameters
        --------
        progress_bar: bool
            Turns on the progress bar for the polar transformation
    
        Returns
        --------
        im_pca: np,array
            rgb image array
        coef_pca: np.array
            radial PCA component selected

        """

        # PCA decomposition
        shape = self.radial_median.shape
        A = np.reshape(self.radial_median, (shape[0]*shape[1],shape[2]))
        if normalize_mean:
            A -= np.mean(A,axis=0)
        if normalize_std:
            A /= np.std(A,axis=0)
        pca = PCA(n_components=np.maximum(pca_index+1,2))
        pca.fit(A)

        components = pca.components_
        loadings = pca.transform(A)

        # output image data
        sig_pca = np.reshape(loadings[:,pca_index], shape[0:2])
        sig_pca -= intensity_range[0]
        sig_pca /= intensity_range[1] - intensity_range[0]
        sig_pca = np.clip(sig_pca,0,1)
        im_pca = np.tile(sig_pca[:,:,None],(1,1,3))

        # output PCA coefficient
        coef_pca = np.vstack((
            self.radial_bins,
            components[pca_index,:]
        )).T

        if plot_result:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(
                im_pca,
                vmin = 0,
                vmax = 5,
            )
        if plot_coef:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(
                coef_pca[:,0],
                coef_pca[:,1]
                )

        return im_pca, coef_pca


# Progressbar taken from stackexchange:
# https://stackoverflow.com/questions/3160699/python-progress-bar
def update_progress(progress):
    barLength = 60 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), 
        np.round(progress*100,4), 
        status)
    sys.stdout.write(text)
    sys.stdout.flush()
