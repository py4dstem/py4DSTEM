"""
This sub-module contains functions for polar transform peak detection of amorphous / semicrystalline datasets.

"""

from py4DSTEM import tqdmnd
from scipy.ndimage import gaussian_filter
from itertools import product
from typing import Optional

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
                inds = np.squeeze(np.argwhere(sub_peaks[:,a0]))
                vals = im[inds,a0]

                inds_sort = np.argsort(vals)[::-1]
                inds_keep = inds_sort[:num_peaks_fit]

                peaks_val = np.maximum(vals[inds_keep] - self.radial_median[rx,ry,a0], 0)
                peaks_ind = inds[inds_keep]
                peaks_angle = self.annular_bins[peaks_ind]

                # TODO - add subpixel peak fitting?

                # output
                self.radial_peaks[rx,ry,a0,:,0] = peaks_angle
                self.radial_peaks[rx,ry,a0,:,1] = peaks_val





        # fig, ax = plt.subplots(figsize=(12,8))
        # ax.imshow(
        #     np.hstack((
        #         im,
        #         sub_peaks * im,
        #     )),
        #     vmin = 0,
        #     vmax = 5,
        #     )
        # ax.plot(
        #     im[:,6]
        #     )
