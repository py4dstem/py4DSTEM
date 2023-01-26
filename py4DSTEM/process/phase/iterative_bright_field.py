"""
Module for reconstructing virtual bright field images by aligning each pixel.
"""

import matplotlib.pyplot as plt
import numpy as np
from py4DSTEM import show
from py4DSTEM.utils.tqdmnd import tqdmnd

class BFreconstruction():
    """
    A class for reconstructing aligned virtual bright field images.

    """

    def __init__(
        self,
        dataset,
        threshold_intensity = 0.5,
        padding = (64,64),
        edge_blend = 8,
        progress_bar = True,
        ):

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

        # padding mask
        # self.padding_mask = np.zeros((
        #     dataset.data.shape[0] + padding[0],
        #     dataset.data.shape[1] + padding[1]),
        #     dtype='bool')
        # self.padding_mask[:padding[0]//2,:] = True
        # self.padding_mask[:,:padding[0]//2] = True
        # self.padding_mask[dataset.data.shape[0] + padding[0]//2:,:] = True
        # self.padding_mask[:,dataset.data.shape[0] + padding[0]//2:] = True

        # init and populate virtual image array
        self.stack_BF = np.zeros((
            self.num_images,
            dataset.data.shape[0] + padding[0],
            dataset.data.shape[1] + padding[1],
        ))
        for a0 in tqdmnd(
            self.num_images,
            desc="Getting BF pixels",
            unit=" images",
            disable=not progress_bar,
            ):
            int_mean = np.mean(dataset.data[:,:,self.xy_inds[a0,0],self.xy_inds[a0,1]])
            # self.stack_BF[a0,:,:][self.padding_mask] = int_mean

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
                ] = self.window_inv * int_mean \
                + dataset.data[:,:,self.xy_inds[a0,0],self.xy_inds[a0,1]] * self.window_edge \

        # test plotting
        # show(
        #     self.window_edge,
        #     intensity_range='absolute',
        #     vmin=0,
        #     vmax=1,
        # )        
        # show(np.mean(self.stack_BF,axis=0))