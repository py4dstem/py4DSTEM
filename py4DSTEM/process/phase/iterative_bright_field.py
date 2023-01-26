"""
Module for reconstructing virtual bright field images by aligning each pixel.
"""

import matplotlib.pyplot as plt
import numpy as np
from py4DSTEM import show


class BFreconstruction():
    """
    A class for reconstructing aligned virtual bright field images.

    """

    def __init__(
        self,
        dataset,
        threshold_intensity = 0.5,
        padding = (128,128),
        edge_blend = 4,
        ):

        # Get mean diffraction pattern
        
        self.dp_mean = dataset.get_dp_mean().data

        # Select virtual detector pixels
        self.dp_mask = self.dp_mean >= np.max(self.dp_mean) * threshold_intensity




        # test plotting
        # show(self.dp_mean)        
        show(self.dp_mask)