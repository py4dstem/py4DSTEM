# Functions for generating diffraction images

import numpy as np

def get_max_dp(datacube):
    """
    Returns the maximal value of each diffraction space detector pixel, over the
    entire dataset.

    Args:
        datacube (Datacube)

    Returns:
        (2D array): the maximal diffraction pattern
    """
    max_dp = np.max(datacube.data, axis=(0,1))
    return max_dp


