# Functions for generating diffraction images

import numpy as np
from ...io import DataCube

def get_max_dp(datacube):
    """
    Returns the maximal value of each diffraction space detector pixel, over the
    entire dataset.

    Args:
        datacube (Datacube)

    Returns:
        (2D array): the maximal diffraction pattern
    """
    assert isinstance(datacube, DataCube)
    max_dp = np.max(datacube.data, axis=(0,1))
    return max_dp


