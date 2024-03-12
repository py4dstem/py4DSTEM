import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np



def bin2D(array, factor, dtype=np.float64):
    """
    Bin a 2D ndarray by binfactor.

    Parameters
    ----------
    array : 2D numpy array
    factor : int
        the binning factor
    dtype : numpy dtype
        datatype for binned array. default is numpy default for np.zeros()

    Returns
    -------
    the binned array
    """
    x, y = array.shape
    binx, biny = x // factor, y // factor
    xx, yy = binx * factor, biny * factor

    # Make a binned array on the device
    binned_ar = np.zeros((binx, biny), dtype=dtype)
    array = array.astype(dtype)

    # Collect pixel sums into new bins
    for ix in range(factor):
        for iy in range(factor):
            binned_ar += array[0 + ix : xx + ix : factor, 0 + iy : yy + iy : factor]
    return binned_ar



