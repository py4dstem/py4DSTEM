# Functions for generating diffraction images

import numpy as np

def get_dp(datacube,where=(0,0)):
    """
    Returns a single diffraction pattern from the specified scan position.

    Args:
        datacube (DataCube)
        where (2-tuple of ints): the scan position

    Returns:
        (2D array): the diffraction pattern
    """
    assert(len(where)==2 and all([isinstance(i,(int,np.integer)) for i in where])), "Scan position was specified incorrectly, must be a pair of integers"
    assert(where[0]<datacube.R_Nx and where[1]<datacube.R_Ny), "The requested scan position is outside the dataset"
    return datacube.data[where[0],where[1],:,:]

def get_max_dp(datacube,where=None):
    """
    Returns the maximal value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        where (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the max dp over.  If `where` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the maximal diffraction pattern
    """
    if where is not None:
        if isinstance(where,np.ndarray):
            assert(where.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            where = where.astype(bool)
            max_dp = np.max(datacube.data[where,:,:], axis=(0))
        else:
            assert(len(where)==4 and all([isinstance(i,(int,np.integer)) for i in where])), "`where` must be a boolean array or a length 4 tuple of ints"
            max_dp = np.max(
                datacube.data[where[0]:where[1],where[2]:where[3],:,:],
                axis=(0,1))
    else:
        max_dp = np.max(datacube.data, axis=(0,1))
    return max_dp

def get_mean_dp(datacube,where=None):
    """
    Returns the mean value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        where (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the mean dp over.  If `where` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the mean diffraction pattern
    """
    if where is not None:
        if isinstance(where,np.ndarray):
            assert(where.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            where = where.astype(bool)
            mean_dp = np.mean(datacube.data[where,:,:], axis=(0))
        else:
            assert(len(where)==4 and all([isinstance(i,(int,np.integer)) for i in where])), "`where` must be a boolean array or a length 4 tuple of ints"
            mean_dp = np.mean(
                datacube.data[where[0]:where[1],where[2]:where[3],:,:],
                axis=(0,1))
    else:
        mean_dp = np.mean(datacube.data, axis=(0,1))
    return mean_dp

def get_median_dp(datacube,where=None):
    """
    Returns the median value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        where (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the median dp over.  If `where` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the median diffraction pattern
    """
    if where is not None:
        if isinstance(where,np.ndarray):
            assert(where.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            where = where.astype(bool)
            median_dp = np.median(datacube.data[where,:,:], axis=(0))
        else:
            assert(len(where)==4 and all([isinstance(i,(int,np.integer)) for i in where])), "`where` must be a boolean array or a length 4 tuple of ints"
            median_dp = np.median(
                datacube.data[where[0]:where[1],where[2]:where[3],:,:],
                axis=(0,1))
    else:
        median_dp = np.median(datacube.data, axis=(0,1))
    return median_dp


