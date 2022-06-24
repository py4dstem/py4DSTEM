# Functions for generating diffraction images

import numpy as np

def get_dp_max(datacube,region=None):
    """
    Returns the maximal value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        region (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the max dp over.  If `region` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the maximal diffraction pattern
    """
    # whole dataset
    if region is None:
        dp_max = np.max(datacube.data, axis=(0,1))

    # subset of data
    else:
        if isinstance(region,np.ndarray):
            assert(region.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            region = region.astype(bool)
            dp_max = np.max(datacube.data[region,:,:], axis=(0))
        else:
            assert(len(region)==4 and all([isinstance(i,(int,np.integer)) for i in region])), "`where` must be a boolean array or a length 4 tuple of ints"
            dp_max = np.max(
                datacube.data[region[0]:region[1],region[2]:region[3],:,:],
                axis=(0,1))

    # add to tree and return
    datacube.tree['dp_max'] = dp_max
    return dp_max

def get_dp_mean(datacube,region=None):
    """
    Returns the mean value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        region (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the mean dp over.  If `region` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the mean diffraction pattern
    """
    # whole dataset
    if region is None:
        dp_mean = np.mean(datacube.data, axis=(0,1))

    # subset of data
    else:
        if isinstance(region,np.ndarray):
            assert(region.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            region = region.astype(bool)
            dp_mean = np.mean(datacube.data[region,:,:], axis=(0))
        else:
            assert(len(region)==4 and all([isinstance(i,(int,np.integer)) for i in region])), "`region` must be a boolean array or a length 4 tuple of ints"
            dp_mean = np.mean(
                datacube.data[region[0]:region[1],region[2]:region[3],:,:],
                axis=(0,1))

    # add to tree and return
    datacube.tree['dp_mean'] = dp_mean
    return dp_mean

def get_dp_median(datacube,region=None):
    """
    Returns the median value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        region (None, or 4-tuple of ints, or boolean array): specifies which
            diffraction patterns to compute the median dp over.  If `region` is
            None, uses the whole datase. If it is a 4-tuple, uses a rectangular
            region (xi,xf,yi,yf) of real space. If if is a boolean mask, the mask
            shape must match real space, and only True pixels are used.

    Returns:
        (2D array): the median diffraction pattern
    """
    # whole dataset
    if region is None:
        dp_median = np.median(datacube.data, axis=(0,1))

    # subset of data
    else:
        if isinstance(region,np.ndarray):
            assert(region.shape==(datacube.R_Nx,datacube.R_Ny)), "mask must have the same shape as real space"
            region = region.astype(bool)
            dp_median = np.median(datacube.data[region,:,:], axis=(0))
        else:
            assert(len(region)==4 and all([isinstance(i,(int,np.integer)) for i in region])), "`region` must be a boolean array or a length 4 tuple of ints"
            dp_median = np.median(
                datacube.data[region[0]:region[1],region[2]:region[3],:,:],
                axis=(0,1))

    # add to tree and return
    datacube.tree['dp_median'] = dp_median
    return dp_median




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




