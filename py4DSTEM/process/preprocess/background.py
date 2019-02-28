# Functions for background fitting and subtraction.

import numpy as np
from ...file import log

#### Get background subtracted DPs ####

@log
def get_bksbtr_DP(datacube, background, Rx, Ry):
    """
    Returns a background subtracted diffraction pattern.

    Accepts:
        datacube        (DataCube) data to background subtract
        backround       (ndarray) background diffraction pattern,
                                  must have shape (datacube.Q_Nx, datacube.Q_Ny)
        Rx, Ry          (ints) the scan position of the diffraction pattern of interest

    Returns:
        bksbtr_DP       (ndarray) the background subtracted diffraction pattern
    """
    assert background.shape==(datacube.Q_Nx,datacube.Q_Ny), "background must have shape (datacube.Q_Nx, datacube.Q_Ny)"
    return datacube.data4D[Rx,Ry,:,:].astype(float) - background.astype(float)


#### Get background streaks ####

# 2D diffraction pattern shaped outputs

@log
def get_background_streaks(datacube, width, N_DPs, side='start', direction='x'):
    """
    Gets background streaking in either the x- or y-direction, by finding the average of a strip of
    pixels along the edge of the detector over a random selection of diffraction patterns.

    Note that the data is cast to float before computing the background, and should similarly be
    cast to float before performing a subtraction. This avoids integer clipping and wraparound
    errors.

    Accepts:
        datacube        (DataCube) data to background subtract
        width           (int) width of the ROI strip for background identification
        N_DPs           (int) number of random diffraction patterns to use
        side (optional) (str) use a strip from the start or end of the array. Must be 'start' or
                        'end', defaults to 'start'
        directions      (str) the direction of background streaks to find. Must be either 'x' or 'y'
                        defaults to 'x'

    Returns:
        bkgrnd_DP       (ndarray) a 2D ndarray of shape (datacube.Q_Nx, datacube.Ny) giving the
                        background.
    """
    assert ((direction=='x') or (direction=='y')), "direction must be 'x' or 'y'."
    if direction=='x':
        return get_background_streaks_x(datacube=datacube, width=width, N_DPs=N_DPs, side=side)
    else:
        return get_background_streaks_y(datacube=datacube, width=width, N_DPs=N_DPs, side=side)

@log
def get_background_streaks_x(datacube, width, N_DPs, side='start'):
    """
    Gets background streaking in the x-direction, by finding the average of a strip of
    pixels along the edge of the detector over a random selection of diffraction patterns.

    See the get_background_streaks docstring for more info.
    """
    bkgrnd_streaks_1D = get_background_streaks_1D_x(datacube, width, N_DPs, side)
    return streaks_to_DP_x(bkgrnd_streaks_1D, datacube.Q_Nx, datacube.Q_Ny)

@log
def get_background_streaks_y(datacube, width, N_DPs, side='start'):
    """
    Gets background streaking in the y-direction, by finding the average of a strip of
    pixels along the edge of the detector over a random selection of diffraction patterns.

    See the get_background_streaks docstring for more info.
    """
    bkgrnd_streaks_1D = get_background_streaks_1D_y(datacube, width, N_DPs, side)
    return streaks_to_DP_y(bkgrnd_streaks_1D, datacube.Q_Nx, datacube.Q_Ny)

# 1D array shaped outputs

@log
def get_background_streaks_1D(datacube, width, N_DPs, side='start', direction='x'):
    """
    Gets background streaking in either the x- or y-direction, by finding the average of a strip of
    pixels along the edge of the detector over a random selection of diffraction patterns, and
    returns a 1D array corresponding to the streak values along the rows or columns.

    Note that the data is cast to float before computing the background, and should similarly be
    cast to float before performing a subtraction. This avoids integer clipping and wraparound
    errors.

    Accepts:
        datacube        (DataCube) data to background subtract
        width           (int) width of the ROI strip for background identification
        N_DPs           (int) number of random diffraction patterns to use
        side (optional) (str) use a strip from the start or end of the array. Must be 'start' or
                        'end', defaults to 'start'
        directions      (str) the direction of background streaks to find. Must be either 'x' or 'y'
                        defaults to 'x'

    Returns:
        bkgrnd_streaks  (ndarray) a 1D ndarray of length datacube.Q_N or datacube.Q_Ny,
                        corresponding to the value of the x- or y-direction background streaks
                        at each row or column of the diffraction plane.
    """
    assert ((direction=='x') or (direction=='y')), "direction must be 'x' or 'y'."
    if direction=='x':
        return get_background_streaks_1D_x(datacube=datacube, width=width, N_DPs=N_DPs, side=side)
    else:
        return get_background_streaks_1D_y(datacube=datacube, width=width, N_DPs=N_DPs, side=side)

@log
def get_background_streaks_1D_x(datacube, width, N_DPs, side='start'):
    """
    Gets background streaking, by finding the average of a strip of pixels along the y-edge of the
    detector over a random selection of diffraction patterns.

    See docstring for get_background_streaks_1D() for more info.
    """
    assert N_DPs <= datacube.R_Nx*datacube.R_Ny, "N_DPs must be less than or equal to the total number of diffraction patterns."
    assert ((side=='start') or (side=='end')), "side must be 'start' or 'end'."

    # Get random subset of DPs
    indices = np.arange(datacube.R_Nx*datacube.R_Ny)
    np.random.shuffle(indices)
    indices = indices[:N_DPs]
    indices_x, indices_y = np.unravel_index(indices, (datacube.R_Nx,datacube.R_Ny))

    # Make a reference strip array
    refstrip = np.zeros((width, datacube.Q_Ny))
    if side=='start':
        for i in range(N_DPs):
            refstrip += datacube.data4D[indices_x[i], indices_y[i], :width, :].astype(float)
    else:
        for i in range(N_DPs):
            refstrip += datacube.data4D[indices_x[i], indices_y[i], -width:, :].astype(float)

    # Calculate mean and return
    bkgrnd_streaks = np.sum(refstrip, axis=0) / width / N_DPs  # TODO: check with Hamish:
                                                               # why was he using integer division?
    return bkgrnd_streaks

@log
def get_background_streaks_1D_y(datacube, width, N_DPs, side='start'):
    """
    Gets background streaking, by finding the average of a strip of pixels along the x-edge of the
    detector over a random selection of diffraction patterns.

    See docstring for get_background_streaks_1D() for more info.
    """
    assert N_DPs <= datacube.R_Nx*datacube.R_Ny, "N_DPs must be less than or equal to the total number of diffraction patterns."
    assert ((side=='start') or (side=='end')), "side must be 'start' or 'end'."

    # Get random subset of DPs
    indices = np.arange(datacube.R_Nx*datacube.R_Ny)
    np.random.shuffle(indices)
    indices = indices[:N_DPs]
    indices_x, indices_y = np.unravel_index(indices, (datacube.R_Nx,datacube.R_Ny))

    # Make a reference strip array
    refstrip = np.zeros((datacube.Q_Nx, width))
    if side=='start':
        for i in range(N_DPs):
            refstrip += datacube.data4D[indices_x[i], indices_y[i], :, :width].astype(float)
    else:
        for i in range(N_DPs):
            refstrip += datacube.data4D[indices_x[i], indices_y[i], :, -width:].astype(float)

    # Calculate mean and return
    bkgrnd_streaks = np.sum(refstrip, axis=1) / width / N_DPs  # TODO: check with Hamish:
                                                               # why was he using integer division?
    return bkgrnd_streaks


#### Helper functions ####

def streaks_to_DP(bkgrnd_streaks, Q_Nx, Q_Ny, direction='x'):
    """
    Casts 1D bkgrnd_streaks into a 2D diffraction pattern along direction
    """
    assert ((direction=='x') or (direction=='y')), "direction must be 'x' or 'y'."
    if direction=='x':
        return streaks_to_DP_x(bkgrnd_streaks_x=bkgrnd_streaks, Q_Nx=Q_Nx, Q_Ny=Q_Ny)
    else:
        return streaks_to_DP_y(bkgrnd_streaks_y=bkgrnd_streaks, Q_Nx=Q_Nx, Q_Ny=Q_Ny)

def streaks_to_DP_x(bkgrnd_streaks_x, Q_Nx, Q_Ny):
    """
    Casts 1D bkgrnd_streaks into a 2D diffraction pattern along the x-direction
    """
    bkgrnd_x = np.zeros((Q_Nx,Q_Ny))
    bkgrnd_x += bkgrnd_streaks_x[np.newaxis,:]
    return bkgrnd_x

def streaks_to_DP_y(bkgrnd_streaks_y, Q_Nx, Q_Ny):
    """
    Casts 1D bkgrnd_streaks into a 2D diffraction pattern along the y-direction
    """
    bkgrnd_y = np.zeros((Q_Nx,Q_Ny))
    bkgrnd_y += bkgrnd_streaks_y[:,np.newaxis]
    return bkgrnd_y


