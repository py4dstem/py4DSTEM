# Functions for background fitting and subtraction.

import numpy as np

#### Subtrack darkreference from datacube frame at (Rx,Ry) ####

def get_bksbtr_DP(datacube, darkref, Rx, Ry):
    """
    Returns a background subtracted diffraction pattern.

    Args:
        datacube (DataCube): data to background subtract
        darkref (ndarray): dark reference. must have shape (datacube.Q_Nx, datacube.Q_Ny)
        Rx,Ry (int): the scan position of the diffraction pattern of interest

    Returns:
        (ndarray) the background subtracted diffraction pattern
    """
    assert darkref.shape==(datacube.Q_Nx,datacube.Q_Ny), "background must have shape (datacube.Q_Nx, datacube.Q_Ny)"
    return datacube.data[Rx,Ry,:,:].astype(float) - darkref.astype(float)


#### Get dark reference ####

def get_darkreference(datacube, N_frames, width_x=0, width_y=0, side_x='end',
                                                                side_y='end'):
    """
    Gets a dark reference image.

    Select N_frames random frames (DPs) from datacube.  Find streaking noise in the
    horizontal and vertical directions, by finding the average values along a thin strip
    of width_x/width_y pixels along the detector edges. Which edges are used is
    controlled by side_x/side_y, which must be 'start' or 'end'. Streaks along only one
    direction can be used by setting width_x or width_y to 0, which disables correcting
    streaks in this direction.

    Note that the data is cast to float before computing the background, and should
    similarly be cast to float before performing a subtraction. This avoids integer
    clipping and wraparound errors.

    Args:
        datacube (DataCube): data to background subtract
        N_frames (int): number of random diffraction patterns to use
        width_x (int): width of the ROI strip for finding streaking in x
        width_y (int): see above
        side_x (str): use a strip from the start or end of the array. Must be 'start' or
            'end', defaults to 'end'
        side_y (str): see above

    Returns:
        (ndarray): a 2D ndarray of shape (datacube.Q_Nx, datacube.Ny) giving the
        background.
    """
    if width_x==0 and width_y==0:
        print("Warning: either width_x or width_y should be a positive integer. Returning an empty dark reference.")
        return np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    elif width_x==0:
        return get_background_streaks_y(datacube=datacube, N_frames=N_frames,
                                           width=width_y,side=side_y)
    elif width_y==0:
        return get_background_streaks_x(datacube=datacube, N_frames=N_frames,
                                           width=width_x,side=side_x)
    else:
       darkref_x = get_background_streaks_x(datacube=datacube, N_frames=N_frames,
                                               width=width_x,side=side_x)
       darkref_y = get_background_streaks_y(datacube=datacube, N_frames=N_frames,
                                               width=width_y,side=side_y)
       return darkref_x + darkref_y - (np.mean(darkref_x)*width_x + \
                                        np.mean(darkref_y)*width_y)/(width_x+width_y)
                                        # Mean has been added twice; subtract one off

def get_background_streaks(datacube, N_frames, width, side='end', direction='x'):
    """
    Gets background streaking in either the x- or y-direction, by finding the average of
    a strip of pixels along the edge of the detector over a random selection of
    diffraction patterns, and returns a dark reference array.

    Note that the data is cast to float before computing the background, and should
    similarly be cast to float before performing a subtraction. This avoids integer
    clipping and wraparound errors.

    Args:
        datacube (DataCube): data to background subtract
        N_frames (int): number of random frames to use
        width (int): width of the ROI strip for background identification
        side (str, optional): use a strip from the start or end of the array. Must be
            'start' or 'end', defaults to 'end'
        directions (str): the direction of background streaks to find. Must be either
            'x' or 'y' defaults to 'x'

    Returns:
        (ndarray): a 2D ndarray of shape (datacube.Q_Nx,datacube.Q_Ny), giving the
        the x- or y-direction background streaking.
    """
    assert ((direction=='x') or (direction=='y')), "direction must be 'x' or 'y'."
    if direction=='x':
        return get_background_streaks_x(datacube=datacube, N_frames=N_frames, width=width, side=side)
    else:
        return get_background_streaks_y(datacube=datacube, N_frames=N_frames, width=width, side=side)

def get_background_streaks_x(datacube, width, N_frames, side='start'):
    """
    Gets background streaking, by finding the average of a strip of pixels along the
    y-edge of the detector over a random selection of diffraction patterns.

    See docstring for get_background_streaks() for more info.
    """
    assert N_frames <= datacube.R_Nx*datacube.R_Ny, "N_frames must be less than or equal to the total number of diffraction patterns."
    assert ((side=='start') or (side=='end')), "side must be 'start' or 'end'."

    # Get random subset of DPs
    indices = np.arange(datacube.R_Nx*datacube.R_Ny)
    np.random.shuffle(indices)
    indices = indices[:N_frames]
    indices_x, indices_y = np.unravel_index(indices, (datacube.R_Nx,datacube.R_Ny))

    # Make a reference strip array
    refstrip = np.zeros((width, datacube.Q_Ny))
    if side=='start':
        for i in range(N_frames):
            refstrip += datacube.data[indices_x[i], indices_y[i], :width, :].astype(float)
    else:
        for i in range(N_frames):
            refstrip += datacube.data[indices_x[i], indices_y[i], -width:, :].astype(float)

    # Calculate mean and return 1D array of streaks
    bkgrnd_streaks = np.sum(refstrip, axis=0) // width // N_frames

    # Broadcast to 2D array
    darkref = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    darkref += bkgrnd_streaks[np.newaxis,:]
    return darkref

def get_background_streaks_y(datacube, N_frames, width, side='start'):
    """
    Gets background streaking, by finding the average of a strip of pixels along the
    x-edge of the detector over a random selection of diffraction patterns.

    See docstring for get_background_streaks_1D() for more info.
    """
    assert N_frames <= datacube.R_Nx*datacube.R_Ny, "N_frames must be less than or equal to the total number of diffraction patterns."
    assert ((side=='start') or (side=='end')), "side must be 'start' or 'end'."

    # Get random subset of DPs
    indices = np.arange(datacube.R_Nx*datacube.R_Ny)
    np.random.shuffle(indices)
    indices = indices[:N_frames]
    indices_x, indices_y = np.unravel_index(indices, (datacube.R_Nx,datacube.R_Ny))

    # Make a reference strip array
    refstrip = np.zeros((datacube.Q_Nx, width))
    if side=='start':
        for i in range(N_frames):
            refstrip += datacube.data[indices_x[i], indices_y[i], :, :width].astype(float)
    else:
        for i in range(N_frames):
            refstrip += datacube.data[indices_x[i], indices_y[i], :, -width:].astype(float)

    # Calculate mean and return 1D array of streaks
    bkgrnd_streaks = np.sum(refstrip, axis=1) // width // N_frames

    # Broadcast to 2D array
    darkref = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    darkref += bkgrnd_streaks[:,np.newaxis]
    return darkref




