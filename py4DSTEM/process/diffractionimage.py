# Functions for generating diffraction images

import numpy as np

def get_diffraction_image(
    datacube,
    type,
    mode,
    geometry,
    shift_center = True,
    verbose = True,
    return_mask = False,
):

    '''
    Computes and returns a diffraction image from `datacube`. The
    kind of diffraction image (max, mean, median) is specified by the
    `mode` argument, and the region it is computed over is specified
    by the `geometry` argument.

    Args:
        datacube (Datacube)
        mode (str): must be in ('max','mean','median')
        geometry (variable): indicates the region the image will
            be computed over. Behavior depends on the argument type:
                - None: uses the whole image
                - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                - 2-tuple: (rx,ry), which are length L arrays.
                    Uses the specified scan positions.
                - `mask`: boolean 2D array
                - `mask_float`: floating point 2D array. Valid only for
                    `mean` mode
        shift_corr (bool): if True, correct for beam shift

    Returns:
        (2D array): the diffraction image

    '''

    assert type in ('max', 'median', 'mean'),\
        'check doc strings for supported types'

    #create mask
    if geometry is not None: 
        from py4DSTEM.process.virtual_image import make_detector
        assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
        'check doc strings for supported modes'
        g = geometry

        # Get mask
        mask = make_detector(datacube.Rshape, mode, g)
        # if return_mask is True, skip computation
        if return_mask == True and shift_center == False:
            return mask

    #if no mask 
    else: 
        mask = np.ones(datacube.Rshape)

    # Calculate diffracton patterns

    # no center shifting
    if shift_center == False:
        # compute
        diffraction_image = np.zeros(datacube.Qshape)
        for qx,qy in tqdmnd(
            datacube.Q_Nx,
            datacube.Q_Ny,
            disable = not verbose,
        ):
            if type == 'mean':
                diffraction_image[qx,qy] = np.sum(datacube.data[:,:,qx,qy]*mask[:,:,np.newaxis,np.newaxis], axis=(0,1))
            if type == 'max':
                diffraction_image[qx,qy] = np.sum(datacube.data[:,:,qx,qy]*mask[:,:,np.newaxis,np.newaxis], axis=(0,1))
            if type == 'median':
                diffraction_image[qx,qy] = np.sum(datacube.data[:,:,qx,qy]*mask[:,:,np.newaxis,np.newaxis], axis=(0,1))

    # with center shifting
    else:


    return diffraction_image