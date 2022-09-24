# Functions for generating diffraction images

import numpy as np
from py4DSTEM.utils.tqdmnd import tqdmnd

def get_virtual_diffraction(
    datacube,
    type,
    mode = None,
    geometry = None,
    shift_center = False,
    calibrated = False,
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
        from py4DSTEM.process.virtualimage import make_detector
        assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
        'check doc strings for supported modes'
        g = geometry

        # Get mask
        mask = make_detector(datacube.Rshape, mode, g)
        # if return_mask is True, skip computation

    #if no mask 
    else: 
        mask = np.ones(datacube.Rshape, dtype = bool)

    if return_mask == True:
            return mask
    
    # no center shifting
    if shift_center == False:

        # Calculate diffracton pattern
        virtual_diffraction = np.zeros(datacube.Qshape)

        for qx,qy in tqdmnd(
            datacube.Q_Nx,
            datacube.Q_Ny,
            disable = not verbose,
        ):
            if type == 'mean':
                virtual_diffraction[qx,qy] = np.sum(   np.squeeze(datacube.data[:,:,qx,qy])*mask)
            elif type == 'max':
                virtual_diffraction[qx,qy] = np.max(   np.squeeze(datacube.data[:,:,qx,qy])*mask)
            elif type == 'median':
                virtual_diffraction[qx,qy] = np.median(np.squeeze(datacube.data[:,:,qx,qy])*mask)

    # with center shifting
    else:
        assert type in ('max', 'mean'),\
        'check doc strings for supported types'

        # Get calibration metadata
        assert datacube.calibration.get_origin(), "origin need to be calibrated"
        x0, y0 = datacube.calibration.get_origin()
        x0_mean = np.mean(x0)
        y0_mean = np.mean(y0)

        # get shifts
        qx_shift = (x0-x0_mean).round().astype(int)
        qy_shift = (y0-y0_mean).round().astype(int)


        # compute
        virtual_diffraction = np.zeros(datacube.Qshape)
        for rx,ry in tqdmnd(
            datacube.R_Nx,
            datacube.R_Ny,
            disable = not verbose,
        ):
            if mask == True: 
                # get shifted DP
                DP = np.roll(
                    datacube.data[rx, ry, :,:,],
                    (qx_shift[rx,ry], qy_shift[rx,ry]),
                    axis=(0,1),
                    )
                if type == 'mean':
                    virtual_diffraction += DP     
                elif type == 'max':
                    virtual_diffraction = np.maximum(virtual_diffraction, DP)
            virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*_mask)

    return virtual_diffraction