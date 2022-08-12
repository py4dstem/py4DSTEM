# Functions for generating virtual images

import numpy as np
import dask.array as da
import h5py
import warnings

from ..utils.tqdmnd import tqdmnd

def get_virtual_image(
    datacube, 
    mode, 
    qx = None,
    qy = None,
    radius_i = None,
    radius_o = None,
    x_min = None, 
    x_max = None, 
    y_min = None, 
    y_max = None,
    virtual_mask = None
):
    """

    """
    
    #point mode 
    if mode == 'point':
        assert(qx,qy), "specify qx and qy"
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        mask[qx,qy] = 1

    #circular mask
    if mode in('circle', 'circular') :
        assert(qx qy,radius_i), "specify qx, qy, radius_i"

        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask = (qxa - qx) ** 2 + (y-jc) ** 2 < radius_i ** 2

    #annular mask 
    if mode == 'annulus'
        assert(qx qy,radius_i, radius_o), "specify qx, qy, radius_i, radius_o"
        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask1 = (qxa - qx) ** 2 + (y-jc) ** 2 > radius_i ** 2
        mask2 = (qxa - qx) ** 2 + (y-jc) ** 2 < radius_i ** 2
        mask = np.logical(mask1, mask2)

    #rectangle mask 
    if mode in('rectangle', 'square', 'rectangular')
        assert(x_min, x_max, y_min, y_max), "x_min, x_max, y_min, y_max"
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        mask[x_min:x_max, y_min:y_max] = 1

    #flexible mask
    if mode == 'mask'
        mask = virtual_mask

    virtual_image = np.sum(datacube*mask, axis = (2,3))

    return virtual_image