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
    virtual_mask = None, 
    verbose = True,
):
    '''
    Function to calculate virtual image
    
    Args: 
        datacube (Datacube)     : datacube class object which stores 4D-dataset
                                needed for calculation
        mode (string)           : defines geometry mode for calculating virtual image
                                  options:
                                    "point" uses singular point as detector
                                    "circle" or "circular" uses round detector,
                                        like bright field
                                    "annular" or "annulus" uses annular detector,
                                         like dark field
                                    "rectangle", "square", "rectangular", uses square detector
                                    "mask" flexible detector, any 2D array
                    
        qx (float)              : x index for center of detector for point, 
                                circle and annular detector (pixels)
        qy (float)              : y index for center of detector for point, 
                                circle and annular detector (pixels)
        radius_i (float)        : radius of circlular detector or inner radius 
                                 of annulus (pixels)
        radius_o (float)        : outer radius of annulus (pixels)
        x_min (float)           : minimum x value of rectangular detector (pixels)
        x_max (float)           : maximum x value of rectangular detector (pixels)
        y_min (float)           : minimum y value of rectangular detector (pixels)
        y_max (float)           : maximum y value of rectangular detector (pixels)
        virtual_mask (2D array) : 2D array with shape Q_Nx, Q_Ny (same shape as 
                                  diffraction space)
        verbose (bool)          : if True, show progress bar

    Returns:
        virtual image (2D-array)
    '''

    #point mode 
    if mode == 'point':
        assert(qx,qy) is not None, "specify qx and qy"
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        
        if qx%1 > 1e-6: 
            print("warning: rounding qx to integer")

        if qy%1 > 1e-6: 
            print("warning: rounding qy to integer")
            
        qx = int(qx)
        qy = int(qy)
        
        mask[qx,qy] = 1

    #circular mask
    if mode in('circle', 'circular') :
        assert (qx, qy,radius_i) is not None, "specify qx, qy, radius_i"

        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask = (qxa - qx) ** 2 + (qya - qy) ** 2 < radius_i ** 2

    #annular mask 
    if mode in('annulus', 'annular') :
        assert (qx, qy,radius_i, radius_o) is not None, "specify qx, qy, radius_i, radius_o"
        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask1 = (qxa - qx) ** 2 + (qya - qy) ** 2 > radius_i ** 2
        mask2 = (qxa - qx) ** 2 + (qya - qy) ** 2 < radius_o ** 2
        mask = np.logical_and(mask1, mask2)

    #rectangle mask 
    if mode in('rectangle', 'square', 'rectangular') :
        assert (x_min, x_max, y_min, y_max) is not None, "specify x_min, x_max, y_min, y_max"
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        mask[x_min:x_max, y_min:y_max] = 1

    #flexible mask
    if mode == 'mask' :
        assert (virtual_mask.shape == (datacube.Q_Nx, datacube.Q_Ny)), "check mask dimensions"
        mask = virtual_mask

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny)) 
    for rx,ry in tqdmnd(
        datacube.R_Nx, 
        datacube.R_Ny,
        disable = not verbose,
    ):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)

    return virtual_image