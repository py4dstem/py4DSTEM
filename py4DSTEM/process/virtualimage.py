# Functions for generating virtual images

import numpy as np
#import dask.array as da
#import h5py
#import warnings

from ..utils.tqdmnd import tqdmnd

def get_virtual_image(
    datacube, 
    mode, 
    geometry,
    shift_center = False,
    verbose = True,
):
    '''
    Function to calculate virtual image
    
    Args: 
        datacube (Datacube) : datacube class object which stores 4D-dataset
                              needed for calculation
        mode (str)          : defines geometry mode for calculating virtual image
                                options:
                                    - 'point' uses singular point as detector
                                    - 'circle' or 'circular' uses round detector,like bright field
                                    - 'annular' or 'annulus' uses annular detector, like dark field
                                    - 'rectangle', 'square', 'rectangular', uses rectangular detector
                                    - 'mask' flexible detector, any 2D array
        geometry (variable) : valid entries are determined by the `mode`, values in pixels
                                argument, as follows:
                                    - 'point': 2-tuple, (qx,qy)
                                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius)
                                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o))
                                    - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                                    - `mask`: flexible detector, any 2D array, same size as datacube.QShape         
        verbose (bool)      : if True, show progress bar

    Returns:

        virtual image (2D-array)
    '''
    g = geometry

    #point mode 
    if mode == 'point':
        assert(isinstance(g,tuple) and len(g)==2), 'specify qx and qy as tuple (qx, qy)'
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        
        if g[0]%1 > 1e-6: 
            print('warning: rounding qx to integer')

        if g[1]%1 > 1e-6: 
            print('warning: rounding qy to integer')
            
        qx = int(g[0])
        qy = int(g[1])
        
        mask[qx,qy] = 1

    #circular mask
    if mode in('circle', 'circular') :
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and isinstance(g[1],float or int)), \
        'specify qx, qy, radius_i as ((qx, qy), radius)'

        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1] ** 2

    #annular mask 
    if mode in('annulus', 'annular') :
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and len(g[1])==2), \
        'specify qx, qy, radius_i, radius_0 as ((qx, qy), radius_i, radius_o)'
        qxa, qya = np.indices((datacube.Q_Nx, datacube.Q_Ny))
        mask1 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 > g[1][0] ** 2
        mask2 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1][1] ** 2
        mask = np.logical_and(mask1, mask2)

    #rectangle mask 
    if mode in('rectangle', 'square', 'rectangular') :
        assert(isinstance(g,tuple) and len(g)==4), \
       'specify x_min, x_max, y_min, y_max as (x_min, x_max, y_min, y_max)'
        mask = np.zeros((datacube.Q_Nx, datacube.Q_Ny))
        mask[g[0]:g[1], g[2]:g[3]] = 1

    #flexible mask
    if mode == 'mask' :
        assert type(g) == np.ndarray, '`geometry` type should be `np.ndarray`'
        assert (g.shape == datacube.Qshape), 'mask and diffraction pattern shapes do not match'
        mask = g

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny)) 
    for rx,ry in tqdmnd(
        datacube.R_Nx, 
        datacube.R_Ny,
        disable = not verbose,
    ):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)

    return virtual_image