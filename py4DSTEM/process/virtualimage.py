# Functions for generating virtual images
import numpy as np
import dask.array as da

from py4DSTEM.utils.tqdmnd import tqdmnd

def get_virtual_image(
    datacube, 
    mode, 
    geometry,
    shift_center = False,
    verbose = True,
    dask = False,
):
    '''
    Function to calculate virtual image
    
    Args: 
        datacube (Datacube) : datacube class object which stores 4D-dataset
                              needed for calculation
        mode (str)          : defines geometry mode for calculating virtual image
                                options:
                                    - 'point' uses singular point as detector
                                    - 'circle' or 'circular' uses round detector, like bright field
                                    - 'annular' or 'annulus' uses annular detector, like dark field
                                    - 'rectangle', 'square', 'rectangular', uses rectangular detector
                                    - 'mask' flexible detector, any 2D array
        geometry (variable) : valid entries are determined by the `mode`, values in pixels
                                argument, as follows:
                                    - 'point': 2-tuple, (qx,qy), 
                                       qx and qy are each single float or int to define center
                                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius), 
                                       qx, qy and radius, are each single float or int 
                                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                                       qx, qy, radius_i, and radius_o are each single float or integer 
                                    - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                                    - `mask`: flexible detector, any 2D array, same size as datacube.QShape         
        shift_center (bool) : if True, qx and qx are shifted for each position in real space
                                supported for 'point', 'circle', and 'annular' geometry. 
                                For the shifting center mode, the geometry argument shape
                                should be modified so that qx and qy are the same size as Rshape
                                    - 'point': 2-tuple, (qx,qy) 
                                       where qx.shape and qx.shape == datacube.Rshape
                                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius) 
                                       where qx.shape and qx.shape == datacube.Rshape
                                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o))
                                       where qx.shape and qx.shape == datacube.Rshape
        verbose (bool)      : if True, show progress bar
        dask (bool)         : if True, use dask arrays

    Returns:
        virtual image (2D-array)
    '''
    
    g = geometry

    assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
    'check doc strings for supported modes'

    if shift_center == False:
        #point mask 
        if mode == 'point':
            assert(isinstance(g,tuple) and len(g)==2), 'specify qx and qy as tuple (qx, qy)'
            mask = np.zeros(datacube.Qshape)
            
            if g[0]%1 > 1e-6: 
                print('warning: rounding qx to integer')

            if g[1]%1 > 1e-6: 
                print('warning: rounding qy to integer')
                
            qx = int(g[0])
            qy = int(g[1])
            
            mask[qx,qy] = 1

        #circular mask
        if mode in('circle', 'circular'):
            assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and isinstance(g[1],(float,int))), \
            'specify qx, qy, radius_i as ((qx, qy), radius)'

            qxa, qya = np.indices(datacube.Qshape)
            mask = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1] ** 2

        #annular mask 
        if mode in('annulus', 'annular'):
            assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and len(g[1])==2), \
            'specify qx, qy, radius_i, radius_0 as ((qx, qy), radius_i, radius_o)'
            qxa, qya = np.indices(datacube.Qshape)
            mask1 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 > g[1][0] ** 2
            mask2 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1][1] ** 2
            mask = np.logical_and(mask1, mask2)

        #rectangle mask 
        if mode in('rectangle', 'square', 'rectangular') :
            assert(isinstance(g,tuple) and len(g)==4), \
           'specify x_min, x_max, y_min, y_max as (x_min, x_max, y_min, y_max)'
            mask = np.zeros(datacube.Qshape)

            if g[0]%1 > 1e-6: 
                print('warning: rounding xmin to integer')

            if g[1]%1 > 1e-6: 
                print('warning: rounding xmax to integer')

            if g[2]%1 > 1e-6: 
                print('warning: rounding ymin to integer')

            if g[3]%1 > 1e-6: 
                print('warning: rounding ymax to integer')
                
            xmin = int(g[0])
            xmax = int(g[1])
            ymin = int(g[2])
            ymax = int(g[3])
            
            mask[xmin:xmax, ymin:ymax] = 1

        #flexible mask
        if mode == 'mask':
            assert type(g) == np.ndarray, '`geometry` type should be `np.ndarray`'
            assert (g.shape == datacube.Qshape), 'mask and diffraction pattern shapes do not match'
            mask = g

        #dask 
        def _apply_mask_dask(datacube,mask):
            virtual_image = np.sum(np.multiply(datacube.data,mask), dtype=np.float64)
        
        #calculate images
        if dask == True:
            apply_mask_dask = da.as_gufunc(_apply_mask_dask,signature='(i,j),(i,j)->()', output_dtypes=np.float64, axes=[(2,3),(0,1),()], vectorize=True)
            virtual_image = apply_mask_dask(datacube, mask)
        else: 
            virtual_image = np.zeros(datacube.Rshape) 
            for rx,ry in tqdmnd(
                datacube.R_Nx, 
                datacube.R_Ny,
                disable = not verbose,
            ):
                virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)
    else: 
        assert mode in ('point', 'circle', 'circular','annulus', 'annular'), \
        'only point, circular, and annular detectors supported for shift_center'
        
        #point mask
        if mode == 'point':
                assert(isinstance(g,tuple) and len(g)==2), 'specify qx and qy as tuple (qx, qy)'
                qx_scan = np.asarray(g[0])
                qy_scan = np.asarray(g[1])
                assert(qx_scan.shape == datacube.Rshape and qy_scan.shape == datacube.Rshape), 'qx and qy should match real space size'

                virtual_image = np.zeros(datacube.Rshape) 

                for rx,ry in tqdmnd(
                    datacube.R_Nx, 
                    datacube.R_Ny,
                    disable = not verbose,
                ):
                        mask = np.zeros(datacube.Qshape)
                        
                        qx = int(qx_scan[rx,ry])
                        qy = int(qy_scan[rx,ry])

                        mask[qx,qy] = 1

                        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)
        #circular mask 
        if mode in('circle', 'circular') :
                assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and isinstance(g[1],(float,int))), \
                'specify qx, qy, radius_i as ((qx, qy), radius)'
                
                qx_scan = np.asarray(g[0][0])
                qy_scan = np.asarray(g[0][1])
                
                assert(qx_scan.shape == datacube.Rshape and qy_scan.shape == datacube.Rshape), 'qx and qy should match real space size'

                qxa, qya = np.indices(datacube.Qshape)
                
                virtual_image = np.zeros(datacube.Rshape) 

                for rx,ry in tqdmnd(
                    datacube.R_Nx, 
                    datacube.R_Ny,
                    disable = not verbose,
                ):
                        mask = (qxa - qx_scan[rx,ry]) ** 2 + (qya - qx_scan[rx,ry]) ** 2 < g[1] ** 2

                        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)

        #annular mask
        if mode in('annulus', 'annular'):
                assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and len(g[1])==2),\
                'specify qx, qy, radius_i, radius_0 as ((qx, qy), radius_i, radius_o)'
                
                qx_scan = np.asarray(g[0][0])
                qy_scan = np.asarray(g[0][1])
                assert(qx_scan.shape == datacube.Rshape and qy_scan.shape == datacube.Rshape),\
                'qx and qy should match real space size'
                
                qxa, qya = np.indices(datacube.Qshape)

                virtual_image = np.zeros(datacube.Rshape) 

                for rx,ry in tqdmnd(
                    datacube.R_Nx, 
                    datacube.R_Ny,
                    disable = not verbose,
                ):
                        mask1 = (qxa - qx_scan[rx,ry]) ** 2 + (qya - qx_scan[rx,ry]) ** 2 > g[1][0] ** 2
                        mask2 = (qxa - qx_scan[rx,ry]) ** 2 + (qya - qx_scan[rx,ry]) ** 2 < g[1][1] ** 2
                        mask = np.logical_and(mask1, mask2)

                        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)

    return virtual_image
