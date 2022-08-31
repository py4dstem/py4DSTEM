# Functions for generating virtual images
import numpy as np
import dask.array as da
from py4DSTEM.utils.tqdmnd import tqdmnd

def get_virtual_image(
    datacube, 
    mode, 
    geometry,
    centered = False, 
    calibrated = False,
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
        centered (bool)     : if false (default), the origin is in the upper left corner.
                              If True, the mean measured origin in the datacube calibrations 
                              is set as center. In this case, for example, a centered bright field image 
                              could be defined by geometry = ((0,0), R).
        calibrated (bool)   : if True, geometry is specified in units of 'A^-1' isntead of pixels. 
                              The datacube must have updated calibration metadata.
        shift_center (bool) : if True, qx and qx are shifted for each position in real space
                                supported for 'point', 'circle', and 'annular' geometry. 
                                For the shifting center mode, the geometry argument shape
                                should be modified so that qx and qy are the same size as Rshape
                                    - 'point': 2-tuple, (qx,qy) 
                                       where qx.shape and qy.shape == datacube.Rshape
                                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius) 
                                       where qx.shape and qx.shape == datacube.Rshape
                                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o))
                                       where qx.shape and qx.shape == datacube.Rshape
        verbose (bool)      : if True, show progress bar
        dask (bool)         : if True, use dask arrays

    Returns:
        virtual image (2D-array)
    '''
    
    assert mode in ('point', 'circle', 'circular', 'annulus', 'annular', 'rectangle', 'square', 'rectangular', 'mask'),\
    'check doc strings for supported modes'
    g = geometry

    if centered == True: 
        assert datacube.calibration.get_origin_meas(), "origin need to be calibrated"
        x0, y0 = datacube.calibration.get_origin_meas()
        x0_mean = np.mean(x0)
        y0_mean = np.mean(y0)
        if mode == 'point':
            g = (g[0] + x0_mean, g[1] + y0_mean)
        if mode in('circle', 'circular', 'annulus', 'annular'):
            g = ((g[0][0] + x0_mean, g[0][1] + y0_mean), g[1])
        if mode in('rectangle', 'square', 'rectangular') :
             g = (g[0] + x0_mean, g[1] + x0_mean, g[2] + y0_mean, g[3] + y0_mean)

    if calibrated == True:
        assert datacube.calibration['Q_pixel_units'] == 'A^-1', \
        'check datacube.calibration. datacube must be calibrated in A^-1'

        unit_conversion = datacube.calibration['Q_pixel_size']
        if mode == 'point':
            g = (g[0]/unit_conversion, g[1]/unit_conversion)
        if mode in('circle', 'circular'):
            g = ((g[0][0]/unit_conversion, g[0][1]/unit_conversion), 
                (g[1]/unit_conversion))
        if mode in('annulus', 'annular'):
            g = ((g[0][0]/unit_conversion, g[0][1]/unit_conversion), 
                (g[1][0]/unit_conversion, g[1][1]/unit_conversion))
        if mode in('rectangle', 'square', 'rectangular') :
            g = (g[0]/unit_conversion, g[1]/unit_conversion, 
                 g[2]/unit_conversion, g[3]/unit_conversion)

    if shift_center == False: 
        mask = make_detector(datacube.Qshape, mode, g)
        
        #dask 
        def _apply_mask_dask(datacube,mask):
            virtual_image = np.sum(np.multiply(datacube.data,mask), dtype=np.float64)
        
        #calculate images
        if dask == True:
            apply_mask_dask = da.as_gufunc(_apply_mask_dask,signature='(i,j),(i,j)->()', output_dtypes=np.float64, axes=[(2,3),(0,1),()], vectorize=True)
            virtual_image = apply_mask_dask(datacube.data, mask)
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
            qx_scan = np.asarray(g[0])
            qy_scan = np.asarray(g[1])
            assert(qx_scan.shape == datacube.Rshape and qy_scan.shape == datacube.Rshape), 'qx and qy should match real space size'

            virtual_image = np.zeros(datacube.Rshape) 

            for rx,ry in tqdmnd(
                datacube.R_Nx, 
                datacube.R_Ny,
                disable = not verbose,
            ):
                geometry_shift = (qx_scan[rx,ry], qy_scan[rx,ry])
                mask = make_detector(datacube.Qshape, mode, geometry_shift)
                virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*mask)


        #circular or annular mask 
        if mode in('circle', 'circular', 'annulus', 'annular') :
            qx_scan = np.asarray(g[0][0])
            qy_scan = np.asarray(g[0][1])
            
            assert(qx_scan.shape == datacube.Rshape and qy_scan.shape == datacube.Rshape), 'qx and qy should match real space size'

            #make mask using small subset of diffraction space
            if mode in('circle', 'circular'): 
                R = g[1]
            if mode in('annulus', 'annular'): 
                R = g[1][1]

            xmin,xmax = max(0,int(np.floor(np.min(qx_scan)-R))),min(datacube.Q_Nx,int(np.ceil(np.max(qx_scan)+R)))
            ymin,ymax = max(0,int(np.round(np.min(qy_scan)-R))),min(datacube.Q_Ny,int(np.ceil(np.max(qy_scan)+R)))
            xsize,ysize = xmax-xmin,ymax-ymin
            qx_scan_crop,qy_scan_crop = qx_scan-xmin,qy_scan-ymin
            virtual_image = np.zeros(datacube.Rshape) 

            for rx,ry in tqdmnd(
                datacube.R_Nx, 
                datacube.R_Ny,
                disable = not verbose,
            ):
                geometry_shift = ((qx_scan_crop[rx, ry], qy_scan_crop[rx, ry]), g[1])
                mask = make_detector((xsize,ysize), mode, geometry_shift)

                full_mask = np.zeros(shape=datacube.Qshape, dtype=np.bool_)
                full_mask[xmin:xmax,ymin:ymax] = mask
            
                virtual_image[rx,ry] = np.sum(datacube.data[rx,ry]*full_mask)

    return virtual_image

def make_detector(
    Qshape,
    mode, 
    geometry,
):
    '''
    Function to return 2D mask
    
    Args: 
        Qshape (tuple)     : defines shape of mask (Q_Nx, Q_Ny) where Q_Nx and Q_Ny are mask sizes
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

    Returns:
        virtual detector in the form of a 2D mask (array)
    '''
    g = geometry

    #point mask 
    if mode == 'point':
        assert(isinstance(g,tuple) and len(g)==2), 'specify qx and qy as tuple (qx, qy)'
        mask = np.zeros(Qshape)
            
        qx = int(g[0])
        qy = int(g[1])
        
        mask[qx,qy] = 1

    #circular mask
    if mode in('circle', 'circular'):
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and isinstance(g[1],(float,int))), \
        'specify qx, qy, radius_i as ((qx, qy), radius)'

        qxa, qya = np.indices(Qshape)
        mask = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1] ** 2

    #annular mask 
    if mode in('annulus', 'annular'):
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2 and len(g[1])==2), \
        'specify qx, qy, radius_i, radius_0 as ((qx, qy), (radius_i, radius_o))'

        assert g[1][1] > g[1][0], "Inner radius must be smaller than outer radius"

        qxa, qya = np.indices(Qshape)
        mask1 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 > g[1][0] ** 2
        mask2 = (qxa - g[0][0]) ** 2 + (qya - g[0][1]) ** 2 < g[1][1] ** 2
        mask = np.logical_and(mask1, mask2)

    #rectangle mask 
    if mode in('rectangle', 'square', 'rectangular') :
        assert(isinstance(g,tuple) and len(g)==4), \
       'specify x_min, x_max, y_min, y_max as (x_min, x_max, y_min, y_max)'
        mask = np.zeros(Qshape)
            
        xmin = int(g[0])
        xmax = int(g[1])
        ymin = int(g[2])
        ymax = int(g[3])
        
        mask[xmin:xmax, ymin:ymax] = 1

    #flexible mask
    if mode == 'mask':
        assert type(g) == np.ndarray, '`geometry` type should be `np.ndarray`'
        assert (g.shape == Qshape), 'mask and diffraction pattern shapes do not match'
        mask = g

    return mask
