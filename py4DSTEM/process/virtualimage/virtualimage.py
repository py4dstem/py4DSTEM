# Functions for generating virtual images

import numpy as np
from ...io.datastructure import DataCube
from ..utils import tqdmnd
import numba as nb
import dask.array as da
def test():
    return True

def get_virtualimage_rect(datacube, xmin, xmax, ymin, ymax, verbose=True):
    """
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.

    Args:
        datacube (DataCube):
        xmin,xmax (ints): x limits of the detector
        ymin,ymax (ints): y limits of the detector

    Returns:
        (2D array): the virtual image
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.round(xmin))),min(datacube.Q_Nx,int(np.round(xmax)))
    ymin,ymax = max(0,int(np.round(ymin))),min(datacube.Q_Ny,int(np.round(ymax)))

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny,disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax])
    return virtual_image

def get_virtualimage_circ(datacube, x0, y0, R, verbose=True):
    """
    Get a virtual image using a circular detector centered at (x0,y0) and with radius R
    in the diffraction plane.

    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector

    Returns:
        (2D array): the virtual image
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.floor(x0-R))),min(datacube.Q_Nx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(datacube.Q_Ny,int(np.ceil(y0+R)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize)) # Avoids making meshgrids

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny, disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax]*mask)
    return virtual_image

def get_virtualimage_ann(datacube, x0, y0, Ri, Ro, verbose=True):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro.

    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        Ri,Ro (numbers): inner/outer detector radii

    Returns:
        (2D array): the virtual image
    """
    assert isinstance(datacube, DataCube)
    assert Ro>Ri, "Inner radius must be smaller than outer radius"
    xmin,xmax = max(0,int(np.floor(x0-Ro))),min(datacube.Q_Nx,int(np.ceil(x0+Ro)))
    ymin,ymax = max(0,int(np.round(y0-Ro))),min(datacube.Q_Ny,int(np.ceil(y0+Ro)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask_o = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < Ro**2, (xsize,ysize))
    mask_i = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < Ri**2, (xsize,ysize))
    mask = np.logical_xor(mask_o,mask_i)

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny, disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax]*mask)
    return virtual_image


#### Mask Making Functions ####
def make_circ_mask(datacube, x0, y0, R, return_crop_vals=False):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.floor(x0-R))),min(datacube.Q_Nx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(datacube.Q_Ny,int(np.ceil(y0+R)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize)) # Avoids making meshgrids
    
    
    full_mask = np.zeros(shape=datacube.data.shape[2:], dtype=np.bool_)
    full_mask[xmin:xmax,ymin:ymax] = mask

    if return_crop_vals:
        return full_mask, (xmin,xmax,ymin,ymax)
    else:
        return full_mask

def make_ann_mask(datacube, x0, y0, Ri, Ro):
    """
    Make an annular boolean mask centered at (x0,y0), with inner/outer
    radii of Ri/Ro.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        Ri,Ro (numbers): inner/outer detector radii
    Returns:
        (2D array): Boolean mask
    """
    assert isinstance(datacube, DataCube)
    assert Ro>Ri, "Inner radius must be smaller than outer radius"
    xmin,xmax = max(0,int(np.floor(x0-Ro))),min(datacube.Q_Nx,int(np.ceil(x0+Ro)))
    ymin,ymax = max(0,int(np.round(y0-Ro))),min(datacube.Q_Ny,int(np.ceil(y0+Ro)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask_o = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < Ro**2, (xsize,ysize))
    mask_i = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < Ri**2, (xsize,ysize))
    mask = np.logical_xor(mask_o,mask_i)
    
    
    full_mask = np.zeros(shape=datacube.data.shape[2:], dtype=np.bool_)
    full_mask[xmin:xmax,ymin:ymax] = mask
    return full_mask

def make_rect_mask(datacube, xmin, xmax, ymin, ymax, return_crop_vals=False):
    """
    Make a rectangular boolean mask with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.

    Args:
        datacube (DataCube):
        xmin,xmax (ints): x limits of the detector
        ymin,ymax (ints): y limits of the detector

    Returns:
        (2D array): Boolean mask
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.round(xmin))),min(datacube.Q_Nx,int(np.round(xmax)))
    ymin,ymax = max(0,int(np.round(ymin))),min(datacube.Q_Ny,int(np.round(ymax)))

    
    full_mask = np.zeros(shape=datacube.data.shape[2:], dtype=np.bool_)
    full_mask[xmin:xmax,ymin:ymax] = True
    
    if return_crop_vals:
        return full_mask, (xmin, xmax, ymin, ymax)
    else:
        return full_mask

def combine_masks(masks, operator='or'):
    """
    
    Args:
        masks (list,tuple): collection of 2D boolean masks
        operator (str): choice of operator either (or, xor)
    
    Returns:
        (2D array): Boolean mask
    """
    assert(operator.lower() in ('or', 'xor'))

    if operator.lower() == 'or': 
        return np.logical_or.reduce(masks)
    elif operator.lower() == 'xor':
        return np.logical_xor.reduce(masks)
    else: 
        print('specified operator not supported, must be "or" or "xor"')


#### In memory functions ####

@nb.jit(nopython=True, parallel=True, fastmath=True)
def make_virtual_image_numba(datacube, mask, out):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """    
    for rx, ry in np.ndindex(datacube.data.shape[:-2]):
        out[rx,ry] = np.sum(datacube.data[rx,ry]*mask)
    return out

@nb.jit(nopython=True, parallel=False, cache=False, fastmath=True)
def make_virtual_image_BF_numba(datacube, mask, xmin,xmax,ymin,ymax,out):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    
    mask = mask[xmin:xmax,ymin:ymax]
    arr = datacube.data[:,:,xmin:xmax,ymin:ymax]
    for rx, ry in np.ndindex(arr.shape[:-2]):
        out[rx,ry] = np.sum(np.multiply(arr[rx,ry],mask))
    return out



@nb.jit(nopython=True, parallel=True, fastmath=True)
def make_virtual_image_numba(datacube, mask):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    
    out = np.zeros(datacube.data.shape[:-2])
    for rx, ry in np.ndindex(datacube.dataray.shape[:-2]):
        out[rx,ry] = np.sum(datacube.data[rx,ry]*mask)
    return out

# I can't get this to work quicker than you're brightfield function for some unknown reason.
@nb.jit(nopython=True, parallel=False, cache=False, fastmath=True)
def make_virtual_image_BF_numba(datacube, mask, xmin,xmax,ymin,ymax):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    out = np.zeros(datacube.data.shape[:-2])
    mask = mask[xmin:xmax,ymin:ymax]
    arr = datacube.data[:,:,xmin:xmax,ymin:ymax]
    for rx, ry in np.ndindex(arr[:-2]):
        out[rx,ry] = np.sum(np.multiply(arr[rx,ry],mask))
    return out

# Not as good way to do it, have to specify types etc. Leaving in for now incase I want to use something similar elsewhere 
@nb.guvectorize(signature='(i,j),(i,j)->()', ftylist=[(nb.uint[:,:], nb.boolean[:,:], nb.uint)], nopython=True, target='parallel')
def make_vitual_image_numba(array, mask, out):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        x0,y0 (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    out = np.sum(array * mask)




#### Dask Functions #### 

# can use a list of different output_dtype if that makes sense, I've set to np.uint for now. 
# only works on dask array. 
@da.as_gufunc(signature='(i,j),(i,j)->()', output_dtypes=np.uint, vectorize=True)
def make_virtual_image_dask(array, mask):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        array (dask_array):
        mask (numbers): center of detector
        R (number): radius of detector
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """

    val = np.sum(np.multipy(array,mask), dtype=np.uint)
    return val
