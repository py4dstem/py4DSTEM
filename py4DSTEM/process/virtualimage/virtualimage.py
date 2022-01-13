# Functions for generating virtual images

import numpy as np
from ...io.datastructure import DataCube
from ..utils import tqdmnd
import numba as nb
import dask.array as da
import matplotlib.pyplot as plt
import warnings

#TODO clean up all the old code snippets
#TODO add automagic functions that will pick dask or normal depending on the array type. 
#TODO add alias names for get get_BF, get_ADF? 


#### Mask Making Functions ####
# lifted from py4DSTEM old funcs
#TODO Add symmetry mask maker, e.g. define one spot, add symmetry related reflections 
#TODO Add multiple mask maker, e.g. given list of coordinate tuples create circular masks at each point

def make_circ_mask(datacube, x0, y0, R, return_crop_vals=False):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane. The mask returned is the same shape as each diffraction slice.
    If return_crop_vals is True, then they can be used to acceleate.  
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


    # full mask is same size as each diffraction slice. 
    # If the xmin, xmax, ymin, ymax are returned index values are returned, they could be used when indexing values. 
    if return_crop_vals:
        return full_mask, (xmin,xmax,ymin,ymax)
    else:
        return full_mask

def make_annular_mask(datacube, x0, y0, Ri, Ro):
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
    
    #TODO Should this be made more similar to other mask functions, i.e. pass a center coordinate and size of rectangle?
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
    Function to combine multiple masks into single boolean mask, using np.logical_or , np.logical_xor functions. 

    Args:
        masks (list,tuple): collection of 2D boolean masks
        operator (str): choice of operator either (or, xor) defaults to 'or'  
    
    Returns:
        (2D array): Boolean mask
    """
    assert(operator.lower() in ('or', 'xor')), 'specified operator not supported, must be "or" or "xor"'

    if operator.lower() == 'or': 
        return np.logical_or.reduce(masks)
    elif operator.lower() == 'xor':
        return np.logical_xor.reduce(masks)

def plot_mask_overlay(mask, dp=None, datacube=None, reduce_func=np.mean, alpha=0.5, *args, **kwargs):
    """
    Function to plot the overaly of the mask on a diffraction slice. A diffraction slice or a datacube object may be passed to the function as dp, datacube.
    If a datacube is passed the function diffraction pattern for the dataset will be calculated (note this could be expesnive for large datasets) 

    Args:
        mask (numpy array): 2D array mask
        dp (numpy array, optional): a single 2D diffraction pattern . Defaults to None.
        datacube (py4DSTEM datacube, optional): py4DSTEM datacube object from which a mean or max diffraction pattern will be calculated. Defaults to None.
        reduce_func (str, optional): function to generate the 2D diffraction pattern from the 4D datacube. Function passed must reduce 4D datacube to 2D diffraction pattern . Defaults to np.mean.
        alpha (float, optional): [description]. Defaults to 0.5.
    """

    #TODO clean this function up and bring closer to the standard py4DSTEM way
    #check the mask is 2D
    assert mask.ndim == 2, "mask must be 2D slice"

    #check that at least one of diffraction pattern (dp) or a datacube is passed
    assert dp is not None or datacube is not None, "Must pass a diffraction pattern (dp) or datacube object"
    
    # check if the diffraction pattern is passed, if it isnt use the datacube and . 
    if dp is None:
        
        # apply the function over the first to axes
        dp = np.apply_over_axes(reduce_func, datacube.data, axes=[0,1])
        # quick check that dp is 2D and or can trivally converted to 2D i.e. (1,1,n,m)

        # assume that the last two array axis are correct
        if dp.ndim != 2 and np.all(dp.shape[:-2]) == 1:
            warnings.warn("Diffraction pattern (dp) returned from function was not 2D, but was trivially shaped (1,1,...,n,m) converting to 2D assuming last two axes are correct. This could be wrong")
            m,n = dp.shape[-2:]
            dp = dp.flat[:m*n].reshape((m,n))
        elif dp.ndim != 2 and np.all(dp.shape[:-2]) != 1:
            raise Exception("Diffraction pattern (dp) returned from function was not 2D and non-trivial to convert to 2D")
        else:
            pass
        # assert func == 'mean' or func == 'max', "func must be 'mean' or 'max' "

        # if func == 'mean':
        #     dp = datacube.data.mean(axis=(0,1))
        # elif func == 'max':
        #     dp = datacube.data.max(axis=(0,1))
    else:

        # quick check that dp is 2D and or can trivally converted to 2D i.e. (1,1,n,m)
        # assume that the last two array axis are correct
        if dp.ndim != 2 and np.all(dp.shape[:-2]) == 1:
            warnings.warn("Diffraction pattern (dp) passed was not 2D, but was trivially shaped (1,1,...,n,m) converting to 2D assuming last two axes are correct. This could be wrong")
            m,n = dp.shape[-2:]
            dp = dp.flat[:m*n].reshape((m,n))
        elif dp.ndim != 2 and np.all(dp.shape[:-2]) != 1:
            raise Exception("Diffraction pattern (dp) passed was not 2D and non-trivial to convert to 2D")
        else:
            pass

    # explicitly check the diffraction pattern is 2D should be caught above but just incase,  catch before plotting 
    assert dp.ndim == 2, "Diffraction pattern (dp) must be a 2D slice"

    #plot the diffraction pattern
    plt.imshow(dp, cmap='Greys', *args,**kwargs)
    #plot the mask overlay
    plt.imshow(mask, alpha=alpha, cmap='Reds', *args, **kwargs)
    # add a few options 
    plt.tight_layout()
    #show the plot
    plt.show()

#### End of Mask Making Features ####

#### Virtual Imaging Functions ####

##### old py4dSTEM funcs ####
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

#### End of old py4DSTEM funcs ####


#### Dask Function #### 

#TODO add complementary functions to above operating on dask e.g. BF, annular, circular etc. 
#TODO change to passing datacube
#TODO add automagic application of Dask or Numba functions depending on type(datacube.data)
# can use a list of different output_dtype if that makes sense, I've set to np.uint for now which seems sensible to me
# only works on dask array. 
#TODO add extra wrapper functions for BF etc, all use this as the underlying function, creating masks inside
#TODO change the output type so it is compaitable with floating data
@da.as_gufunc(signature='(i,j),(i,j)->()', output_dtypes=np.float64, axes=[(2,3),(0,1),()], vectorize=True)
def _get_virtual_image_dask(array, mask):
    """
    Make a virutal for all probe posistions from a dask array object using a mask in the diffraction plane.
    Example:
    image = make_virtual_image_dask(dataset.data, mask).compute()

    Args:
        array (dask array): dask array of 4DSTEM data with shape rx,ry,qx,qy
        mask (2D numpy array): mask from which virtual image is generated, must be the same shape as diffraction pattern
    Returns:
        out (2D numpy array): virtual image lazy virtual image, requires explict computation
    """    
    val = np.sum(np.multiply(array,mask), dtype=np.uint)
    return val

def get_virtualimage_from_mask_dask(datacube, mask, eager_compute=True):
    """
    Create a virtual image from a generic mask, i.e. both boolean or non-boolean,  The mask and diffraction slices must be the same shape 

    This should only be used on datasets which are dask arrays. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        mask (2D array): This can be any mask i.e. both boolean or non-boolean, but it must be the same size as the diffraction slice
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    if eager_compute:
        return _get_virtual_image_dask(datacube.data, mask).compute()

    else:
        return _get_virtual_image_dask(datacube.data, mask)

def get_virtualimage_ann_dask(datacube,x0,x1, Ri, Ro, eager_compute=True):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro. 

    This should only be used on datasets which are dask arrays. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        Ri,Ro (numbers): inner/outer detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """
    
    # make the annular mask
    mask = make_annular_mask(datacube, x0, x1, Ri, Ro)
    
    if eager_compute:
            return _get_virtual_image_dask(datacube.data, mask).compute()
    else:
        
        return _get_virtual_image_dask(datacube.data, mask)

def get_virtualimage_circ_dask(datacube, x0, y0, R , eager_compute=True):
    
    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro. 

    This should only be used on datasets which are dask arrays. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        R (numbers): detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the circular mask

    mask = make_circ_mask(datacube, x0, y0, R)

    if eager_compute:
        return _get_virtual_image_dask(datacube.data, mask).compute()
    else:
        return _get_virtual_image_dask(datacube.data, mask)

def get_virutalimage_rect_dask(datacube, xmin, xmax, ymin, ymax, eager_compute=True):
    """        
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.


    Args:
       datacube (DataCube):
        xmin,xmax (ints): x limits of the detector
        ymin,ymax (ints): y limits of the detector
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the rectangular mask
    mask = make_rect_mask(datacube, xmin, xmax, ymin, ymax)


    if eager_compute:
        return _get_virtual_image_dask(datacube.data, mask).compute()
    else:
        return _get_virtual_image_dask(datacube.data, mask)


#### End of Dask Functions ####

#### Einsum Powered Functions ####
# TODO possible quicker for circ, rectangular masks to do np.einsum('ijnm,nm->ij', data[:,:,xmin:xmax,ymin:ymax],mask[xmin:xmax, ymin:ymax]) or similar
# I can get xmin,xmax etc from mask making functions
# TODO I could probably use the boolean array indexes as well rather than multiplication



def get_virtualimage_from_mask_einsum(datacube, mask):
    """
    Create a virtual image from a generic mask, i.e. both boolean or non-boolean, the mask and diffraction slices must be the same shape 



    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        mask (2D array): This can be any mask i.e. both boolean or non-boolean, but it must be the same size as the diffraction slice
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    return np.einsum('ijnm,nm->ij', datacube.data, mask)

def get_virtualimage_ann_einsum(datacube,x0,x1, Ri, Ro):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        Ri,Ro (numbers): inner/outer detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """
    
    # make the annular mask
    mask = make_annular_mask(datacube, x0, x1, Ri, Ro)
    
    return np.einsum('ijnm,nm->ij', datacube.data, mask)

def get_virtualimage_circ_einsum(datacube, x0, y0, R):
    
    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro. 

    This should only be used on datasets which are dask arrays. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        R (numbers): detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the circular mask

    mask, (xmin,xmax,ymin,ymax) = make_circ_mask(datacube, x0, y0, R, return_crop_vals=True)

    return np.einsum('ijnm,nm->ij', datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax])

def get_virutalimage_rect_einsum(datacube, xmin, xmax, ymin, ymax):
    """        
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.


    Args:
       datacube (DataCube):
        xmin,xmax (ints): x limits of the detector
        ymin,ymax (ints): y limits of the detector
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the rectangular mask
    mask, (xmin,xmax,ymin,ymax) = make_rect_mask(datacube, xmin, xmax, ymin, ymax, return_crop_vals=True)

    return np.einsum('ijnm,nm->ij', datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax])

#### End of Einsum Powered Functions ####

#### Tensordot Powered Functions ####

def get_virtualimage_from_mask_tensordot(datacube, mask):
    """
    Create a virtual image from a generic mask, i.e. both boolean or non-boolean, the mask and diffraction slices must be the same shape 



    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        mask (2D array): This can be any mask i.e. both boolean or non-boolean, but it must be the same size as the diffraction slice
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    return np.tensordot(datacube.data, mask)

def get_virtualimage_ann_tensordot(datacube,x0,x1, Ri, Ro):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        Ri,Ro (numbers): inner/outer detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """
    
    # make the annular mask
    mask = make_annular_mask(datacube, x0, x1, Ri, Ro)
    
    return np.tensordot(datacube.data, mask)

def get_virtualimage_circ_tensordot(datacube, x0, y0, R):
    
    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro. 

    This should only be used on datasets which are dask arrays. 

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        x0,y0 (numbers): center of detector
        R (numbers): detector radii
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the circular mask

    mask, (xmin,xmax,ymin,ymax) = make_circ_mask(datacube, x0, y0, R, return_crop_vals=True)

    return np.tensordot(datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax])

def get_virutalimage_rect_tensordot(datacube, xmin, xmax, ymin, ymax):
    """        
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.


    Args:
       datacube (DataCube):
        xmin,xmax (ints): x limits of the detector
        ymin,ymax (ints): y limits of the detector
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image 
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the rectangular mask
    mask, (xmin,xmax,ymin,ymax) = make_rect_mask(datacube, xmin, xmax, ymin, ymax, return_crop_vals=True)

    return np.tensordot(datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax])


### End of Tensordot Powered Functions ####






# Leaving these in for now as they may be recoverable. 
#### NUMBA Powered Functions ####
# #TODO decide if fastmath is appropriate or not  
# #TODO add assertions if desired about shape of mask and diffraction image being the same. a little tricky with numba
# #TODO add ability to shift patterns from coordinates / calibration centers - after v 13.0 update
# #TODO add simplified boolean functions i.e. abstract away passing mask,
# #TODO add alias functions 
# #TODO Jitted functions don't work, either need to restructure into 

# # Most generalised function 
# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def get_virtual_image_numba(datacube, mask):
#     """
#     Make a virutal for all probe posistions from a py4DSTEM datacube object. This is the most generalised and flexible method to make a virtual image, as any mask (boolean, int, float) may be passed. 
#     There are other virtual image functions which are simpler to use and may be preferable for your use case. see:
#         get_virtual_BF_numba
#         get_virtual_ADF_numba
#         get_virtual_rectangular_image_numba
#         get_virtual_circular_image_numba
#         get_virtual_annular_image_numba

#     Note:
#         this function is accelerated using numba.jit with the fastmath flag set as true. see https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath) for more details.
#     Args:
#         datacube (DataCube): py4DSTEM datacube, rx,ry,qx,qy, 
#         mask (2D numpy array): mask from which virtual image is generated, must be the same shape as diffraction pattern
#         out (2D numpy array, optional): pre-allocated output array 
#     Returns:
#         out (2D numpy array): virtual image 
#     """    

#     # assert datacube.data.shape[-2:] == mask.shape, "mask and diffraction pattern sizes are mismatched"
#     #TODO should we be using fastmath = True ? 

#     # if out is None:
#     out = np.zeros(datacube.data.shape[:-2], dtype=datacube.data.dtype)

#     for rx, ry in np.ndindex(datacube.data.shape[:-2]):
#         out[rx,ry] = np.sum(np.multiply(datacube.data[rx,ry],mask)) # multiply used here so that it can take any mask dtype
#     return out


# @nb.jit(nopython=True, parallel=False, cache=True, fastmath=True)
# def get_virtual_BF_numba(datacube, R, center=None):
#     """
#     Make a virtual bright field image from a py4DSTEM datacube object, with a radius of R pixels.

#     Note this function is accelerated using numba.jit with the fastmath flag set as true. see https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath) for more details.
#     Args:
#         datacube (py4DSTEM.datacube): py4DSTEM datacube object 
#         R (int): radius of the bright field detector
#         center (tuple, optional): coordinates to center the BF detector, if None it will pick the center pixel via datacube.data.shape // 2 . Defaults to None.

#     Returns:
#         out (2D numpy array): BF image generated from the Datacube.
#     """
    
#     #TODO accept center coordinates from datacube and roll mask to fit
#     #TODO add ability to use function to get center pixel, instead of a constant center pixel value? 
#     #TODO check if parallel = True is faster or not, a few tests I ran parallel = True was slower
    
#     # if no center is passed it will assume the center of the diffraction pattern
#     if center is None:
#         center = np.array(datacube.data.shape) // 2

#     # create the array for the output image
#     out = np.zeros(datacube.data.shape[:-2])

#     # Create the BF mask and get the crop values 
#     mask, (xmin,xmax,ymin,ymax) = make_circ_mask(datacube, x0=center[0], y0=center[1], R=R, return_crop_vals=True)
    
#     # This might not be correct I need to look through the shapes... and see if it makes sense. 
#     # Crop the mask to the required shape
#     mask = mask[xmin:xmax,ymin:ymax]
#     # Crop the data array to 
#     arr = datacube.data[:,:,xmin:xmax,ymin:ymax]
#     # Loop over the realspace coordinates and use boolean mask to index the array and sum
#     for rx, ry in np.ndindex(arr.shape[:-2]):
#         # out[rx,ry] = np.sum(np.multiply(arr[rx,ry],mask)) # Multiply by the mask and sum 
#         out[rx,ry] = np.sum(arr[rx,ry][mask]) # in a few tests it seems quicker to use Boolean index and sum than multiply for BF images

#     return out

# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def get_virtual_ADF_numba(datacube, Ri, Ro, center=None):
#     """
#     Make a virtual annular dark field image from a py4DSTEM datacube object with a inner radius of Ri, and a outer radius of R0. 

#     Note this function is accelerated using numba.jit with the fastmath flag set as true. see https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath) for more details.

#     Args:
#         datacube (py4DSTEM datacube): py4DSTEM datacube object
#         Ri (int): Inner detector angle radius in pixels
#         Ro (int): Outter detector angle radius in pixels
#         center (tuple, optional): coordinates to center the BF detector, if None it will pick the center pixel via datacube.data.shape // 2 . Defaults to None.

#     Returns:
#         out (2D numpy array): ADF image generated from the Datacube.
#     """
#     #TODO accept center coordinates from datacube and roll mask to fit
#     #TODO use function to get center pixel, will be slower? 
#     # if no center is passed it will assume the center of the diffraction pattern
#     if center is None:
#         center = np.array(datacube.data.shape) // 2

#     # create the array for the output image
#     out = np.zeros(datacube.data.shape[:-2])

#     # if no center is passed it will assume the center of the diffraction pattern
#     mask = make_annular_mask(datacube, x0=center[0],y0=center[1], Ri=Ri, Ro=Ro)
    
#     # loop over 
#     for rx, ry in np.ndindex(datacube.data.shape[:-2]):
#         out[rx,ry] = np.sum(np.multiply(datacube.data[rx,ry],mask))# in a few tests it seems quicker to multiply than boolean index for ADF images 
#         # out[rx,ry] = np.sum(arr[rx,ry][mask]) 
#     return out
    
    
# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def get_virtual_circular_image_numba(datacube, R, center):
#     """ 
#     Make a virtual image from a py4DSTEM datacube object with a circular detector with radius R centere at center.

#     Args:
#         datacube (py4DSTEM datacube): py4DSTEM datacube object
#         R (int): Radius of the circular detector 
#         center (tuple): center coordinates for the circular detector

#     Returns:
#         out (2D numpy array: Virtual image generated
#     """
#     # create the array for the output image
#     out = np.zeros(datacube.data.shape[:-2])

#     # make the mask center at the desired coordinates
#     mask = make_circ_mask(datacube, x0=center[0],y0=center[1], R=R)

#     # loop over realspace coordinates
#     for rx, ry in np.ndindex(datacube.data.shape[:-2]):
#         # out[rx,ry] = np.sum(np.multiply(arr[rx,ry],mask)) # Multiply by the mask and sum 
#         out[rx,ry] = np.sum(datacube.data[rx,ry][mask]) # in a few tests it seems quicker to use Boolean index and sum than multiply for BF images
#     return out

# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def get_virtual_rectangular_image_numba(datacube, xmin, xmax, ymin, ymax, center):
    
#     # create the array for the output image
#     out = np.zeros(datacube.data.shape[:-2])

#     # make the mask center at the desired coordinates
#     mask = make_rect_mask(datacube, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
#     # loop over realspace coordinates
#     for rx, ry in np.ndindex(datacube.data.shape[:-2]):
#         # out[rx,ry] = np.sum(np.multiply(arr[rx,ry],mask)) # Multiply by the mask and sum 
#         out[rx,ry] = np.sum(datacube.data[rx,ry][mask]) # in a few tests it seems quicker to use Boolean index and sum than multiply for BF images
#     return out


# @nb.jit(nopython=True, parallel=True, fastmath=True)
# def get_virtual_annular_image_numba(datacube, Ri, Ro, center):

#     # create the array for the output image
#     out = np.zeros(datacube.data.shape[:-2])

#     # if no center is passed it will assume the center of the diffraction pattern
#     mask = make_annular_mask(datacube, x0=center[0],y0=center[1], Ri=Ri, Ro=Ro)
    
#     # loop over 
#     for rx, ry in np.ndindex(datacube.data.shape[:-2]):
#         out[rx,ry] = np.sum(np.multiply(datacube.data[rx,ry],mask))# in a few tests it seems quicker to multiply than boolean index for ADF images 
#         # out[rx,ry] = np.sum(arr[rx,ry][mask]) 
#     return out

#### End of Virtual Image Functions ####

