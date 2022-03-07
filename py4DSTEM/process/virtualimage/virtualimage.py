# Functions for generating virtual images
import numpy as np
from ...io import DataCube
from ..utils import tqdmnd
import dask.array as da
import matplotlib.pyplot as plt
import warnings
import h5py

__all__ = [
    'make_circ_mask',
    'make_annular_mask',
    'make_rect_mask',
    'combine_masks',
    'plot_mask_overlay',
    'get_virtualimage',
    '_get_virtualimage_rect_old',
    '_get_virtualimage_circ_old',
    '_get_virtualimage_ann_old',
    '_infer_image_type_from_geometry',
    '_get_virtualimage_from_mask_dask',
    '_get_virtualimage_from_mask_einsum',
    '_get_virtualimage_from_mask_tensordot'
    ]

#TODO clean up all the old code snippets - in progress
#TODO add automagic functions that will pick dask or normal depending on the array type - in progress 
#TODO add alias names for get get_BF, get_ADF? 
#TODO Work out how to handle name space to access underlying __functions__, use __all__ or something like that 

#### Mask Making Functions ####
# lifted from py4DSTEM old funcs
#TODO Add symmetry mask maker, e.g. define one spot, add symmetry related reflections 
#TODO Add multiple mask maker, e.g. given list of coordinate tuples create circular masks at each point
#TODO Add assertion statements 
def make_circ_mask(datacube, geometry, return_crop_vals=False):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane. The mask returned is the same shape as each diffraction slice.
    If return_crop_vals is True, then they can be used to acceleate.
    Args:
        datacube (DataCube):
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
        and radius is a number
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """
    assert isinstance(datacube, DataCube)

    (x0,y0),R = geometry

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

def make_annular_mask(datacube, geometry):
    """
    Make an annular boolean mask centered at (x0,y0), with inner/outer
    radii of Ri/Ro.
    Args:
        datacube (DataCube):
        geometry (2-tuple): (center,radii), where center is the 2-tuple (qx0,qy0),
        and radii is the 2-tuple (ri,ro)
    Returns:
        (2D array): Boolean mask
    """

    (x0,y0),(Ri,Ro) = geometry

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

def make_rect_mask(datacube, geometry, return_crop_vals=False):
    """
    Make a rectangular boolean mask with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.

    Args:
        datacube (DataCube):
        geometry (4-tuple of ints): (qxmin,qxmax,qymin,qymax)
        return_crop_vals (Boolean): boolean toggle to return indicies for cropping diffraction pattern
    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern (xmin,xmax,ymin,ymax)
    """

    xmin,xmax,ymin,ymax = geometry

    assert isinstance(datacube, DataCube)

    xmin,xmax = max(0,int(np.round(xmin))),min(datacube.Q_Nx,int(np.round(xmax)))
    ymin,ymax = max(0,int(np.round(ymin))),min(datacube.Q_Ny,int(np.round(ymax)))


    full_mask = np.zeros(shape=datacube.data.shape[2:], dtype=np.bool_)
    full_mask[xmin:xmax,ymin:ymax] = True

    if return_crop_vals:
        return full_mask, (xmin, xmax, ymin, ymax)
    else:
        return full_mask

# TODO add logical_and, logical_not functions as well? 
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

# REMOVE THIS ?
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

##### py4DSTEM funcs V0.13.0 ####
def _get_virtualimage_rect_old(datacube, geometry, verbose=True, *args, **kwargs):
    """
    Get a virtual image using a rectagular detector.
    Args:
        datacube (DataCube):
        geometry (4-tuple of ints): (qxmin,qxmax,qymin,qymax)
    Returns:
        (2D array): the virtual image
    """
    assert(len(geometry)==4 and all([isinstance(i,(int,np.integer)) for i in geometry])), "Detector geometry was specified incorrectly, must be a set of 4 of integers"
    xmin,xmax,ymin,ymax = geometry
    assert(xmax>xmin and ymax>ymin)

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny,disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax])
    return virtual_image

def _get_virtualimage_circ_old(datacube, geometry, verbose=True, *args, **kwargs):
    """
    Get a virtual image using a circular detector centered at (x0,y0) and with radius R
    in the diffraction plane.
    Args:
        datacube (DataCube):
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
            and radius is a number
    Returns:
        (2D array): the virtual image
    """
    (x0,y0),R = geometry
    xmin,xmax = max(0,int(np.floor(x0-R))),min(datacube.Q_Nx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(datacube.Q_Ny,int(np.ceil(y0+R)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize)) # Avoids making meshgrids

    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny, disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax]*mask)
    return virtual_image

def _get_virtualimage_ann_old(datacube, geometry, verbose=True, *args, **kwargs):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro.
    Args:
        datacube (DataCube):
        geometry (2-tuple): (center,radii), where center is the 2-tuple (qx0,qy0),
        and radii is the 2-tuple (ri,ro)
    Returns:
        (2D array): the virtual image
    """
    (x0,y0),(Ri,Ro) = geometry
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
#### End of py4DSTEM funcs V0.13.0 ####

#### Dask Function #### 

@da.as_gufunc(signature='(i,j),(i,j)->()', output_dtypes=np.float64, axes=[(2,3),(0,1),()], vectorize=True)
def _get_virtual_image_dask(array, mask):
    """
    Make a virtual for all probe posistions from a dask array object using a mask in the diffraction plane.
    Example:
    image = make_virtual_image_dask(dataset.data, mask).compute()

    Args:
        array (dask array): dask array of 4DSTEM data with shape rx,ry,qx,qy
        mask (2D numpy array): mask from which virtual image is generated, must be the same shape as diffraction pattern
    Returns:
        out (2D numpy array): virtual image lazy virtual image, requires explict computation
    """
    val = np.sum(np.multiply(array,mask), dtype=np.float64)
    return val

def _get_virtualimage_from_mask_dask(datacube, mask, eager_compute=True, *args, **kwargs):
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

def _get_virtualimage_ann_dask(datacube, geometry, eager_compute=True, *args, **kwargs):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro.

    This should only be used on datasets which are dask arrays.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radii), where center is the 2-tuple (qx0,qy0),
        and radii is the 2-tuple (ri,ro)
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the annular mask
    mask = make_annular_mask(datacube, geometry)

    if eager_compute:
            return _get_virtual_image_dask(datacube.data, mask).compute()
    else:

        return _get_virtual_image_dask(datacube.data, mask)

def _get_virtualimage_circ_dask(datacube, geometry , eager_compute=True, *args, **kwargs):

    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro.

    This should only be used on datasets which are dask arrays.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
        and radius is a number
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the circular mask

    mask = make_circ_mask(datacube, geometry)

    if eager_compute:
        return _get_virtual_image_dask(datacube.data, mask).compute()
    else:
        return _get_virtual_image_dask(datacube.data, mask)

def _get_virtualimage_rect_dask(datacube, geometry, eager_compute=True, *args, **kwargs):
    """
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.


    Args:
        datacube (DataCube):
        geometry (4-tuple of ints): (qxmin,qxmax,qymin,qymax)
        eager_compute (bool, optional): Flag if it should return virtual image as a numpy or dask array. Defaults to True.

    Returns:
        if eager_compute == True:
            (2D array): the virtual image
        else:
            (Lazy dask array): Returns a lazy dask array object which can be subsequently computed using array.compute() syntax
    """

    # make the rectangular mask
    mask = make_rect_mask(datacube, geometry)


    if eager_compute:
        return _get_virtual_image_dask(datacube.data, mask).compute()
    else:
        return _get_virtual_image_dask(datacube.data, mask)


#### End of Dask Functions ####

#### Einsum Powered Functions ####
# TODO I could probably use the boolean array indexes as well rather than multiplication - need to check speeds

def _get_virtualimage_from_mask_einsum(datacube, mask, dtype=np.float64, *args, **kwargs):
    """
    Create a virtual image from a generic mask, i.e. both boolean or non-boolean, the mask and diffraction slices must be the same shape

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        mask (2D array): This can be any mask i.e. both boolean or non-boolean, but it must be the same size as the diffraction slice

    Returns:
        (2D array): the virtual image

    """

    return np.einsum('ijnm,nm->ij', datacube.data, mask, dtype=dtype)

def _get_virtualimage_ann_einsum(datacube, geometry, dtype=np.float64, *args, **kwargs):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer
    radii of Ri/Ro.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radii), where center is the 2-tuple (qx0,qy0),
        and radii is the 2-tuple (ri,ro)

    Returns:
        (2D array): the virtual image
    """

    # make the annular mask
    mask = make_annular_mask(datacube, geometry)

    return np.einsum('ijnm,nm->ij', datacube.data, mask, dtype=dtype)

def _get_virtualimage_circ_einsum(datacube, geometry, dtype=np.float64, *args, **kwargs):

    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
        and radius is a number

    Returns:
        (2D array): the virtual image
    """
    # make the circular mask
    mask, (xmin,xmax,ymin,ymax) = make_circ_mask(datacube, geometry, return_crop_vals=True)

    return np.einsum('ijnm,nm->ij', datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax], dtype=dtype)

def _get_virtualimage_rect_einsum(datacube, geometry, dtype=np.float64, *args, **kwargs):
    """
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.

    Args:
        datacube (DataCube):
        geometry (4-tuple of ints): (qxmin,qxmax,qymin,qymax)

    Returns:
        (2D array): the virtual image
    """

    # make the rectangular mask
    mask, (xmin,xmax,ymin,ymax) = make_rect_mask(datacube, geometry, return_crop_vals=True)

    return np.einsum('ijnm,nm->ij', datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax], dtype=dtype)

#### End of Einsum Powered Functions ####

#### Tensordot Powered Functions ####
def _get_virtualimage_from_mask_tensordot(datacube, mask, *args, **kwargs):
    """
    Create a virtual image from a generic mask, i.e. both boolean or non-boolean, the mask and diffraction slices must be the same shape

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        mask (2D array): This can be any mask i.e. both boolean or non-boolean, but it must be the same size as the diffraction slice

    Returns:
        (2D array): the virtual image
    """

    return np.tensordot(datacube.data, mask, axes=((2,3),(0,1)))

def _get_virtualimage_ann_tensordot(datacube, geometry, *args, **kwargs):
    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
        and radius is a number

    Returns:
        (2D array): the virtual image
    """

    # make the annular mask
    mask = make_annular_mask(datacube, geometry)

    return np.tensordot(datacube.data, mask, axes=((2,3),(0,1)))

def _get_virtualimage_circ_tensordot(datacube, geometry, spicy=False, *args, **kwargs):

    """
    Get a virtual image using an circular detector centered at (x0,y0), with a
    radius of Ri/Ro.

    Args:
        datacube (DataCube): DataCube object where datacube.data is a dask array
        geometry (2-tuple): (center,radius), where center is the 2-tuple (qx0,qy0),
        and radius is a number
        spicy (bool): dictates if fancy indexing is used (True) or not (False). This will be depreciated after testing

    Returns:
        (2D array): the virtual image
    """

    # make the circular mask
    mask, (xmin,xmax,ymin,ymax) = make_circ_mask(datacube, geometry, return_crop_vals=True)

    if spicy:
        return np.tensordot(datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax], axes=((2,3),(0,1)))
    else:
        return np.tensordot(datacube.data, mask, axes=((2,3),(0,1)))

def _get_virtualimage_rect_tensordot(datacube, geometry, spicy=False, *args, **kwargs):
    """
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax)
    in the diffraction plane. Floating point limits will be rounded and cast to ints.

    Args:
        datacube (DataCube):
        geometry (4-tuple of ints): (qxmin,qxmax,qymin,qymax)
        spicy (bool): dictates if fancy indexing is used (True) or not (False). This will be depreciated after testing

    Returns:
        (2D array): the virtual image
    """

    # make the rectangular mask
    mask, (xmin,xmax,ymin,ymax) = make_rect_mask(datacube, geometry, return_crop_vals=True)

    if spicy:
        return np.tensordot(datacube.data[:,:,xmin:xmax, ymin:ymax], mask[xmin:xmax, ymin:ymax], axes=((2,3),(0,1)))
    else:
        return np.tensordot(datacube.data, mask, axes=((2,3),(0,1)))

### End of Tensordot Powered Functions ####

### Facade Function and Helper Functions ####

def _infer_image_type_from_geometry(geometry, *args, **kwargs):

    """
    Takes a geometry of nested tuples and infers the detector type and returns corresponding string.

    This isa hardcoded method reliant on the max depth of the nesting is 1 i.e. ((10,10), (20,20))

    There is probably a nicer way to do this
    """

    # extract the 
    shape = [np.array(a).shape for a  in np.array(geometry, dtype='object')]

    # cycle through the mask types
    # I could do this as a dictionary but this is simple as well

    # cicular detector
    if shape == [(2,),()]:
        return 'circ'
    # annular dectector
    elif shape == [(2,), (2,)]:
        return 'ann'
    # rectangular detector
    elif shape == [(),(),(),()]:
        return 'rect'
    elif shape == [(),()]:
        return 'point'

    # TODO add square detector, but that should be formatted as (qx0,qy0,s) -> [(3,)] so as not to conflict with circular  
    # elif shape == [(3,)]:
    #     return 'square'

    # Raise Exception for incorrectly formatted geometries
    else:
        raise Exception("Could not infer the detector type from the geometry")

def _make_function_dict():
    """
    Function which creates a dictionary with the prefered image functions for various datacube array dtype, mask/geometry, mask type etc.
    """
    function_dict = {
        # mode
        'geometry' : {
            # detector_geometry
            'circ' : {
                # data_type
                'numpy' :_get_virtualimage_circ_old, # changed from tensordot
                'dask' : _get_virtualimage_circ_dask
            },
            # detector_geometry
            'ann' : {
                # data_type
                'numpy' : _get_virtualimage_ann_old, # changed from tensordot
                'dask' : _get_virtualimage_ann_dask,

            },
            # detector_geometry
            'rect' : {
                # data_type
                'numpy' : _get_virtualimage_rect_old, # changed from tensordot
                'dask' : _get_virtualimage_rect_dask
            },
        },
        # mode
        'mask' : {
            # data_type
            'numpy' : {
                # mask_type
                'bool' : _get_virtualimage_from_mask_einsum, # changed from tensordot
                'non-bool' : _get_virtualimage_from_mask_einsum
            },
            # data_type
            'dask' : {
                # mask_type
                'bool' : _get_virtualimage_from_mask_dask,
                'non-bool' : _get_virtualimage_from_mask_dask
            }
        }
    }
    return function_dict


def get_virtualimage(datacube, geometry=None, mask=None, eager_compute=True, *args, **kwargs):

    """
    Get a virtual image from a py4DSTEM datacube object, and will operate on in memory (np.ndarray), memory mapped (np.memmap) or dask arrays (da.Array)
    This function can be operated in two modes:
        - passing a detector geometry, will generate a boolean mask corresponding to detector geometry these require a tuple to define the detector geometry:
        - passing a mask:

    This function is a high level function and calls sub functions from within. Users may prefer to use these subfunctions:

    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_dask - operating on dask array objects
    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_einsum - operating on numpy objects with non-boolean masks
    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_tensordot - operating on numpy objects with boolean masks
    py4DSTEM.process.virtualimage.make_circ_mask - make a circular boolean mask
    py4DSTEM.process.virtualimage.make_annular_mask' - make a annular boolean mask
    py4DSTEM.process.virtualimage.make_rect_mask - make rectangular boolean mask
    py4DSTEM.process.virtualimage.combine_masks - function to combine boolean masks
    py4DSTEM.process.virtualimage.plot_mask_overlay - tool for visualising a detector boolean or non-boolean masks

    Args:
        datacube (DataCube):
        geometry (nested tuple, optional): Tuple defining the geoemtry of the detector,
                    - 'rect':   (4-tuple) the corners (qx0,qxf,qy0,qyf)
                    - 'circ':   (2-tuple) (center,radius) where center=(qx0,qy0)
                    - 'ann':    (2-tuple) (center,radii) where center=(qx0,qy0) and
                                radii=(ri,ro)
        mask (2D array, optional): numpy array defining a mask, either boolean or non-boolean.  Must be the same size as the
            diffraction pattern
        eager_compute(boolean, optional): if datacube.data is a dask.Array defines if it returns a 2D image or lazy dask object,
            or if datacube.data is numpy.ndarray this does nothing.
    Returns:
        if dask.Array & eager_compute:
            (2D array): the virtual image
        if dask.Array & eager_compute ==False:
            (lazy 2D array): Lazy dask object which maybe computed to generate virtual image
        if numpy.ndarray or numpy.memmap:
            (2D array): the virtual image


    """
    # TODO add ability to pass both mask and geometry where mask acts as bad pixels e.g. beam stop
    # I decided to do this with switch like statements using a dictionary, in python 3.10, we could use them explicitly. 
    # This should make it easier to split into two functions if that is the prefered route

    # check one of geometry or mask is passed
    # I could use np.all(mask) != None, but I want to check its a numpy array as well
    assert (geometry is not None) ^ (mask is not None), "Either, neither or both geometry or mask passed"

    # create the dictionary with all prefered virtual image functions
    function_dict = _make_function_dict()

    ### Set flags for deciding what function to use ### 

    # check the datacube data type, three data types we should expect are, np.ndarray, np.memmap, da.Array 
    # dask array
    if type(datacube.data) == da.Array:
        data_type = 'dask'
    # numpy array or memory mapped array or h5py dataset which are conveted to numpy objects at operation
    elif type(datacube.data) == np.ndarray or type(datacube.data) == np.memmap or type(datacube.data) == h5py.Dataset:
        data_type = 'numpy'
    # handle unexpected type, this shouldn't be possible but just incase
    else:
        raise Exception(f"Unexpected datacube array data type, {type(datacube.data)}")

    # if geometry settings are passed, this will take the highest prioirty compared to passing a mask
    if geometry != None:
        mode = 'geometry'
        # _infer_image_type_from_geometry will raise exception if it cannot parse the tupple
        detector_geometry = _infer_image_type_from_geometry(geometry)

        # key will be 'geometry','circ'/'ann'/'rect'..., 
        image_function = function_dict[mode][detector_geometry][data_type]

        return image_function(datacube, geometry, eager_compute=eager_compute)

    # if mask is passed and geometry is not passed
    elif type(mask) == np.ndarray:
        #check the mask is the same shape as the diffraction slices 

        assert datacube.data[0,0].shape == mask.shape, "mask and diffraction pattern shapes do not match"
        mode = 'mask'
        # check if the mask is boolean or not
        if mask.dtype == bool:
            mask_type = 'bool'
        else:
            mask_type = 'non-bool'

        # key will be 'mask','dask'/'numpy','bool'/'non-bool' to find corresponding function 
        image_function = function_dict[mode][data_type][mask_type]

        return image_function(datacube, mask, eager_compute=eager_compute)
    else:
        raise Exception("Neither Geometry or Mask were passed")
