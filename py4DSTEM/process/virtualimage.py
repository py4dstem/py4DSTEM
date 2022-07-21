# Functions for generating virtual images

import numpy as np
import dask.array as da
import h5py
import warnings

from ..utils.tqdmnd import tqdmnd




def get_virtual_image(
    datacube,
    mode,
    geometry,
    shift_corr = False,
    eager_compute = True
    ):
    """
    Computes and returns a virtual image from `datacube`. The
    kind of virtual image is specified by the `mode` argument.

    Args:
        datacube (Datacube)
        mode (str): must be in
            ('point','circle','annulus','rectangle',
            'cpoint','ccircle','cannulus','csquare',
            'qpoint','qcircle','qannulus','qsquare',
            'mask').  The first four modes represent point, circular,
            annular, and rectangular detectors with geomtries specified
            in pixels, relative to the uncalibrated origin, i.e. the upper
            left corner of the diffraction plane. The next four modes
            represent point, circular, annular, and square detectors with
            geometries specified in pixels, relative to the calibrated origin,
            taken to be the mean posiion of the origin over all scans.
            'ccircle','cannulus', and 'csquare' are automatically centered
            about the origin. The next four modes are identical to these,
            except that the geometry is specified in q-space units, rather
            than pixels. In the last mode the geometry is specified with a
            user provided mask, which can be either boolean or floating point.
            Floating point masks are normalized by setting their maximum value
            to 1.
        geometry (variable): valid entries are determined by the `mode`
            argument, as follows:
                - 'point': 2-tuple, (qx,qy)
                - 'circle': nested 2-tuple, ((qx,qy),r)
                - 'annulus': nested 2-tuple, ((qx,qy),(ri,ro))
                - 'rectangle': 4-tuple, (xmin,xmax,ymin,ymax)
                - 'cpoint': 2-tuple, (qx,qy)
                - 'ccircle': number, r
                - 'cannulus': 2-tuple, (ri,ro)
                - 'csquare': number, s
                - 'qpoint': 2-tuple, (qx,qy)
                - 'qcircle': number, r
                - 'qannulus': 2-tuple, (ri,ro)
                - 'qsquare': number, s
                - `mask`: 2D array
        shift_corr (bool): if True, correct for beam shift. Works only with
            'c' and 'q' modes - uses the calibrated origin for each pixel,
            instead of the mean origin position.

    Returns:
        (2D array): the virtual image

    """
    # parse args
    modes = ('point','circle','annulus','rectangle',
             'cpoint','ccircle','cannulus','csquare',
             'qpoint','qcircle','qannulus','qsquare',
             'mask')
    assert( mode in modes), f"`mode` was {mode}; must be in {modes}"
    g=geometry
    er = 'mode/geometry are mismatched'
    if mode in ('ccircle','csquare','qcircle','qsquare'):
        assert(isinstance(g,Number)), er
    elif mode in ('point','cpoint','cannulus','qpoint','qannulus'):
        assert(isinstance(g,tuple) and len(g)==2), er
    elif mode in ('circle'):
        assert(isinstance(g,tuple) and len(g)==2 and len(g[0])==2), er
    elif mode in ('annulus'):
        assert(isinstance(g,tuple) and len(g)==2 and
               all([len(g[i])==2 for i in (0,1)])), er
    elif mode in ('mask'):
        assert type(mask) == np.ndarray, "`mask` type should be `np.ndarray`"
        er = "mask and diffraction pattern shapes do not match"
        assert mask.shape == datacube.Qshape, er
        mode = 'mask' if g.dtype==bool else 'mask_float'
    else:
        raise Exception(f"Unknown mode {mode}")



    # select a function
    dtype = _infer_dtype(datacube)
    fn_dict = _make_function_dict()
    fn = fn_dict[mode][shift_corr][dtype]


    # run and return
    im = fn(datacube, geometry)
    return im




def _make_function_dict():
    """
    Creates a dictionary for selecting an imaging function
    """
    function_dict = {
        # mode
        'point' : {
            # shift corr
            True : {
                # data_type
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : lambda d:d.data[g[0],g[1],:,:],
                'dask' : _get_virtual_image_fn
            },
        },

        'circle' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtualimage_circ_old,
                'dask' : _get_virtual_image_fn
            },
        },

        'annulus' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtualimage_ann_old,
                'dask' : _get_virtual_image_fn
            },
        },

        'rectangle' : {
            True : {
                'numpy' : _get_virtualimage_rect_old,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        # c modes
        'cpoint' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'ccircle' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'cannulus' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'csquare' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        # q modes
        'qpoint' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'qcircle' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'qannulus' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'qsquare' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'mask' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        },

        'mask_float' : {
            True : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
            False : {
                'numpy' : _get_virtual_image_fn,
                'dask' : _get_virtual_image_fn
            },
        }
    }
    return function_dict



def _infer_dtype(datacube):

    # numpy array /
    # mem mapped np.array /
    # h5py dataset 
    if (type(datacube.data) == np.ndarray or
        isinstance(datacube.data, np.memmap) or
        isinstance(datacube.data, h5py.Dataset)):
        data_type = 'numpy'

    # dask array
    elif type(datacube.data) == da.Array:
        data_type = 'dask'

    else:
        er = f"Unexpected datacube array data type, {type(datacube.data)}"
        raise Exception(er)

    return data_type



def _get_virtual_image_fn(datacube):
    raise Exception("This functions doesn't exist yet!")




































def make_circ_mask(
    datacube,
    geometry,
    return_crop_vals=False
    ):
    """
    Make a circular boolean mask centered at (x0,y0) and with radius R
    in the diffraction plane. The mask returned is the same shape as each
    diffraction slice.

    If return_crop_vals is True, then they can be used to acceleate.

    Args:
        datacube (DataCube):
        geometry (2-tuple): (center,radius), where center is
            a 2-tuple (qx0,qy0), and radius is a number
        return_crop_vals (Boolean): boolean toggle to return
            indicies for cropping diffraction pattern

    Returns:
        (2D array): Boolean mask
        (tuple) : index values for croping diffraction pattern
            (xmin,xmax,ymin,ymax)
    """
    assert isinstance(datacube, DataCube)

    # unpack geometry
    (x0,y0),R = geometry

    # make mask
    xmin,xmax = max(0,int(np.floor(x0-R))),min(datacube.Q_Nx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(datacube.Q_Ny,int(np.ceil(y0+R)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize)) # Avoids making meshgrids

    full_mask = np.zeros(shape=datacube.data.shape[2:], dtype=np.bool_)
    full_mask[xmin:xmax,ymin:ymax] = mask


    # full mask is same size as diffraction space
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


#### BraggVector Functions ####

# TODO - add pointlistarrays as a possible input for the overall fascade function
def get_virtualimage_pointlistarray(
    peaks,
    geometry = None,
    ):
    """
    Make an annular boolean mask centered at (x0,y0), with inner/outer
    radii of Ri/Ro.
    Args:
        peaks (PointListArray): List of all peaks and intensities.
        geometry (2-tuple): (center,radii), where center is the 2-tuple (qx0,qy0),
                            and radii is either max angle, or a 2-tuple (ri,ro)
                            describing the inner and outer radial ranges.
    Returns:
        im_virtual (2D numpy array): the output virtual image
    """

    # Set geometry
    if geometry is None:
        radial_range = None
    else:
        if len(geometry[0]) == 0:
            origin = None
        else:
            origin = np.array(geometry[0])
        if isinstance(geometry[1], int) or isinstance(geometry[1], float):
            radial_range = np.array((0,geometry[1]))
        elif len(geometry[1]) == 0:
            radial_range = None
        else:
            radial_range = np.array(geometry[1])

    # init
    im_virtual = np.zeros(peaks.shape)

    # Generate image
    for rx,ry in tqdmnd(peaks.shape[0],peaks.shape[1]):
        p = peaks.get_pointlist(rx,ry)
        if p.data.shape[0] > 0:
            if radial_range is None:
                im_virtual[rx,ry] = np.sum(p.data['intensity'])
            else:
                if origin is None:
                    qr = np.hypot(p.data['qx'],p.data['qy'])
                else:
                    qr = np.hypot(p.data['qx'] - origin[0],p.data['qy'] - origin[1])
                sub = np.logical_and(
                    qr >= radial_range[0],
                    qr <  radial_range[1])
                if np.sum(sub) > 0:
                    im_virtual[rx,ry] = np.sum(p.data['intensity'][sub])

    return im_virtual


#### Mask Making Functions ####
# lifted from py4DSTEM old funcs
#TODO Add symmetry mask maker, e.g. define one spot, add symmetry related reflections 
#TODO Add multiple mask maker, e.g. given list of coordinate tuples create circular masks at each point
#TODO Add assertion statements 






#TODO add automagic functions that will pick dask or normal depending on the array type - in progress 
#TODO add alias names for get get_BF, get_ADF? 







#TODO Work out how to handle name space to access underlying __functions__, use __all__ or something like that 

#__all__ = [
#    'make_circ_mask',
#    'make_annular_mask',
#    'make_rect_mask',
#    'combine_masks',
#    'plot_mask_overlay',
#    'get_virtualimage',
#    '_get_virtualimage_rect_old',
#    '_get_virtualimage_circ_old',
#    '_get_virtualimage_ann_old',
#    '_infer_detector_geometry',
#    '_get_virtualimage_from_mask_dask',
#    '_get_virtualimage_from_mask_einsum',
#    '_get_virtualimage_from_mask_tensordot'
#    ]




#    This function is a high level function and calls sub functions from within. Users may prefer to use these subfunctions:
#
#    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_dask - operating on dask array objects
#    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_einsum - operating on numpy objects with non-boolean masks
#    py4DSTEM.process.virtualimage._get_virtualimage_from_mask_tensordot - operating on numpy objects with boolean masks
#    py4DSTEM.process.virtualimage.make_circ_mask - make a circular boolean mask
#    py4DSTEM.process.virtualimage.make_annular_mask' - make a annular boolean mask
#    py4DSTEM.process.virtualimage.make_rect_mask - make rectangular boolean mask
#    py4DSTEM.process.virtualimage.combine_masks - function to combine boolean masks
#    py4DSTEM.process.virtualimage.plot_mask_overlay - tool for visualising a detector boolean or non-boolean masks





# Notes from top of get_virtualimage():

    # TODO add ability to pass both mask and geometry where mask acts as bad pixels e.g. beam stop
    # I decided to do this with switch like statements using a dictionary, in python 3.10, we could use them explicitly. 
    # This should make it easier to split into two functions if that is the prefered route

    # check one of geometry or mask is passed
    # I could use np.all(mask) != None, but I want to check its a numpy array as well


