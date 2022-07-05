# Functions for generating diffraction images

import numpy as np



def get_diffraction_image(
    datacube,
    mode,
    geometry = None,
    shift_corr = False
    ):
    """
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

    """
    # parse args
    assert( mode in ('max','mean','median') )
    if geometry is None:
        region = 'all'
    elif isinstance(geometry, tuple):
        if len(geometry) == 4:
            region = 'subcube'
        elif len(geometry) == 2:
            region = 'selected'
        else:
            raise Exception("invalid `geometry` argument passed")
    elif isinstance(geometry, np.ndarray):
        er = "`geometry` and `datacube` diffraction space shapes must match"
        assert( geometry.shape == datacube.Rshape ), er
        if geometry.dtype == bool:
            region = 'mask'
        else:
            er = "non-boolean masks are only supported for 'mean' mode"
            assert( mode == 'mean' ), er
            region = 'mask_float'


    # select a function
    function_dict = _make_function_dict()
    fn = function_dict[mode][region][shift_corr]

    # run and return
    dp = fn(datacube, geometry)
    return dp



def _make_function_dict():
    """
    Creates a dictionary to select which function to call
    """
    function_dict = {
        # mode
        'max' : {
            # geometry
            'all' : {
                # shift correction
                True : _exception_shiftcorr,
                False : _get_dp_max_all
            },
            'subcube' : {
                True : _exception_shiftcorr,
                False : _get_dp_max_subcube
            },
            'selected' : {
                True : _exception_shiftcorr,
                False : _get_dp_max_selected
            },
            'mask' : {
                True : _exception_shiftcorr,
                False : _get_dp_max_mask
            },
            'mask_float' : {
                True : _exception_shiftcorr,
                False : _exception_nofloat
            },
        },

        # mode
        'mean' : {
            # geometry
            'all' : {
                # shift correction
                True : _exception_shiftcorr,
                False : _get_dp_mean_all
            },
            'subcube' : {
                True : _exception_shiftcorr,
                False : _get_dp_mean_subcube
            },
            'selected' : {
                True : _exception_shiftcorr,
                False : _get_dp_mean_selected
            },
            'mask' : {
                True : _exception_shiftcorr,
                False : _get_dp_mean_mask
            },
            'mask_float' : {
                True : _exception_shiftcorr,
                False : _get_dp_mean_mask_float
            },
        },

        # mode
        'median' : {
            # geometry
            'all' : {
                # shift correction
                True : _exception_shiftcorr,
                False : _get_dp_median_all
            },
            'subcube' : {
                True : _exception_shiftcorr,
                False : _get_dp_median_subcube
            },
            'selected' : {
                True : _exception_shiftcorr,
                False : _get_dp_median_selected
            },
            'mask' : {
                True : _exception_shiftcorr,
                False : _get_dp_median_mask
            },
            'mask_float' : {
                True : _exception_shiftcorr,
                False : _exception_nofloat
            },
        },

    }

    return function_dict


def _exception(**kwargs):
    raise Exception('this function has not been added yet!')
def _exception_shiftcorr(**kwargs):
    raise Exception('shift corrected functions have not been added yet!')
def _exception_nofloat(**kwargs):
    raise Exception('floating point masks are only valid in "mean" mode!')




# Max

def get_dp_max(
    datacube,
    geometry = None,
    shift_corr = False
    ):
    """
    Returns the maximal value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        geometry (variable): specifies the region used in the computation.
            See the `py4DSTEM.process.virtualimage.get_diffraction_image`
            docstring for details.
        shift_corr (bool): if True, corrects for beam shift

    Returns:
        (2D array): the maximal diffraction pattern
    """
    return get_diffraction_image(
        datacube,
        mode = 'max',
        geometry = geometry,
        shift_corr = shift_corr
    )


# no shift correction

def _get_dp_max_all(datacube,geometry=None):
    """
    """
    dp = np.max(datacube.data, axis=(0,1))
    return dp

def _get_dp_max_subcube(datacube,geometry):
    """
    """
    xmin,xmax,ymin,ymax = geometry
    dp = np.max(datacube.data[xmin:xmax,ymin:ymax,:,:], axis=(0,1))
    return dp

def _get_dp_max_selected(datacube,geometry):
    """
    """
    rx,ry = geometry
    dp = np.max(datacube.data[rx,ry,:,:], axis=(0))
    return dp

def _get_dp_max_mask(datacube,geometry):
    """
    """
    geometry = geometry.astype(bool)
    dp = np.max(datacube.data[geometry,:,:], axis=(0))
    return dp



# TODO add shift correction







# Median

def get_dp_median(
    datacube,
    geometry = None,
    shift_corr = False
    ):
    """
    Returns the median value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        geometry (variable): specifies the region used in the computation.
            See the `py4DSTEM.process.virtualimage.get_diffraction_image`
            docstring for details.
        shift_corr (bool): if True, corrects for beam shift

    Returns:
        (2D array): the median diffraction pattern
    """
    return get_diffraction_image(
        datacube,
        mode = 'median',
        geometry = geometry,
        shift_corr = shift_corr
    )


# no shift correction

def _get_dp_median_all(datacube,geometry=None):
    """
    """
    dp = np.median(datacube.data, axis=(0,1))
    return dp

def _get_dp_median_subcube(datacube,geometry):
    """
    """
    xmin,xmax,ymin,ymax = geometry
    dp = np.median(datacube.data[xmin:xmax,ymin:ymax,:,:], axis=(0,1))
    return dp

def _get_dp_median_selected(datacube,geometry):
    """
    """
    rx,ry = geometry
    dp = np.median(datacube.data[rx,ry,:,:], axis=(0))
    return dp

def _get_dp_median_mask(datacube,geometry):
    """
    """
    geometry = geometry.astype(bool)
    dp = np.median(datacube.data[geometry,:,:], axis=(0))
    return dp



# TODO add shift correction







# Median

def get_dp_mean(
    datacube,
    geometry = None,
    shift_corr = False
    ):
    """
    Returns the mean value of each diffraction space detector pixel.

    Args:
        datacube (Datacube)
        geometry (variable): specifies the region used in the computation.
            See the `py4DSTEM.process.virtualimage.get_diffraction_image`
            docstring for details.
        shift_corr (bool): if True, corrects for beam shift

    Returns:
        (2D array): the mean diffraction pattern
    """
    return get_diffraction_image(
        datacube,
        mode = 'mean',
        geometry = geometry,
        shift_corr = shift_corr
    )


# no shift correction

def _get_dp_mean_all(datacube,geometry=None):
    """
    """
    dp = np.mean(datacube.data, axis=(0,1))
    return dp

def _get_dp_mean_subcube(datacube,geometry):
    """
    """
    xmin,xmax,ymin,ymax = geometry
    dp = np.mean(datacube.data[xmin:xmax,ymin:ymax,:,:], axis=(0,1))
    return dp

def _get_dp_mean_selected(datacube,geometry):
    """
    """
    rx,ry = geometry
    dp = np.mean(datacube.data[rx,ry,:,:], axis=(0))
    return dp

def _get_dp_mean_mask(datacube,geometry):
    """
    """
    geometry = geometry.astype(bool)
    dp = np.mean(datacube.data[geometry,:,:], axis=(0))
    return dp

def _get_dp_mean_mask_float(datacube,geometry):
    """
    """
    mask = geometry>0
    dp = np.average(datacube.data[mask,:,:],
        weights=geometry[mask], axis=(0))
    return dp



# TODO add shift correction









