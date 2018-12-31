# Functions for generating virtual images
#
# The general form of these functions is:
#       virtual_image = function(datacube, slice_x, slice_y, *args)
# where virtual_image is an ndarray of shape (R_Nx, R_Ny), datacube is a DataCube instance, and 
# slice_x and slice_y are python slice objects which refer to a rectangular ROI within the 
# diffraction plane. For non-rectangular detectors, additional input parameters specify the
# subset of this ROI used in generating virtual images.
#
# Most of these functions are also included as DataCube class methods.  Thus
#       virtual_image = function(datacube, slice_x, slice_y, *args)
# will be identical to
#       virtual_image = datacube.function(slice_x, slice_y, *args)

import numpy as np

# Utility functions

def get_ROI_dataslice_rect(datacube, slice_x, slice_y):
    """
    Returns a subset of datacube corresponding to a rectangular ROI in the diffraction plane
    specified by slice_x, slice_y
    """
    return datacube.data4D[:,:,slice_x,slice_y]

def get_circ_mask(size_x,size_y,R=1):
    """
    Returns a mask of shape (size_x,size_y) which is True inside an ellipse with major/minor
    diameters of R*size_x, R*size_y.  Thus if R=1 and size_x=size_y, returns a ciruclar mask which
    is inscribed inside a square array.
    Note that an ellipse, rather than a circle, is used to prevent failure when the slice objects
    returned when calling getArraySlice on a pyqtgraph circular ROI are off-by-one in length.
    """
    return np.fromfunction(lambda x,y: ( ((x+0.5)/(size_x/2.)-1)**2 + ((y+0.5)/(size_y/2.)-1)**2 ) < R**2, (size_x,size_y))


# Virtual images

def get_virtual_image_rect(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector in integration
    mode. Also returns a bool indicating success or failure.
    """
    try:
        return datacube.data4D[:,:,slice_x,slice_y].sum(axis=(2,3)), 1
    except ValueError:
        return 0,0

def get_virtual_image_circ(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in integration
    mode. Also returns a bool indicating success or failure.
    """
    try:
        return np.sum(datacube.data4D[:,:,slice_x,slice_y]*get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start), axis=(2,3)), 1
    except ValueError:
        return 0,0

def get_virtual_image_annular(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in integration
    mode. Also returns a bool indicating success or failure. The input parameter R is the ratio of
    the inner to the outer detector radii.
    """
    mask = np.logical_xor(get_circ_mask(slice_x.stop-slice_x.start,slice_y.stop-slice_y.start), get_circ_mask(slice_x.stop-slice_x.start,slice_y.stop-slice_y.start,R))
    try:
        return np.sum(datacube.data4D[:,:,slice_x,slice_y]*mask, axis=(2,3)), 1
    except ValueError:
        return 0,0














