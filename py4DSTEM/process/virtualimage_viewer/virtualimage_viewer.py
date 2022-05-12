# Functions for generating virtual images
#
# The general form of these functions is:
#       virtual_image = function(datacube, slice_x, slice_y, *args)
# where virtual_image is an ndarray of shape (R_Nx, R_Ny), datacube is a DataCube
# instance, and slice_x and slice_y are python slice objects which refer to a rectangular
# ROI within the diffraction plane. For non-rectangular detectors, additional input
# parameters specify the subset of this ROI used in generating virtual images.
#
# Most of these functions are also included as DataCube class methods.  Thus
#       virtual_image = function(datacube, slice_x, slice_y, *args)
# will be identical to
#       virtual_image = datacube.function(slice_x, slice_y, *args)

import numpy as np


# Utility functions

def get_ROI_dataslice_rect(datacube, slice_x, slice_y):
    """
    Returns a subset of datacube corresponding to a rectangular ROI in the diffraction
    plane specified by slice_x, slice_y
    """
    return datacube.data[:,:,slice_x,slice_y]

def get_circ_mask(size_x,size_y,R=1):
    """
    Returns a mask of shape (size_x,size_y) which is True inside an ellipse with
    major/minor diameters of R*size_x, R*size_y.  Thus if R=1 and size_x=size_y, returns
    a ciruclar mask which is inscribed inside a square array. Note that an ellipse,
    rather than a circle, is used to prevent failure when the slice objects returned when
    calling getArraySlice on a pyqtgraph circular ROI are off-by-one in length.
    """
    return np.fromfunction(lambda x,y: ( ((x+0.5)/(size_x/2.)-1)**2 + ((y+0.5)/(size_y/2.)-1)**2 ) < R**2, (size_x,size_y))

def get_annular_mask(size_x,size_y,R):
    """
    Returns an annular mask, where the outer annulus is inscribed in a rectangle of shape
    (size_x,size_y) - and can thus be elliptical - and the inner radius to outer radius
    ratio is R.
    """
    return np.logical_xor(get_circ_mask(size_x,size_y), get_circ_mask(size_x,size_y,R))


# Virtual images -- integrating detector

def get_virtual_image_rect_integrate(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector in
    integration mode. Also returns a bool indicating success or failure.
    """
    try:
        return datacube.data[:,:,slice_x,slice_y].sum(axis=(2,3)), 1
    except ValueError:
        return 0,0

def get_virtual_image_circ_integrate(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in
    integration mode. Also returns a bool indicating success or failure.
    """
    try:
        return np.sum(datacube.data[:,:,slice_x,slice_y]*get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start), axis=(2,3)), 1
    except ValueError:
        return 0,0

def get_virtual_image_annular_integrate(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in
    integration mode. Also returns a bool indicating success or failure. The input
    parameter R is the ratio of the inner to the outer detector radii.
    """
    mask = get_annular_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start, R)
    try:
        return np.sum(datacube.data[:,:,slice_x,slice_y]*mask, axis=(2,3)), 1
    except ValueError:
        return 0,0


# Virtual images -- difference

def get_virtual_image_rect_diffX(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector, in
    difference mode. Also returns a bool indicating success or failure.
    """
    try:
        midpoint = slice_x.start + (slice_x.stop-slice_x.start)/2
        slice_left = slice(slice_x.start, int(np.floor(midpoint)))
        slice_right = slice(int(np.ceil(midpoint)), slice_x.stop)
        img = datacube.data[:,:,slice_left,slice_y].sum(axis=(2,3)).astype('int64') - datacube.data[:,:,slice_right,slice_y].sum(axis=(2,3)).astype('int64')
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_rect_diffY(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector in
    difference mode. Also returns a bool indicating success or failure.
    """
    try:
        midpoint = slice_y.start + (slice_y.stop-slice_y.start)/2
        slice_bottom = slice(slice_y.start, int(np.floor(midpoint)))
        slice_top = slice(int(np.ceil(midpoint)), slice_y.stop)
        img = datacube.data[:,:,slice_x,slice_bottom].sum(axis=(2,3)).astype('int64') - datacube.data[:,:,slice_x,slice_top].sum(axis=(2,3)).astype('int64')
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_circ_diffX(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in
    difference mode. Also returns a bool indicating success or failure.
    """
    mask = get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start)
    try:
        midpoint = slice_x.start + (slice_x.stop-slice_x.start)/2
        slice_left = slice(slice_x.start, int(np.floor(midpoint)))
        slice_right = slice(int(np.ceil(midpoint)), slice_x.stop)
        img = np.ndarray.astype(np.sum(datacube.data[:,:,slice_left,slice_y]*mask[:slice_left.stop-slice_left.start,:],axis=(2,3)) - np.sum(datacube.data[:,:,slice_right,slice_y]*mask[slice_right.start-slice_right.stop:,:],axis=(2,3)), 'int64')
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_circ_diffY(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in
    differenece mode. Also returns a bool indicating success or failure.
    """
    mask = get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start)
    try:
        midpoint = slice_y.start + (slice_y.stop-slice_y.start)/2
        slice_bottom = slice(slice_y.start, int(np.floor(midpoint)))
        slice_top = slice(int(np.ceil(midpoint)), slice_y.stop)
        img = np.ndarray.astype(np.sum(datacube.data[:,:,slice_x,slice_bottom]*mask[:,:slice_bottom.stop-slice_bottom.start],axis=(2,3)) - np.sum(datacube.data[:,:,slice_x,slice_top]*mask[:,slice_top.start-slice_top.stop:],axis=(2,3)), 'int64')
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_annular_diffX(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in
    difference mode. Also returns a bool indicating success or failure. The input
    parameter R is the ratio of the inner to the outer detector radii.
    """
    mask = get_annular_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start, R)
    try:
        midpoint = slice_x.start + (slice_x.stop-slice_x.start)/2
        slice_left = slice(slice_x.start, int(np.floor(midpoint)))
        slice_right = slice(int(np.ceil(midpoint)), slice_x.stop)
        img = np.ndarray.astype(np.sum(datacube.data[:,:,slice_left,slice_y]*mask[:slice_left.stop-slice_left.start,:],axis=(2,3)) - np.sum(datacube.data[:,:,slice_right,slice_y]*mask[slice_right.start-slice_right.stop:,:],axis=(2,3)), 'int64')
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_annular_diffY(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in
    difference mode. Also returns a bool indicating success or failure. The input
    parameter R is the ratio of the inner to the outer detector radii.
    """
    mask = get_annular_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start, R)
    try:
        midpoint = slice_y.start + (slice_y.stop-slice_y.start)/2
        slice_bottom = slice(slice_y.start, int(np.floor(midpoint)))
        slice_top = slice(int(np.ceil(midpoint)), slice_y.stop)
        img = np.ndarray.astype(np.sum(datacube.data[:,:,slice_x,slice_bottom]*mask[:,:slice_bottom.stop-slice_bottom.start],axis=(2,3)) - np.sum(datacube.data[:,:,slice_x,slice_top]*mask[:,slice_top.start-slice_top.stop:],axis=(2,3)), 'int64')
        return img, 1
    except ValueError:
        return 0,0


# Virtual images -- center of mass

def get_virtual_image_rect_CoMX(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector, in CoM
    mode. Also returns a bool indicating success or failure.
    """
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*rx,axis=(2,3))/datacube.data[:,:,slice_x,slice_y].sum(axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_rect_CoMY(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a rectangular detector, in CoM
    mode. Also returns a bool indicating success or failure.
    """
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*ry,axis=(2,3))/datacube.data[:,:,slice_x,slice_y].sum(axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_circ_CoMX(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in CoM
    mode. Also returns a bool indicating success or failure.
    """
    mask = get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start)
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*rx*mask,axis=(2,3)) / np.sum(datacube.data[:,:,slice_x,slice_y]*mask,axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_circ_CoMY(datacube, slice_x, slice_y):
    """
    Returns a virtual image as an ndarray, generated from a circular detector in CoM
    mode. Also returns a bool indicating success or failure.
    """
    mask = get_circ_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start)
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*ry*mask,axis=(2,3)) / np.sum(datacube.data[:,:,slice_x,slice_y]*mask,axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_annular_CoMX(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in CoM
    mode. Also returns a bool indicating success or failure. The input parameter R is
    the ratio of the inner to the outer detector radii.
    """
    mask = get_annular_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start, R)
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*rx*mask,axis=(2,3)) / np.sum(datacube.data[:,:,slice_x,slice_y]*mask,axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0

def get_virtual_image_annular_CoMY(datacube, slice_x, slice_y, R):
    """
    Returns a virtual image as an ndarray, generated from an annular detector in CoM
    mode. Also returns a bool indicating success or failure. The input parameter R is the
    ratio of the inner to the outer detector radii.
    """
    mask = get_annular_mask(slice_x.stop-slice_x.start, slice_y.stop-slice_y.start, R)
    ry,rx = np.meshgrid(np.arange(slice_y.stop-slice_y.start),np.arange(slice_x.stop-slice_x.start))
    try:
        img = np.sum(datacube.data[:,:,slice_x,slice_y]*ry*mask,axis=(2,3)) / np.sum(datacube.data[:,:,slice_x,slice_y]*mask,axis=(2,3))
        return img, 1
    except ValueError:
        return 0,0







