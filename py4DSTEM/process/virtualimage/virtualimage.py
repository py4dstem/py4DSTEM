# Functions for generating virtual images

import numpy as np
from ...io import DataCube
from ..utils import tqdmnd

def get_im(datacube,geometry=(0,0),detector='point'):
    """
    Return a virtual image.  Which image depends on the arguments `geometry` and
    `detector`.

    Args:
        datacube (DataCube)
        geometry (variable): the type and meaning of this argument depend on the
            value of the `detector` argument:
                - 'point': (2-tuple of ints) the (qx,qy) position of a point detector
                - 'rect': (4-tuple) the corners (qx0,qxf,qy0,qyf)
                - 'square': (2-tuple) ((qx0,qy0),s) where s is the sidelength and
                  (qx0,qy0) is the upper left corner
                - 'circ': (2-tuple) (center,radius) where center=(qx0,qy0)
                - 'ann': (2-tuple) (center,radii) where center=(qx0,qy0) and
                  radii=(ri,ro)
                - 'mask': (2D boolean array)
        detector (str): the detector type. Must be one of: 'point','rect','square',
            'circ', 'ann', or 'mask'

    Returns:
        (2D array): the virtual image
    """
    assert(detector in ('point','p','rect','rectangle','rectangular','r','square','sq','s','circ','circle','circular','c','ann','annulus','annular','a','mask','m')), "detector type '{}' not recognized".format(detector)
    if detector in ('point','p'): detector = 'point'
    if detector in ('rect','rectangle','rectangular','r'): detector = 'rect'
    if detector in ('square','sq','s'): detector = 'square'
    if detector in ('circ','circle','circular','c'): detector = 'circ'
    if detector in ('ann','annulus','annular','a'): detector = 'ann'
    if detector in ('mask','m'): detector = 'mask'

    if detector == 'point':
        im = get_virtualimage_point(datacube,geometry)
    elif detector == 'rect':
        im = get_virtualimage_rect(datacube,geometry)
    elif detector == 'square':
        (x0,y0),s = geometry
        g = (x0,x0+s,y0,y0+s)
        im = get_virtualimage_rect(datacube,g)
    elif detector == 'circ':
        im = get_virtualimage_circ(datacube,geometry)
    elif detector == 'ann':
        im = get_virtualimage_ann(datacube,geometry)
    elif detector == 'mask':
        im = get_virtualimage_mask(datacube,geometry)
    else:
        raise Exception("Unrecognized `detector` value '{}'".format(detector))

    return im

def get_virtualimage_point(datacube, geometry, verbose=True):
    """
    Get a virtual image using a point detector.

    Args:
        datacube (DataCube):
        geometry (2-tuple of ints): (qx,qy)

    Returns:
        (2D array): the virtual image
    """
    assert(len(geometry)==2 and all([isinstance(i,(int,np.integer)) for i in geometry])), "Scan position was specified incorrectly, must be a pair of integers"
    assert(geometry[0]<datacube.Q_Nx and geometry[1]<datacube.Q_Ny), "The requested scan position is outside the dataset"
    im = datacube.data[:,:,geometry[0],geometry[1]]
    return im

def get_virtualimage_rect(datacube, geometry, verbose=True):
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

def get_virtualimage_circ(datacube, geometry, verbose=True):
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

def get_virtualimage_ann(datacube, geometry, verbose=True):
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


def get_virtualimage_mask(
    datacube,
    mask,
    verbose=True,
    ):
    """
    Get a virtual image using an arbitrary boolean mask

    Args:
        datacube (DataCube):    Input datacube with dimensions (R_Nx, R_Nx, Q_Nx, Q_Ny)
        mask (bool):            Mask with dimensions (Q_Nx, Q_Ny)
        verbose (bool):         Use progress bar

    Returns:
        (2D array): the virtual image
    """
    assert isinstance(datacube, DataCube)

    # find range of True values in boolean mask
    x = np.where(np.max(mask, axis=1))
    y = np.where(np.max(mask, axis=0))
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    mask_sub = mask[xmin:xmax,ymin:ymax]

    # Generate virtual image
    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))
    for rx,ry in tqdmnd(datacube.R_Nx, datacube.R_Ny, disable=not verbose):
        virtual_image[rx,ry] = np.sum(datacube.data[rx,ry,xmin:xmax,ymin:ymax][mask_sub])

    return virtual_image


