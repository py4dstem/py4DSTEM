# Functions for generating virtual images

import numpy as np
from ...file.datastructure import DataCube

def test():
    return True

def get_virtualimage_rect(datacube, xmin, xmax, ymin, ymax):
    """
    Get a virtual image using a rectagular detector with limits (xmin,xmax,ymin,ymax) in the diffraction plane.
    Floating point limits will be rounded and cast to ints.

    Accepts:
        datacube        (DataCube)
        xmin,xmax       (ints) x limits of the detector
        ymin,ymax       (ints) y limits of the detector

    Returns:
        virtual_image   (2D array)
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.round(xmin))),min(datacube.Q_Nx,int(np.round(xmax)))
    ymin,ymax = max(0,int(np.round(ymin))),min(datacube.Q_Ny,int(np.round(ymax)))

    virtual_image = np.sum(datacube.data[:,:,xmin:xmax,ymin:ymax], axis=(2,3))
    return virtual_image

def get_virtualimage_circ(datacube, x0, y0, R):
    """
    Get a virtual image using a circular detector centered at (x0,y0) and with radius R in the diffraction plane.

    Accepts:
        datacube        (DataCube)
        x0,y0           (numbers) center of detector
        R               (number) radius of detector

    Returns:
        virtual_image   (2D array)
    """
    assert isinstance(datacube, DataCube)
    xmin,xmax = max(0,int(np.floor(x0-R))),min(datacube.Q_Nx,int(np.ceil(x0+R)))
    ymin,ymax = max(0,int(np.round(y0-R))),min(datacube.Q_Ny,int(np.ceil(y0+R)))

    xsize,ysize = xmax-xmin,ymax-ymin
    x0_s,y0_s = x0-xmin,y0-ymin
    mask = np.fromfunction(lambda x,y: ((x-x0_s+0.5)**2 + (y-y0_s+0.5)**2) < R**2, (xsize,ysize)) # Avoids making meshgrids

    virtual_image = np.sum(datacube.data[:,:,xmin:xmax,ymin:ymax]*mask, axis=(2,3))
    return virtual_image

def get_virtualimage_ann(datacube, x0, y0, Ri, Ro):
    """
    Get a virtual image using an annular detector centered at (x0,y0), with inner/outer radii of Ri/Ro.

    Accepts:
        datacube        (DataCube)
        x0,y0           (numbers) center of detector
        Ri,Ro           (numbers) inner/outer detector radii

    Returns:
        virtual_image   (2D array)
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

    virtual_image = np.sum(datacube.data[:,:,xmin:xmax,ymin:ymax]*mask, axis=(2,3))
    return virtual_image






