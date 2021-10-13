# Functions for generating virtual images

import numpy as np
from ...io import DataCube
from ..utils import tqdmnd

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



def get_virtualimage(
    datacube, 
    mask, 
    verbose=True,
    return_mask_coords=False,
    ):
    """
    Get a virtual image using an arbitrary boolean mask

    Args:
        datacube (DataCube):    Input datacube with dimensions (R_Nx, R_Nx, Q_Nx, Q_Ny)
        mask (bool):            Mask with dimensions (Q_Nx, Q_Ny)

    Returns:
        (2D array): the virtual image
    """
    assert isinstance(datacube, DataCube)

    # Segment mask into rectangular regions
    # Adapted from: https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix
    # init arrays
    w = np.zeros(dtype=int, shape=mask.shape)
    h = np.zeros(dtype=int, shape=mask.shape)
    coords = []
    count = 0
    mask_mark = ~mask.copy()

    while not np.all(mask_mark):
        area_max = (0, [])
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask_mark[r][c]:
                    continue
                if r == 0:
                    h[r][c] = 1
                else:
                    h[r][c] = h[r-1][c]+1
                if c == 0:
                    w[r][c] = 1
                else:
                    w[r][c] = w[r][c-1]+1
                minw = w[r][c]
                for dh in range(h[r][c]):
                    minw = min(minw, w[r-dh][c])
                    area = (dh+1)*minw
                    if area > area_max[0]:
                        area_max = (area, [(r-dh, c-minw+1, r, c)])


        mask_mark[ \
            area_max[1][0][0]:area_max[1][0][2]+1,\
            area_max[1][0][1]:area_max[1][0][3]+1] = True
        coords.append(area_max[1][0])

    # init virtual image
    virtual_image = np.zeros((datacube.R_Nx, datacube.R_Ny))

    # Generate virtual image
    for c in tqdmnd(coords, disable=not verbose):
        virtual_image += np.sum(datacube.data[:,:,c[0]:c[2]+1,c[1]:c[3]+1], axis=(2,3))
    
    if return_mask_coords:
        return virtual_image, coords
    else:
        return virtual_image


