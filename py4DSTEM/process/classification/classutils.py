# Utility functions for classification routines

import numpy as np
from ..utils import get_shifted_ar, tqdmnd
from ...io import DataCube, PointListArray

def get_class_DP(datacube, class_image, thresh=0.01, xshifts=None, yshifts=None,
                 darkref=None, intshifts=True):
    """
    Get the average diffraction pattern for the class described in real space by
    class_image.

    Args:
        datacube (DataCube): a datacube
        class_image (2D array): the weight of the class at each position in real space
        thresh (float): only include diffraction patterns for scan positions with a value
            greater than or equal to thresh in class_image
        xshifts (2D array, or None): the x diffraction shifts at each real space pixel.
            If None, no shifting is performed.
        yshifts (2D array, or None): the y diffraction shifts at each real space pixel.
            If None, no shifting is performed.
        darkref (2D array, or None): background to remove from each diffraction pattern
        intshifts (bool): if True, round shifts to the nearest integer to speed up
            computation

    Returns:
        (2D array): the average diffraction pattern for the class
    """
    assert isinstance(datacube,DataCube)
    assert class_image.shape == (datacube.R_Nx,datacube.R_Ny)
    if xshifts is not None:
        assert xshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if yshifts is not None:
        assert yshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if darkref is not None:
        assert darkref.shape == (datacube.Q_Nx,datacube.Q_Ny)
    assert isinstance(intshifts, bool)

    class_DP = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    for (Rx,Ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Computing class diffraction pattern',unit='DP',unit_scale=True):
        if class_image[Rx,Ry] >= thresh:
            curr_DP = class_image[Rx,Ry]*datacube.data[Rx,Ry,:,:]
            if xshifts is not None and yshifts is not None:
                xshift = xshifts[Rx,Ry]
                yshift = yshifts[Rx,Ry]
                if intshifts is True:
                    xshift = int(np.round(xshift))
                    yshift = int(np.round(yshift))
                    curr_DP = np.roll(curr_DP,-xshift,axis=0)
                    curr_DP = np.roll(curr_DP,-yshift,axis=1)
                else:
                    curr_DP = get_shifted_ar(curr_DP,-xshift,-yshift)
            class_DP += curr_DP
            if darkref is not None:
                class_DP -= darkref*class_image[Rx,Ry]
    class_DP /= np.sum(class_image[class_image>=thresh])
    class_DP = np.where(class_DP>0,class_DP,0)
    return class_DP

def get_class_DP_without_Bragg_scattering(datacube,class_image,braggpeaks,radius,
                                          x0,y0,thresh=0.01,xshifts=None,yshifts=None,
                                          darkref=None,intshifts=True):
    """
    Get the average diffraction pattern, removing any Bragg scattering, for the class
    described in real space by class_image.

    Bragg scattering is eliminated by masking circles of size radius about each of the
    detected peaks in braggpeaks in each diffraction pattern before adding to the average
    image. Importantly, braggpeaks refers to the peak positions in the raw data - i.e.
    BEFORE any shift correction is applied.  Passing shifted Bragg peaks will yield
    incorrect results.  For speed, the Bragg peaks are removed with a binary mask, rather
    than a continuous sigmoid, so selecting a radius that is slightly (~1 pix) larger
    than the disk size is recommended.

    Args:
        datacube (DataCube): a datacube
        class_image (2D array): the weight of the class at each position in real space
        braggpeaks (PointListArray): the detected Bragg peak positions, with respect to
            the raw data (i.e. not diffraction shift or ellipse corrected)
        radius (number): the radius to mask about each detected Bragg peak - should be
            slightly larger than the disk radius
        x0 (number): x-position of the optic axis
        y0 (number): y-position of the optic axis
        thresh (float): only include diffraction patterns for scan positions with a value
            greater than or equal to thresh in class_image
        xshifts (2D array, or None): the x diffraction shifts at each real space pixel.
            If None, no shifting is performed.
        yshifts (2D array, or None): the y diffraction shifts at each real space pixel.
            If None, no shifting is performed.
        darkref (2D array, or None): background to remove from each diffraction pattern
        intshifts (bool): if True, round shifts to the nearest integer to speed up
            computation

    Returns:
        class_DP        (2D array) the average diffraction pattern for the class
    """
    assert isinstance(datacube,DataCube)
    assert class_image.shape == (datacube.R_Nx,datacube.R_Ny)
    assert isinstance(braggpeaks, PointListArray)
    if xshifts is not None:
        assert xshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if yshifts is not None:
        assert yshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if darkref is not None:
        assert darkref.shape == (datacube.Q_Nx,datacube.Q_Ny)
    assert isinstance(intshifts,bool)

    class_DP = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    mask_weights = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    yy,xx = np.meshgrid(np.arange(datacube.Q_Ny),np.arange(datacube.Q_Nx))
    for (Rx,Ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Computing class diffraction pattern',unit='DP',unit_scale=True):
        weight = class_image[Rx,Ry]
        if weight >= thresh:
            braggpeaks_curr = braggpeaks.get_pointlist(Rx,Ry)
            mask = np.ones((datacube.Q_Nx,datacube.Q_Ny))
            if braggpeaks_curr.length > 1:
                center_index = np.argmin(np.hypot(braggpeaks_curr.data['qx']-x0,
                                                  braggpeaks_curr.data['qy']-y0))
                for i in range(braggpeaks_curr.length):
                    if i != center_index:
                        mask_ = ((xx-braggpeaks_curr.data['qx'][i])**2 + \
                                 (yy-braggpeaks_curr.data['qy'][i])**2) >= radius**2
                        mask = np.logical_and(mask,mask_)
            curr_DP = datacube.data[Rx,Ry,:,:]*mask*weight
            if xshifts is not None and yshifts is not None:
                xshift = xshifts[Rx,Ry]
                yshift = yshifts[Rx,Ry]
                if intshifts:
                    xshift = int(np.round(xshift))
                    yshift = int(np.round(yshift))
                    curr_DP = np.roll(curr_DP,-xshift,axis=0)
                    curr_DP = np.roll(curr_DP,-yshift,axis=1)
                    mask = np.roll(mask,-xshift,axis=0)
                    mask = np.roll(mask,-yshift,axis=1)
                else:
                    curr_DP = get_shifted_ar(curr_DP,-xshift,-yshift)
                    mask = get_shifted_ar(mask,-xshift,-yshift)
            if darkref is not None:
                curr_DP -= darkref*weight
            class_DP += curr_DP
            mask_weights += mask*weight
    class_DP = np.divide(class_DP,mask_weights,where=mask_weights!=0,
                         out=np.zeros((datacube.Q_Nx,datacube.Q_Ny)))
    return class_DP


