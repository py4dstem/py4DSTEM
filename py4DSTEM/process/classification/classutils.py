# Utility functions for classification routines

import numpy as np
from ..utils import get_shifted_ar
from ...file.datastructure import DataCube, PointListArray

def get_class_DP(datacube, class_image, thresh=0.01, xshifts=None, yshifts=None, darkref=None):
    """
    Get the average diffraction pattern for the class described in real space by class_image.

    Accepts:
        datacube        (DataCube) a datacube
        class_image     (2D array) the weight of the class at each position in real space
        thresh          (float) only include diffraction patterns for scan positions with a value
                        greater than or equal to thresh in class_image
        xshifts         (2D array, or None) the x diffraction shifts at each real space pixel.
                        If None, no shifting is performed.
        yshifts         (2D array, or None) the y diffraction shifts at each real space pixel.
                        If None, no shifting is performed.
        darkref         (2D array, or None) background to remove from each diffraction pattern

    Returns:
        class_DP        (2D array) the average diffraction pattern for the class
    """
    assert isinstance(datacube,DataCube)
    assert class_image.shape == (datacube.R_Nx,datacube.R_Ny)
    if xshifts is not None:
        assert xshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if yshifts is not None:
        assert yshifts.shape == (datacube.R_Nx,datacube.R_Ny)
    if darkref is not None:
        assert darkref.shape == (datacube.Q_Nx,datacube.Q_Ny)

    class_DP = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    for Rx in range(datacube.R_Nx):
        for Ry in range(datacube.R_Ny):
            if class_image[Rx,Ry] >= thresh:
                curr_DP = class_image[Rx,Ry]*datacube.data[Rx,Ry,:,:]
                if xshifts is not None and yshifts is not None:
                    xshift = xshifts[Rx,Ry]
                    yshift = yshifts[Rx,Ry]
                    curr_DP = get_shifted_ar(curr_DP,-xshift,-yshift)
                class_DP += curr_DP
                if darkref is not None:
                    class_DP -= darkref*class_image[Rx,Ry]
    class_DP /= np.sum(class_image[class_image>=thresh])
    return class_DP

def get_class_DP_without_Bragg_scattering(datacube, class_image, braggpeaks, radius, x0, y0, thresh=0.01, xshifts=None, yshifts=None, darkref=None):
    """
    Get the average diffraction pattern, removing any Bragg scattering, for the class described in
    real space by class_image.

    Bragg scattering is eliminated by masking circles of size radius about each of the detected
    peaks in braggpeaks in each diffraction pattern before adding to the average image. Importantly,
    braggpeaks refers to the peak positions in the raw data - i.e. BEFORE any shift correction is
    applied.  Passing shifted Bragg peaks will yield incorrect results.  For speed, the Bragg peaks
    are removed with a binary mask, rather than a continuous sigmoid, so selecting a radius that is
    slightly (~1 pix) larger than the disk size is recommended.

    Accepts:
        datacube        (DataCube) a datacube
        class_image     (2D array) the weight of the class at each position in real space
        braggpeaks      (PointListArray) the detected Bragg peak positions, with respect to the
                        raw data (i.e. not diffraction shift or ellipse corrected)
        radius          (number) the radius to mask about each detected Bragg peak - should be
                        slightly larger than the disk radius
        x0              (number) x-position of the optic axis
        y0              (number) y-position of the optic axis
        thresh          (float) only include diffraction patterns for scan positions with a value
                        greater than or equal to thresh in class_image
        xshifts         (2D array, or None) the x diffraction shifts at each real space pixel.
                        If None, no shifting is performed.
        yshifts         (2D array, or None) the y diffraction shifts at each real space pixel.
                        If None, no shifting is performed.
        darkref         (2D array, or None) background to remove from each diffraction pattern

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

    class_DP = np.zeros((datacube.Q_Nx,datacube.Q_Ny))
    yy,xx = np.meshgrid(np.arange(datacube.Q_Ny),np.arange(datacube.Q_Nx))
    for Rx in range(datacube.R_Nx):
        for Ry in range(datacube.R_Ny):
            if class_image[Rx,Ry] >= thresh:
                braggpeaks_curr = braggpeaks.get_pointlist(Rx,Ry)
                mask = np.ones((datacube.Q_Nx,datacube.Q_Ny))
                if braggpeaks_curr.length != 1:
                    center_index = np.argmin(np.hypot(braggpeaks_curr.data['qx']-x0,
                                                      braggpeaks_curr.data['qy']-y0))
                    for i in range(braggpeaks_curr.length):
                        if i != center_index:
                            mask *= ((xx-braggpeaks_curr.data['qx'][i])**2 + \
                                     (yy-braggpeaks_curr.data['qy'][i])**2) >= radius**2
                curr_DP = class_image[Rx,Ry]*datacube.data[Rx,Ry,:,:]*mask
                if xshifts is not None and yshifts is not None:
                    xshift = xshifts[Rx,Ry]
                    yshift = yshifts[Rx,Ry]
                    curr_DP = get_shifted_ar(curr_DP,-xshift,-yshift)
                class_DP += curr_DP
                if darkref is not None:
                    class_DP -= darkref*class_image[Rx,Ry]
    class_DP /= np.sum(class_image[class_image>=thresh])
    return class_DP


