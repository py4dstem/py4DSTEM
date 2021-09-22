"""
Functions for generating radially averaged backgrounds
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from ..utils import cartesian_to_polarelliptical_transform

## Create look up table for background subtraction
def get_1D_polar_background(data, qx0, qy0 ,e, phi,
                    maskUpdateIter=3,
                    min_relative_threshold = 4,
                    smoothing = False,
                    smoothingWindowSize = 3,
                    smoothingPolyOrder = 4,
                    smoothing_log = True,
                    min_background_value=1E-3,
                    return_polararr=False):
    """
    Gets the median polar background for a diffraction pattern

    Args:
        data (ndarray): the data for which to find the polar eliptical background,
            usually a diffraction pattern
        qx0,qy0 (floats): the ellipse center; if the braggpeaks have been centered,
            these should be zero
        e (float): the length ratio of semiminor/semimajor axes
        phi (float): tilt of the major axis with respect  to (qx, qy) axis, in radians
        maskUpdate_iter (integer):
        min_relative_threshold (float):
        smoothing (bool): if true savgol filter smoothing is applied
        smoothingWindowSize (integer): size of the smoothing window, must be odd number
        smoothingPolyOrder (number): order of the polynomial smoothing to be applied
        smoothing_log (bool): if true log smoothing is performed
        min_background_value (float): if log smoothing is true, a zero value will be
            replaced with a small nonzero float
        return_polar_arr (bool): if True the polar transform with the masked high
            intensity peaks will be returned

    Returns:
        (2- or 3-tuple of ndarrays): A 2- or 3-tuple of ndarrays:

            * **background1D**: 1D polar elliptical background
            * **r_bins**: the elliptically transformed radius associated with
              background1D
            * **polarData** (optional): the masked polar transform from which the
              background is computed, returned iff `return_polar_arr==True`
    """
    # assert data is proper form 
    assert isinstance(smoothing, bool), "Smoothing must be bool"
    assert smoothingWindowSize%2==1, 'Smoothing window must be odd'
    assert isinstance(return_polararr, bool), "return_polararr must be bool"

    # Compute Polar Transform
    polarData, rr, tt = cartesian_to_polarelliptical_transform(data,tuple([qx0,qy0, 1, e, phi]))

    # Crop polar data to maximum distance which contains information from original image
    if (polarData.mask.sum(axis = (0))==polarData.shape[0]).any():
            ii = polar.data.shape[1]
            while(polar.mask[:,ii].all()==True):
                ii = ii-1
            maximalDistance = ii
            polarData = polarData[:,0:maximalDistance]
            r_bins = rr[0,0:maximalDistance]
    else:
            r_bins = rr[0,:]

    # Iteratively mask off high intensity peaks
    maskPolar = np.copy(polarData.mask)
    for ii in range(maskUpdateIter+1):
        if ii > 0:
            maskUpdate = np.logical_or(maskPolar,
                                        polarData/background1D > min_relative_threshold)
            # Prevent entire columns from being masked off 
            colMaskMin = np.all(maskUpdate, axis = 0)  # Detect columns that are empty
            maskUpdate[:,colMaskMin] = polarData.mask[:,colMaskMin] # reset empty columns to values of previous iterations
            polarData.mask  = maskUpdate  # Update Mask

        background1D = np.ma.median(polarData, axis = 0)

    background1D = np.maximum(background1D, min_background_value)

    if smoothing == True:
        if smoothing_log==True:
            background1D = np.log(background1D)

        background1D = savgol_filter(background1D,
                                     smoothingWindowSize,
                                     smoothingPolyOrder)
        if smoothing_log==True:
            background1D = np.exp(background1D)
    if return_polararr ==True:
        return(background1D, r_bins, polarData)
    else:
        return(background1D, r_bins)

#Create 2D Background 
def get_2D_polar_background(data, background1D, r_bins, qx0, qy0, phi, e):
    """
    Gets 2D polar elliptical background from linear 1D background

    Args:
        data (ndarray): the data for which to find the polar eliptical background,
            usually a diffraction pattern
        background1D (ndarray): a vector representing the radial elliptical background
        r_bins (ndarray): a vector of the elliptically transformed radius associated with
            background1D
        qx0,qy0 (floats): the ellipse center; if the braggpeaks have been centered, these
            should be zero
        e (float): the length ratio of semiminor/semimajor axes
        phi (float): tilt of the major axis with respect  to (qx, qy) axis, in radians

    Returns:
        (ndarray) 2D polar elliptical median background image
    """
    assert r_bins.shape==background1D.shape, "1D background and r_bins must be same length"
    # Define centered 2D cartesian coordinate system
    yc, xc = np.meshgrid(np.arange(0,data.shape[1])-qy0,
                         np.arange(0,data.shape[0])-qx0)


    # Calculate the semimajor axis distance for each point in the 2D array
    r = np.sqrt(((xc*np.cos(phi)+yc*np.sin(phi))**2)+
                (((xc*np.sin(phi)-yc*np.cos(phi))**2)/(e**2)))

    # Create a 2D eliptical background using linear interpolation  
    f = interp1d(r_bins, background1D, fill_value = 'extrapolate')
    background2D = f(r)

    return(background2D)

