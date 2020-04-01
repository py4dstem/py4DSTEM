# Functions for calibrating the pixel size in the diffraction plane.

import numpy as np
from scipy.optimize import leastsq
from py4DSTEM.process.utils import get_CoM

def get_dq(q,d):
    """
    Get dq, the size of the detector pixels in the diffraction plane, in inverse length units.

    Accepts:
        q       (number) a measured diffraction space distance, in pixels
        d       (number) the known corresponding length, in *real space* length units
                (e.g. in Angstroms)

    Returns:
        dq      (number) the detector pixel size
    """
    return 1/(q*d)

def get_dq_from_indexed_peaks(qs,hkl,a):
    """
    Get dq, the size of the detector pixels in the diffraction plane, in inverse length units,
    using a set of measured peak distances from the optic axis, their Miller indices, and the known
    unit cell size.

    Accepts:
        qs      (array) the measured peak positions
        hkl     (list/tuple of length-3 tuples) the Miller indices of the peak positions qs.
                The length of qs and hkl must be the same.  To ignore any peaks, for this peak
                set (h,k,l)=(0,0,0).
        a       (number) the unit cell size

    Returns:
        dq      (number) the detector pixel size
        qs_fit  (array) the fit positions of the peaks
        hkl_fit (list/tuple of length-3 tuples) the Miller indices of the fit peaks
        mask    (array of bools) False wherever hkl[i]==(0,0,0)
    """
    assert len(qs)==len(hkl), "qs and hkl must have same length"

    # Get spacings
    d_inv = np.array([np.sqrt(a**2+b**2+c**2) for (a,b,c) in hkl])
    mask = d_inv!=0

    # Get scaling factor
    c0 = np.average(qs[mask]/d_inv[mask])
    fiterr = lambda c: qs[mask] - c*d_inv[mask]
    popt,_ = leastsq(fiterr,c0)
    c = popt[0]

    # Get pixel size
    dq = 1/(c*a)
    qs_fit = d_inv[mask]/a
    hkl_fit = [hkl[i] for i in range(len(hkl)) if mask[i]==True]

    return dq, qs_fit, hkl_fit

def get_probe_size(DP, thresh_lower=0.01, thresh_upper=0.99, N=100):
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower,thresh_upper,N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax*thresh
        r_vals[i] = np.sqrt(np.sum(mask)/np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2*np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax*thresh
    x0,y0 = get_CoM(DP*mask)

    return r,x0,y0



