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



