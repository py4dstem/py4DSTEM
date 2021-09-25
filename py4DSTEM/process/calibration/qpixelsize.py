# Functions for calibrating the pixel size in the diffraction plane.

import numpy as np
from scipy.optimize import leastsq
from typing import Union, Optional
from py4DSTEM.process.utils import get_CoM, tqdmnd
from ...io.datastructure import Coordinates, PointListArray


def get_dq(q, d):
    """
    Get dq, the size of the detector pixels in the diffraction plane, in inverse length
    units.

    Args:
        q (number): a measured diffraction space distance, in pixels
        d (number): the known corresponding length, in *real space* length units (e.g.
            in Angstroms)

    Returns:
        (number): the detector pixel size
    """
    return 1 / (q * d)


def get_dq_from_indexed_peaks(qs, hkl, a):
    """
    Get dq, the size of the detector pixels in the diffraction plane, in inverse length
    units, using a set of measured peak distances from the optic axis, their Miller
    indices, and the known unit cell size.

    Args:
        qs (array): the measured peak positions
        hkl (list/tuple of length-3 tuples): the Miller indices of the peak positions qs.
            The length of qs and hkl must be the same.  To ignore any peaks, for this
            peak set (h,k,l)=(0,0,0).
        a (number): the unit cell size

    Returns:
        (4-tuple): A 4-tuple containing:

            * **dq**: *(number)* the detector pixel size
            * **qs_fit**: *(array)* the fit positions of the peaks
            * **hkl_fit**: *(list/tuple of length-3 tuples)* the Miller indices of the
              fit peaks
            * **mask**: *(array of bools)* False wherever hkl[i]==(0,0,0)
    """
    assert len(qs) == len(hkl), "qs and hkl must have same length"

    # Get spacings
    d_inv = np.array([np.sqrt(a ** 2 + b ** 2 + c ** 2) for (a, b, c) in hkl])
    mask = d_inv != 0

    # Get scaling factor
    c0 = np.average(qs[mask] / d_inv[mask])
    fiterr = lambda c: qs[mask] - c * d_inv[mask]
    popt, _ = leastsq(fiterr, c0)
    c = popt[0]

    # Get pixel size
    dq = 1 / (c * a)
    qs_fit = d_inv[mask] / a
    hkl_fit = [hkl[i] for i in range(len(hkl)) if mask[i] == True]

    return dq, qs_fit, hkl_fit


def calibrate_Bragg_peaks_pixel_size(
    braggpeaks: PointListArray,
    q_pixel_size: Optional[float] = None,
    coords: Optional[Coordinates] = None,
    name: Optional[str] = None,
) -> PointListArray:
    """
    Calibrate reciprocal space measurements of Bragg peak positions, using
    either `q_pixel_size` or the `Q_pixel_size` field of a
    Coordinates object

    Accepts:
        braggpeaks  (PointListArray) the detected, unscaled bragg peaks
        q_pixel_size (float) Q pixel size in inverse Ångström
        coords      (Coordinates) an object containing pixel size
        name        (str, optional) a name for the returned PointListArray.
                    If unspecified, takes the old PLA name, removes '_raw'
                    if present at the end of the string, then appends
                    '_calibrated'.

    Returns:
        braggpeaks_calibrated  (PointListArray) the calibrated Bragg peaks
    """
    assert isinstance(braggpeaks, PointListArray)
    assert (q_pixel_size is not None) != (
        coords is not None
    ), "Either (qx0,qy0) or coords must be specified"

    if coords is not None:
        assert isinstance(coords, Coordinates), "coords must be a Coordinates object."
        q_pixel_size = coords.get_Q_pixel_size()
        assert q_pixel_size is not None, "coords did not contain center position"

    if q_pixel_size is not None:
        assert isinstance(q_pixel_size, float), "q_pixel_size must be a float."

    if name is None:
        sl = braggpeaks.name.split("_")
        _name = "_".join(
            [s for i, s in enumerate(sl) if not (s == "raw" and i == len(sl) - 1)]
        )
        name = _name + "_calibrated"
    assert isinstance(name, str)

    braggpeaks_calibrated = braggpeaks.copy(name=name)

    for Rx, Ry in tqdmnd(
        braggpeaks_calibrated.shape[0], braggpeaks_calibrated.shape[1]
    ):
        pointlist = braggpeaks_calibrated.get_pointlist(Rx, Ry)
        pointlist.data["qx"] *= q_pixel_size
        pointlist.data["qy"] *= q_pixel_size

    return braggpeaks_calibrated
