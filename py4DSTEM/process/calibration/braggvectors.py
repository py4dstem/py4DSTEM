
import numpy as np
from typing import Optional, Union
from numbers import Number

from ...io.datastructure.emd import PointListArray
from ...io.datastructure.py4dstem import Calibration


def calibrate(
        braggpeaks,
        calibration,
        use_fitted_origin = True,
        **params
    ):
    """
    Determines which calibrations are present in `calibrations` (of origin,
    elliptical, pixel, rotational), and applies any it finds to `braggpeaks`.

    Note that this function updates the data pointed to by whatever
    PointListArray that's passed to it, so consider copying the PLA before
    running this function!

    Args:
        braggpeaks (PointListArray)
        calibration (Calibration)
        use_fitted_origin (bool): determine if using fitted origin or measured origin

    Returns:
        (PointListArray)
    """
    assert(isinstance(braggpeaks,PointListArray))
    assert(isinstance(calibration,Calibration))

    # find calibrations
    c = calibration
    origin = c.get_origin() if use_fitted_origin else c.get_origin_meas()
    ellipse = c.get_ellipse()
    pixel_size = c.get_Q_pixel_size()
    pixel_units = c.get_Q_pixel_units()
    rotflip = c.get_QR_rotflip()

    # determine if there are calibrations to perform
    pixel = None if (pixel_size==1 or pixel_units=='pixel') else True
    if all([x is None for x in (origin,ellipse,pixel,rotflip)]):
        raise Exception("no calibrations found")


    # calibrate

    # origin
    if origin is not None:
        braggpeaks = center_braggpeaks(
            braggpeaks,
            origin
        )

    # ellipse
    if ellipse is not None:
        braggpeaks = correct_braggpeak_elliptical_distortions(
            braggpeaks,
            p_ellipse = (0,0)+ellipse,
            centered = True
        )

    # pixel size
    if pixel is not None:
        braggpeaks = calibrate_Bragg_peaks_pixel_size(
            braggpeaks,
            q_pixel_size = pixel_size
        )

    # Q/R rotation
    if rotflip is not None:
        rot,flip = rotflip
        braggpeaks = calibrate_bragg_peaks_rotation(
            braggpeaks,
            theta = rot,
            flip = flip
        )


    # return
    return braggpeaks





def center_braggpeaks(
    braggpeaks,
    origin,
    ):
    """
    Shift the braggpeaks positions to center them about the origin.

    Accepts:
        braggpeaks (PointListArray): the unshifted peak positions
        origin (2-tuple): (qx0,qy0) either as scalars or as (R_Nx,R_Ny)-
            shaped arrays

    Returns:
        (PointListArray): the centered Bragg peaks
    """
    assert isinstance(braggpeaks, PointListArray)
    assert(len(origin)==2)
    qx0,qy0 = origin

    if np.isscalar(qx0) & np.isscalar(qy0):
        for Rx in range(braggpeaks.shape[0]):
            for Ry in range(braggpeaks.shape[1]):
                pointlist = braggpeaks.get_pointlist(Rx, Ry)
                pointlist.data["qx"] -= qx0
                pointlist.data["qy"] -= qy0
    else:
        assert(all([q.shape==braggpeaks.shape for q in origin]))
        for Rx in range(braggpeaks.shape[0]):
            for Ry in range(braggpeaks.shape[1]):
                pointlist = braggpeaks.get_pointlist(Rx, Ry)
                qx, qy = qx0[Rx, Ry], qy0[Rx, Ry]
                pointlist.data["qx"] -= qx
                pointlist.data["qy"] -= qy

    return braggpeaks





### Correct Bragg peak positions, making a circular coordinate system

def correct_braggpeak_elliptical_distortions(
    braggpeaks,
    p_ellipse,
    centered=True
    ):
    """
    Correct the elliptical distortions in a BraggPeaks instance.

    Accepts:
        braggpeaks (PointListArray): the detected, unshifted bragg peaks
        p_ellipse (5-tuple): the ellipse parameters (x0,y0,a,b,theta)
        centered (bool): if True, assumes that the braggpeaks PointListArray has been
            centered, and uses (x0,y0)=(0,0). Otherwise, uses the (x0,y0) from
            `p_ellipse`

    Returns:
        (PointListArray): the corrected Bragg peaks
    """
    assert(isinstance(braggpeaks,PointListArray))

    # Unpack parameters
    x0,y0,a,b,theta = p_ellipse
    if centered:
        x0,y0 = 0,0

    # Get the transformation matrix
    e = b/a
    sint, cost = np.sin(theta-np.pi/2.), np.cos(theta-np.pi/2.)
    T = np.array(
            [
                [e*sint**2 + cost**2, sint*cost*(1-e)],
                [sint*cost*(1-e), sint**2 + e*cost**2]
            ]
        )

    # Correct distortions
    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            x, y = pointlist.data["qx"] - x0, pointlist.data["qy"] - y0
            xyar_i = np.vstack([x, y])
            xyar_f = np.matmul(T, xyar_i)
            pointlist.data["qx"] = xyar_f[0, :] + x0
            pointlist.data["qy"] = xyar_f[1, :] + y0
    return braggpeaks



def calibrate_Bragg_peaks_pixel_size(
    braggpeaks: PointListArray,
    q_pixel_size: Number,
    ):
    """
    Calibrate the reciprocal length of Bragg peak positions.

    Accepts:
        braggpeaks (PointListArray) the detected, unscaled bragg peaks
        q_pixel_size (float) Q pixel size in inverse Ångström

    Returns:
        (PointListArray)
    """
    assert isinstance(braggpeaks, PointListArray)
    assert isinstance(q_pixel_size, Number)

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)
            pointlist.data["qx"] *= q_pixel_size
            pointlist.data["qy"] *= q_pixel_size

    return braggpeaks



def calibrate_bragg_peaks_rotation(
    braggpeaks: PointListArray,
    theta: float,
    flip: bool,
    ) -> PointListArray:
    """
    Calibrate rotation of Bragg peak positions, using either the R/Q rotation `theta`
    or the `QR_rotation` value inside a Calibration object.

    Accepts:
        braggpeaks  (PointListArray) the CENTERED Bragg peaks
        theta       (float) the rotation between real and reciprocal space in radians
        flip        (bool) whether there is a flip between real and reciprocal space

    Returns:
        braggpeaks_rotated  (PointListArray) the rotated Bragg peaks
    """

    assert isinstance(braggpeaks, PointListArray)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pointlist = braggpeaks.get_pointlist(Rx, Ry)

            if flip:
                positions = R @ np.vstack((pointlist.data["qy"], pointlist.data["qx"]))
            else:
                positions = R @ np.vstack((pointlist.data["qx"], pointlist.data["qy"]))

            rotated_pointlist = braggpeaks.get_pointlist(Rx, Ry)
            rotated_pointlist.data["qx"] = positions[0, :]
            rotated_pointlist.data["qy"] = positions[1, :]

    return braggpeaks






