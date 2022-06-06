# Rotational calibrations

import numpy as np
from typing import Optional
from ...io.datastructure import Calibrations, PointListArray
from ...tqdmnd import tqdmnd


def calibrate_bragg_peaks_rotation(
    braggpeaks: PointListArray,
    theta: float,
    flip: bool,
) -> PointListArray:
    """
    Calibrate rotation of Bragg peak positions, using either the R/Q rotation `theta`
    or the `QR_rotation` value inside a Calibrations object.

    Accepts:
        braggpeaks  (PointListArray) the CENTERED Bragg peaks
        theta       (float) the rotation between real and reciprocal space in radians
        flip        (bool) whether there is a flip between real and reciprocal space

    Returns:
        braggpeaks_rotated  (PointListArray) the rotated Bragg peaks
    """

    assert isinstance(braggpeaks, PointListArray)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for Rx, Ry in tqdmnd(braggpeaks.shape[0], braggpeaks.shape[1]):
        pointlist = braggpeaks.get_pointlist(Rx, Ry)

        if flip:
            positions = R @ np.vstack((pointlist.data["qy"], pointlist.data["qx"]))
        else:
            positions = R @ np.vstack((pointlist.data["qx"], pointlist.data["qy"]))

        rotated_pointlist = braggpeaks.get_pointlist(Rx, Ry)
        rotated_pointlist.data["qx"] = positions[0, :]
        rotated_pointlist.data["qy"] = positions[1, :]

    return braggpeaks


def get_Qvector_from_Rvector(vx, vy, QR_rotation):
    """
    For some vector (vx,vy) in real space, and some rotation QR between real and
    reciprocal space, determine the corresponding orientation in diffraction space.
    Returns both R and Q vectors, normalized.

    Args:
        vx,vy (numbers): the (x,y) components of a real space vector
        QR_rotation (number): the offset angle between real and reciprocal space.
        Specifically, the counterclockwise rotation of real space with respect to
        diffraction space.  In degrees.

    Returns:
        (4-tuple): 4-tuple consisting of:

            * **vx_R**: the x component of the normalized real space vector
            * **vy_R**: the y component of the normalized real space vector
            * **vx_Q**: the x component of the normalized reciprocal space vector
            * **vy_Q**: the y component of the normalized reciprocal space vector
    """
    phi = np.radians(QR_rotation)
    vL = np.hypot(vx, vy)
    vx_R, vy_R = vx / vL, vy / vL

    vx_Q = np.cos(phi) * vx_R + np.sin(phi) * vy_R
    vy_Q = -np.sin(phi) * vx_R + np.cos(phi) * vy_R

    return vx_R, vy_R, vx_Q, vy_Q


def get_Rvector_from_Qvector(vx, vy, QR_rotation):
    """
    For some vector (vx,vy) in diffraction space, and some rotation QR between real and
    reciprocal space, determine the corresponding orientation in diffraction space.
    Returns both R and Q vectors, normalized.

    Args:
        vx,vy (numbers): the (x,y) components of a reciprocal space vector
        QR_rotation (number): the offset angle between real and reciprocal space.
            Specifically, the counterclockwise rotation of real space with respect to
            diffraction space.  In degrees.

    Returns:
        (4-tuple): 4-tuple consisting of:

            * **vx_R**: the x component of the normalized real space vector
            * **vy_R**: the y component of the normalized real space vector
            * **vx_Q**: the x component of the normalized reciprocal space vector
            * **vy_Q**: the y component of the normalized reciprocal space vector
    """
    phi = np.radians(QR_rotation)
    vL = np.hypot(vx, vy)
    vx_Q, vy_Q = vx / vL, vy / vL

    vx_R = np.cos(phi) * vx_Q - np.sin(phi) * vy_Q
    vy_R = np.sin(phi) * vx_Q + np.cos(phi) * vy_Q

    return vx_R, vy_R, vx_Q, vy_Q
