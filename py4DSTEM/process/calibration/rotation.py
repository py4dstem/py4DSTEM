# Rotational calibrations

import numpy as np
from typing import Optional
from ...io.datastructure import Coordinates, PointListArray
from py4DSTEM.process.utils import tqdmnd


def calibrate_Bragg_peaks_rotation(
    braggpeaks: PointListArray,
    theta: Optional[float] = None,
    flip: Optional[bool] = None,
    coords: Optional[Coordinates] = None,
    name: Optional[str] = None,
) -> PointListArray:
    """
    Calibrate rotation of Bragg peak positions, using either the R/Q rotation `theta`
    or the `QR_rotation` value inside a Coordinates object.

    Accepts:
        braggpeaks  (PointListArray) the CENTERED Bragg peaks
        theta       (float) the rotation between real and reciprocal space in radians
        flip        (bool) whether there is a flip between real and reciprocal space
        coords      (Coordinates) an object containing QR_rotation
        name        (str, optional) a name for the returned PointListArray.
                    If unspecified, takes the old PLA name, removes '_centered'
                    if present at the end of the string, then appends
                    '_rotated'.

    Returns:
        braggpeaks_rotated  (PointListArray) the rotated Bragg peaks
    """

    assert isinstance(braggpeaks, PointListArray)
    assert (theta is not None and flip is not None) != (
        coords is not None
    ), "Either (qx0,qy0) or coords must be specified"

    if coords is not None:
        assert isinstance(coords, Coordinates), "coords must be a Coordinates object."
        theta = coords.get_QR_rotation()
        flip = coords.get_QR_flip()
        assert theta is not None, "coords did not contain center position"

    if theta is not None:
        assert isinstance(theta, float), "theta must be a float."
    if flip is not None:
        assert isinstance(flip, bool), "flip must be a boolean."

    if name is None:
        sl = braggpeaks.name.split("_")
        _name = "_".join(
            [s for i, s in enumerate(sl) if not (s == "centered" and i == len(sl) - 1)]
        )
        name = _name + "_rotated"
    assert isinstance(name, str)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    braggpeaks_rotated = braggpeaks.copy(name=name)

    for Rx, Ry in tqdmnd(braggpeaks_rotated.shape[0], braggpeaks_rotated.shape[1]):
        pointlist = braggpeaks.get_pointlist(Rx, Ry)

        if flip:
            positions = R @ np.vstack((pointlist.data["qy"], pointlist.data["qx"]))
        else:
            positions = R @ np.vstack((pointlist.data["qx"], pointlist.data["qy"]))

        rotated_pointlist = braggpeaks_rotated.get_pointlist(Rx, Ry)
        rotated_pointlist.data["qx"] = positions[0, :]
        rotated_pointlist.data["qy"] = positions[1, :]

    return braggpeaks_rotated


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
