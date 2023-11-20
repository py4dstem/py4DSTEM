# Rotational calibrations

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from py4DSTEM import show


def compare_QR_rotation(
    im_R,
    im_Q,
    QR_rotation,
    R_rotation=0,
    R_position=None,
    Q_position=None,
    R_pos_anchor="center",
    Q_pos_anchor="center",
    R_length=0.33,
    Q_length=0.33,
    R_width=0.001,
    Q_width=0.001,
    R_head_length_adjust=1,
    Q_head_length_adjust=1,
    R_head_width_adjust=1,
    Q_head_width_adjust=1,
    R_color="r",
    Q_color="r",
    figsize=(10, 5),
    returnfig=False,
):
    """
    Visualize a rotational offset between an image in real space, e.g. a STEM
    virtual image, and an image in diffraction space, e.g. a defocused CBED
    shadow image of the same region, by displaying an arrow overlaid over each
    of these two images with the specified QR rotation applied.  The QR rotation
    is defined as the counter-clockwise rotation from real space to diffraction
    space, in degrees.

    Parameters
    ----------
    im_R : numpy array or other 2D image-like object (e.g. a VirtualImage)
        A real space image, e.g. a STEM virtual image
    im_Q : numpy array or other 2D image-like object
        A diffraction space image, e.g. a defocused CBED image
    QR_rotation : number
        The counterclockwise rotation from real space to diffraction space,
        in degrees
    R_rotation : number
        The orientation of the arrow drawn in real space, in degrees
    R_position : None or 2-tuple
        The position of the anchor point for the R-space arrow. If None, defaults
        to the center of the image
    Q_position : None or 2-tuple
        The position of the anchor point for the Q-space arrow. If None, defaults
        to the center of the image
    R_pos_anchor : 'center' or 'tail' or 'head'
        The anchor point for the R-space arrow, i.e. the point being specified by
        the `R_position` parameter
    Q_pos_anchor : 'center' or 'tail' or 'head'
        The anchor point for the Q-space arrow, i.e. the point being specified by
        the `Q_position` parameter
    R_length : number or None
        The length of the R-space arrow, as a fraction of the mean size of the
        image
    Q_length : number or None
        The length of the Q-space arrow, as a fraction of the mean size of the
        image
    R_width : number
        The width of the R-space arrow
    Q_width : number
        The width of the R-space arrow
    R_head_length_adjust : number
        Scaling factor for the R-space arrow head length
    Q_head_length_adjust : number
        Scaling factor for the Q-space arrow head length
    R_head_width_adjust : number
        Scaling factor for the R-space arrow head width
    Q_head_width_adjust : number
        Scaling factor for the Q-space arrow head width
    R_color : color
        Color of the R-space arrow
    Q_color : color
        Color of the Q-space arrow
    figsize : 2-tuple
        The figure size
    returnfig : bool
        Toggles returning the figure and axes
    """
    # parse inputs
    if R_position is None:
        R_position = (
            im_R.shape[0] / 2,
            im_R.shape[1] / 2,
        )
    if Q_position is None:
        Q_position = (
            im_Q.shape[0] / 2,
            im_Q.shape[1] / 2,
        )
    R_length = np.mean(im_R.shape) * R_length
    Q_length = np.mean(im_Q.shape) * Q_length
    assert R_pos_anchor in ("center", "tail", "head")
    assert Q_pos_anchor in ("center", "tail", "head")

    # compute positions
    rpos_x, rpos_y = R_position
    qpos_x, qpos_y = Q_position
    R_rot_rad = np.radians(R_rotation)
    Q_rot_rad = np.radians(R_rotation + QR_rotation)
    rvecx = np.cos(R_rot_rad)
    rvecy = np.sin(R_rot_rad)
    qvecx = np.cos(Q_rot_rad)
    qvecy = np.sin(Q_rot_rad)
    if R_pos_anchor == "center":
        x0_r = rpos_x - rvecx * R_length / 2
        y0_r = rpos_y - rvecy * R_length / 2
        x1_r = rpos_x + rvecx * R_length / 2
        y1_r = rpos_y + rvecy * R_length / 2
    elif R_pos_anchor == "tail":
        x0_r = rpos_x
        y0_r = rpos_y
        x1_r = rpos_x + rvecx * R_length
        y1_r = rpos_y + rvecy * R_length
    elif R_pos_anchor == "head":
        x0_r = rpos_x - rvecx * R_length
        y0_r = rpos_y - rvecy * R_length
        x1_r = rpos_x
        y1_r = rpos_y
    else:
        raise Exception(f"Invalid value for R_pos_anchor {R_pos_anchor}")
    if Q_pos_anchor == "center":
        x0_q = qpos_x - qvecx * Q_length / 2
        y0_q = qpos_y - qvecy * Q_length / 2
        x1_q = qpos_x + qvecx * Q_length / 2
        y1_q = qpos_y + qvecy * Q_length / 2
    elif Q_pos_anchor == "tail":
        x0_q = qpos_x
        y0_q = qpos_y
        x1_q = qpos_x + qvecx * Q_length
        y1_q = qpos_y + qvecy * Q_length
    elif Q_pos_anchor == "head":
        x0_q = qpos_x - qvecx * Q_length
        y0_q = qpos_y - qvecy * Q_length
        x1_q = qpos_x
        y1_q = qpos_y
    else:
        raise Exception(f"Invalid value for Q_pos_anchor {Q_pos_anchor}")

    # make the figure
    axsize = (figsize[0] / 2, figsize[1])
    fig, axs = show([im_R, im_Q], returnfig=True, axsize=axsize)
    axs[0, 0].arrow(
        x=y0_r,
        y=x0_r,
        dx=y1_r - y0_r,
        dy=x1_r - x0_r,
        color=R_color,
        length_includes_head=True,
        width=R_width,
        head_width=R_length * R_head_width_adjust * 0.072,
        head_length=R_length * R_head_length_adjust * 0.1,
    )
    axs[0, 1].arrow(
        x=y0_q,
        y=x0_q,
        dx=y1_q - y0_q,
        dy=x1_q - x0_q,
        color=Q_color,
        length_includes_head=True,
        width=Q_width,
        head_width=Q_length * Q_head_width_adjust * 0.072,
        head_length=Q_length * Q_head_length_adjust * 0.1,
    )
    if returnfig:
        return fig, axs
    else:
        plt.show()


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
