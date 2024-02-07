# Functions for generating masks

import numpy as np
from scipy.ndimage import binary_dilation



def make_circular_mask(shape, qxy0, radius):
    """
    Create a hard circular mask, for use in DPC integration or
    or to use as a filter in diffraction or real space.

    Args:
        shape       (2-tuple of ints) image size, in pixels
        qxy0        (2-tuple of floats) center coordinates, in pixels.  Must be in (row, column) format.
        radius      (float) radius of mask, in pixels

    Returns:
        mask        (2D boolean array) the mask

    """
    # coordinates
    qx = np.arange(shape[0]) - qxy0[0]
    qy = np.arange(shape[1]) - qxy0[1]
    [qya, qxa] = np.meshgrid(qy, qx)

    # return circular mask
    return qxa**2 + qya**2 < radius**2



def get_beamstop_mask(dp, qx0, qy0, theta, dtheta=1, w=10, r=10):
    """
    Generates a beamstop shaped mask.

    Args:
        dp (2d array): a diffraction pattern
        qx0,qy0 (numbers): the center position of the beamstop
        theta (number): the orientation of the beamstop, in degrees
        dtheta (number): angular span of the wedge representing the beamstop, in degrees
        w (integer): half the width of the beamstop arm, in pixels
        r (number): the radius of a circle at the end of the beamstop, in pixels

    Returns:
        (2d boolean array): the mask
    """
    # Handle inputs
    theta = np.mod(np.radians(theta), 2 * np.pi)
    dtheta = np.abs(np.radians(dtheta))

    # Get a meshgrid
    Q_Nx, Q_Ny = dp.shape
    qyy, qxx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
    qyy, qxx = qyy - qy0, qxx - qx0

    # wedge handles
    if dtheta > 0:
        qzz = qxx + qyy * 1j
        phi = np.mod(np.angle(qzz), 2 * np.pi)
        # Handle the branch cut in the complex plane
        if theta - dtheta < 0:
            phi, theta = np.mod(phi + dtheta, 2 * np.pi), theta + dtheta
        elif theta + dtheta > 2 * np.pi:
            phi, theta = np.mod(phi - dtheta, 2 * np.pi), theta - dtheta
        mask1 = np.abs(phi - theta) < dtheta
        if w > 0:
            mask1 = binary_dilation(mask1, iterations=w)

    # straight handles
    else:
        pass

    # circle mask
    qrr = np.hypot(qxx, qyy)
    mask2 = qrr < r

    # combine masks
    mask = np.logical_or(mask1, mask2)

    return mask


def sector_mask(shape, centre, radius, angle_range=(0, 360)):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.

    Args:
        shape: 2D shape of the mask
        centre: 2D center of the circular sector
        radius: radius of the circular mask
        angle_range: angular range of the circular mask
    """
    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= 2 * np.pi

    # circular mask
    circmask = r2 <= radius * radius

    # print 'radius - ', radius

    # angular mask
    anglemask = theta < (tmax - tmin)

    return circmask * anglemask



