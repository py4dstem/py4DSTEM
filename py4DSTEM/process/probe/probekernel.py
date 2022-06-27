# Functions for preparing the probe for cross-correlative template matching.


import numpy as np

from ..utils import get_shifted_ar
from ..calibration import get_probe_size





def get_probe_kernel(
    probe,
    origin=None
    ):
    """
    Creates a convolution kernel from an average probe, by normalizing, then
    shifting the center of the probe to the corners of the array.

    Args:
        probe (ndarray): the diffraction pattern corresponding to the probe over
            vacuum
        origin (2-tuple or None): if None (default), finds the origin using
            get_probe_radius. Otherwise, should be a 2-tuple (x0,y0) specifying
            the origin position

    Returns:
        (ndarray): the convolution kernel corresponding to the probe, in real
            space
    """
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    if origin is None:
        _,xCoM,yCoM = get_probe_size(probe)
    else:
        xCoM,yCoM = origin

    # Normalize
    probe = probe/np.sum(probe)

    # Shift center to corners of array
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM)

    return probe_kernel


def get_probe_kernel_edge_gaussian(
        probe,
        sigma,
        origin=None
        ):
    """
    Creates a convolution kernel from an average probe, subtracting a gaussian
    from the normalized probe such that the kernel integrates to zero, then
    shifting the center of the probe to the array corners.

    Args:
        probe (ndarray): the diffraction pattern corresponding to the probe
            over vacuum
        sigma (float): the width of the gaussian to subtract, relative to
            the standard deviation of the probe
        origin (2-tuple or None): if None (default), finds the origin using
            get_probe_radius. Otherwise, should be a 2-tuple (x0,y0) specifying
            the origin position

    Returns:
        (ndarray) the convolution kernel corresponding to the probe
    """
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    if origin is None:
        _,xCoM,yCoM = get_probe_size(probe)
    else:
        xCoM,yCoM = origin

    # Shift probe to origin
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM)

    # Generate normalization kernel
    # Coordinates
    qy,qx = np.meshgrid(
        np.mod(np.arange(Q_Ny) + Q_Ny//2, Q_Ny) - Q_Ny//2,
        np.mod(np.arange(Q_Nx) + Q_Nx//2, Q_Nx) - Q_Nx//2)
    qr2 = (qx**2 + qy**2)
    # Calculate Gaussian normalization kernel
    qstd2 = np.sum(qr2*probe_kernel) / np.sum(probe_kernel)
    kernel_norm = np.exp(-qr2 / (2*qstd2*sigma**2))

    # Output normalized kernel
    probe_kernel = probe_kernel/np.sum(probe_kernel) - kernel_norm/np.sum(kernel_norm)

    return probe_kernel


def get_probe_kernel_edge_sigmoid(
    probe,
    ri,
    ro,
    origin=None,
    type='sine_squared'
    ):
    """
    Creates a convolution kernel from an average probe, subtracting an annular
    trench about the probe such that the kernel integrates to zero, then
    shifting the center of the probe to the array corners.

    Args:
        probe (ndarray): the diffraction pattern corresponding to the probe over
            vacuum
        ri (float): the sigmoid inner radius, from the probe center
        ro (float): the sigmoid outer radius
        origin (2-tuple or None): if None (default), finds the origin using
            get_probe_radius. Otherwise, should be a 2-tuple (x0,y0) specifying
            the origin position
        type (string): must be 'logistic' or 'sine_squared'

    Returns:
        (ndarray): the convolution kernel corresponding to the probe
    """
    valid_types = ('logistic','sine_squared')
    assert(type in valid_types), "type must be in {}".format(valid_types)
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    if origin is None:
        _,xCoM,yCoM = get_probe_size(probe)
    else:
        xCoM,yCoM = origin

    # Shift probe to origin
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM)

    # Generate normalization kernel
    # Coordinates
    qy,qx = np.meshgrid(
        np.mod(np.arange(Q_Ny) + Q_Ny//2, Q_Ny) - Q_Ny//2,
        np.mod(np.arange(Q_Nx) + Q_Nx//2, Q_Nx) - Q_Nx//2)
    qr = np.sqrt(qx**2 + qy**2)
    # Calculate sigmoid
    if type == 'logistic':
        r0 = 0.5*(ro+ri)
        sigma = 0.25*(ro-ri)
        sigmoid = 1/(1+np.exp((qr-r0)/sigma))
    elif type == 'sine_squared':
        sigmoid = (qr - ri) / (ro - ri)
        sigmoid = np.minimum(np.maximum(sigmoid, 0.0), 1.0)
        sigmoid = np.cos((np.pi/2)*sigmoid)**2
    else:
        raise Exception("type must be in {}".format(valid_types))

    # Output normalized kernel
    probe_kernel = probe_kernel/np.sum(probe_kernel) - sigmoid/np.sum(sigmoid)

    return probe_kernel







