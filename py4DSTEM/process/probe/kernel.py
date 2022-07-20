# Functions for preparing the probe for cross-correlative template matching.


import numpy as np

from ..utils import get_shifted_ar
from ..calibration import get_probe_size


def get_kernel(
    probe,
    mode = 'flat',
    **kwargs
    ):
    """
    Creates a kernel from the probe for cross-correlative template matching.

    Precise behavior and valid keyword arguments depend on the `mode`
    selected.  In each case, the center of the probe is shifted to the
    origin and the kernel normalized such that it sums to 1. In 'flat'
    mode, this is the only processing performed. In the remaining modes,
    some additional processing is performed which adds a ring of
    negative intensity around the central probe, which results in
    edge-filetering-like behavior during cross correlation. Valid modes
    are:

        - 'flat': creates a flat probe kernel. For bullseye or other
            structured probes, this mode is recommended.
        - 'gaussian': subtracts a gaussian with a width of standard
            deviation 'sigma'
        - 'sigmoid': subtracts an annulus with inner and outer radii
            of (ri,ro) and a sine-squared sigmoid radial profile from
            the probe template.
        - 'sigmoid_log': subtracts an annulus with inner and outer radii
            of (ri,ro) and a logistic sigmoid radial profile from
            the probe template.

    Each mode accepts 'center' (2-tuple) as a kwarg to manually specify
    the center of the probe, which is otherwise autodetected. Modes which
    accept additional kwargs and those arguments are:

        - 'gaussian':
            sigma (number)
        - 'sigmoid':
            radii (2-tuple)
        - 'sigmoid_log':
            radii (2-tuple)

    Accepts:
        probe (2D array):
        mode (str): must be in 'flat','gaussian','sigmoid','sigmoid_log'
        **kwargs: depend on `mode`, see above

    Returns:
        (2D array)
    """

    modes = [
        'flat',
        'gaussian',
        'sigmoid',
        'sigmoid_log'
    ]

    # parse args
    assert mode in modes, f"mode must be in {modes}. Received {mode}"

    # get function
    fn_dict = _make_function_dict()
    fn = fn_dict[mode]

    # compute and return
    kernel = fn(probe, **kwargs)
    return kernel



def _make_function_dict():
    d = {
        'flat' : get_probe_kernel,
        'gaussian' : get_probe_kernel_edge_gaussian,
        'sigmoid' : _get_probe_kernel_edge_sigmoid_sine_squared,
        'sigmoid_log' : _get_probe_kernel_edge_sigmoid_sine_squared
    }
    return d





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
        origin=None,
        bilinear=False,
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
        bilinear (bool): By default probe is shifted via a Fourier transform. 
                         Setting this to true overrides it and uses bilinear shifting.
                         Not recommended!

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
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM, bilinear=bilinear)

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
    radii,
    origin=None,
    type='sine_squared',
    bilinear=False,
    ):
    """
    Creates a convolution kernel from an average probe, subtracting an annular
    trench about the probe such that the kernel integrates to zero, then
    shifting the center of the probe to the array corners.

    Args:
        probe (ndarray): the diffraction pattern corresponding to the probe over
            vacuum
        radii (2-tuple): the sigmoid inner and outer radii, from the probe center
        origin (2-tuple or None): if None (default), finds the origin using
            get_probe_radius. Otherwise, should be a 2-tuple (x0,y0) specifying
            the origin position
        type (string): must be 'logistic' or 'sine_squared'
        bilinear (bool): By default probe is shifted via a Fourier transform. 
                 Setting this to true overrides it and uses bilinear shifting.
                 Not recommended!

    Returns:
        (ndarray): the convolution kernel corresponding to the probe
    """
    valid_types = ('logistic','sine_squared')
    assert(type in valid_types), "type must be in {}".format(valid_types)
    Q_Nx, Q_Ny = probe.shape
    ri,ro = radii

    # Get CoM
    if origin is None:
        _,xCoM,yCoM = get_probe_size(probe)
    else:
        xCoM,yCoM = origin

    # Shift probe to origin
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM, bilinear=bilinear)

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



def _get_probe_kernel_edge_sigmoid_sine_squared(
    probe,
    radii,
    origin=None,
    ):
    return get_probe_kernel_edge_sigmoid(
        probe,
        radii,
        origin = origin,
        type='sine_squared'
    )

def _get_probe_kernel_edge_sigmoid_logistic(
    probe,
    radii,
    origin=None,
    ):
    return get_probe_kernel_edge_sigmoid(
        probe,
        radii,
        origin = origin,
        type='logistic'
    )



