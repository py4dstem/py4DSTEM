# Functions for obtaining vacuum probe templates.
#
# Probe templates can be generated from vacuum scans, from a selected ROI of a vacuum region of a 
# scan, or synthetic probes.  Ultimately the purpose is to generate a kernel for convolution with 
# individual diffraction patterns to identify Bragg disks.  Kernel generation will generally proceed 
# in two steps, which will each correspond to a function call: first, obtaining  or creating the 
# diffraction pattern of a probe over vacuum, and second, turning the probe DP into a convolution
# kernel by shifting and normalizing.

import numpy as np
from scipy.ndimage.morphology import binary_opening, binary_dilation
from ..utils import get_shifted_ar, get_CoM, get_shift, tqdmnd

#### Get the vacuum probe ####

def get_average_probe_from_vacuum_scan(datacube, mask_threshold=0.2,
                                                 mask_expansion=12,
                                                 mask_opening=3,
                                                 verbose=False):
    """
    Aligns and averages all diffraction patterns in a datacube, assumed to be taken over vacuum,
    to create and average vacuum probe.

    Values outisde the average probe are zeroed, using a binary mask determined by the optional
    parameters mask_threshold, mask_expansion, and mask_opening.  An initial binary mask is created
    using a threshold of less than mask_threshold times the maximal probe value. A morphological
    opening of mask_opening pixels is performed to eliminate stray pixels (e.g. from x-rays),
    followed by a dilation of mask_expansion pixels to ensure the entire probe is captured.

    Accepts:
        datacube        (DataCube) a vacuum scan
        mask_threshold  (float) threshold determining mask which zeros values outside of probe
        mask_expansion  (int) number of pixels by which the zeroing mask is expanded to capture
                        the full probe
        mask_opening    (int) size of binary opening used to eliminate stray bright pixels
        verbose         (bool) if True, prints progress updates

    Returns:
        probe           (ndarray of shape (datacube.Q_Nx,datacube.Q_Ny)) the average probe
    """
    probe = datacube.data[0,0,:,:]
    for n in tqdmnd(range(1,datacube.R_N)):
        Rx,Ry = np.unravel_index(n,datacube.data.shape[:2])
        curr_DP = datacube.data[Rx,Ry,:,:]
        if verbose:
            print("Shifting and averaging diffraction pattern {} of {}.".format(n,datacube.R_N))

        xshift,yshift = get_shift(probe, curr_DP)
        curr_DP_shifted = get_shifted_ar(curr_DP, xshift, yshift)
        probe = probe*(n-1)/n + curr_DP_shifted/n

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=mask_expansion)

    return probe*mask


def get_average_probe_from_vacuum_stack(data, mask_threshold=0.2,
                                              mask_expansion=12,
                                              mask_opening=3):
    """
    Averages all diffraction patterns in a 3D stack of diffraction patterns, assumed to be taken
    over vacuum, to create and average vacuum probe. No alignment is performed - i.e. it is assumed
    that the beam was stationary during acquisition of the stack.

    Values outisde the average probe are zeroed, using a binary mask determined by the optional
    parameters mask_threshold, mask_expansion, and mask_opening.  An initial binary mask is created
    using a threshold of less than mask_threshold times the maximal probe value. A morphological
    opening of mask_opening pixels is performed to eliminate stray pixels (e.g. from x-rays),
    followed by a dilation of mask_expansion pixels to ensure the entire probe is captured.

    Accepts:
        data            (array) a 3D stack of vacuum diffraction patterns, shape (Q_Nx,Q_Ny,N)
        mask_threshold  (float) threshold determining mask which zeros values outside of probe
        mask_expansion  (int) number of pixels by which the zeroing mask is expanded to capture
                        the full probe
        mask_opening    (int) size of binary opening used to eliminate stray bright pixels

    Returns:
        probe           (array of shape (Q_Nx,Q_Ny)) the average probe
    """
    probe = np.average(data,axis=2)

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=mask_expansion)

    return probe*mask


def get_average_probe_from_ROI(datacube, ROI, mask_threshold=0.2,
                                              mask_expansion=12,
                                              mask_opening=3,
                                              verbose=False,
                                              DP_mask=1):
    """
    Aligns and averages all diffraction patterns within a specified ROI of a datacube to create an
    average vacuum probe.

    See documentation for get_average_probe_from_vacuum_scan for more detailed discussion of the
    algorithm.

    Accepts:
        datacube        (DataCube) a vacuum scan
        ROI             (ndarray of dtype=bool and shape (datacube.R_Nx,datacube.R_Ny))
                        An array of boolean variables shaped like the real space scan. Only scan
                        positions where ROI==True are used to create the average probe.
        mask_threshold  (float) threshold determining mask which zeros values outside of probe
        mask_expansion  (int) number of pixels by which the zeroing mask is expanded to capture
                        the full probe
        mask_opening    (int) size of binary opening used to eliminate stray bright pixels
        verbose         (bool) if True, prints progress updates
        DP_mask         (array) array of same shape as diffraction pattern to mask probes

    Returns:
        probe           (ndarray of shape (datacube.Q_Nx,datacube.Q_Ny)) the average probe
    """
    assert ROI.shape==(datacube.R_Nx,datacube.R_Ny)
    length = ROI.sum()
    xy = np.vstack(np.nonzero(ROI))
    probe = datacube.data[xy[0,0],xy[1,0],:,:]
    for n in tqdmnd(range(1,length)):
        curr_DP = datacube.data[xy[0,n],xy[1,n],:,:] * DP_mask

        xshift,yshift = get_shift(probe, curr_DP)
        curr_DP_shifted = get_shifted_ar(curr_DP, xshift, yshift)
        probe = probe*(n-1)/n + curr_DP_shifted/n

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=mask_expansion)

    return probe*mask


def get_synthetic_probe(radius, width, Q_Nx, Q_Ny):
    """
    Makes a synthetic probe, with the functional form of a disk blurred by a sigmoid (a logistic
    function).

    Accepts:
        radius        (float) the probe radius
        width         (float) the blurring of the probe edge. width represents the full width of the
                      blur, with x=-w/2 to x=+w/2 about the edge spanning values of ~0.12 to 0.88
        Q_Nx, Q_Ny    (int) the diffraction plane dimensions

    Returns:
        probe         (ndarray of shape (Q_Nx,Q_Ny)) the probe
    """
    # Make coords
    qy,qx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    qy,qx = qy - Q_Ny/2., qx-Q_Nx/2.
    qr = np.sqrt(qx**2+qy**2)

    # Shift zero to disk edge
    qr = qr - radius

    # Calculate logistic function
    probe = 1/(1+np.exp(4*qr/width))

    return probe



#### Get the probe kernel ####

def get_probe_kernel(probe):
    """
    Creates a convolution kernel from an average probe, by normalizing, then shifting the center of
    the probe to the corners of the array.

    Accepts:
        probe           (ndarray) the diffraction pattern corresponding to the probe over vacuum

    Returns:
        probe_kernel    (ndarray) the convolution kernel corresponding to the probe, in real space
    """
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    xCoM, yCoM = get_CoM(probe)

    # Normalize
    probe = probe/np.sum(probe)

    # Shift center to corners of array
    probe_kernel = get_shifted_ar(probe, -xCoM, -yCoM)

    return probe_kernel


def get_probe_kernel_subtrgaussian(probe, sigma_probe_scale):
    """
    Creates a convolution kernel from an average probe, subtracting a gaussian from the normalized
    probe such that the kernel integrates to zero, then shifting the center of the probe to the
    array corners.

    Accepts:
        probe              (ndarray) the diffraction pattern corresponding to the probe over vacuum
        sigma_probe_scale  (float) the width of the gaussian to subtract, relative to the standard
                           deviation of the probe

    Returns:
        probe_kernel       (ndarray) the convolution kernel corresponding to the probe
    """
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    xCoM, yCoM = get_CoM(probe)

    # Get probe size
    qy,qx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    q2 = (qx-xCoM)**2 + (qy-yCoM)**2
    qstd2 = np.sum(q2*probe) / np.sum(probe)

    # Normalize to one, then subtract of normed gaussian, yielding kernel which integrates to zero
    probe_template_norm = probe/np.sum(probe)
    subtr_gaussian = np.exp(-q2 / (2*qstd2*sigma_probe_scale**2))
    subtr_gaussian = subtr_gaussian/np.sum(subtr_gaussian)
    probe_kernel = probe_template_norm - subtr_gaussian

    # Shift center to array corners
    probe_kernel = get_shifted_ar(probe_kernel, -xCoM, -yCoM)

    return probe_kernel


def get_probe_kernel_logistictrench(probe, radius, trenchwidth, blurwidth):
    """
    Creates a convolution kernel from an average probe, subtracting an annular trench about the
    probe such that the kernel integrates to zero, then shifting the center of the probe to the
    array corners.

    Accepts:
        probe           (ndarray) the diffraction pattern corresponding to the probe over vacuum
        radius          (float) the inner radius of the trench, from the probe center
        trenchwidth     (float) the trench annulus width (r_outer - r_inner)
        blurwidth       (float) the full width of the blurring of the trench walls

    Returns:
        probe_kernel    (ndarray) the convolution kernel corresponding to the probe
    """
    Q_Nx, Q_Ny = probe.shape

    # Get CoM
    xCoM, yCoM = get_CoM(probe)

    # Get probe size
    qy,qx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    qr = np.sqrt((qx-xCoM)**2 + (qy-yCoM)**2)
    qr = qr-radius                                        # Shift qr=0 to disk edge

    # Calculate logistic function
    logistic_annulus = 1/(1+np.exp(4*qr/blurwidth)) - 1/(1+np.exp(4*(qr-trenchwidth)/blurwidth))

    # Normalize to one, then subtract off logistic annulus, yielding kernel which integrates to zero
    probe_template_norm = probe/np.sum(probe)
    logistic_annulus_norm = logistic_annulus/np.sum(logistic_annulus)
    probe_kernel = probe_template_norm - logistic_annulus_norm

    # Shift center to array corners
    probe_kernel = get_shifted_ar(probe_kernel, -xCoM, -yCoM)

    return probe_kernel








