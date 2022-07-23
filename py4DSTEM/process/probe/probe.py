# Functions for getting images of the vacuum probe

import numpy as np
from scipy.ndimage.morphology import (
    binary_opening, binary_dilation, distance_transform_edt)

from ..utils import get_shifted_ar, get_shift
from ...utils.tqdmnd import tqdmnd
from ...io.datastructure import DataCube, Probe





def get_vacuum_probe(
    data,
    **kwargs
    ):
    """
    Takes some data and computes a vacuum probe, using a method
    selected based on the shape of `data`, and on other arguments
    passed.

    Valid modes are:
        - ...

    Args:
        data:

    Returns:
        (Probe) a Probe instance
    """

    # select mode of operation

    if isinstance(data,DataCube):
        mode = '4D'
    elif isinstance(data,np.ndarray):
        mode = str(data.ndim)+'D'
    elif data is None:
        mode = 'synth'
    else:
        er = f"invalid type {type(data)} for `data`."
        er += f"must be in (DataCube,np.ndarray,None)"
        raise Exception(er)

    if mode == '4D':
        if 'ROI' in kwargs.keys():
            roi = kwargs['ROI']
            if isinstance(roi, np.ndarray):
                mode = '4D_roi_mask'
            else:
                mode = '4D_roi_lims'
        else:
            mode = '4D_full'



    # choose and run a function
    functions = {
        '4D_full' : get_probe_from_vacuum_4Dscan,
        '4D_roi_mask' : get_probe_from_4Dscan_ROI_mask,
        '4D_roi_lims' : get_probe_from_4Dscan_ROI_lims,
        '3D' : get_probe_from_vacuum_3Dstack,
        '2D' : get_probe_from_vacuum_2Dimage,
        'synth' : get_probe_synthetic
        }
    fn = functions[mode]
    probe = fn(data,**kwargs)
    return probe





def get_probe_from_vacuum_4Dscan(
    datacube,
    mask_threshold=0.2,
    mask_expansion=12,
    mask_opening=3,
    verbose=False,
    align=True
    ):
    """
    Averages all diffraction patterns in a datacube, assumed to be taken
    over vacuum, to create and average vacuum probe. Optionally (default)
    aligns the patterns.

    Values outisde the average probe are zeroed, using a binary mask determined
    by the optional parameters mask_threshold, mask_expansion, and mask_opening.
    An initial binary mask is created using a threshold of less than
    mask_threshold times the maximal probe value. A morphological opening of
    mask_opening pixels is performed to eliminate stray pixels (e.g. from
    x-rays), followed by a dilation of mask_expansion pixels to ensure the
    entire probe is captured.

    Args:
        datacube (DataCube): a vacuum scan
        mask_threshold (float): threshold determining mask which zeros values
            outside of probe
        mask_expansion (int): number of pixels by which the zeroing mask is
        expanded to capture the full probe
        mask_opening (int): size of binary opening used to eliminate stray
            bright pixels
        verbose (bool): if True, prints progress updates
        align (bool): if True, aligns the probes before averaging

    Returns:
        (ndarray of shape (datacube.Q_Nx,datacube.Q_Ny)): the average probe
    """

    probe = datacube.data[0,0,:,:]
    for n in tqdmnd(range(1,datacube.R_N)):
        Rx,Ry = np.unravel_index(n,datacube.data.shape[:2])
        curr_DP = datacube.data[Rx,Ry,:,:]
        if verbose:
            print(f"Shifting and averaging diffraction pattern {n} of {datacube.R_N}.")
        if align:
            xshift,yshift = get_shift(probe, curr_DP)
            curr_DP = get_shifted_ar(curr_DP, xshift, yshift)
        probe = probe*(n-1)/n + curr_DP/n

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=1)
    mask = np.cos((np.pi/2)*np.minimum(distance_transform_edt(np.logical_not(mask)) / mask_expansion, 1))**2

    return probe*mask




def get_probe_from_4Dscan_ROI_lims(
    datacube,
    ROI,
    mask_threshold=0.2,
    mask_expansion=12,
    mask_opening=3,
    verbose=False,
    align=True
    ):
    """
    Averages all diffraction patterns within a specified ROI of a datacube to
    create an average vacuum probe. Optionally (default) aligns the patterns.

    See documentation for get_average_probe_from_vacuum_scan for more detailed
    discussion of the algorithm.

    Args:
        datacube (DataCube): a vacuum scan
        ROI (len 4 list or tuple): the limits (rx_min, rx_max, ry_min, ry_max)
            of the selected region.
        mask_threshold (float): threshold determining mask which zeros values
            outside of probe
        mask_expansion (int): number of pixels by which the zeroing mask is
            expanded to capture the full probe
        mask_opening (int): size of binary opening used to eliminate stray
            bright pixels
        verbose (bool): if True, prints progress updates
        align (bool): if True, aligns the probes before averaging
        DP_mask (array): array of same shape as diffraction pattern to mask
            probes

    Returns:
        (ndarray of shape (datacube.Q_Nx,datacube.Q_Ny)): the average probe
    """
    assert len(ROI) == 4

    datacube = DataCube(
        data = datacube.data[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    )
    return get_probe_from_vacuum_4Dscan(
        datacube,
        mask_threshold = mask_threshold,
        mask_expansion = mask_expansion,
        mask_opening = mask_opening,
        verbose = verbose,
        align = align)




def get_probe_from_4Dscan_ROI_mask(
    datacube,
    ROI,
    mask_threshold=0.2,
    mask_expansion=12,
    mask_opening=3,
    verbose=False,
    align=True,
    DP_mask=1
    ):
    """
    Averages all diffraction patterns within a specified ROI of a datacube to
    create an average vacuum probe. Optionally (default) aligns the patterns.

    See documentation for get_average_probe_from_vacuum_scan for more detailed
    discussion of the algorithm.

    Args:
        datacube (DataCube): a vacuum scan
        ROI (ndarray of dtype=bool and shape (datacube.R_Nx,datacube.R_Ny)): An
            array of boolean variables shaped like the real space scan. Only scan
            positions where ROI==True are used to create the average probe.
        mask_threshold (float): threshold determining mask which zeros values
            outside of probe
        mask_expansion (int): number of pixels by which the zeroing mask is
            expanded to capture the full probe
        mask_opening (int): size of binary opening used to eliminate stray
            bright pixels
        verbose (bool): if True, prints progress updates
        align (bool): if True, aligns the probes before averaging
        DP_mask (array): array of same shape as diffraction pattern to mask
            probes

    Returns:
        (ndarray of shape (datacube.Q_Nx,datacube.Q_Ny)): the average probe
    """
    assert ROI.shape==(datacube.R_Nx,datacube.R_Ny)
    length = ROI.sum()
    xy = np.vstack(np.nonzero(ROI))
    probe = datacube.data[xy[0,0],xy[1,0],:,:]
    for n in tqdmnd(range(1,length)):
        curr_DP = datacube.data[xy[0,n],xy[1,n],:,:] * DP_mask
        if align:
            xshift,yshift = get_shift(probe, curr_DP)
            curr_DP = get_shifted_ar(curr_DP, xshift, yshift)
        probe = probe*(n-1)/n + curr_DP/n

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=1)
    mask = np.cos((np.pi/2)*np.minimum(distance_transform_edt(np.logical_not(mask)) / mask_expansion, 1))**2

    return probe*mask





def get_probe_from_vacuum_3Dstack(
    data,
    mask_threshold=0.2,
    mask_expansion=12,
    mask_opening=3
    ):
    """
    Averages all diffraction patterns in a 3D stack of diffraction patterns,
    assumed to be taken over vacuum, to create and average vacuum probe. No
    alignment is performed - i.e. it is assumed that the beam was stationary
    during acquisition of the stack.

    Values outisde the average probe are zeroed, using a binary mask determined
    by the optional parameters mask_threshold, mask_expansion, and mask_opening.
    An initial binary mask is created using a threshold of less than
    mask_threshold times the maximal probe value. A morphological opening of
    mask_opening pixels is performed to eliminate stray pixels (e.g. from
    x-rays), followed by a dilation of mask_expansion pixels to ensure the
    entire probe is captured.

    Args:
        data (array): a 3D stack of vacuum diffraction patterns, shape
            (Q_Nx,Q_Ny,N)
        mask_threshold (float): threshold determining mask which zeros values
            outside of probe
        mask_expansion (int): number of pixels by which the zeroing mask is
            expanded to capture the full probe
        mask_opening (int): size of binary opening used to eliminate stray
            bright pixels

    Returns:
        (array of shape (Q_Nx,Q_Ny)): the average probe
    """
    probe = np.average(data,axis=2)

    mask = probe > np.max(probe)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=1)
    mask = np.cos((np.pi/2)*np.minimum(distance_transform_edt(np.logical_not(mask)) / mask_expansion, 1))**2

    return probe*mask




def get_probe_from_vacuum_2Dimage(
    data,
    mask_threshold=0.2,
    mask_expansion=12,
    mask_opening=3
    ):
    """
    A single image of the probe over vacuum is processed by zeroing values
    outside the central disk, using a binary mask determined by the optional
    parameters mask_threshold, mask_expansion, and mask_opening.  An initial
    binary mask is created using a threshold of less than mask_threshold time
    the maximal probe value. A morphological opening of mask_opening pixels is
    performed to eliminate stray pixels (e.g. from x-rays), followed by a
    dilation of mask_expansion pixels to ensure the entire probe is captured.

    Args:
        data (array): a 2D array of the vacuum diffraction pattern, shape
            (Q_Nx,Q_Ny)
        mask_threshold (float): threshold determining mask which zeros values
            outside of probe
        mask_expansion (int): number of pixels by which the zeroing mask is
            expanded to capture the full probe
        mask_opening (int): size of binary opening used to eliminate stray
            bright pixels

    Returns:
        (array of shape (Q_Nx,Q_Ny)) the average probe
    """
    mask = data > np.max(data)*mask_threshold
    mask = binary_opening(mask, iterations=mask_opening)
    mask = binary_dilation(mask, iterations=1)
    mask = np.cos((np.pi/2)*np.minimum(distance_transform_edt(np.logical_not(mask)) / mask_expansion, 1))**2

    return data*mask




def get_probe_synthetic(
    radius,
    width,
    Q_Nx,
    Q_Ny
    ):
    """
    Makes a synthetic probe, with the functional form of a disk blurred by a
    sigmoid (a logistic function).

    Args:
        radius (float): the probe radius
        width (float): the blurring of the probe edge. width represents the
            full width of the blur, with x=-w/2 to x=+w/2 about the edge
            spanning values of ~0.12 to 0.88
        Q_Nx, Q_Ny (int): the diffraction plane dimensions

    Returns:
        (ndarray of shape (Q_Nx,Q_Ny)): the probe
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








# Probe templates can be generated from vacuum scans, from a selected ROI of a
# vacuum region of a scan, or synthetic probes.  Ultimately the purpose is to
# generate a kernel for convolution with individual diffraction patterns to
# identify Bragg disks.  Kernel generation will generally proceed in two steps,
# which will each correspond to a function call: first, obtaining  or creating
# the diffraction pattern of a probe over vacuum, and second, turning the probe
# DP into a convolution kernel by shifting and normalizing.



