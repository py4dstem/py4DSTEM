# Find the origin of diffraction space

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq

from .qpixelsize import get_probe_size
from ..fit import plane,parabola,fit_2D
from ..diskdetection import get_bragg_vector_map
from ..utils import get_CoM, add_to_2D_array_from_floats,tqdmnd
from ...io.datastructure import PointListArray


### Functions for finding the origin

def get_origin_single_dp(dp,r,rscale=1.2):
    """
    Find the origin for a single diffraction pattern, assuming:
        - There is no beam stop
        - The center beam contains the highest intensity

    Accepts:
        dp          (ndarray) a diffraction pattern
        r           (number) the approximate radius of the center disk
        rscale      (number) expand 'r' by this amount to form a mask about the
                    center disk when taking its center of mass

    Returns:
        qx0,qy0     (numbers) the origin
    """
    Q_Nx,Q_Ny = dp.shape
    _qx0,_qy0 = np.unravel_index(np.argmax(gaussian_filter(dp,r)),(Q_Nx,Q_Ny))
    qyy,qxx = np.meshgrid(np.arange(Q_Ny),np.arange(Q_Nx))
    mask = np.hypot(qxx-_qx0,qyy-_qy0) < r*rscale
    qx0,qy0 = get_CoM(dp*mask)
    return qx0,qy0

def get_origin(datacube,r=None,rscale=1.2,dp_max=None):
    """
    Find the origin for all diffraction patterns in a datacube, assuming:
        - There is no beam stop
        - The center beam contains the highest intensity

    Accepts:
        datacube    (DataCube) the data
        r           (number or None) the approximate radius of the center disk.
                    If None (default), tries to compute r using the get_probe_size
                    method.  The data used for this is controlled by dp_max.
        rscale      (number) expand 'r' by this amount to form a mask about the
                    center disk when taking its center of mass
        dp_max      (ndarray or None) the diffraction pattern or dp-shaped array
                    used to compute the center disk radius, if r is left unspecified.
                    if dp_max==None (default), computes and uses the maximal diffraction
                    pattern. Otherwise, this should be a (Q_Nx,Q_Ny) shaped array.
                    diffraction

    Returns:
        qx0,qy0     (numbers) the origin
    """
    if r is None:
        if dp_max is None:
            dp_max = np.max(datacube.data,axis=(0,1))
        else:
            assert dp_max.shape == (datacube.Q_Nx,datacube.Q_Ny)
        r,_,_ = get_probe_size(dp_max)

    qx0 = np.zeros((datacube.R_Nx,datacube.R_Ny))
    qy0 = np.zeros((datacube.R_Nx,datacube.R_Ny))
    qyy,qxx = np.meshgrid(np.arange(datacube.Q_Ny),np.arange(datacube.Q_Nx))
    for (rx,ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding origins',unit='DP',unit_scale=True):
        dp = datacube.data[rx,ry,:,:]
        _qx0,_qy0 = np.unravel_index(np.argmax(gaussian_filter(dp,r)),
                                     (datacube.Q_Nx,datacube.Q_Ny))
        mask = np.hypot(qxx-_qx0,qyy-_qy0) < r*rscale
        qx0[rx,ry],qy0[rx,ry] = get_CoM(dp*mask)
    return qx0,qy0

def get_origin_single_dp_beamstop(dp,**kwargs):
    """
    Find the origin for a single diffraction pattern, assuming there is a beam stop.

    Accepts:

    Returns:

    """
    return

def get_origin_beamstop(dp,**kwargs):
    """
    Find the origin for all single diffraction patterns in a datacube,
    assuming there is a beam stop.

    Accepts:

    Returns:

    """
    return


### Functions for fitting the origin

def fit_origin(qx0_meas,qy0_meas,mask=None,fitfunction='plane',returnfitp=False):
    """
    Fits the position of the origin of diffraction space to a plane or parabola,
    given some 2D arrays (qx0_meas,qy0_meas) of measured center positions, optionally
    masked by the Boolean array `mask`.

    Accepts:
        qx0_meas,qy0_meas       (2d arrays)
        mask                    (2b boolean array)
        fitfunction             (str) must be 'plane' or 'parabola'
        returnfitp              (bool) if True, returns the fit parameters

    Returns:
        (qx0_fit,qy0_fit,qx0_residuals,qy0_residuals)
                             or
        (qx0_fit,qy0_fit,qx0_residuals,qy0_residuals),(popt_x,popt_y,pcov_x,pcov_y)
    """
    assert isinstance(qx0_meas,np.ndarray) and len(qx0_meas.shape)==2
    assert isinstance(qx0_meas,np.ndarray) and len(qy0_meas.shape)==2
    assert qx0_meas.shape == qy0_meas.shape
    assert mask is None or mask.shape==qx0_meas.shape and mask.dtype==bool
    assert fitfunction in ('plane','parabola')
    if fitfunction=='plane':
        f = plane
    elif fitfunction=='parabola':
        f = parabola
    else:
        raise Exception("Invalid fitfunction '{}'".format(fitfunction))

    # Fit data
    if mask is None:
        popt_x, pcov_x, qx0_fit = fit_2D(f, qx0_meas)
        popt_y, pcov_y, qy0_fit = fit_2D(f, qy0_meas)
    else:
        popt_x, pcov_x, qx0_fit = fit_2D(f, qx0_meas, data_mask=mask==False)
        popt_y, pcov_y, qy0_fit = fit_2D(f, qy0_meas, data_mask=mask==False)

    # Compute residuals
    qx0_residuals = qx0_meas-qx0_fit
    qy0_residuals = qy0_meas-qy0_fit

    # Return
    if not returnfitp:
        return qx0_fit,qy0_fit,qx0_residuals,qy0_residuals
    else:
        return (qx0_fit,qy0_fit,qx0_residuals,qy0_residuals),(popt_x,popt_y,pcov_x,pcov_y)








### Older / soon-to-be-deprecated functions for finding the origin

def get_diffraction_shifts(Braggpeaks, Q_Nx, Q_Ny, findcenter='CoM'):
    """
    Gets the diffraction shifts.

    First, an guess at the unscattered beam position is determined, either by taking the CoM of the
    Bragg vector map, or by taking its maximal pixel.  If the CoM is used, an additional
    refinement step is used where we take the CoM of a Bragg vector map contructed from a first guess
    at the central Bragg peaks (as opposed to the BVM of all BPs). Once a
    unscattered beam position is determined, the Bragg peak closest to this position is identified.
    The shifts in these peaks positions from their average are returned as the diffraction shifts.

    Accepts:
        Braggpeaks      (PointListArray) the Bragg peak positions
        Q_Nx, Q_Ny      (ints) the shape of diffration space
        findcenter      (str) specifies the method for determining the unscattered beam position
                        options: 'CoM', or 'max'

    Returns:
        xshifts         ((R_Nx,R_Ny)-shaped array) the shifts in x
        yshifts         ((R_Nx,R_Ny)-shaped array) the shifts in y
        braggvectormap  ((R_Nx,R_Ny)-shaped array) the Bragg vector map of only the Bragg peaks
                        identified with the unscattered beam. Useful for diagnostic purposes.
    """
    assert isinstance(Braggpeaks, PointListArray), "Braggpeaks must be a PointListArray"
    assert all([isinstance(item, (int,np.integer)) for item in [Q_Nx,Q_Ny]])
    assert isinstance(findcenter, str), "center must be a str"
    assert findcenter in ['CoM','max'], "center must be either 'CoM' or 'max'"
    R_Nx,R_Ny = Braggpeaks.shape

    # Get guess at position of unscattered beam
    braggvectormap_all = get_bragg_vector_map(Braggpeaks, Q_Nx, Q_Ny)
    if findcenter=='max':
        x0,y0 = np.unravel_index(np.argmax(gaussian_filter(braggvectormap_all,10)),(Q_Nx,Q_Ny))
    else:
        x0,y0 = get_CoM(braggvectormap_all)
        braggvectormap = np.zeros_like(braggvectormap_all)
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                pointlist = Braggpeaks.get_pointlist(Rx,Ry)
                if pointlist.length > 0:
                    r2 = (pointlist.data['qx']-x0)**2 + (pointlist.data['qy']-y0)**2
                    index = np.argmin(r2)
                    braggvectormap = add_to_2D_array_from_floats(braggvectormap,
                                                                pointlist.data['qx'][index],
                                                                pointlist.data['qy'][index],
                                                                pointlist.data['intensity'][index])
        x0,y0 = get_CoM(braggvectormap)

    # Get Bragg peak closest to unscattered beam at each scan position
    braggvectormap = np.zeros_like(braggvectormap_all)
    xshifts = np.zeros((R_Nx,R_Ny))
    yshifts = np.zeros((R_Nx,R_Ny))
    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            pointlist = Braggpeaks.get_pointlist(Rx,Ry)
            if pointlist.length > 0:
                r2 = (pointlist.data['qx']-x0)**2 + (pointlist.data['qy']-y0)**2
                index = np.argmin(r2)
                braggvectormap = add_to_2D_array_from_floats(braggvectormap,
                                                            pointlist.data['qx'][index],
                                                            pointlist.data['qy'][index],
                                                            pointlist.data['intensity'][index])
                xshifts[Rx,Ry] = pointlist.data['qx'][index]
                yshifts[Rx,Ry] = pointlist.data['qy'][index]

    xshifts -= np.average(xshifts)
    yshifts -= np.average(yshifts)
    return xshifts, yshifts, braggvectormap

def find_outlier_shifts(xshifts, yshifts, n_sigma=10, edge_boundary=0):
    """
    Finds outliers in the shift matrices.

    Gets a score function for each scan position Rx,Ry, given by the sum of the absolute values of
    the difference between the shifts at this position and all 8 NNs. Calculates a histogram of the
    scoring function, fits a gaussian to its initial peak, and sets a cutoff value to n_sigma times
    its standard deviation. Values beyond this cutoff are deemed outliers, as are scan positions
    within edge_boundary pixels of the edge of real space.

    Accepts:
        xshifts         ((R_Nx,R_Ny)-shaped array) the shifts in x
        yshifts         ((R_Nx,R_Ny)-shaped array) the shifts in y
        n_sigma         (float) the cutoff value for the score function, in number of std
        edge_boundary   (int) number of pixels near the mask edge to mark as outliers

    Returns:
        mask            ((R_nx,R_ny)-shaped array of bools) the outlier mask
        score           ((R_nx,R_ny)-shaped array) the outlier scores
        cutoff          (float) the score cutoff value
    """
    # Get score
    score = np.zeros_like(xshifts)
    score[:-1,:] += np.abs(xshifts[:-1,:]-np.roll(xshifts,(-1, 0),axis=(0,1))[:-1,:])
    score[ 1:,:] += np.abs(xshifts[ 1:,:]-np.roll(xshifts,( 1, 0),axis=(0,1))[ 1:,:])
    score[:,:-1] += np.abs(xshifts[:,:-1]-np.roll(xshifts,( 0,-1),axis=(0,1))[:,:-1])
    score[:, 1:] += np.abs(xshifts[:, 1:]-np.roll(xshifts,( 0, 1),axis=(0,1))[:, 1:])
    score[:-1,:-1] += np.abs(xshifts[:-1,:-1]-np.roll(xshifts,(-1,-1),axis=(0,1))[:-1,:-1])
    score[ 1:,:-1] += np.abs(xshifts[ 1:,:-1]-np.roll(xshifts,( 1,-1),axis=(0,1))[ 1:,:-1])
    score[:-1, 1:] += np.abs(xshifts[:-1, 1:]-np.roll(xshifts,(-1, 1),axis=(0,1))[:-1, 1:])
    score[ 1:, 1:] += np.abs(xshifts[ 1:, 1:]-np.roll(xshifts,( 1, 1),axis=(0,1))[ 1:, 1:])
    score[:-1,:] += np.abs(yshifts[:-1,:]-np.roll(yshifts,(-1, 0),axis=(0,1))[:-1,:])
    score[ 1:,:] += np.abs(yshifts[ 1:,:]-np.roll(yshifts,( 1, 0),axis=(0,1))[ 1:,:])
    score[:,:-1] += np.abs(yshifts[:,:-1]-np.roll(yshifts,( 0,-1),axis=(0,1))[:,:-1])
    score[:, 1:] += np.abs(yshifts[:, 1:]-np.roll(yshifts,( 0, 1),axis=(0,1))[:, 1:])
    score[:-1,:-1] += np.abs(yshifts[:-1,:-1]-np.roll(yshifts,(-1,-1),axis=(0,1))[:-1,:-1])
    score[ 1:,:-1] += np.abs(yshifts[ 1:,:-1]-np.roll(yshifts,( 1,-1),axis=(0,1))[ 1:,:-1])
    score[:-1, 1:] += np.abs(yshifts[:-1, 1:]-np.roll(yshifts,(-1, 1),axis=(0,1))[:-1, 1:])
    score[ 1:, 1:] += np.abs(yshifts[ 1:, 1:]-np.roll(yshifts,( 1, 1),axis=(0,1))[ 1:, 1:])
    score[1:-1,1:-1] /= 8.
    score[: 1,1:-1] /= 5.
    score[-1:,1:-1] /= 5.
    score[1:-1,: 1] /= 5.
    score[1:-1,-1:] /= 5.
    score[ 0, 0] /= 3.
    score[ 0,-1] /= 3.
    score[-1, 0] /= 3.
    score[-1,-1] /= 3.

    # Get mask and return
    cutoff = np.std(score)*n_sigma
    mask = score > cutoff
    if edge_boundary > 0:
        mask[:edge_boundary,:] = True
        mask[-edge_boundary:,:] = True
        mask[:,:edge_boundary] = True
        mask[:,-edge_boundary:] = True

    return mask, score, cutoff

def shift_braggpeaks(Braggpeaks, xshifts, yshifts):
    """
    Applies shifts xshifts,yshifts to Braggpeaks.

    Accepts:
        Braggpeaks  (PointListArray) the Bragg unshifted Bragg peaks
        xshifts     ((R_Nx,R_Ny)-shaped array) the shifts in x
        yshifts     ((R_Nx,R_Ny)-shaped array) the shifts in y

    Returns:
        shifted_Braggpeaks  (PointListArray) the shifted Bragg peaks
    """
    assert isinstance(Braggpeaks, PointListArray)
    shifted_Braggpeaks = Braggpeaks.copy(name=Braggpeaks.name+"_shiftcorrected")

    for Rx in range(shifted_Braggpeaks.shape[0]):
        for Ry in range(shifted_Braggpeaks.shape[1]):
            pointlist = shifted_Braggpeaks.get_pointlist(Rx,Ry)
            shifts_qx = xshifts[Rx,Ry]
            shifts_qy = yshifts[Rx,Ry]
            pointlist.data['qx'] -= shifts_qx
            pointlist.data['qy'] -= shifts_qy

    return shifted_Braggpeaks



