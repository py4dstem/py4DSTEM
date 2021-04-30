# Find the origin of diffraction space

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq

from ..fit import plane,parabola,fit_2D
from ..diskdetection import get_bragg_vector_map
from ..utils import get_CoM, add_to_2D_array_from_floats,tqdmnd
from ...io.datastructure import PointListArray


### Functions for finding the origin

def get_probe_size(DP, thresh_lower=0.01, thresh_upper=0.99, N=100):
    """
    Gets the center and radius of the probe in the diffraction plane.

    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern DP with a
    linspace of N thresholds from thresh_lower to thresh_upper, measured relative to the maximum
    intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r should
    change very little over a wide range of intermediate values of the threshold. The range in which
    r is trustworthy is found by taking the derivative of r(thresh) and finding identifying where it
    is small.  The radius is taken to be the mean of these r values.
    Using the threshold corresponding to this r, a mask is created and the CoM of the DP times this
    mask it taken.  This is taken to be the origin x0,y0.

    Accepts:
        DP              (2D array) the diffraction pattern in which to find the central disk.
                        A position averaged, or shift-corrected and averaged, DP work well.
        thresh_lower    (float, 0 to 1) the lower limit of threshold values
        thresh_upper    (float, 0 to 1) the upper limit of threshold values
        N               (int) the number of thresholds / masks to use

    Returns:
        r               (float) the central disk radius, in pixels
        x0              (float) the x position of the central disk center
        y0              (float) the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower,thresh_upper,N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax*thresh
        r_vals[i] = np.sqrt(np.sum(mask)/np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2*np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax*thresh
    x0,y0 = get_CoM(DP*mask)

    return r,x0,y0

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

def get_origin(datacube,r=None,rscale=1.2,dp_max=None,mask=None):
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
        mask        (ndarray or None) if not None, should be an (R_Nx,R_Ny) shaped
                    boolean array. Origin is found only where mask==True, and masked
                    arrays are returned for qx0,qy0

    Returns:
        qx0,qy0     (ndarrays) the origin
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

    if mask is None:
        for (rx,ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding origins',unit='DP',unit_scale=True):
            dp = datacube.data[rx,ry,:,:]
            _qx0,_qy0 = np.unravel_index(np.argmax(gaussian_filter(dp,r)),
                                         (datacube.Q_Nx,datacube.Q_Ny))
            _mask = np.hypot(qxx-_qx0,qyy-_qy0) < r*rscale
            qx0[rx,ry],qy0[rx,ry] = get_CoM(dp*_mask)

    else:
        assert mask.shape==(datacube.R_Nx,datacube.R_Ny)
        assert mask.dtype==bool
        qx0 = np.ma.array(data=qx0,mask=np.zeros((datacube.R_Nx,datacube.R_Ny),dtype=bool))
        qy0 = np.ma.array(data=qy0,mask=np.zeros((datacube.R_Nx,datacube.R_Ny),dtype=bool))
        for (rx,ry) in tqdmnd(datacube.R_Nx,datacube.R_Ny,desc='Finding origins',unit='DP',unit_scale=True):
            if mask[rx,ry]:
                dp = datacube.data[rx,ry,:,:]
                _qx0,_qy0 = np.unravel_index(np.argmax(gaussian_filter(dp,r)),
                                             (datacube.Q_Nx,datacube.Q_Ny))
                _mask = np.hypot(qxx-_qx0,qyy-_qy0) < r*rscale
                qx0.data[rx,ry],qy0.data[rx,ry] = get_CoM(dp*_mask)
            else:
                qx0.mask,qy0.mask = True,True

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
        mask                    (2b boolean array) ignore points where mask=True
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

def center_braggpeaks(braggpeaks, qx0=None, qy0=None, coords=None, name=None):
    """
    Shift the braggpeaks positions to center them about the origin, given
    either by (qx0,qy0) or by the Coordinates instance coords. Either
    (qx0,qy0) or coords must be specified.

    Accepts:
        braggpeaks  (PointListArray) the detected, unshifted bragg peaks
        qx0,qy0     ((R_Nx,R_Ny)-shaped arrays) the position of the origin
        coords      (Coordinates) an object containing the origin positions
        name        (str, optional) a name for the returned PointListArray.
                    If unspecified, takes the old PLA name, removes '_raw'
                    if present at the end of the string, then appends
                    '_centered'.

    Returns:
        braggpeaks_centered  (PointListArray) the centered Bragg peaks
    """
    assert isinstance(braggpeaks, PointListArray)
    assert (qx0 is not None and qy0 is not None) != (coords is not None), (
                                "Either (qx0,qy0) or coords must be specified")
    if coords is not None:
        qx0,qy0 = coords.get_center()
        assert qx0 is not None and qy0 is not None, "coords did not contain center position"
    if name is None:
        sl = braggpeaks.name.split('_')
        _name = '_'.join([s for i,s in enumerate(sl) if not (s=='raw' and i==len(sl)-1)])
        name = _name+"_centered"
    assert isinstance(name,str)
    braggpeaks_centered = braggpeaks.copy(name=name)

    for Rx in range(braggpeaks_centered.shape[0]):
        for Ry in range(braggpeaks_centered.shape[1]):
            pointlist = braggpeaks_centered.get_pointlist(Rx,Ry)
            qx,qy = qx0[Rx,Ry],qy0[Rx,Ry]
            pointlist.data['qx'] -= qx
            pointlist.data['qy'] -= qy

    return braggpeaks_centered



