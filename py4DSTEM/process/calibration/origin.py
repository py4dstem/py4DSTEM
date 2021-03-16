# Find the origin of diffraction space

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq

from .qpixelsize import get_probe_size
from ..diskdetection import get_bragg_vector_map
from ..utils import get_CoM, add_to_2D_array_from_floats
from ...io.datastructure import PointListArray

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
    for rx in range(datacube.R_Nx):
        for ry in range(datacube.R_Ny):
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



# Older functions for finding the origin

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

def find_outlier_shifts(xshifts, yshifts, n_sigma=10, edge_boundary=0, n_bins=50):
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
        n_bins          (int) number of histogram bins

    Returns:
        mask            ((R_Nx,R_Ny)-shaped array of bools) the outlier mask
        n               (1D array of length n_bins-1) the histogram counts
        bins            (1D array of length n_bins) the histogram bins
        cutoff          (float) the score cutoff value
    """
    # Get score
    score = np.zeros_like(xshifts)
    score += np.abs(xshifts-np.roll(xshifts,(-1, 0),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,( 1, 0),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,( 0,-1),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,( 0, 1),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,(-1,-1),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,( 1,-1),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,(-1, 1),axis=(0,1))) + \
             np.abs(xshifts-np.roll(xshifts,( 1, 1),axis=(0,1)))
    score += np.abs(yshifts-np.roll(yshifts,(-1, 0),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,( 1, 0),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,( 0,-1),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,( 0, 1),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,(-1,-1),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,( 1,-1),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,(-1, 1),axis=(0,1))) + \
             np.abs(yshifts-np.roll(yshifts,( 1, 1),axis=(0,1)))

    # Make histogram
    bins = np.linspace(0,np.max(score),n_bins)
    n,bins = np.histogram(score,bins=bins)
    width = bins[1]-bins[0]

    # Fit gaussian
    fitfunc = lambda p,x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
    errfunc = lambda p,x,y: fitfunc(p,x) - y
    p0_0 = np.max(gaussian_filter(n,2))
    p0_1 = np.average(bins[:-1]+width/2.,weights=n)
    p0_2 = np.sqrt(np.average((bins[:-1]+width/2. - p0_1)**2,weights=n))
    p0 = [p0_0,p0_1,p0_2]
    p1,success = leastsq(errfunc, p0, args=(bins[:-1]+width/2.,n))

    # Get mask and return
    cutoff = p1[2]*n_sigma
    mask = score > cutoff
    mask[:edge_boundary,:] = True
    mask[-edge_boundary:,:] = True
    mask[:,:edge_boundary] = True
    mask[:,-edge_boundary:] = True

    return mask, n, bins, cutoff

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



