# Functions for fitting lattice vectors to measured Bragg peak positions

import numpy as np
from numpy.linalg import lstsq

from ...io.datastructure import PointList, PointListArray, RealSlice
from ..utils import tqdmnd

def fit_lattice_vectors(bragg_peaks, bragg_directions, x0, y0, maxPeakSpacing=20, minNumPeaks=5):
    """
    Fits lattice vectors u,v to bragg_peaks given some known (h,k) indexing in bragg_directions.

    Accepts:
        bragg_peaks         (PointList) A 3 coordinate Pointlist containing the data to fit.
                            Coords 'qx','qy' specify the bragg peak positions, and coord 'intensity'
                            is used as a weighting factor when fitting
        bragg_directions    (PointList) A 4 coordinate Poinlist containing the (h,k) indexing of
                            the bragg peaks. Coords 'qx','qy' specify the positions of the bragg
                            directions, and coords 'h','k' specify their indexing.
        x0                  (float) x-coord of the origin
        y0                  (float) y-coord of the origin
        maxPeakSpacing      (float) When identifying the index of each peak in bragg_peaks, it must
                            be within maxPeakSpacing of one of the bragg_diretions, or it is ignored
        minNumPeaks         (int) if there are fewer than minNumPeaks peaks found in bragg_peaks
                            which can be indexed, return None for all return parameters

    Returns:
        ux                  (float) x-coord of the first lattice vector
        uy                  (float) y-coord of the first lattice vector
        vx                  (float) x-coord of the second lattice vector
        vy                  (float) y-coord of the second lattice vector
        error               (float) the fit error
    """
    assert isinstance(bragg_peaks, PointList)
    assert np.all([name in bragg_peaks.dtype.names for name in ('qx','qy','intensity')])
    assert isinstance(bragg_directions, PointList)
    assert np.all([name in bragg_directions.dtype.names for name in ('qx','qy','h','k')])
    bragg_peaks = bragg_peaks.copy()

    # Get indices
    h = np.zeros(bragg_peaks.length,dtype=int)
    k = np.zeros(bragg_peaks.length,dtype=int)
    deletemask = np.zeros(bragg_peaks.length,dtype=bool)
    for i in range(bragg_peaks.length):
        r2 = (bragg_peaks.data['qx'][i]-bragg_directions.data['qx'])**2 + \
             (bragg_peaks.data['qy'][i]-bragg_directions.data['qy'])**2
        ind = np.argmin(r2)
        h[i] = bragg_directions.data['h'][ind]
        k[i] = bragg_directions.data['k'][ind]
        if r2[ind] > maxPeakSpacing**2:
            deletemask[i] = True
    bragg_peaks.remove_points(deletemask)
    h = np.delete(h,deletemask.nonzero()[0])
    k = np.delete(k,deletemask.nonzero()[0])

    # Check to ensure enough peaks are present
    if bragg_peaks.length < minNumPeaks:
        return None,None,None,None,None

    # Get M, the matrix of (h,k) indices
    M = np.vstack((np.ones_like(h,dtype=int),h,k)).T

    # Get alpha, the matrix of measured Bragg peak positions
    alpha = np.vstack((bragg_peaks.data['qx']-x0, bragg_peaks.data['qy']-y0)).T

    # Get weighted matrices
    weights = bragg_peaks.data['intensity']
    weighted_M = M*weights[:,np.newaxis]
    weighted_alpha = alpha*weights[:,np.newaxis]

    # Solve for lattice vectors
    beta = lstsq(weighted_M, weighted_alpha, rcond=None)[0]
    ux,uy = beta[1,0],beta[1,1]
    vx,vy = beta[2,0],beta[2,1]

    # Calculate the error
    alpha_calculated = np.matmul(M,beta)
    error = np.sqrt(np.sum((alpha-alpha_calculated)**2,axis=1))
    error = np.sum(error*weights)/np.sum(weights)

    return ux,uy,vx,vy,error

def fit_lattice_vectors_all_DPs(bragg_peaks, bragg_directions, x0, y0, maxPeakSpacing=20,
                                                                       minNumPeaks=5):
    """
    Fits lattice vectors u,v to each diffraction pattern in bragg_peaks, given some known (h,k)
    indexing in bragg_directions.

    Accepts:
        bragg_peaks         (PointListArray) A 3 coordinate Pointlist containing the data to fit.
                            Coords 'qx','qy' specify the bragg peak positions, and coord 'intensity'
                            is used as a weighting factor when fitting
        bragg_directions    (PointList) A 4 coordinate Poinlist containing the (h,k) indexing of
                            the bragg peaks. Coords 'qx','qy' specify the positions of the bragg
                            directions, and coords 'h','k' specify their indexing.
        x0                  (float) x-coord of the origin
        y0                  (float) y-coord of the origin
        maxPeakSpacing      (float) When identifying the index of each peak in bragg_peaks, it must
                            be within maxPeakSpacing of one of the bragg_diretions, or it is ignored
        minNumPeaks         (int) if there are fewer than minNumPeaks peaks found in bragg_peaks
                            which can be indexed, return None for all return parameters

    Returns:
        uv_map                  (RealSlice) a RealSlice containing the following 6 arrays:
        uv_map.slices['ux']     x-coord of the first lattice vector
        uv_map.slices['uy']     y-coord of the first lattice vector
        uv_map.slices['vx']     x-coord of the second lattice vector
        uv_map.slices['vy']     y-coord of the second lattice vector
        uv_map.slices['error']  the fit error
        uv_map.slices['mask']   1 for successful fits, 0 for unsuccessful fits
    """
    assert isinstance(bragg_peaks, PointListArray)
    assert np.all([name in bragg_peaks.dtype.names for name in ('qx','qy','intensity')])
    assert isinstance(bragg_directions, PointList)
    assert np.all([name in bragg_directions.dtype.names for name in ('qx','qy','h','k')])

    # Make RealSlice to contain outputs
    slicelabels = ('ux','uy','vx','vy','error','mask')
    uv_map = RealSlice(data=np.zeros((bragg_peaks.shape[0],bragg_peaks.shape[1],6)),
                       slicelabels=slicelabels, name='uv_map')

    # Fit lattice vectors
    for (Rx, Ry) in tqdmnd(bragg_peaks.shape[0],bragg_peaks.shape[1]):
        bragg_peaks_curr = bragg_peaks.get_pointlist(Rx,Ry)
        ux,uy,vx,vy,error = fit_lattice_vectors(bragg_peaks_curr, bragg_directions, x0, y0,
                                                maxPeakSpacing, minNumPeaks)
        # Store data
        if ux is not None:
            uv_map.slices['ux'][Rx,Ry] = ux
            uv_map.slices['uy'][Rx,Ry] = uy
            uv_map.slices['vx'][Rx,Ry] = vx
            uv_map.slices['vy'][Rx,Ry] = vy
            uv_map.slices['error'][Rx,Ry] = error
            uv_map.slices['mask'][Rx,Ry] = 1

    return uv_map

def fit_lattice_vectors_masked(bragg_peaks, bragg_directions, x0, y0, mask, maxPeakSpacing=20,
                                                                            minNumPeaks=5):
    """
    Fits lattice vectors u,v to each diffraction pattern in bragg_peaks corresponding to a scan
    position for which mask==True.

    Accepts:
        bragg_peaks         (PointListArray) A 3 coordinate Pointlist containing the data to fit.
                            Coords 'qx','qy' specify the bragg peak positions, and coord 'intensity'
                            is used as a weighting factor when fitting
        bragg_directions    (PointList) A 4 coordinate Poinlist containing the (h,k) indexing of
                            the bragg peaks. Coords 'qx','qy' specify the positions of the bragg
                            directions, and coords 'h','k' specify their indexing.
        x0                  (float) x-coord of the origin
        y0                  (float) y-coord of the origin
        mask                (boolean array) real space shaped (R_Nx,R_Ny); fit lattice vectors where
                            mask is True
        maxPeakSpacing      (float) When identifying the index of each peak in bragg_peaks, it must
                            be within maxPeakSpacing of one of the bragg_diretions, or it is ignored
        minNumPeaks         (int) if there are fewer than minNumPeaks peaks found in bragg_peaks
                            which can be indexed, return None for all return parameters

    Returns:
        uv_map                  (RealSlice) a RealSlice containing the following 6 arrays:
        uv_map.slices['ux']     x-coord of the first lattice vector
        uv_map.slices['uy']     y-coord of the first lattice vector
        uv_map.slices['vx']     x-coord of the second lattice vector
        uv_map.slices['vy']     y-coord of the second lattice vector
        uv_map.slices['error']  the fit error
        uv_map.slices['mask']   1 for successful fits, 0 for unsuccessful fits
    """
    assert isinstance(bragg_peaks, PointListArray)
    assert np.all([name in bragg_peaks.dtype.names for name in ('qx','qy','intensity')])
    assert isinstance(bragg_directions, PointList)
    assert np.all([name in bragg_directions.dtype.names for name in ('qx','qy','h','k')])

    # Make RealSlice to contain outputs
    slicelabels = ('ux','uy','vx','vy','error','mask')
    uv_map = RealSlice(data=np.zeros((bragg_peaks.shape[0],bragg_peaks.shape[1],6)),
                       slicelabels=slicelabels, name='uv_map')

    # Fit lattice vectors
    for (Rx, Ry) in tqdmnd(bragg_peaks.shape[0],bragg_peaks.shape[1]):
        if mask[Rx,Ry]:
            bragg_peaks_curr = bragg_peaks.get_pointlist(Rx,Ry)
            ux,uy,vx,vy,error = fit_lattice_vectors(bragg_peaks_curr, bragg_directions, x0, y0,
                                                    maxPeakSpacing, minNumPeaks)
            # Store data
            if ux is not None:
                uv_map.slices['ux'][Rx,Ry] = ux
                uv_map.slices['uy'][Rx,Ry] = uy
                uv_map.slices['vx'][Rx,Ry] = vx
                uv_map.slices['vy'][Rx,Ry] = vy
                uv_map.slices['error'][Rx,Ry] = error
                uv_map.slices['mask'][Rx,Ry] = 1

    return uv_map

