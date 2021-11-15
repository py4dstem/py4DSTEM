# Functions for fitting lattice vectors to measured Bragg peak positions

import numpy as np
from numpy.linalg import lstsq

from ...io import PointList, PointListArray, RealSlice
from ..utils import tqdmnd

def fit_lattice_vectors(braggpeaks, x0=0, y0=0, minNumPeaks=5):
    """
    Fits lattice vectors g1,g2 to braggpeaks given some known (h,k) indexing.

    Args:
        braggpeaks (PointList): A 6 coordinate PointList containing the data to fit.
            Coords are 'qx','qy' (the bragg peak positions), 'intensity' (used as a
            weighting factor when fitting), 'h','k' (indexing). May optionally also
            contain 'index_mask' (bool), indicating which peaks have been successfully
            indixed and should be used.
        x0 (float): x-coord of the origin
        y0 (float): y-coord of the origin
        minNumPeaks (int): if there are fewer than minNumPeaks peaks found in braggpeaks
            which can be indexed, return None for all return parameters

    Returns:
        (7-tuple) A 7-tuple containing:

            * **x0**: *(float)* the x-coord of the origin of the best-fit lattice.
            * **y0**: *(float)* the y-coord of the origin
            * **g1x**: *(float)* x-coord of the first lattice vector
            * **g1y**: *(float)* y-coord of the first lattice vector
            * **g2x**: *(float)* x-coord of the second lattice vector
            * **g2y**: *(float)* y-coord of the second lattice vector
            * **error**: *(float)* the fit error
    """
    assert isinstance(braggpeaks, PointList)
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy','intensity','h','k')])
    braggpeaks = braggpeaks.copy()

    # Remove unindexed peaks
    if 'index_mask' in braggpeaks.dtype.names:
        deletemask = braggpeaks.data['index_mask'] == False
        braggpeaks.remove_points(deletemask)

    # Check to ensure enough peaks are present
    if braggpeaks.length < minNumPeaks:
        return None,None,None,None,None,None,None

    # Get M, the matrix of (h,k) indices
    h,k = braggpeaks.data['h'],braggpeaks.data['k']
    M = np.vstack((np.ones_like(h,dtype=int),h,k)).T

    # Get alpha, the matrix of measured Bragg peak positions
    alpha = np.vstack((braggpeaks.data['qx']-x0, braggpeaks.data['qy']-y0)).T

    # Get weighted matrices
    weights = braggpeaks.data['intensity']
    weighted_M = M*weights[:,np.newaxis]
    weighted_alpha = alpha*weights[:,np.newaxis]

    # Solve for lattice vectors
    beta = lstsq(weighted_M, weighted_alpha, rcond=None)[0]
    x0,y0 = beta[0,0],beta[0,1]
    g1x,g1y = beta[1,0],beta[1,1]
    g2x,g2y = beta[2,0],beta[2,1]

    # Calculate the error
    alpha_calculated = np.matmul(M,beta)
    error = np.sqrt(np.sum((alpha-alpha_calculated)**2,axis=1))
    error = np.sum(error*weights)/np.sum(weights)

    return x0,y0,g1x,g1y,g2x,g2y,error

def fit_lattice_vectors_all_DPs(braggpeaks, x0=0, y0=0, minNumPeaks=5):
    """
    Fits lattice vectors g1,g2 to each diffraction pattern in braggpeaks, given some
    known (h,k) indexing.

    Args:
        braggpeaks (PointList): A 6 coordinate PointList containing the data to fit.
            Coords are 'qx','qy' (the bragg peak positions), 'intensity' (used as a
            weighting factor when fitting), 'h','k' (indexing). May optionally also
            contain 'index_mask' (bool), indicating which peaks have been successfully
            indixed and should be used.
        x0 (float): x-coord of the origin
        y0 (float): y-coord of the origin
        minNumPeaks (int): if there are fewer than minNumPeaks peaks found in braggpeaks
            which can be indexed, return None for all return parameters

    Returns:
        (RealSlice): A RealSlice ``g1g2map`` containing the following 8 arrays:

            * ``g1g2_map.slices['x0']``     x-coord of the origin of the best fit lattice
            * ``g1g2_map.slices['y0']``     y-coord of the origin
            * ``g1g2_map.slices['g1x']``    x-coord of the first lattice vector
            * ``g1g2_map.slices['g1y']``    y-coord of the first lattice vector
            * ``g1g2_map.slices['g2x']``    x-coord of the second lattice vector
            * ``g1g2_map.slices['g2y']``    y-coord of the second lattice vector
            * ``g1g2_map.slices['error']``  the fit error
            * ``g1g2_map.slices['mask']``   1 for successful fits, 0 for unsuccessful
              fits
    """
    assert isinstance(braggpeaks, PointListArray)
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy','intensity','h','k')])

    # Make RealSlice to contain outputs
    slicelabels = ('x0','y0','g1x','g1y','g2x','g2y','error','mask')
    g1g2_map = RealSlice(data=np.zeros((braggpeaks.shape[0],braggpeaks.shape[1],8)),
                       slicelabels=slicelabels, name='g1g2_map')

    # Fit lattice vectors
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        braggpeaks_curr = braggpeaks.get_pointlist(Rx,Ry)
        qx0,qy0,g1x,g1y,g2x,g2y,error = fit_lattice_vectors(braggpeaks_curr, x0, y0, minNumPeaks)
        # Store data
        if g1x is not None:
            g1g2_map.slices['x0'][Rx,Ry] = qx0
            g1g2_map.slices['y0'][Rx,Ry] = qy0
            g1g2_map.slices['g1x'][Rx,Ry] = g1x
            g1g2_map.slices['g1y'][Rx,Ry] = g1y
            g1g2_map.slices['g2x'][Rx,Ry] = g2x
            g1g2_map.slices['g2y'][Rx,Ry] = g2y
            g1g2_map.slices['error'][Rx,Ry] = error
            g1g2_map.slices['mask'][Rx,Ry] = 1

    return g1g2_map

def fit_lattice_vectors_masked(braggpeaks, mask, x0=0, y0=0, minNumPeaks=5):
    """
    Fits lattice vectors g1,g2 to each diffraction pattern in braggpeaks corresponding
    to a scan position for which mask==True.

    Args:
        braggpeaks (PointList): A 6 coordinate PointList containing the data to fit.
            Coords are 'qx','qy' (the bragg peak positions), 'intensity' (used as a
            weighting factor when fitting), 'h','k' (indexing). May optionally also
            contain 'index_mask' (bool), indicating which peaks have been successfully
            indixed and should be used.
        mask (boolean array): real space shaped (R_Nx,R_Ny); fit lattice vectors where
            mask is True
        x0 (float): x-coord of the origin
        y0 (float): y-coord of the origin
        minNumPeaks (int): if there are fewer than minNumPeaks peaks found in braggpeaks
            which can be indexed, return None for all return parameters

    Returns:
        (RealSlice): A RealSlice ``g1g2map`` containing the following 8 arrays:

            * ``g1g2_map.slices['x0']``     x-coord of the origin of the best fit lattice
            * ``g1g2_map.slices['y0']``     y-coord of the origin
            * ``g1g2_map.slices['g1x']``    x-coord of the first lattice vector
            * ``g1g2_map.slices['g1y']``    y-coord of the first lattice vector
            * ``g1g2_map.slices['g2x']``    x-coord of the second lattice vector
            * ``g1g2_map.slices['g2y']``    y-coord of the second lattice vector
            * ``g1g2_map.slices['error']``  the fit error
            * ``g1g2_map.slices['mask']``   1 for successful fits, 0 for unsuccessful
              fits
    """
    assert isinstance(braggpeaks, PointListArray)
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy','intensity')])

    # Make RealSlice to contain outputs
    slicelabels = ('x0','y0','g1x','g1y','g2x','g2y','error','mask')
    g1g2_map = RealSlice(data=np.zeros((braggpeaks.shape[0],braggpeaks.shape[1],8)),
                         slicelabels=slicelabels, name='g1g2_map')

    # Fit lattice vectors
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        if mask[Rx,Ry]:
            braggpeaks_curr = braggpeaks.get_pointlist(Rx,Ry)
            qx0,qy0,g1x,g1y,g2x,g2y,error = fit_lattice_vectors(braggpeaks_curr, x0, y0, minNumPeaks)
            # Store data
            if g1x is not None:
                g1g2_map.slices['x0'][Rx,Ry] = qx0
                g1g2_map.slices['y0'][Rx,Ry] = qy0
                g1g2_map.slices['g1x'][Rx,Ry] = g1x
                g1g2_map.slices['g1y'][Rx,Ry] = g1y
                g1g2_map.slices['g2x'][Rx,Ry] = g2x
                g1g2_map.slices['g2y'][Rx,Ry] = g2y
                g1g2_map.slices['error'][Rx,Ry] = error
                g1g2_map.slices['mask'][Rx,Ry] = 1

    return g1g2_map

