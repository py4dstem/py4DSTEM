# Functions for indexing the Bragg directions

import numpy as np
from numpy.linalg import lstsq

from ...io.datastructure import PointList, PointListArray

def index_bragg_directions(x0, y0, ux, uy, vx, vy, bragg_x, bragg_y):
    """
    From an origin (x0,y0), an pair of lattice vectors (ux,uy), (vx,vy), and a set of measured
    bragg directions (bragg_x,bragg_y), find the indices (h,k) of all the bragg directions.

    The approach is to solve the matrix equation
            alpha = beta * M
    where alpha is the 2xN array of the (x,y) coordinates of N measured bragg directions, beta is the
    2x2 array of the two lattice vectors u,v, and M is the 2xN array of the h,k indices.

    Accepts:
        x0                  (float) x-coord of origin
        y0                  (float) y-coord of origin
        ux                  (float) x-coord of first lattice vector
        uy                  (float) y-coord of first lattice vector
        vx                  (float) x-coord of second lattice vector
        vy                  (float) y-coord of second lattice vector
        bragg_x             (ndarray of floats) x-coords of bragg directions
        bragg_y             (ndarray of floats) y-coords of bragg directions

    Returns:
        h                   (ndarray of ints) first index of the bragg directions
        k                   (ndarray of ints) second index of the bragg directions
        bragg_directions    (PointList) a 4-coordinate PointList with the indexed bragg directions;
                            coords 'qx' and 'qy' contain bragg_x and bragg_y
                            coords 'h' and 'k' contain h and k.
    """
    # Get beta, the matrix of lattice vectors
    beta = np.array([[ux,vx],[uy,vy]])

    # Get alpha, the matrix of measured bragg angles
    alpha = np.vstack([bragg_x-x0,bragg_y-y0])

    # Calculate M, the matrix of peak positions
    M = lstsq(beta, alpha, rcond=None)[0].T
    M = np.round(M).astype(int)

    # Get h,k
    h = M[:,0]
    k = M[:,1]

    # Store in a PointList
    coords = [('qx',float),('qy',float),('h',int),('k',int)]
    bragg_directions = PointList(coordinates=coords)
    bragg_directions.add_tuple_of_nparrays((bragg_x,bragg_y,h,k))

    return h,k, bragg_directions

def generate_lattice(ux,uy,vx,vy,x0,y0,Q_Nx,Q_Ny,h_max=None,k_max=None):
    """
    Returns a full reciprocal lattice stretching to the limits of the diffraction pattern
    by making linear combinations of the lattice vectors up to (±h_max,±k_max).

    This can be useful when there are false peaks or missing peaks in the braggvectormap, which can
    cause errors in the strain finding routines that rely on those peaks for indexing. This allows
    us to create a reference lattice that has all combinations of the lattice vectors all the way
    out to the edges of the frame, and excluding any erroneous intermediate peaks.

    Accepts:
        ux, uy, vx, vy          (float) x and y coords of the u,v lattice vectors
        x0, y0                  (float) x,y origin of the lattice
        Q_Nx, Q_Ny              (int) diffraction pattern size (i.e. dc.Q_Nx, dc.Q_Ny)
        h_max, k_max            (int) maximal indices for generating the lattice
                                    (the lattive is always trimmed to fit inside the
                                     pattern so you can overestimate these, or leave
                                     unspecified and they will be automatically found)

    Returns:
        ideal_lattice           (PointList) A 4-coordinate PointList, ('qx','qy','h','k'),
                                    containing points corresponding to linear combinations
                                    of the u and v vectors, with associated indices
    """

    # Matrix of lattice vectors
    beta = np.array([[ux,uy],[vx,vy]])

    # If no max index is specified, (over)estimate based on image size
    if (h_max is None) or (k_max is None):
        (y,x) = np.mgrid[0:Q_Ny,0:Q_Nx]
        x = x - x0
        y = y - y0
        h_max = np.max(np.ceil(np.abs((x/ux,y/uy))))
        k_max = np.max(np.ceil(np.abs((x/vx,y/vy))))

    (hlist,klist) = np.meshgrid(np.arange(-h_max,h_max+1),np.arange(-k_max,k_max+1))

    M_ideal = np.vstack((hlist.ravel(),klist.ravel())).T
    ideal_peaks = np.matmul(M_ideal,beta)

    coords = [('qx',float),('qy',float),('h',int),('k',int)]

    ideal_data = np.zeros(len(ideal_peaks[:,0]),dtype=coords)
    ideal_data['qx'] = ideal_peaks[:,0]
    ideal_data['qy'] = ideal_peaks[:,1]
    ideal_data['h'] = M_ideal[:,0]
    ideal_data['k'] = M_ideal[:,1]

    ideal_lattice = PointList(coordinates=coords)
    ideal_lattice.add_dataarray(ideal_data)

    #shift to the DP center
    ideal_lattice.data['qx'] += x0
    ideal_lattice.data['qy'] += y0

    # trim peaks outside the image
    deletePeaks = (ideal_lattice.data['qx'] > Q_Nx) | \
            (ideal_lattice.data['qx'] < 0) | \
            (ideal_lattice.data['qy'] > Q_Ny) | \
            (ideal_lattice.data['qy'] < 0)
    ideal_lattice.remove_points(deletePeaks)

    return ideal_lattice

def add_indices_to_braggpeaks(braggpeaks, lattice, maxPeakSpacing, mask=None):
    """
    Using the peak positions (qx,qy) and indices (h,k) in the PointList lattice,
    identify the indices for each peak in the PointListArray braggpeaks.
    Return a new braggpeaks_indexed PointListArray, containing a copy of braggpeaks plus
    three additional data columns -- 'h','k', and 'index_mask' -- specifying the peak indices
    with the ints (h,k) and indicating whether the peak was successfully indexed or not with
    the bool index_mask. If `mask` is specified, only the locations where mask is True are
    indexed.

    Accepts:
        braggpeaks              (PointListArray) the braggpeaks to index. Must contain
                                    the coordinates 'qx', 'qy', and 'intensity'
        lattice                 (PointList) the positions (qx,qy) of the (h,k) lattice points.
                                    Must contain the coordinates 'qx', 'qy', 'h', and 'k'
        maxPeakSpacing          (float) Maximum distance from the ideal lattice points
                                    to include a peak for indexing
        mask                    (bool)  Boolean mask, same shape as the pointlistarray,
                                    indicating which locations should be indexed. This
                                    can be used to index different regions of the scan
                                    with different lattices

    Returns:
        indexed_braggpeaks      (PointListArray) The original braggpeaks pointlistarray, with new
                                    coordinates 'h', 'k', and 'index_mask', containing the indices
                                    of each indexable peak and a bool indicating if each peak has
                                    been successfully indexed
    """

    assert isinstance(braggpeaks,PointListArray)
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy','intensity')])
    assert isinstance(lattice, PointList)
    assert np.all([name in lattice.dtype.names for name in ('qx','qy','h','k')])

    if mask is None:
        mask = np.ones(braggpeaks.shape,dtype=bool)

    assert mask.shape == braggpeaks.shape, 'mask must have same shape as pointlistarray'
    assert mask.dtype == bool, 'mask must be boolean'

    indexed_braggpeaks = braggpeaks.copy()

    # add the coordinates if they don't exist
    if not ('h' in braggpeaks.dtype.names):
        indexed_braggpeaks = indexed_braggpeaks.add_coordinates([('h',int)])
    if not ('k' in braggpeaks.dtype.names):
        indexed_braggpeaks = indexed_braggpeaks.add_coordinates([('k',int)])
    if not ('hindex_mask' in braggpeaks.dtype.names):
        indexed_braggpeaks = indexed_braggpeaks.add_coordinates([('index_mask',bool)])

    # loop over all the scan positions
    for Rx in range(mask.shape[0]):
        for Ry in range(mask.shape[1]):
            if mask[Rx,Ry]:
                pl = indexed_braggpeaks.get_pointlist(Rx,Ry)

                for i in range(pl.length):
                    r2 = (pl.data['qx'][i]-lattice.data['qx'])**2 + \
                         (pl.data['qy'][i]-lattice.data['qy'])**2
                    ind = np.argmin(r2)
                    if r2[ind] <= maxPeakSpacing**2:
                        pl.data['h'][i] = lattice.data['h'][ind]
                        pl.data['k'][i] = lattice.data['k'][ind]
                        pl.data['index_mask'][i] = True
                    else:
                        pl.data['index_mask'][i] = False

    indexed_braggpeaks.name = braggpeaks.name + "_indexed"
    return indexed_braggpeaks


def bragg_vector_intensity_map_by_index(braggpeaks,h,k, symmetric=False):
    """
    Returns a correlation intensity map for an indexed (h,k) Bragg vector
    Used to obtain a darkfield image corresponding to the (h,k) reflection
    or a bightfield image when h=k=0

    Accepts:
        braggpeaks          (PointListArray) must contain the coordinates 'h','k', and 'intensity'
        h, k                (ints) indices for the reflection to generate an intensity map from
        symmetric           (bool) if set to true, returns sum of intensity of (h,k), (-h,k),
                                (h,-k), (-h,-k)

    Returns:
        intensitty_map      (numpy array) a map of the intensity of the (h,k) Bragg vector
                                correlation. same shape as the pointlistarray.
    """
    assert isinstance(braggpeaks,PointListArray), "braggpeaks must be a PointListArray"
    assert np.all([name in braggpeaks.dtype.names for name in ('h','k','intensity')])
    intensity_map = np.zeros(braggpeaks.shape,dtype=float)

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pl = braggpeaks.get_pointlist(Rx,Ry)
            if pl.length > 0:
                if symmetric:
                    matches = np.logical_and(np.abs(pl.data['h']) == np.abs(h), np.abs(pl.data['k']) == np.abs(k))
                else:
                    matches = np.logical_and(pl.data['h'] == h, pl.data['k'] == k)

                # now apply the indexing mask
                matches = np.logical_and(matches, pl.data['index_mask'])
                if len(matches)>0:
                    intensity_map[Rx,Ry] = np.sum(pl.data['intensity'][matches])

    return intensity_map


