# Functions for indexing the Bragg directions

import numpy as np
from numpy.linalg import lstsq

from emdfile import tqdmnd, PointList, PointListArray


def get_selected_lattice_vectors(gx, gy, i0, i1, i2):
    """
    From a set of reciprocal lattice points (gx,gy), and indices in those arrays which
    specify the center beam, the first basis lattice vector, and the second basis lattice
    vector, computes and returns the lattice vectors g1 and g2.

    Args:
        gx (1d array): the reciprocal lattice points x-coords
        gy (1d array): the reciprocal lattice points y-coords
        i0 (int): index in the (gx,gy) arrays specifying the center beam
        i1 (int): index in the (gx,gy) arrays specifying the first basis lattice vector
        i2 (int): index in the (gx,gy) arrays specifying the second basis lattice vector

    Returns:
        (2-tuple of 2-tuples) A 2-tuple containing

            * **g1**: *(2-tuple)* the first lattice vector, (g1x,g1y)
            * **g2**: *(2-tuple)* the second lattice vector, (g2x,g2y)
    """
    for i in (i0, i1, i2):
        assert isinstance(i, (int, np.integer))
    g1x = gx[i1] - gx[i0]
    g1y = gy[i1] - gy[i0]
    g2x = gx[i2] - gx[i0]
    g2y = gy[i2] - gy[i0]
    return (g1x, g1y), (g2x, g2y)


def generate_lattice(ux, uy, vx, vy, x0, y0, Q_Nx, Q_Ny, h_max=None, k_max=None):
    """
    Returns a full reciprocal lattice stretching to the limits of the diffraction pattern
    by making linear combinations of the lattice vectors up to (±h_max,±k_max).

    This can be useful when there are false peaks or missing peaks in the braggvectormap,
    which can cause errors in the strain finding routines that rely on those peaks for
    indexing. This allows us to create a reference lattice that has all combinations of
    the lattice vectors all the way out to the edges of the frame, and excluding any
    erroneous intermediate peaks.

    Args:
        ux (float): x-coord of the u lattice vector
        uy (float): y-coord of the u lattice vector
        vx (float): x-coord of the v lattice vector
        vy (float): y-coord of the v lattice vector
        x0 (float): x-coord of the lattice origin
        y0 (float): y-coord of the lattice origin
        Q_Nx (int): diffraction pattern size in the x-direction
        Q_Ny (int): diffraction pattern size in the y-direction
        h_max, k_max (int): maximal indices for generating the lattice (the lattive is
            always trimmed to fit inside the pattern so you can overestimate these, or
            leave unspecified and they will be automatically found)

    Returns:
        (PointList): A 4-coordinate PointList, ('qx','qy','h','k'), containing points
        corresponding to linear combinations of the u and v vectors, with associated
        indices
    """

    # Matrix of lattice vectors
    beta = np.array([[ux, uy], [vx, vy]])

    # If no max index is specified, (over)estimate based on image size
    if (h_max is None) or (k_max is None):
        (y, x) = np.mgrid[0:Q_Ny, 0:Q_Nx]
        x = x - x0
        y = y - y0
        h_max = np.max(np.ceil(np.abs((x / ux, y / uy))))
        k_max = np.max(np.ceil(np.abs((x / vx, y / vy))))

    (hlist, klist) = np.meshgrid(
        np.arange(-h_max, h_max + 1), np.arange(-k_max, k_max + 1)
    )

    M_ideal = np.vstack((hlist.ravel(), klist.ravel())).T
    ideal_peaks = np.matmul(M_ideal, beta)

    coords = [("qx", float), ("qy", float), ("h", int), ("k", int)]

    ideal_data = np.zeros(len(ideal_peaks[:, 0]), dtype=coords)
    ideal_data["qx"] = ideal_peaks[:, 0]
    ideal_data["qy"] = ideal_peaks[:, 1]
    ideal_data["h"] = M_ideal[:, 0]
    ideal_data["k"] = M_ideal[:, 1]

    ideal_lattice = PointList(data=ideal_data)

    # shift to the DP center
    ideal_lattice.data["qx"] += x0
    ideal_lattice.data["qy"] += y0

    # trim peaks outside the image
    deletePeaks = (
        (ideal_lattice.data["qx"] > Q_Nx)
        | (ideal_lattice.data["qx"] < 0)
        | (ideal_lattice.data["qy"] > Q_Ny)
        | (ideal_lattice.data["qy"] < 0)
    )
    ideal_lattice.remove(deletePeaks)

    return ideal_lattice




def bragg_vector_intensity_map_by_index(braggpeaks, h, k, symmetric=False):
    """
    Returns a correlation intensity map for an indexed (h,k) Bragg vector
    Used to obtain a darkfield image corresponding to the (h,k) reflection
    or a bightfield image when h=k=0

    Args:
        braggpeaks (PointListArray): must contain the coordinates 'h','k', and
            'intensity'
        h, k (int): indices for the reflection to generate an intensity map from
        symmetric (bool): if set to true, returns sum of intensity of (h,k), (-h,k),
            (h,-k), (-h,-k)

    Returns:
        (numpy array): a map of the intensity of the (h,k) Bragg vector correlation.
        Same shape as the pointlistarray.
    """
    assert isinstance(braggpeaks, PointListArray), "braggpeaks must be a PointListArray"
    assert np.all([name in braggpeaks.dtype.names for name in ("h", "k", "intensity")])
    intensity_map = np.zeros(braggpeaks.shape, dtype=float)

    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            pl = braggpeaks.get_pointlist(Rx, Ry)
            if pl.length > 0:
                if symmetric:
                    matches = np.logical_and(
                        np.abs(pl.data["h"]) == np.abs(h),
                        np.abs(pl.data["k"]) == np.abs(k),
                    )
                else:
                    matches = np.logical_and(pl.data["h"] == h, pl.data["k"] == k)

                if len(matches) > 0:
                    intensity_map[Rx, Ry] = np.sum(pl.data["intensity"][matches])

    return intensity_map
