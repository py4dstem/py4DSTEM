# Functions for fitting lattice vectors to measured Bragg peak positions

import numpy as np
from numpy.linalg import lstsq

from emdfile import tqdmnd, PointList, PointListArray
from py4DSTEM.data import RealSlice


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

            * ``g1g2_map.get_slice('x0')``     x-coord of the origin of the best fit lattice
            * ``g1g2_map.get_slice('y0')``     y-coord of the origin
            * ``g1g2_map.get_slice('g1x')``    x-coord of the first lattice vector
            * ``g1g2_map.get_slice('g1y')``    y-coord of the first lattice vector
            * ``g1g2_map.get_slice('g2x')``    x-coord of the second lattice vector
            * ``g1g2_map.get_slice('g2y')``    y-coord of the second lattice vector
            * ``g1g2_map.get_slice('error')``  the fit error
            * ``g1g2_map.get_slice('mask')``   1 for successful fits, 0 for unsuccessful
              fits
    """
    assert isinstance(braggpeaks, PointListArray)
    assert np.all(
        [name in braggpeaks.dtype.names for name in ("qx", "qy", "intensity")]
    )

    # Make RealSlice to contain outputs
    slicelabels = ("x0", "y0", "g1x", "g1y", "g2x", "g2y", "error", "mask")
    g1g2_map = RealSlice(
        data=np.zeros((braggpeaks.shape[0], braggpeaks.shape[1], 8)),
        slicelabels=slicelabels,
        name="g1g2_map",
    )

    # Fit lattice vectors
    for Rx, Ry in tqdmnd(braggpeaks.shape[0], braggpeaks.shape[1]):
        if mask[Rx, Ry]:
            braggpeaks_curr = braggpeaks.get_pointlist(Rx, Ry)
            qx0, qy0, g1x, g1y, g2x, g2y, error = fit_lattice_vectors(
                braggpeaks_curr, x0, y0, minNumPeaks
            )
            # Store data
            if g1x is not None:
                g1g2_map.get_slice("x0").data[Rx, Ry] = qx0
                g1g2_map.get_slice("y0").data[Rx, Ry] = qx0
                g1g2_map.get_slice("g1x").data[Rx, Ry] = g1x
                g1g2_map.get_slice("g1y").data[Rx, Ry] = g1y
                g1g2_map.get_slice("g2x").data[Rx, Ry] = g2x
                g1g2_map.get_slice("g2y").data[Rx, Ry] = g2y
                g1g2_map.get_slice("error").data[Rx, Ry] = error
                g1g2_map.get_slice("mask").data[Rx, Ry] = 1
    return g1g2_map
