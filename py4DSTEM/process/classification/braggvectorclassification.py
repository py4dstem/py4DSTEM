# Functions for classification using a method which initially classifies the detected bragg
# vectors, then uses these labels to classify real space scan positions

import numpy as np
from ...file.datastructure import PointListArray

def get_braggpeak_labels_by_scan_position(braggpeaks, Qx, Qy):
    """
    For each scan position, gets a set of integers, specifying the bragg peaks at this scan
    position.

    From a set of positions in diffraction space (Qx,Qy), assign each detected bragg peak in the
    PointListArray braggpeaks a label corresponding to the index of the closest position; thus
    for a bragg peak at (qx,qy), if the closest position in (Qx,Qy) is (Qx[i],Qy[i]), assign
    this peak the label i. This is equivalent to assigning each bragg peak (qx,qy) a label
    according to the Voronoi region it lives in, given a voronoi tesselation seeded from the
    points (Qx,Qy).

    For each scan position, get the set of all indices i for all bragg peaks found at this scan
    position.

    Accepts:
        braggpeaks          (PointListArray) Bragg peaks; must have coords 'qx' and 'qy'
        Qx                  (ndarray of floats) x-coords of the voronoi points
        Qy                  (ndarray of floats) y-coords of the voronoi points

    Returns:
        braggpeak_labels    (list of lists of sets) the labels found at each scan position.
                            Scan position (Rx,Ry) is accessed via braggpeak_labels[Rx][Ry]
    """
    assert isinstance(braggpeaks,PointListArray), "braggpeaks must be a PointListArray"
    assert np.all([name in braggpeaks.dtype.names for name in ('qx','qy')]), "braggpeaks must contain coords 'qx' and 'qy'"

    braggpeak_labels = [[set() for i in range(braggpeaks.shape[1])] for j in range(braggpeaks.shape[0])]
    for Rx in range(braggpeaks.shape[0]):
        for Ry in range(braggpeaks.shape[1]):
            s = braggpeak_labels[Rx][Ry]
            pointlist = braggpeaks.get_pointlist(Rx,Ry)
            for i in range(pointlist.length):
                label = np.argmin(np.hypot(Qx-pointlist.data['qx'][i],Qy-pointlist.data['qy'][i]))
                s.add(label)

    return braggpeak_labels






