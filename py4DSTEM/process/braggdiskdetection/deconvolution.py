# Functions for calculating and making use of the average deconvolution

import numpy as np

def get_deconvolution(pointlistarray, Q_Nx, Q_Ny):
    """
    Calculates the average deconvolution from a PointListArray of Bragg peak positions.

    Accepts:
        pointlistarray      (PointListArray)
        Q_Nx,Q_Ny           (ints)

    Returns:
        deconvolution       (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"
    deconvolution = np.zeros((Q_Nx,Q_Ny))
    for Rx in range(pointlistarray.shape[0]):
        for Ry in range(pointlistarray.shape[1]):
            pointlist = pointlistarray.get_pointlist(Rx,Ry)
            qx = np.round(pointlist.data['qx']).astype(int)
            qy = np.round(pointlist.data['qy']).astype(int)
            deconvolution[qx,qy] += pointlist.data['intensity']
    return deconvolution


