# Functions for calculating and making use of the Bragg vector map

import numpy as np
from ..utils import add_to_2D_array_from_floats, tqdmnd

def get_bragg_vector_map(pointlistarray, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions.

    Accepts:
        pointlistarray      (PointListArray)
        Q_Nx,Q_Ny           (ints)

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']]), "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        for i in range(pointlist.length):
            qx = pointlist.data['qx'][i]
            qy = pointlist.data['qy'][i]
            I = pointlist.data['intensity'][i]
            braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I)
    return braggvectormap

def get_bragg_vector_maxima_map(pointlistarray, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector maxima map from a PointListArray of Bragg peak positions.

    Accepts:
        pointlistarray      (PointListArray)
        Q_Nx,Q_Ny           (ints)

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']]), "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)
        for i in range(pointlist.length):
            qx = int(np.round(pointlist.data['qx'][i]))
            qy = int(np.round(pointlist.data['qy'][i]))
            I = pointlist.data['intensity'][i]
            braggvectormap[qx,qy] = max(I,braggvectormap[qx,qy])
    return braggvectormap

def get_weighted_bragg_vector_map(pointlistarray, Q_Nx, Q_Ny, weights):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, weighting the
    Bragg peaks at each scan position according to the array weights.

    Accepts:
        pointlistarray      (PointListArray)
        Q_Nx,Q_Ny           (ints)
        weights             (2D array) The shape of weights must be (R_Nx,R_Ny)

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']]), "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"
    assert weights.shape == pointlistarray.shape, "weights must have shape (R_Nx,R_Ny)"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
        if weights[Rx,Ry] != 0:
            pointlist = pointlistarray.get_pointlist(Rx,Ry)
            for i in range(pointlist.length):
                qx = pointlist.data['qx'][i]
                qy = pointlist.data['qy'][i]
                I = pointlist.data['intensity'][i]
                braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I*weights[Rx,Ry])
    return braggvectormap




