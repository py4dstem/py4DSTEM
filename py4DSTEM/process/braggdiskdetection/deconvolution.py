# Functions for calculating and making use of the average deconvolution

import numpy as np
from ..utils import add_to_2D_array_from_floats

def get_deconvolution(pointlistarray, Q_Nx, Q_Ny):
    """
    Calculates the average deconvolution from a PointListArray of Bragg peak positions.

    Accepts:
        pointlistarray      (PointListArray)
        Q_Nx,Q_Ny           (ints)

    Returns:
        deconvolution       (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    #assert np.all([name in pointlistarray.dtype.names for name in ['qx','qy','intensity']), "pointlistarray coords must include coordinates: 'qx', 'qy', 'intensity'."
    #assert 'qx' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    #assert 'qy' in pointlistarray.dtype.names, "pointlistarray coords must include 'qx' and 'qy'"
    #assert 'intensity' in pointlistarray.dtype.names, "pointlistarray coords must include 'intensity'"
    deconvolution = np.zeros((Q_Nx,Q_Ny))
    for Rx in range(pointlistarray.shape[0]):
        for Ry in range(pointlistarray.shape[1]):
            pointlist = pointlistarray.get_pointlist(Rx,Ry)
            for i in range(pointlist.length):
                qx = pointlist.data['qx'][i]
                qy = pointlist.data['qy'][i]
                I = pointlist.data['intensity'][i]
                deconvolution = add_to_2D_array_from_floats(deconvolution,qx,qy,I)
    return deconvolution


