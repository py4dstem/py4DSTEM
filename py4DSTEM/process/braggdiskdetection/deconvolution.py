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
                qx0,qx1 = int(np.floor(qx)),int(np.ceil(qx))
                qy0,qy1 = int(np.floor(qy)),int(np.ceil(qy))
                if (qx0>=0) and (qy0>0) and (qx1<Q_Nx) and (qy1<Q_Ny):
                    I = pointlist.data['intensity'][i]
                    dqx = qx-qx0
                    dqy = qy-qy0
                    deconvolution[qx0,qy0] += (1-dqx)*(1-dqy)*I
                    deconvolution[qx0,qy1] += (1-dqx)*dqy*I
                    deconvolution[qx1,qy0] += dqx*(1-dqy)*I
                    deconvolution[qx1,qy1] += dqx*dqy*I
    return deconvolution


