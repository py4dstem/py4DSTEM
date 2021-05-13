# Functions for calculating and making use of the Bragg vector map

import numpy as np
from ..utils import add_to_2D_array_from_floats, tqdmnd

def get_bragg_vector_map(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, given
    braggpeak positions which have been centered about the origin. In the returned array
    braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    qx0,qy0 = Q_Nx/2.,Q_Ny/2.
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        peaks = braggpeaks.get_pointlist(Rx,Ry)
        for i in range(peaks.length):
            qx = peaks.data['qx'][i]+qx0
            qy = peaks.data['qy'][i]+qy0
            I = peaks.data['intensity'][i]
            braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I)
    return braggvectormap

def get_bragg_vector_maxima_map(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector maxima map from a PointListArray of Bragg peak positions,
    given braggpeak positions which have been centered about the origin. In the returned
    array braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    qx0,qy0 = Q_Nx/2.,Q_Ny/2.
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        peaks = braggpeaks.get_pointlist(Rx,Ry)
        for i in range(peaks.length):
            qx = int(np.round(peaks.data['qx'][i]))+qx0
            qy = int(np.round(peaks.data['qy'][i]))+qy0
            I = peaks.data['intensity'][i]
            braggvectormap[qx,qy] = max(I,braggvectormap[qx,qy])
    return braggvectormap

def get_weighted_bragg_vector_map(braggpeaks, Q_Nx, Q_Ny, weights):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, given
    bragg peak positions which have been centered about the origin, weighting the peaks
    at each scan position according to the array weights. In the returned array
    braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels
        weights             (2D array) The shape of weights must be (R_Nx,R_Ny)

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert weights.shape == braggpeaks.shape, "weights must have shape (R_Nx,R_Ny)"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    qx0,qy0 = Q_Nx/2.,Q_Ny/2.
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        if weights[Rx,Ry] != 0:
            peaks = braggpeaks.get_pointlist(Rx,Ry)
            for i in range(peaks.length):
                qx = peaks.data['qx'][i]+qx0
                qy = peaks.data['qy'][i]+qy0
                I = peaks.data['intensity'][i]
                braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I*weights[Rx,Ry])
    return braggvectormap


# Functions for getting bragg vector maps from raw / uncentered braggpeak data

def get_bragg_vector_map_raw(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, where
    the peak positions have not been centered.

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        peaks = braggpeaks.get_pointlist(Rx,Ry)
        for i in range(peaks.length):
            qx = peaks.data['qx'][i]
            qy = peaks.data['qy'][i]
            I = peaks.data['intensity'][i]
            braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I)
    return braggvectormap

def get_bragg_vector_maxima_map_raw(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector maxima map from a PointListArray of Bragg peak positions,
    where the peak positions have not been centered.

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        peaks = braggpeaks.get_pointlist(Rx,Ry)
        for i in range(peaks.length):
            qx = int(np.round(peaks.data['qx'][i]))
            qy = int(np.round(peaks.data['qy'][i]))
            I = peaks.data['intensity'][i]
            braggvectormap[qx,qy] = max(I,braggvectormap[qx,qy])
    return braggvectormap

def get_weighted_bragg_vector_map_raw(braggpeaks, Q_Nx, Q_Ny, weights):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, where
    the peak positions have not been centered, and weighting the peaks at each scan position
    according to the array weights.

    Accepts:
        braggpeaks          (PointListArray) Must have the coords 'qx','qy','intensity',
                            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny           (ints) the size of diffraction space in pixels
        weights             (2D array) The shape of weights must be (R_Nx,R_Ny)

    Returns:
        braggvectormap      (2D ndarray, shape (Q_Nx,Q_Ny))
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert weights.shape == braggpeaks.shape, "weights must have shape (R_Nx,R_Ny)"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1]):
        if weights[Rx,Ry] != 0:
            peaks = braggpeaks.get_pointlist(Rx,Ry)
            for i in range(peaks.length):
                qx = peaks.data['qx'][i]
                qy = peaks.data['qy'][i]
                I = peaks.data['intensity'][i]
                braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I*weights[Rx,Ry])
    return braggvectormap


# Aliases
get_bvm = get_bragg_vector_map
get_bvm_maxima = get_bragg_vector_maxima_map
get_bvm_weighted = get_weighted_bragg_vector_map

get_bvm_raw = get_bragg_vector_map_raw
get_bvm_maxima_raw = get_bragg_vector_maxima_map_raw
get_bvm_weighted_raw = get_weighted_bragg_vector_map_raw


