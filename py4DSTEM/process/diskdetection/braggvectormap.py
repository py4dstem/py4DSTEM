# Functions for calculating and making use of the Bragg vector map

import numpy as np
from py4DSTEM.process.utils import add_to_2D_array_from_floats
from py4DSTEM.io.datastructure.emd import PointListArray
from py4DSTEM.utils.tqdmnd import tqdmnd



def get_bvm(
    braggpeaks,
    Qshape,
    mode = 'centered',
    Q_pixel_size = 1.0,
    ):
    """
    Gets a Bragg vector map, a 2D histogram of Bragg scattering vectors.

    Args:
        braggpeaks (PointListArray): the Bragg vectors and intensities.
        Qshape (2 tuple): diffraction space shape
        mode (str): must be 'raw' or 'centered'. TODO, sampling selection

    Args:
        braggpeaks (PointListArray):
    """
    # parse args
    assert isinstance(braggpeaks, PointListArray)
    modes = ('raw','centered')
    assert mode in modes, f"{mode} must be in {modes}"


    # select a function
    fn_dict = {
        'raw' : get_bragg_vector_map_raw,
        'centered' : get_bragg_vector_map,
    }
    fn = fn_dict[mode]


    # run
    bvm = fn(
        braggpeaks,
        Q_Nx = Qshape[0],
        Q_Ny = Qshape[1],
        Q_pixel_size=Q_pixel_size,
    )

    return bvm











def get_bragg_vector_map(braggpeaks, Q_Nx, Q_Ny, Q_pixel_size=1):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, given
    braggpeak positions which have been centered about the origin. In the returned array
    braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity', the
            default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels
        Q_pixel_size (number): the size of the diffraction space p[ixels

    Returns:
        (ndarray): the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    # Concatenate all PointList data together for speed
    b = braggpeaks
    bigpl = np.concatenate([b[i,j].data for i in range(b.shape[0]) for j in range(b.shape[1])])
    qx = bigpl['qx']/Q_pixel_size + (Q_Nx/2.)
    qy = bigpl['qy']/Q_pixel_size + (Q_Ny/2.)
    I = bigpl['intensity']

    # Precompute rounded coordinates
    floorx = np.floor(qx).astype(np.int64)
    ceilx = np.ceil(qx).astype(np.int64)
    floory = np.floor(qy).astype(np.int64)
    ceily = np.ceil(qy).astype(np.int64)

    # Remove any points outside [0, Q_Nx] & [0, Q_Ny]
    mask = np.logical_and.reduce(((floorx>=0),(floory>=0),(ceilx<Q_Nx),(ceily<Q_Ny)))
    qx = qx[mask]
    qy = qy[mask]
    I = I[mask]
    floorx = floorx[mask]
    floory = floory[mask]
    ceilx = ceilx[mask]
    ceily = ceily[mask]

    dx = qx - floorx
    dy = qy - floory

    # Compute indices of the 4 neighbors to (qx,qy)
    # floor x, floor y
    inds00 = np.ravel_multi_index([floorx,floory],(Q_Nx,Q_Ny))
    # floor x, ceil y
    inds01 = np.ravel_multi_index([floorx,ceily],(Q_Nx,Q_Ny))
    # ceil x, floor y
    inds10 = np.ravel_multi_index([ceilx,floory],(Q_Nx,Q_Ny))
    # ceil x, ceil y
    inds11 = np.ravel_multi_index([ceilx,ceily],(Q_Nx,Q_Ny))

    # Compute the BVM by accumulating intensity in each neighbor weighted by linear interpolation
    bvm = (np.bincount(inds00, I * (1.-dx) * (1.-dy), minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds01, I * (1.-dx) * dy, minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds10, I * dx * (1.-dy), minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds11, I * dx * dy, minlength=Q_Nx*Q_Ny)).reshape(Q_Nx,Q_Ny)

    return bvm

def get_bragg_vector_maxima_map(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector maxima map from a PointListArray of Bragg peak positions,
    given braggpeak positions which have been centered about the origin. In the returned
    array braggvectormap, the origin is placed at (Q_Nx/2.,Q_Ny/2.)

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels

    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)) the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    qx0,qy0 = Q_Nx/2.,Q_Ny/2.
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1],
                           desc='Computing Bragg vector map',unit='DP',unit_scale=True):
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

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (int): the size of diffraction space in pixels
        weights (2D array): The shape of weights must be (R_Nx,R_Ny)

    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)): the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert weights.shape == braggpeaks.shape, "weights must have shape (R_Nx,R_Ny)"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    qx0,qy0 = Q_Nx/2.,Q_Ny/2.
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1],
                           desc='Computing Bragg vector map',unit='DP',unit_scale=True):
        if weights[Rx,Ry] != 0:
            peaks = braggpeaks.get_pointlist(Rx,Ry)
            qx = peaks.data['qx']+qx0
            qy = peaks.data['qy']+qy0
            I = peaks.data['intensity']
            add_to_2D_array_from_floats(braggvectormap,qx,qy,I*weights[Rx,Ry])
    return braggvectormap


# Functions for getting bragg vector maps from raw / uncentered braggpeak data

def get_bragg_vector_map_raw(braggpeaks, Q_Nx, Q_Ny, Q_pixel_size=1):
    """
    Calculates the Bragg vector map from a PointListArray of Bragg peak positions, where
    the peak positions have not been centered.

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels
        Q_pixel_size (number): the size of the diffraction space p[ixels

    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)): the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    # Concatenate all PointList data together for speeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed
    b = braggpeaks
    bigpl = np.concatenate([b[i,j].data for i in range(b.shape[0]) for j in range(b.shape[1])])
    qx = bigpl['qx']/Q_pixel_size
    qy = bigpl['qy']/Q_pixel_size
    I = bigpl['intensity']

    # Precompute rounded coordinates
    floorx = np.floor(qx).astype(np.int64)
    ceilx = np.ceil(qx).astype(np.int64)
    floory = np.floor(qy).astype(np.int64)
    ceily = np.ceil(qy).astype(np.int64)

    # Remove any points outside [0, Q_Nx] & [0, Q_Ny]
    mask = np.logical_and.reduce(((floorx>=0),(floory>=0),(ceilx<Q_Nx),(ceily<Q_Ny)))
    qx = qx[mask]
    qy = qy[mask]
    I = I[mask]
    floorx = floorx[mask]
    floory = floory[mask]
    ceilx = ceilx[mask]
    ceily = ceily[mask]

    dx = qx - floorx
    dy = qy - floory

    # Compute indices of the 4 neighbors to (qx,qy)
    # floor x, floor y
    inds00 = np.ravel_multi_index([floorx,floory],(Q_Nx,Q_Ny))
    # floor x, ceil y
    inds01 = np.ravel_multi_index([floorx,ceily],(Q_Nx,Q_Ny))
    # ceil x, floor y
    inds10 = np.ravel_multi_index([ceilx,floory],(Q_Nx,Q_Ny))
    # ceil x, ceil y
    inds11 = np.ravel_multi_index([ceilx,ceily],(Q_Nx,Q_Ny))

    # Compute the BVM by accumulating intensity in each neighbor weighted by linear interpolation
    bvm = (np.bincount(inds00, I * (1.-dx) * (1.-dy), minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds01, I * (1.-dx) * dy, minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds10, I * dx * (1.-dy), minlength=Q_Nx*Q_Ny) + \
            np.bincount(inds11, I * dx * dy, minlength=Q_Nx*Q_Ny)).reshape(Q_Nx,Q_Ny)

    return bvm

def get_bragg_vector_maxima_map_raw(braggpeaks, Q_Nx, Q_Ny):
    """
    Calculates the Bragg vector maxima map from a PointListArray of Bragg peak positions,
    where the peak positions have not been centered.

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels

    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)): the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1],
                           desc='Computing Bragg vector map',unit='DP',unit_scale=True):
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
    the peak positions have not been centered, and weighting the peaks at each scan
    position according to the array weights.

    Args:
        braggpeaks (PointListArray): Must have the coords 'qx','qy','intensity',
            the default coordinates from the bragg peak detection fns
        Q_Nx,Q_Ny (ints): the size of diffraction space in pixels
        weights (2D array): The shape of weights must be (R_Nx,R_Ny)

    Returns:
        (2D ndarray, shape (Q_Nx,Q_Ny)): the bragg vector map
    """
    assert np.all([name in braggpeaks.dtype.names for name in ['qx','qy','intensity']]), "braggpeaks coords must include coordinates: 'qx', 'qy', 'intensity'."
    assert weights.shape == braggpeaks.shape, "weights must have shape (R_Nx,R_Ny)"

    braggvectormap = np.zeros((Q_Nx,Q_Ny))
    for (Rx, Ry) in tqdmnd(braggpeaks.shape[0],braggpeaks.shape[1],
                           desc='Computing Bragg vector map',unit='DP',unit_scale=True):
        if weights[Rx,Ry] != 0:
            peaks = braggpeaks.get_pointlist(Rx,Ry)
            qx = peaks.data['qx']
            qy = peaks.data['qy']
            I = peaks.data['intensity']
            braggvectormap = add_to_2D_array_from_floats(braggvectormap,qx,qy,I*weights[Rx,Ry])
    return braggvectormap


# Aliases
get_bvm_maxima = get_bragg_vector_maxima_map
get_bvm_weighted = get_weighted_bragg_vector_map

get_bvm_raw = get_bragg_vector_map_raw
get_bvm_maxima_raw = get_bragg_vector_maxima_map_raw
get_bvm_weighted_raw = get_weighted_bragg_vector_map_raw


