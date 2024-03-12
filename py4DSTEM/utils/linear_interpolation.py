import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np



def linear_interpolation_1D(ar, x):
    """
    Calculates the 1D linear interpolation of array ar at position x using the two
    nearest elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    dx = x - x0
    return (1 - dx) * ar[x0] + dx * ar[x1]


def linear_interpolation_2D(ar, x, y):
    """
    Calculates the 2D linear interpolation of array ar at position x,y using the four
    nearest array elements.
    """
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    dx = x - x0
    dy = y - y0
    return (
        (1 - dx) * (1 - dy) * ar[x0, y0]
        + (1 - dx) * dy * ar[x0, y1]
        + dx * (1 - dy) * ar[x1, y0]
        + dx * dy * ar[x1, y1]
    )


def add_to_2D_array_from_floats(ar, x, y, I):
    """
    Adds the values I to array ar, distributing the value between the four pixels nearest
    (x,y) using linear interpolation.  Inputs (x,y,I) may be floats or arrays of floats.

    Note that if the same [x,y] coordinate appears more than once in the input array,
    only the *final* value of I at that coordinate will get added.
    """
    Nx, Ny = ar.shape
    x0, x1 = (np.floor(x)).astype(int), (np.ceil(x)).astype(int)
    y0, y1 = (np.floor(y)).astype(int), (np.ceil(y)).astype(int)
    mask = np.logical_and(
        np.logical_and(np.logical_and((x0 >= 0), (y0 >= 0)), (x1 < Nx)), (y1 < Ny)
    )
    dx = x - x0
    dy = y - y0
    ar[x0[mask], y0[mask]] += (1 - dx[mask]) * (1 - dy[mask]) * I[mask]
    ar[x0[mask], y1[mask]] += (1 - dx[mask]) * (dy[mask]) * I[mask]
    ar[x1[mask], y0[mask]] += (dx[mask]) * (1 - dy[mask]) * I[mask]
    ar[x1[mask], y1[mask]] += (dx[mask]) * (dy[mask]) * I[mask]
    return ar


