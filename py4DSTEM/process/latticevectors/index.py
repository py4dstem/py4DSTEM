# Functions for indexing the Bragg directions

import numpy as np
from numpy.linalg import lstsq

from ...file.datastructure import PointList

def index_bragg_directions(x0, y0, ux, uy, vx, vy, bragg_x, bragg_y):
    """
    From an origin (x0,y0), an pair of lattice vectors (ux,uy), (vx,vy), and a set of measured
    bragg directions (bragg_x,bragg_y), find the indices (h,k) of all the bragg directions.

    The approach is to solve the matrix equation
            alpha = beta * M
    where alpha is the 2xN array of the (x,y) coordinates of N measured bragg directions, beta is the
    2x2 array of the two lattice vectors u,v, and M is the 2xN array of the h,k indices.

    Accepts:
        x0                  (float) x-coord of origin
        y0                  (float) y-coord of origin
        ux                  (float) x-coord of first lattice vector
        uy                  (float) y-coord of first lattice vector
        vx                  (float) x-coord of second lattice vector
        vy                  (float) y-coord of second lattice vector
        bragg_x             (ndarray of floats) x-coords of bragg directions
        bragg_y             (ndarray of floats) y-coords of bragg directions

    Returns:
        h                   (ndarray of ints) first index of the bragg directions
        k                   (ndarray of ints) second index of the bragg directions
        bragg_directions    (PointList) a 4-coordinate PointList with the indexed bragg directions;
                            coords 'qx' and 'qy' contain bragg_x and bragg_y
                            coords 'h' and 'k' contain h and k.
    """
    # Get beta, the matrix of lattice vectors
    beta = np.array([[ux,vx],[uy,vy]])

    # Get alpha, the matrix of measured bragg angles
    alpha = np.vstack([bragg_x-x0,bragg_y-y0])

    # Calculate M, the matrix of peak positions
    M = lstsq(beta, alpha, rcond=None)[0].T
    M = np.round(M).astype(int)

    # Get h,k
    h = M[:,0]
    k = M[:,1]

    # Store in a PointList
    coords = [('qx',float),('qy',float),('h',int),('k',int)]
    bragg_directions = PointList(coordinates=coords)
    bragg_directions.add_tuple_of_nparrays((bragg_x,bragg_y,h,k))

    return h,k, bragg_directions

