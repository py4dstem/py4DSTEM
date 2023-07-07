import numpy as np
from emdfile import tqdmnd






## TODO


def make_bragg_mask(
    Qshape,
    g1,
    g2,
    radius,
    origin,
    max_q,
    return_sum = True,
    **kwargs,
    ):
    '''
    Creates and returns a mask consisting of circular disks
    about the points of a 2D lattice.

    Args:
        Qshape (2 tuple): the shape of diffraction space
        g1,g2 (len 2 array or tuple): the lattice vectors
        radius (number): the disk radius
        origin (len 2 array or tuple): the origin
        max_q (nuumber): the maxima distance to tile to
        return_sum (bool): if False, return a 3D array, where each
            slice contains a single disk; if False, return a single
            2D masks of all disks

    Returns:
        (2 or 3D array) the mask
    '''
    nas = np.asarray
    g1,g2,origin = nas(g1),nas(g2),nas(origin)

    # Get N,M, the maximum indices to tile out to
    L1 = np.sqrt(np.sum(g1**2))
    H = int(max_q/L1) + 1
    L2 = np.hypot(-g2[0]*g1[1],g2[1]*g1[0])/np.sqrt(np.sum(g1**2))
    K = int(max_q/L2) + 1

    # Compute number of points
    N = 0
    for h in range(-H,H+1):
        for k in range(-K,K+1):
            v = h*g1 + k*g2
            if np.sqrt(v.dot(v)) < max_q:
                N += 1

    #create mask
    mask = np.zeros((Qshape[0], Qshape[1], N), dtype=bool)
    N = 0
    for h in range(-H,H+1):
        for k in range(-K,K+1):
            v = h*g1 + k*g2
            if np.sqrt(v.dot(v)) < max_q:
                center = origin + v
                mask[:,:,N] = make_detector(
                    Qshape,
                    mode = 'circle',
                    geometry = (center, radius),
                )
                N += 1


    if return_sum:
        mask = np.sum(mask, axis = 2)
    return mask



