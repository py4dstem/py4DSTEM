# Functions for calculating strain from lattice vector maps

import numpy as np
from numpy.linalg import lstsq

from ...io.datastructure import RealSlice

def get_reference_g1g2(g1g2_map, mask):
    """
    Gets a pair of reference lattice vectors from a region of real space specified by mask.
    Takes the median of the lattice vectors in g1g2_map within the specified region.

    Accepts:
        g1g2_map    (RealSlice) the lattice vector map; contains 2D arrays in g1g2_map.data
                    under the keys 'g1x', 'g1y', 'g2x', and 'g2y'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.
        mask        (ndarray of bools) use lattice vectors from g1g2_map scan positions wherever
                    mask==True

    Returns:
        g1          (2-tuple) first reference lattice vector (x,y)
        g2          (2-tuple) second reference lattice vector (x,y)
    """
    assert isinstance(g1g2_map, RealSlice)
    assert np.all([name in g1g2_map.slices.keys() for name in ('g1x','g1y','g2x','g2y')])
    assert mask.dtype == bool
    g1x = np.median(g1g2_map.slices['g1x'][mask])
    g1y = np.median(g1g2_map.slices['g1y'][mask])
    g2x = np.median(g1g2_map.slices['g2x'][mask])
    g2y = np.median(g1g2_map.slices['g2y'][mask])
    return (g1x,g1y),(g2x,g2y)

def get_strain_from_reference_g1g2(g1g2_map, g1, g2):
    """
    Gets a strain map from the reference lattice vectors g1,g2 and lattice vector map g1g2_map.

    Note that this function will return the strain map oriented with respect to the x/y axes of
    diffraction space - to rotate the coordinate system, use get_rotated_strain_map().
    Calibration of the rotational misalignment between real and diffraction space may also be
    necessary.

    Accepts:
        g1g2_map    (RealSlice) the lattice vector map; contains 2D arrays in g1g2_map.data
                    under the keys 'g1x', 'g1y', 'g2x', and 'g2y'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.
        g1          (2-tuple) first reference lattice vector (x,y)
        g2          (2-tuple) second reference lattice vector (x,y)

    Returns:
        strain_map                  (RealSlice) the strain map; contains the elements of the
                                    infinitessimal strain matrix, in the following 5 arrays:
        strain_map.slices['e_xx']   change in lattice x-components with respect to x
        strain_map.slices['e_yy']   change in lattice y-components with respect to y
        strain_map.slices['e_xy']   change in lattice x-components with respect to y
        strain_map.slices['theta']  rotation of lattice with respect to reference
        strain_map.slices['mask']   0/False indicates unknown values
                                    Note 1: the strain matrix has been symmetrized, so e_xy and
                                    e_yx are identical
    """
    assert isinstance(g1g2_map, RealSlice)
    assert np.all([name in g1g2_map.slices.keys() for name in ('g1x','g1y','g2x','g2y','mask')])

    # Get RealSlice for output storage
    R_Nx,R_Ny = g1g2_map.slices['g1x'].shape
    strain_map = RealSlice(data=np.zeros((R_Nx,R_Ny,5)),
                           slicelabels=('e_xx','e_yy','e_xy','theta','mask'),
                           name='strain_map')

    # Get reference lattice matrix
    g1x,g1y = g1
    g2x,g2y = g2
    M = np.array([[g1x,g1y],[g2x,g2y]])

    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            # Get lattice vectors for DP at Rx,Ry
            alpha = np.array([[g1g2_map.slices['g1x'][Rx,Ry],g1g2_map.slices['g1y'][Rx,Ry]],
                              [g1g2_map.slices['g2x'][Rx,Ry],g1g2_map.slices['g2y'][Rx,Ry]]])
            # Get transformation matrix
            beta = lstsq(M, alpha, rcond=None)[0].T

            # Get the infinitesimal strain matrix
            strain_map.slices['e_xx'][Rx,Ry] = 1 - beta[0,0]
            strain_map.slices['e_yy'][Rx,Ry] = 1 - beta[1,1]
            strain_map.slices['e_xy'][Rx,Ry] = -(beta[0,1]+beta[1,0])/2.
            strain_map.slices['theta'][Rx,Ry] =  (beta[0,1]-beta[1,0])/2.
            strain_map.slices['mask'][Rx,Ry] = g1g2_map.slices['mask'][Rx,Ry]
    return strain_map

def get_strain_from_reference_region(g1g2_map, mask):
    """
    Gets a strain map from the reference region of real space specified by mask and the lattice
    vector map g1g2_map.

    Note that this function will return the strain map oriented with respect to the x/y axes of
    diffraction space - to rotate the coordinate system, use get_rotated_strain_map().
    Calibration of the rotational misalignment between real and diffraction space may also be
    necessary.

    Accepts:
        g1g2_map    (RealSlice) the lattice vector map; contains 2D arrays in g1g2_map.data
                    under the keys 'g1x', 'g1y', 'g2x', and 'g2y'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.
        mask        (ndarray of bools) use lattice vectors from g1g2_map scan positions wherever
                    mask==True

    Returns:
        strain_map                  (RealSlice) the strain map; contains the elements of the
                                    infinitessimal strain matrix, in the following 5 arrays:
        strain_map.slices['e_xx']   change in lattice x-components with respect to x
        strain_map.slices['e_yy']   change in lattice y-components with respect to y
        strain_map.slices['e_xy']   change in lattice x-components with respect to y
        strain_map.slices['theta']  rotation of lattice with respect to reference
        strain_map.slices['mask']   0/False indicates unknown values
                                    Note 1: the strain matrix has been symmetrized, so e_xy and
                                    e_yx are identical
    """
    assert isinstance(g1g2_map, RealSlice)
    assert np.all([name in g1g2_map.slices.keys() for name in ('g1x','g1y','g2x','g2y','mask')])
    assert mask.dtype == bool

    g1,g2 = get_reference_g1g2(g1g2_map,mask)
    strain_map = get_strain_from_reference_g1g2(g1g2_map,g1,g2)
    return strain_map

def get_rotated_strain_map(unrotated_strain_map, xaxis_x, xaxis_y):
    """
    Starting from a strain map defined with respect to the xy coordinate system of diffraction space,
    i.e. where exx and eyy are the compression/tension along the Qx and Qy directions, respectively,
    get a strain map defined with respect to some other right-handed coordinate system, in which the
    x-axis is oriented along (xaxis_x,xaxis_y).

    Accepts:
        xaxis_x,xaxis_y         (float) diffraction space (x,y) coordinates of a vector
                                along the new x-axis
        unrotated_strain_map    (RealSlice) a RealSlice object containing 2D arrays of the
                                infinitessimal strain matrix elements, stored at
                                        unrotated_strain_map.slices['e_xx']
                                        unrotated_strain_map.slices['e_xy']
                                        unrotated_strain_map.slices['e_yy']
                                        unrotated_strain_map.slices['theta']

    Returns:
        rotated_strain_map      (RealSlice) the rotated counterpart to unrotated_strain_map, with
                                the rotated_strain_map.slices['e_xx'] element oriented along the
                                new coordinate system
    """
    assert isinstance(unrotated_strain_map, RealSlice)
    assert np.all([key in ['e_xx','e_xy','e_yy','theta','mask'] for key in unrotated_strain_map.slices.keys()])

    theta = -np.arctan2(xaxis_y,xaxis_x)
    cost = np.cos(theta)
    sint = np.sin(theta)
    cost2 = cost**2
    sint2 = sint**2

    Rx,Ry = unrotated_strain_map.slices['e_xx'].shape
    rotated_strain_map = RealSlice(data=np.zeros((Rx,Ry,5)),
                                   slicelabels=['e_xx','e_xy','e_yy','theta','mask'],
                                   name=unrotated_strain_map.name+"_rotated".format(np.degrees(theta)))

    rotated_strain_map.data[:,:,0] = cost2*unrotated_strain_map.slices['e_xx'] - 2*cost*sint*unrotated_strain_map.slices['e_xy'] + sint2*unrotated_strain_map.slices['e_yy']
    rotated_strain_map.data[:,:,1] = cost*sint*(unrotated_strain_map.slices['e_xx']-unrotated_strain_map.slices['e_yy']) + (cost2-sint2)*unrotated_strain_map.slices['e_xy']
    rotated_strain_map.data[:,:,2] = sint2*unrotated_strain_map.slices['e_xx'] + 2*cost*sint*unrotated_strain_map.slices['e_xy'] + cost2*unrotated_strain_map.slices['e_yy']
    rotated_strain_map.data[:,:,3] = unrotated_strain_map.slices['theta']

    rotated_strain_map.data[:,:,4] = unrotated_strain_map.slices['mask']
    return rotated_strain_map



