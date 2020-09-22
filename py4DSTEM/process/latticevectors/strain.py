# Functions for calculating strain from lattice vector maps

import numpy as np
from numpy.linalg import lstsq

from ...io.datastructure import RealSlice

def get_reference_uv(mask, uv_map):
    """
    Gets a pair of reference lattice vectors from a region of real space specified by mask.
    Takes the median of the lattice vectors in uv_map within the specified region.

    Accepts:
        mask        (ndarray of bools) use lattice vectors from uv_map scan positions wherever
                    mask==True
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data under
                    the keys 'ux', 'uy', 'vx', and 'vy'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

    Returns:
        ux          (float) x-coord of the first reference lattice vector
        uy          (float) y-coord of the first reference lattice vector
        vx          (float) x-coord of the second reference lattice vector
        vy          (float) y-coord of the second reference lattice vector
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.slices.keys() for name in ('ux','uy','vx','vy')])
    assert mask.dtype == bool
    ux = np.median(uv_map.slices['ux'][mask])
    uy = np.median(uv_map.slices['uy'][mask])
    vx = np.median(uv_map.slices['vx'][mask])
    vy = np.median(uv_map.slices['vy'][mask])
    return ux,uy,vx,vy

def get_strain_from_reference_uv(ux, uy, vx, vy, uv_map):
    """
    Gets a strain map from the reference lattice vectors (u,v) and lattice vector map uv_map.

    Accepts:
        ux          (float) x-coord of the first reference lattice vector
        uy          (float) y-coord of the first reference lattice vector
        vx          (float) x-coord of the second reference lattice vector
        vy          (float) y-coord of the second reference lattice vector
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data under
                    the keys 'ux', 'uy', 'vx', 'vy', 'mask'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

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
                                    Note 2: x and y are here the coordinate axes of diffraction
                                    space, where the lattice vectors were measured.  Calibration
                                    of their rotation with respect to real space may be necessary.
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.slices.keys() for name in ('ux','uy','vx','vy','mask')])

    # Get RealSlice for output storage
    R_Nx,R_Ny = uv_map.slices['ux'].shape
    strain_map = RealSlice(data=np.zeros((R_Nx,R_Ny,5)),
                           slicelabels=('e_xx','e_yy','e_xy','theta','mask'),
                           name='strain_map')

    # Get reference lattice matrix
    M = np.array([[ux,uy],[vx,vy]])

    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            # Get lattice vectors for DP at Rx,Ry
            alpha = np.array([[uv_map.slices['ux'][Rx,Ry],uv_map.slices['uy'][Rx,Ry]],
                              [uv_map.slices['vx'][Rx,Ry],uv_map.slices['vy'][Rx,Ry]]])
            # Get transformation matrix
            beta = lstsq(M, alpha, rcond=None)[0].T

            # Get the infinitesimal strain matrix
            strain_map.slices['e_xx'][Rx,Ry] = 1 - beta[0,0]
            strain_map.slices['e_yy'][Rx,Ry] = 1 - beta[1,1]
            strain_map.slices['e_xy'][Rx,Ry] = -(beta[0,1]+beta[1,0])/2.
            strain_map.slices['theta'][Rx,Ry] =  (beta[0,1]-beta[1,0])/2.
            strain_map.slices['mask'][Rx,Ry] = uv_map.slices['mask'][Rx,Ry]
    return strain_map

def get_strain_from_reference_region(mask, uv_map):
    """
    Gets a strain map from the reference region of real space specified by mask and the lattice
    vector map uv_map.

    Accepts:
        mask        (ndarray of bools) use lattice vectors from uv_map scan positions wherever
                    mask==True
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data under
                    the keys 'ux', 'uy', 'vx', 'vy', 'mask'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

    Returns:
        strain_map                  (RealSlice) the strain map; contains the elements of the
                                    infinitessimal strain matrix, in the following 4 arrays:
        strain_map.slices['e_xx']   change in lattice x-components with respect to x
        strain_map.slices['e_yy']   change in lattice y-components with respect to y
        strain_map.slices['e_xy']   change in lattice x-components with respect to y
        strain_map.slices['theta']  rotation of lattice with respect to reference
        strain_map.slices['mask']   0/False indicates unknown values
                                    Note 1: the strain matrix has been symmetrized, so e_xy and
                                    e_yx are identical
                                    Note 2: x and y are here the coordinate axes of diffraction
                                    space, where the lattice vectors were measured.  Calibration
                                    of their rotation with respect to real space may be necessary.
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.slices.keys() for name in ('ux','uy','vx','vy','mask')])
    assert mask.dtype == bool

    ux,uy,vx,vy = get_reference_uv(mask, uv_map)
    strain_map = get_strain_from_reference_uv(ux,uy,vx,vy,uv_map)
    return strain_map

def get_rotated_strain_map(unrotated_strain_map, ux, uy):
    """
    Starting from a strain map defined with respect to the xy coordinate system of diffraction space,
    i.e. where exx and eyy are the compression/tension along the Qx and Qy directions, respectively,
    get a strain map defined with respect to a right-handed uv coordinate system, with the u-axis
    oriented along u=(ux,uy).

    Accepts:
        ux                      (float) diffraction space x coordinate of u
        uy                      (float) diffraction space y coordinate of u
        unrotated_strain_map    (RealSlice) a RealSlice object containing 2D arrays of the
                                infinitessimal strain matrix elements, stored at
                                        unrotated_strain_map.slices['e_xx']
                                        unrotated_strain_map.slices['e_xy']
                                        unrotated_strain_map.slices['e_yy']
                                        unrotated_strain_map.slices['theta']

    Returns:
        rotated_strain_map      (RealSlice) the rotated counterpart to unrotated_strain_map, with
                                the rotated_strain_map.slices['e_xx'] element oriented along (ux,uy)
    """
    assert isinstance(unrotated_strain_map, RealSlice)
    assert np.all([key in ['e_xx','e_xy','e_yy','theta','mask'] for key in unrotated_strain_map.slices.keys()])

    theta = -np.arctan2(uy,ux)
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



