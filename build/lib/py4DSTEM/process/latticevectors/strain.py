# Functions for calculating strain from lattice vector maps

import numpy as np
from numpy.linalg import lstsq

from ...file.datastructure import RealSlice

def get_reference_uv(mask, uv_map):
    """
    Gets a pair of reference lattice vectors from a region of real space specified by mask.
    Takes the median of the lattice vectors in uv_map within the specified region.

    Accepts:
        mask        (ndarray of bools) use lattice vectors from uv_map scan positions wherever
                    mask==True
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data2D under
                    the keys 'ux', 'uy', 'vx', and 'vy'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

    Returns:
        ux          (float) x-coord of the first reference lattice vector
        uy          (float) y-coord of the first reference lattice vector
        vx          (float) x-coord of the second reference lattice vector
        vy          (float) y-coord of the second reference lattice vector
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.data2D.keys() for name in ('ux','uy','vx','vy')])
    assert mask.dtype == bool
    ux = np.median(uv_map.data2D['ux'][mask])
    uy = np.median(uv_map.data2D['uy'][mask])
    vx = np.median(uv_map.data2D['vx'][mask])
    vy = np.median(uv_map.data2D['vy'][mask])
    return ux,uy,vx,vy

def get_strain_from_reference_uv(ux, uy, vx, vy, uv_map):
    """
    Gets a strain map from the reference lattice vectors (u,v) and lattice vector map uv_map.

    Accepts:
        ux          (float) x-coord of the first reference lattice vector
        uy          (float) y-coord of the first reference lattice vector
        vx          (float) x-coord of the second reference lattice vector
        vy          (float) y-coord of the second reference lattice vector
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data2D under
                    the keys 'ux', 'uy', 'vx', and 'vy'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

    Returns:
        strain_map                  (RealSlice) the strain map; contains the elements of the
                                    infinitessimal strain matrix, in the following 4 arrays:
        strain_map.data2D['e_xx']   change in lattice x-components with respect to x
        strain_map.data2D['e_yy']   change in lattice y-components with respect to y
        strain_map.data2D['e_xy']   change in lattice x-components with respect to y
        strain_map.data2D['theta']  rotation of lattice with respect to reference
                                    Note 1: the strain matrix has been symmetrized, so e_xy and
                                    e_yx are identical
                                    Note 2: x and y are here the coordinate axes of diffraction
                                    space, where the lattice vectors were measured.  Calibration
                                    of their rotation with respect to real space may be necessary.
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.data2D.keys() for name in ('ux','uy','vx','vy')])

    # Get RealSlice for output storage
    R_Nx,R_Ny = uv_map.data2D['ux'].shape
    strain_map = RealSlice(data=np.zeros((R_Nx,R_Ny,4)),
                           slicelabels=('e_xx','e_yy','e_xy','theta'),
                           name='strain_map')

    # Get reference lattice matrix
    M = np.array([[ux,uy],[vx,vy]])

    for Rx in range(R_Nx):
        for Ry in range(R_Ny):
            # Get lattice vectors for DP at Rx,Ry
            alpha = np.array([[uv_map.data2D['ux'][Rx,Ry],uv_map.data2D['uy'][Rx,Ry]],
                              [uv_map.data2D['vx'][Rx,Ry],uv_map.data2D['vy'][Rx,Ry]]])
            # Get transformation matrix
            beta = lstsq(M, alpha, rcond=None)[0].T

            # Get the infinitesimal strain matrix
            strain_map.data2D['e_xx'][Rx,Ry] = 1 - beta[0,0]
            strain_map.data2D['e_yy'][Rx,Ry] = 1 - beta[1,1]
            strain_map.data2D['e_xy'][Rx,Ry] = -(beta[0,1]+beta[1,0])/2.
            strain_map.data2D['theta'][Rx,Ry] =  (beta[0,1]-beta[1,0])/2.

    return strain_map

def get_strain_from_reference_region(mask, uv_map):
    """
    Gets a strain map from the reference region of real space specified by mask and the lattice
    vector map uv_map.

    Accepts:
        mask        (ndarray of bools) use lattice vectors from uv_map scan positions wherever
                    mask==True
        uv_map      (RealSlice) the lattice vector map; contains 2D arrays in uv_map.data2D under
                    the keys 'ux', 'uy', 'vx', and 'vy'.  See documentation for
                    fit_lattice_vectors_all_DPs() for more information.

    Returns:
        strain_map                  (RealSlice) the strain map; contains the elements of the
                                    infinitessimal strain matrix, in the following 4 arrays:
        strain_map.data2D['e_xx']   change in lattice x-components with respect to x
        strain_map.data2D['e_yy']   change in lattice y-components with respect to y
        strain_map.data2D['e_xy']   change in lattice x-components with respect to y
        strain_map.data2D['theta']  rotation of lattice with respect to reference
                                    Note 1: the strain matrix has been symmetrized, so e_xy and
                                    e_yx are identical
                                    Note 2: x and y are here the coordinate axes of diffraction
                                    space, where the lattice vectors were measured.  Calibration
                                    of their rotation with respect to real space may be necessary.
    """
    assert isinstance(uv_map, RealSlice)
    assert np.all([name in uv_map.data2D.keys() for name in ('ux','uy','vx','vy')])
    assert mask.dtype == bool

    ux,uy,vx,vy = get_reference_uv(mask, uv_map)
    strain_map = get_strain_from_reference_uv(ux,uy,vx,vy,uv_map)
    return strain_map
