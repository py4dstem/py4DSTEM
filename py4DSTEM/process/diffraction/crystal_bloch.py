import warnings
import numpy as np
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..dpc import get_interaction_constant
from ..utils import electron_wavelength_angstrom

from pdb import set_trace


def generate_dynamical_diffraction_pattern(
    self,
    beams: PointList,
    thickness: Union[float, list, tuple, np.ndarray],
    zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
    foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
) -> PointList:
    """
    Generate a dynamical diffraction pattern (or thickness series of patterns)
    using the Bloch wave method.

    The beams to be included in the Bloch calculation must be pre-calculated
    and passed as a PointList containing at least (qx, qy, h, k, l) fields.

    If ``thickness`` is a single value, one new PointList will be returned.
    If ``thickness`` is a sequence of values, a list of PointLists will be returned,
        corresponding to each thickness value in the input.

    Frequent reference will be made to

    Args:
        beams (PointList):              PointList from the kinematical diffraction generator
                                        which will define the beams included in the Bloch calculation
        thickness (float or list/array) thickness to evaluate diffraction patterns at.
                                        The main Bloch calculation can be reused for multiple thicknesses
                                        without much overhead.
        zone_axis (np float vector):     3 element projection direction for sim pattern
                                         Can also be a 3x3 orientation matrix (zone axis 3rd column)
        foil_normal:                     3 element foil normal - set to None to use zone_axis
        proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)

    Returns:
        bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, h, k, l]
    """

    n_beams = beams.length
    beam_g, beam_h = np.meshgrid(np.arange(n_beams), np.arange(n_beams))

    # Clean up zone axis input (same as in kinematic function)
    zone_axis = np.asarray(zone_axis) / np.linalg.norm(zone_axis)
    if not self.cartesian_directions:
        zone_axis = self.cartesian_to_crystal(zone_axis)

    # Foil normal vector
    if foil_normal is None:
        foil_normal = zone_axis
    else:
        foil_normal = np.asarray(foil_normal, dtype="float")
        if not self.cartesian_directions:
            foil_normal = self.crystal_to_cartesian(foil_normal)
        else:
            foil_normal = foil_normal / np.linalg.norm(foil_normal)

    # Note the difference in notation versus kinematic function:
    # k0 is the scalar magnitude of the wavevector, rather than
    # a vector along the zone axis. That is instead called ``ZA``
    k0 = 1.0 / electron_wavelength_angstrom(self.accel_voltage)
    ZA = zone_axis * k0

    ################################################################
    # Compute the reduced structure matrix \hat{A} in DeGraef 5.52 #
    ################################################################

    hkl = np.vstack((beams.data["h"], beams.data["k"], beams.data["l"])).T

    # get hkl indices of \vec{g} - \vec{h}
    g_minus_h = np.vstack(
        (
            beams.data["h"][beam_g.ravel()] - beams.data["h"][beam_h.ravel()],
            beams.data["k"][beam_g.ravel()] - beams.data["k"][beam_h.ravel()],
            beams.data["l"][beam_g.ravel()] - beams.data["l"][beam_h.ravel()],
        )
    ).T

    # Check if each beam has a computed structure factor. We'll ignore coupling for scattering
    # greater than the k_max used in compute_structure_factors. We also flag
    # beams where g-h=0 to ignore.
    nonzero_beams = [
        gmh in self.hkl.T and not np.array_equal(gmh, [0, 0, 0]) for gmh in g_minus_h
    ]

    # Relativistic correction to the potentials [2.38]
    prefactor = get_interaction_constant(self.accel_voltage) / (
        np.pi * electron_wavelength_angstrom(self.accel_voltage)
    )

    # Get the structure factors for each nonzero element, and zero otherwise
    U_gmh = np.array(
        [
            prefactor
            * self.struct_factors[int(np.where((gmh == self.hkl.T).all(axis=1))[0])]
            if nonzero
            else 0.0 + 0.0j
            for gmh, nonzero in zip(g_minus_h, nonzero_beams)
        ],
        dtype=np.complex64,
    ).reshape(beam_g.shape)

    # Compute the diagonal entries of \hat{A}: 2 k_0 s_g
    g = np.linalg.inv(self.lat_real) @ hkl.T
    cos_alpha = np.sum(
        (ZA[:, None] + g) * foil_normal[:, None], axis=0
    ) / np.linalg.norm(ZA[:, None] + g, axis=0)

    sg = (
        (-0.5)
        * np.sum((2 * ZA[:, None] + g) * g, axis=0)
        / (np.linalg.norm(ZA[:, None] + g, axis=0))
        / cos_alpha
    )

    # Fill in the diagonal, completing the structure mattrx
    np.fill_diagonal(U_gmh, 2 * k0 * sg)

    #############################################################################################
    # Compute eigen-decomposition of \hat{A} to yield C (the matrix containing the eigenvectors #
    # as its columns) and gamma (the reduced eigenvalues), as in DeGraef 5.52                   #
    #############################################################################################

    v, C = np.linalg.eig(U_gmh)  # decompose!
    gamma = v / (2 * ZA @ foil_normal)  # divide by 2 k_n

    # precompute the inverse of C
    C_inv = np.linalg.inv(C)

    ######################################################
    # Compute thickness matrix/matrices E (DeGraef 5.60) #
    ######################################################

    E = [np.diag(np.exp(2 * np.pi * 1j * gamma * z)) for z in np.atleast_1d(thickness)]

    ##############################################################################################
    # Compute diffraction intensities by calculating exit wave \Psi in DeGraef 5.60, and collect #
    # values into PointLists                                                                     #
    ##############################################################################################

    psi_0 = np.zeros((n_beams,))
    psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0

    # calculate the diffraction intensities for each thichness matrix
    # I = |psi|^2 ; psi = C @ E(z) @ C^-1 @ psi_0
    intensities = [np.abs(C @ Ez @ C_inv @ psi_0) ** 2 for Ez in E]

    set_trace()

    # make new pointlists for each thickness case and copy intensities
    pls = []
    for i in range(len(intensities)):
        newpl = beams.copy()
        newpl.data["intensity"] = intensities[i]
        pls.append(newpl)

    if len(pls) == 1:
        return pls[0]
    else:
        return pls
