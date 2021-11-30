import warnings
import numpy as np
from scipy import linalg
from typing import Union, Optional
from time import time

from ...io.datastructure import PointList
from ..utils import electron_wavelength_angstrom
from ..dpc import get_interaction_constant


def setup_dynamical_calculation(
    self, accelerating_voltage: float, cartesian_directions: bool = True
):
    """
    Setup required attributes for dynamical calculation without going
    through the full ACOM pipeline
    """
    self.accel_voltage = accelerating_voltage
    self.wavelength = electron_wavelength_angstrom(self.accel_voltage)
    self.cartesian_directions = cartesian_directions


def calculate_dynamical_structure_factors(self):
    # Store relativistic corrected structure factors in a dictionary for faster lookup in the Bloch code
    # Relativistic correction to the potentials [2.38]
    prefactor = (
        47.86
        * get_interaction_constant(self.accel_voltage)
        / (np.pi * electron_wavelength_angstrom(self.accel_voltage))
    )
    self.Ug_dict = {
        (self.hkl[0, i], self.hkl[1, i], self.hkl[2, i]): prefactor
        * self.struct_factors[i]
        for i in range(self.hkl.shape[1])
    }
    self.Ug_dict[(0, 0, 0)] = 0.0 + 0.0j


def generate_dynamical_diffraction_pattern(
    self,
    beams: PointList,
    thickness: Union[float, list, tuple, np.ndarray],
    zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
    foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
    naive_absorption: bool = False,
    verbose=False,
) -> PointList:
    """
    Generate a dynamical diffraction pattern (or thickness series of patterns)
    using the Bloch wave method.

    The beams to be included in the Bloch calculation must be pre-calculated
    and passed as a PointList containing at least (qx, qy, h, k, l) fields.

    If ``thickness`` is a single value, one new PointList will be returned.
    If ``thickness`` is a sequence of values, a list of PointLists will be returned,
        corresponding to each thickness value in the input.

    Frequent reference will be made to "Introduction to conventional transmission electron microscopy"
        by DeGraef, whose overall approach we follow here.

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
        naive_absorption (bool):        Add an imaginary component that is 10% of the real component
                                        as an __extremely__ simple approximation of absorption

    Returns:
        bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, h, k, l]
    """
    t0 = time()  # start timer for matrix setup

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

    # Get the structure factors for each nonzero element, and zero otherwise
    U_gmh = np.array(
        [self.Ug_dict.get((gmh[0], gmh[1], gmh[2]), 0.0 + 0.0j) for gmh in g_minus_h],
        dtype=np.complex128,
    ).reshape(beam_g.shape)

    if verbose:
        print(f"Bloch matrix has size {U_gmh.shape}")

    if naive_absorption:
        U_gmh *= 1.0 + 0.1j

    # Compute the diagonal entries of \hat{A}: 2 k_0 s_g [5.51]
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

    if verbose:
        print(f"Constructing the A matrix took {(time()-t0)*1000.} ms.")

    #############################################################################################
    # Compute eigen-decomposition of \hat{A} to yield C (the matrix containing the eigenvectors #
    # as its columns) and gamma (the reduced eigenvalues), as in DeGraef 5.52                   #
    #############################################################################################

    t0 = time()  # start timer for eigendecomposition

    v, C = linalg.eig(U_gmh)  # decompose!
    gamma = v / (2.0 * ZA @ foil_normal)  # divide by 2 k_n

    # precompute the inverse of C
    C_inv = np.linalg.inv(C)

    if verbose:
        print(f"Decomposing the A matrix took {(time()-t0)*1000.} ms.")

    ######################################################
    # Compute thickness matrix/matrices E (DeGraef 5.60) #
    ######################################################

    t0 = time()

    E = [np.diag(np.exp(2.0j * np.pi * z * gamma)) for z in np.atleast_1d(thickness)]

    if verbose:
        print(f"Constructing thickness matrices took {1000*(time()-t0)} ms.")

    ##############################################################################################
    # Compute diffraction intensities by calculating exit wave \Psi in DeGraef 5.60, and collect #
    # values into PointLists                                                                     #
    ##############################################################################################

    t0 = time()

    psi_0 = np.zeros((n_beams,))
    psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0

    # calculate the diffraction intensities for each thichness matrix
    # I = |psi|^2 ; psi = C @ E(z) @ C^-1 @ psi_0
    intensities = [np.abs(C @ Ez @ C_inv @ psi_0) ** 2 for Ez in E]

    # set_trace()

    # make new pointlists for each thickness case and copy intensities
    pls = []
    for i in range(len(intensities)):
        newpl = beams.copy()
        newpl.data["intensity"] = intensities[i]
        pls.append(newpl)

    if verbose:
        print(f"Assembling outputs took {1000*(time()-t0)} ms.")

    if len(pls) == 1:
        return pls[0]
    else:
        return pls


def generate_CBED(
    self,
    beams: PointList,
    thickness: Union[float, list, tuple, np.ndarray],
    alpha_mrad: float,
    pixel_size_inv_A: float,
    DP_size_inv_A: float,
    zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
    foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
    naive_absorption: bool = False,
    overlapping_disks: bool = False,
    verbose=False,
) -> np.ndarray:
    """
    Generate a dynamical CBED pattern using the Bloch wave method.

    Args:
        beams (PointList)
    """

    # figure out the projected x and y directions from the beams input
    hkl = np.vstack((beams.data["h"], beams.data["k"], beams.data["l"])).T.astype(
        np.float64
    )
    qxy = np.vstack((beams.data["qx"], beams.data["qy"])).T.astype(np.float64)

    proj = np.linalg.lstsq(qxy, hkl, rcond=-1)[0]
    ZAx = proj[0] / np.linalg.norm(proj[0])
    ZAy = proj[1] / np.linalg.norm(proj[1])

    return
