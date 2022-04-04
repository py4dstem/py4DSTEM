import warnings
import numpy as np
from scipy import linalg
from typing import Union, Optional, Dict, Tuple, List
from time import time
from tqdm import tqdm
from dataclasses import dataclass

from ...io.datastructure import PointList
from ..utils import electron_wavelength_angstrom, single_atom_scatter
from ..dpc import get_interaction_constant
from .WK_scattering_factors import compute_WK_factor


@dataclass
class DynamicalMatrixCache:
    has_valid_cache: bool = False
    cached_U_gmh: np.array = None


def calculate_dynamical_structure_factors(
    self,
    accelerating_voltage: float,
    method: str = "WK-CP",
    k_max: float = 2.0,
    thermal_sigma: float = None,
    tol_structure_factor: float = 1.0e-4,
    verbose=True,
):
    """
    Calculate and store the relativistic corrected structure factors used for Bloch computations
    in a dictionary for faster lookup.

    Args:
        accelerating_voltage (float):   accelerating voltage in eV
        method (str):                   Choose which parameterization of the structure factors to use:
            "Lobato": Uses the kinematic structure factors from crystal.py, using the parameterization from
                Lobato & Van Dyck, Acta Cryst A 70:6 (2014)
            "Lobato-absorptive": Lobato factors plus an imaginary part
                equal to 0.1•f, as a simple but inaccurate way to include absorption, per
                Hashimoto, Howie, & Whelan, Proc R Soc Lond A 269:80-103 (1962)
            "WK":   Uses the Weickenmeier-Kohl parameterization for
                    the elastic form factors, including Debye-Waller factor,
                    with no absorption, as described in
                    Weickenmeier & Kohl, Acta Cryst A 47:5 (1991)
            "WK-C": WK form factors plus the "core" contribution to absorption
                    following H. Rose, Optik 45:2 (1976)
            "WK-P": WK form factors plus the phonon/TDS absorptive contribution
            "WK-CP": WK form factors plus core and phonon absorption (default)

        k_max (float):                  max scattering length to compute structure factors to.
                                        Setting this to 2x the k_max used in generating the beamsn
                                        included in a simulation will retain all possible couplings
        thermal_sigma (float):  RMS atomic diplacement for attenuating form factors to account for thermal
                                        broadening of the potential, only used when a "WK" method is
                                        selected. Required when WK-P or WK-CP are selected.
                                        Units are Å. (This is often written as 〈u〉in papers)
        tol_structure_factor (float):   tolerance for removing low-valued structure factors. Reflections
                                        with structure factor below the tolerance will have zero coupling
                                        in the dynamical calculations (i.e. they are the ignored weak beams)

        See WK_scattering_factors.py for details on the Weickenmeier-Kohl form factors.
    """

    assert method in (
        "Lobato",
        "Lobato-absorptive",
        "WK",
        "WK-C",
        "WK-P",
        "WK-CP",
    ), "Invalid method specified."

    # Calculate the reciprocal lattice points to include based on k_max

    k_max = np.asarray(k_max)

    # Inverse lattice vectors
    lat_inv = np.linalg.inv(self.lat_real)

    # Find shortest lattice vector direction
    k_test = np.vstack(
        [
            lat_inv[0, :],
            lat_inv[1, :],
            lat_inv[2, :],
            lat_inv[0, :] + lat_inv[1, :],
            lat_inv[0, :] + lat_inv[2, :],
            lat_inv[1, :] + lat_inv[2, :],
            lat_inv[0, :] + lat_inv[1, :] + lat_inv[2, :],
            lat_inv[0, :] - lat_inv[1, :] + lat_inv[2, :],
            lat_inv[0, :] + lat_inv[1, :] - lat_inv[2, :],
            lat_inv[0, :] - lat_inv[1, :] - lat_inv[2, :],
        ]
    )
    k_leng_min = np.min(np.linalg.norm(k_test, axis=1))

    # Tile lattice vectors
    num_tile = np.ceil(k_max / k_leng_min)
    ya, xa, za = np.meshgrid(
        np.arange(-num_tile, num_tile + 1),
        np.arange(-num_tile, num_tile + 1),
        np.arange(-num_tile, num_tile + 1),
    )
    hkl = np.vstack([xa.ravel(), ya.ravel(), za.ravel()])
    g_vec_all = lat_inv @ hkl

    # Delete lattice vectors outside of k_max
    keep = np.linalg.norm(g_vec_all, axis=0) <= k_max
    hkl = hkl[:, keep]
    g_vec_all = g_vec_all[:, keep]
    g_vec_leng = np.linalg.norm(g_vec_all, axis=0)

    # We do not precompute form factors here, instead we rely on using
    # automatic caching of the form factors.

    lobato_lookup = single_atom_scatter()

    from functools import lru_cache

    @lru_cache(maxsize=2 ** 12)
    def get_f_e(q, Z, B, method):
        if method == "Lobato":
            # Real lobato factors
            lobato_lookup.get_scattering_factor([Z], [1.0], [q], units="VA")
            return np.complex128(lobato_lookup.fe)
        elif method == "Lobato-absorptive":
            # Fake absorptive Lobato factors
            lobato_lookup.get_scattering_factor([Z], [1.0], [q], units="VA")
            return np.complex128(lobato_lookup.fe + 0.1j * lobato_lookup.fe)
        elif method == "WK":
            # Real WK factor
            return compute_WK_factor(
                float(q),
                int(Z),
                float(accelerating_voltage),
                float(thermal_sigma),
                include_core=False,
                include_phonon=False,
            )
        elif method == "WK-C":
            # WK, core only
            return compute_WK_factor(
                float(q),
                int(Z),
                float(accelerating_voltage),
                float(thermal_sigma),
                include_core=True,
                include_phonon=False,
            )
        elif method == "WK-P":
            # WK, phonon only
            return compute_WK_factor(
                float(q),
                int(Z),
                float(accelerating_voltage),
                float(thermal_sigma),
                include_core=False,
                include_phonon=True,
            )
        elif method == "WK-CP":
            # WK, core + phonon
            return compute_WK_factor(
                float(q),
                int(Z),
                float(accelerating_voltage),
                float(thermal_sigma),
                include_core=True,
                include_phonon=True,
            )

    # Calculate structure factors
    struct_factors = np.zeros(np.size(g_vec_leng, 0), dtype="complex128")
    for i_hkl in tqdm(
        range(hkl.shape[1]),
        desc=f"Computing {method} lookup table",
        disable=not verbose,
    ):
        Freal = 0.0
        Fimag = 0.0
        for i_pos in range(self.positions.shape[0]):
            # Get the appropriate atomic form factor:
            fe = get_f_e(g_vec_leng[i_hkl], self.numbers[i_pos], thermal_sigma, method)

            # accumulate the real and imag portions separately (?)
            Freal += np.real(fe) * np.exp(
                (2.0j * np.pi) * (hkl[:, i_hkl] @ self.positions[i_pos])
            )
            Fimag += np.imag(fe) * np.exp(
                (2.0j * np.pi) * (hkl[:, i_hkl] @ self.positions[i_pos])
            )
        struct_factors[i_hkl] = Freal + 1.0j * Fimag

    # Divide by unit cell volume
    unit_cell_volume = np.abs(np.linalg.det(self.lat_real))
    struct_factors /= unit_cell_volume

    # Remove structure factors below tolerance level
    keep = np.abs(struct_factors) > tol_structure_factor
    hkl = hkl[:, keep]

    g_vec_all = g_vec_all[:, keep]
    g_vec_leng = g_vec_leng[keep]
    struct_factors = struct_factors[keep]

    # Store relativistic corrected structure factors in a dictionary for faster lookup in the Bloch code

    self.accel_voltage = accelerating_voltage
    self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

    self.Ug_dict = {
        (hkl[0, i], hkl[1, i], hkl[2, i]): struct_factors[i]
        for i in range(hkl.shape[1])
    }

    # Clear cached scattering factors to free up RAM
    get_f_e.cache_clear()


def generate_dynamical_diffraction_pattern(
    self,
    beams: PointList,
    thickness: Union[float, list, tuple, np.ndarray],
    zone_axis_lattice: np.ndarray = None,
    zone_axis_cartesian: np.ndarray = None,
    foil_normal_lattice: np.ndarray = None,
    foil_normal_cartesian: np.ndarray = None,
    verbose: bool = False,
    always_return_list: bool = False,
    dynamical_matrix_cache: Optional[DynamicalMatrixCache] = None,
) -> Union[PointList, List[PointList]]:
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
        thickness (float or list/array) thickness in Ångström to evaluate diffraction patterns at.
                                        The main Bloch calculation can be reused for multiple thicknesses
                                        without much overhead.
        zone_axis & foil_normal         Incident beam orientation and foil normal direction.
                                        Each can be specified in the Cartesian or crystallographic basis,
                                        using e.g. zone_axis_lattice or zone_axis_cartesian. These are
                                        internally parsed by Crystal.parse_orientation

    Less commonly used args:
        always_return_list (bool):      When True, the return is always a list of PointLists,
                                        even for a single thickness
        dynamical_matrix_cache:         (DyanmicalMatrixCache) Dataclass used for caching of the
                                        dynamical matrix. If the cached matrix does not exist, it is
                                        computed and stored. Subsequent calls will use the cached matrix
                                        for the off-diagonal components of the A matrix and overwrite
                                        the diagonal elements. This is used for CBED calculations.

    Returns:
        bragg_peaks (PointList):         Bragg peaks with fields [qx, qy, intensity, h, k, l]
            or
        [bragg_peaks,...] (PointList):   If thickness is a list/array, or always_return_list is True,
                                        a list of PointLists is returned.

    """
    t0 = time()  # start timer for matrix setup

    n_beams = beams.length

    beam_g, beam_h = np.meshgrid(np.arange(n_beams), np.arange(n_beams))

    # Parse input orientations:
    zone_axis_rotation_matrix = self.parse_orientation(zone_axis_lattice=zone_axis_lattice, 
                                       zone_axis_cartesian=zone_axis_cartesian)
    if foil_normal_lattice is not None or foil_normal_cartesian is not None:
        foil_normal = self.parse_orientation(zone_axis_lattice=foil_normal_lattice,
                                             zone_axis_cartesian=foil_normal_cartesian)
    else:
        foil_normal = zone_axis_rotation_matrix

    foil_normal = foil_normal[:,2]

    # Note the difference in notation versus kinematic function:
    # k0 is the scalar magnitude of the wavevector, rather than
    # a vector along the zone axis.
    k0 = 1.0 / electron_wavelength_angstrom(self.accel_voltage)

    ################################################################
    # Compute the reduced structure matrix \bar{A} in DeGraef 5.52 #
    ################################################################

    hkl = np.vstack((beams.data["h"], beams.data["k"], beams.data["l"])).T

    # Check if we have a cached dynamical matrix, which saves us from calculating the
    # off-diagonal elements when running this in a loop with the same zone axis
    if dynamical_matrix_cache is not None and dynamical_matrix_cache.has_valid_cache:
        U_gmh = dynamical_matrix_cache.cached_U_gmh
    else:
        # No cached matrix is available/desired, so calculate it:

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
            [
                self.Ug_dict.get((gmh[0], gmh[1], gmh[2]), 0.0 + 0.0j)
                for gmh in g_minus_h
            ],
            dtype=np.complex128,
        ).reshape(beam_g.shape)

    # If we are supposed to cache, but don't have one saved, save this one:
    if (
        dynamical_matrix_cache is not None
        and not dynamical_matrix_cache.has_valid_cache
    ):
        dynamical_matrix_cache.cached_U_gmh = U_gmh
        dynamical_matrix_cache.has_valid_cache = True

    if verbose:
        print(f"Bloch matrix has size {U_gmh.shape}")

    # Compute the diagonal entries of \hat{A}: 2 k_0 s_g [5.51]
    g = (hkl @ self.lat_inv) @ zone_axis_rotation_matrix
    sg = self.excitation_errors(g.T, foil_normal=-foil_normal @ zone_axis_rotation_matrix)

    # import matplotlib.pyplot as plt
    # sgp = np.sign(sg) >= 0
    # c = np.zeros_like(g)
    # c[sgp,:] = np.array([1,0,0])
    # c[~sgp,:] = np.array([0,0,1])
    # fig,ax = plt.subplots(dpi=200)
    # ax.scatter(g[:,0],g[:,1],np.abs(sg)*100,c=c)
    # ax.axis('equal')
    # plt.show()


    # Fill in the diagonal, completing the structure mattrx
    np.fill_diagonal(U_gmh, 2 * k0 * sg + 1.0j * np.imag(self.Ug_dict[(0, 0, 0)]))

    if verbose:
        print(f"Constructing the A matrix took {(time()-t0)*1000.:.3f} ms.")

    #############################################################################################
    # Compute eigen-decomposition of \hat{A} to yield C (the matrix containing the eigenvectors #
    # as its columns) and gamma (the reduced eigenvalues), as in DeGraef 5.52                   #
    #############################################################################################

    t0 = time()  # start timer for eigendecomposition

    v, C = linalg.eig(U_gmh)  # decompose!
    gamma = v / (2.0 * k0 * zone_axis_rotation_matrix[:,2] @ foil_normal)  # divide by 2 k_n

    # precompute the inverse of C
    C_inv = np.linalg.inv(C)

    if verbose:
        print(f"Decomposing the A matrix took {(time()-t0)*1000.:.3f} ms.")

    ##############################################################################################
    # Compute diffraction intensities by calculating exit wave \Psi in DeGraef 5.60, and collect #
    # values into PointLists                                                                     #
    ##############################################################################################

    t0 = time()

    psi_0 = np.zeros((n_beams,))
    psi_0[int(np.where((hkl == [0, 0, 0]).all(axis=1))[0])] = 1.0

    # calculate the diffraction intensities for each thichness matrix
    # I = |psi|^2 ; psi = C @ E(z) @ C^-1 @ psi_0, where E(z) is the thickness matrix
    intensities = [
        np.abs(C @ (np.exp(2.0j * np.pi * z * gamma) * (C_inv @ psi_0))) ** 2
        for z in np.atleast_1d(thickness)
    ]

    # make new pointlists for each thickness case and copy intensities
    pls = []
    for i in range(len(intensities)):
        newpl = beams.copy()
        newpl.data["intensity"] = intensities[i]
        pls.append(newpl)

    if verbose:
        print(f"Assembling outputs took {1000*(time()-t0):.3f} ms.")

    if len(pls) == 1 and not always_return_list:
        return pls[0]
    else:
        return pls


def generate_CBED(
    self,
    beams: PointList,
    thickness: Union[float, list, tuple, np.ndarray],
    alpha_mrad: float,
    pixel_size_inv_A: float,
    DP_size_inv_A: Optional[float] = None,
    zone_axis_lattice: np.ndarray = None,
    zone_axis_cartesian: np.ndarray = None,
    foil_normal_lattice: np.ndarray = None,
    foil_normal_cartesian: np.ndarray = None,
    LACBED: bool = False,
    dtype: np.dtype = np.float32,
    verbose: bool = False,
    progress_bar: bool = True,
    return_mask: bool = False,
) -> Union[np.ndarray, List[np.ndarray], Dict[Tuple[int], np.ndarray]]:
    """
    Generate a dynamical CBED pattern using the Bloch wave method.

    Args:
        beams (PointList):              PointList from the kinematical diffraction generator
                                        which will define the beams included in the Bloch calculation
        thickness (float or list/array) thickness in Ångström to evaluate diffraction patterns at.
                                        The main Bloch calculation can be reused for multiple thicknesses
                                        without much overhead.
        alpha_mrad (float):             Convergence angle for CBED pattern. Note that if disks in the calculation
                                        overlap, they will be added incoherently, and the resulting CBED will
                                        thus represent the average over the unit cell (i.e. a PACBED pattern,
                                        as described in LeBeau et al., Ultramicroscopy 110(2): 2010.)
        pixel_size_inv_A (float):       CBED pixel size in 1/Å.
        DP_size_inv_A (optional float): If specified, defines the extents of the diffraction pattern.
                                        If left unspecified, the DP will be automatically scaled to
                                        fit all of the beams present in the input plus some small buffer.
        zone_axis (np float vector):    3 element projection direction for sim pattern
                                        Can also be a 3x3 orientation matrix (zone axis 3rd column)
        foil_normal:                    3 element foil normal - set to None to use zone_axis
        LACBED (bool)                   Return each diffraction disk as a separate image, in a dictionary
                                        keyed by tuples of (h,k,l).
        proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)

    Returns:
        If thickness is a scalar: CBED pattern as np.ndarray
        If thickness is a sequence: CBED patterns for each thickness value as a list of np.ndarrays
        If LACBED is True and thickness is scalar: Dictionary with tuples of ints (h,k,l) as keys, mapping to np.ndarray.
        If LACBED is True and thickness is a sequence: List of dictionaries, structured as above.
    """

    alpha_rad = alpha_mrad / 1000.0

    # figure out the projected x and y directions from the beams input
    hkl = np.vstack((beams.data["h"], beams.data["k"], beams.data["l"])).T.astype(
        np.float64
    )
    qxy = np.vstack((beams.data["qx"], beams.data["qy"])).T.astype(np.float64)

    proj = np.linalg.lstsq(qxy, hkl, rcond=-1)[0]
    hkl_proj_x = proj[0] / np.linalg.norm(proj[0])
    hkl_proj_y = proj[1] / np.linalg.norm(proj[1])

    # get unit vector in zone axis direction and projected x and y Cartesian directions:
    zone_axis_rotation_matrix = self.parse_orientation(zone_axis_lattice=zone_axis_lattice,
                                       zone_axis_cartesian=zone_axis_cartesian,
                                       proj_x_lattice=hkl_proj_x)
    ZA = np.array(zone_axis_rotation_matrix[:,2]) / np.linalg.norm(np.array(zone_axis_rotation_matrix[:,2]))
    proj_x = zone_axis_rotation_matrix[:,0] / np.linalg.norm(zone_axis_rotation_matrix[:,0])
    proj_y = zone_axis_rotation_matrix[:,1] / np.linalg.norm(zone_axis_rotation_matrix[:,1])

    # the foil normal should be the zone axis if unspecified
    if foil_normal_lattice is None:
        foil_normal_lattice = zone_axis_lattice
    if foil_normal_cartesian is None:
        foil_normal_cartesian = zone_axis_cartesian

    # TODO: refine pixel size to center reflections on pixels

    # Generate list of plane waves inside aperture
    alpha_pix = np.round(
        alpha_rad / self.wavelength / pixel_size_inv_A
    )  # radius of aperture in pixels

    tx_pixels, ty_pixels = np.meshgrid(
        np.arange(-alpha_pix, alpha_pix + 1), np.arange(-alpha_pix, alpha_pix + 1)
    )  # plane waves in pixel units

    # remove those outside circular aperture
    keep_mask = np.hypot(tx_pixels, ty_pixels) < alpha_pix
    tx_pixels = tx_pixels[keep_mask].astype(np.intp)
    ty_pixels = ty_pixels[keep_mask].astype(np.intp)

    tx_rad = tx_pixels / alpha_pix * alpha_rad
    ty_rad = ty_pixels / alpha_pix * alpha_rad

    # calculate plane waves as zone axes using small angle approximation for tilting
    tZA = ZA - (tx_rad[:, None] * proj_x) - (ty_rad[:, None] * proj_y)

    if LACBED:
        # In LACBED mode, the default DP size is the same as one diffraction disk (2ɑ)
        if DP_size_inv_A is None:
            DP_size = [int(2 * alpha_pix), int(2 * alpha_pix)]
        else:
            DP_size = [
                int(2 * DP_size_inv_A / pixel_size_inv_A),
                int(2 * DP_size_inv_A / pixel_size_inv_A),
            ]
    else:
        # determine DP size based on beams present, plus a little extra
        qx_max = np.max(np.abs(beams.data["qx"])) / pixel_size_inv_A
        qy_max = np.max(np.abs(beams.data["qy"])) / pixel_size_inv_A

        if DP_size_inv_A is None:
            DP_size = [
                int(2 * (qx_max + 2 * alpha_pix)),
                int(2 * (qy_max + 2 * alpha_pix)),
            ]
        else:
            DP_size = [
                int(2 * DP_size_inv_A / pixel_size_inv_A),
                int(2 * DP_size_inv_A / pixel_size_inv_A),
            ]

    qx0 = DP_size[0] // 2
    qy0 = DP_size[1] // 2

    thickness = np.atleast_1d(thickness)

    if LACBED:
        # In LACBED mode, the DP datastructure is a list of dicts mapping tuples of ints to numpy arrays
        DP = [
            {
                (d["h"], d["k"], d["l"]): np.zeros(DP_size, dtype=dtype)
                for d in beams.data
            }
            for _ in range(len(thickness))
        ]
    else:
        # In CBED mode, the DP datastructure is a list of arrays
        DP = [np.zeros(DP_size, dtype=dtype) for _ in range(len(thickness))]

    mask = np.zeros(DP_size, dtype=np.bool_)

    Ugmh_cache = DynamicalMatrixCache()

    for i in tqdm(range(len(tZA)), disable=not progress_bar):
        bloch = self.generate_dynamical_diffraction_pattern(
            beams,
            thickness=thickness,
            zone_axis_cartesian=tZA[i],
            foil_normal_cartesian=foil_normal_cartesian,
            foil_normal_lattice=foil_normal_lattice,
            always_return_list=True,
            dynamical_matrix_cache=Ugmh_cache,
        )

        if LACBED:
            # loop over each thickness
            for patt, sim in zip(DP, bloch):
                # loop over each beam
                for refl in sim.data:
                    patt[(refl["h"], refl["k"], refl["l"])][
                        qx0 + tx_pixels[i], qy0 + ty_pixels[i]
                    ] = refl["intensity"]
        else:
            xpix = np.round(
                bloch[0].data["qx"] / pixel_size_inv_A + tx_pixels[i] + qx0
            ).astype(np.intp)
            ypix = np.round(
                bloch[0].data["qy"] / pixel_size_inv_A + ty_pixels[i] + qy0
            ).astype(np.intp)

            keep_mask = np.logical_and.reduce(
                (xpix >= 0, ypix >= 0, xpix < DP_size[0], ypix < DP_size[1])
            )

            xpix = xpix[keep_mask]
            ypix = ypix[keep_mask]

            mask[xpix, ypix] = True

            for patt, sim in zip(DP, bloch):
                patt[xpix, ypix] += sim.data["intensity"][keep_mask]

    if return_mask:
        return (DP[0], mask) if len(thickness) == 1 else (DP, mask)
    else:
        return DP[0] if len(thickness) == 1 else DP
