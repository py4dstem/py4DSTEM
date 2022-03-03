# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, single_atom_scatter, electron_wavelength_angstrom

from .crystal_viz import plot_diffraction_pattern


class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    """

    # Various methods for the Crystal class are implemented in a separate file. This
    # import statement inside the class declaration imports them as methods of the class!
    # (see https://stackoverflow.com/a/47562412)

    # Automated Crystal Orientation Mapping is implemented in crystal_ACOM.py
    from .crystal_ACOM import orientation_plan, match_orientations, match_single_pattern

    from .crystal_viz import (
        plot_structure,
        plot_structure_factors,
        plot_orientation_zones,
        plot_orientation_plan,
        plot_orientation_maps,
    )

    # Dynamical diffraction calculations are implemented in crystal_bloch.py
    from .crystal_bloch import (
        generate_dynamical_diffraction_pattern,
        generate_CBED,
        calculate_dynamical_structure_factors,
    )

    def __init__(
        self,
        positions,
        numbers,
        cell,
    ):
        """
        Args:
            positions (np.array): fractional coordinates of each atom in the cell
            numbers (np.array): Z number for each atom in the cell
            cell (np.array): specify the unit cell, using a variable number of parameters
                1 number: the lattice parameter for a cubic cell
                3 numbers: the three lattice parameters for an orthorhombic cell
                6 numbers: the a,b,c lattice parameters and ɑ,β,ɣ angles for any cell
        """
        # Initialize Crystal
        self.positions = np.asarray(positions)  #: fractional atomic coordinates

        #: atomic numbers - if only one value is provided, assume all atoms are same species
        numbers = np.asarray(numbers, dtype="intp")
        if np.size(numbers) == 1:
            self.numbers = np.ones(self.positions.shape[0], dtype="intp") * numbers
        elif np.size(numbers) == self.positions.shape[0]:
            self.numbers = numbers
        else:
            raise Exception("Number of positions and atomic numbers do not match")

        # unit cell, as either [a a a 90 90 90], [a b c 90 90 90], or [a b c alpha beta gamma]
        cell = np.asarray(cell, dtype="float_")
        if np.size(cell) == 1:
            self.cell = np.hstack([cell, cell, cell, 90, 90, 90])
        elif np.size(cell) == 3:
            self.cell = np.hstack([cell, 90, 90, 90])
        elif np.size(cell) == 6:
            self.cell = cell
        else:
            raise Exception("Cell cannot contain " + np.size(cell) + " elements")

        # calculate unit cell lattice vectors
        a = self.cell[0]
        b = self.cell[1]
        c = self.cell[2]
        alpha = self.cell[3] * np.pi / 180
        beta = self.cell[4] * np.pi / 180
        gamma = self.cell[5] * np.pi / 180
        t = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        self.lat_real = np.array(
            [
                [a, 0, 0],
                [b * np.cos(gamma), b * np.sin(gamma), 0],
                [c * np.cos(beta), c * t, c * np.sqrt(1 - np.cos(beta) ** 2 - t ** 2)],
            ]
        )

    def from_CIF(CIF, conventional_standard_structure=True):
        """
        Create a Crystal object from a CIF file, using pymatgen to import the CIF

        Note that pymatgen typically prefers to return primitive unit cells,
        which can be overridden by setting conventional_standard_structure=True.

        Args:
            CIF: (str or Path) path to the CIF File
            conventional_standard_structure: (bool) if True, conventional standard unit cell will be returned
                instead of the primitive unit cell pymatgen typically returns
        """
        from pymatgen.io.cif import CifParser

        parser = CifParser(CIF)

        structure = parser.get_structures()[0]

        return Crystal.from_pymatgen_structure(
            structure, conventional_standard_structure=conventional_standard_structure
        )

    def from_pymatgen_structure(
        structure=None,
        formula=None,
        space_grp=None,
        MaPKey=None,
        conventional_standard_structure=True,
    ):
        """
        Create a Crystal object from a pymatgen Structure object.
        If a Materials Project API key is installed, you may pass
        the Materials Project ID of a structure, which will be
        fetched through the MP API. For setup information see:
        https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file.
        Alternatively, Materials Porject API key can be pass as an argument through
        the function (MaPKey). To get your API key, please visit Materials Project website
        and login/sign up using your email id. Once logged in, go to the dashboard
        to generate your own API key (https://materialsproject.org/dashboard).

        Note that pymatgen typically prefers to return primitive unit cells,
        which can be overridden by setting conventional_standard_structure=True.

        Args:
            structure:      (pymatgen Structure or str), if specified as a string, it will be considered
                            as a Materials Project ID of a structure, otherwise it will accept only
                            pymatgen Structure object. if None, MP database will be queried using the
                            specified formula and/or space groups for the available structure
            formula:        (str), pretty formula to search in the MP database, (note that the forumlas in MP
                            database are not always formatted in the conventional order. Please
                            visit Materials Project website for information (https://materialsproject.org/)
                            if None, structure argument must not be None
            space_grp:      (int) space group number of the forumula provided to query MP database. If None, MP will search
                            for all the available space groups for the formula provided and will consider the
                            one with lowest unit cell volume, only specify when using formula to search MP
                            database
            MaPKey:         (str) Materials Project API key
            conventional_standard_structure: (bool) if True, conventional standard unit cell will be returned
                            instead of the primitive unit cell pymatgen returns

        """
        import pymatgen as mg
        from pymatgen.ext.matproj import MPRester

        if structure is not None:
            if isinstance(structure, str):
                mpr = MPRester(MaPKey)
                structure = mpr.get_structure_by_material_id(structure)

            assert isinstance(
                structure, mg.core.Structure
            ), "structure must be pymatgen Structure object"

            structure = (
                mg.symmetry.analyzer.SpacegroupAnalyzer(
                    structure
                ).get_conventional_standard_structure()
                if conventional_standard_structure
                else structure
            )
        else:
            mpr = MPRester(MaPKey)
            if formula is None:
                raise Exception(
                    "Atleast a formula needs to be provided to query from MP database!!"
                )
            query = mpr.query(
                criteria={"pretty_formula": formula},
                properties=["structure", "icsd_ids", "spacegroup"],
            )
            if space_grp:
                query = [
                    query[i]
                    for i in range(len(query))
                    if mg.symmetry.analyzer.SpacegroupAnalyzer(
                        query[i]["structure"]
                    ).get_space_group_number()
                    == space_grp
                ]
            selected = query[
                np.argmin(
                    [query[i]["structure"].lattice.volume for i in range(len(query))]
                )
            ]
            structure = (
                mg.symmetry.analyzer.SpacegroupAnalyzer(
                    selected["structure"]
                ).get_conventional_standard_structure()
                if conventional_standard_structure
                else selected["structure"]
            )

        positions = structure.frac_coords  #: fractional atomic coordinates

        cell = np.array(
            [
                structure.lattice.a,
                structure.lattice.b,
                structure.lattice.c,
                structure.lattice.alpha,
                structure.lattice.beta,
                structure.lattice.gamma,
            ]
        )

        numbers = np.array([s.species.elements[0].Z for s in structure])

        return Crystal(positions, numbers, cell)

    def from_unitcell_parameters(
        latt_params,
        elements,
        positions,
        space_group=None,
        lattice_type="cubic",
        from_cartesian=False,
        conventional_standard_structure=True,
    ):

        """
        Create a Crystal using pymatgen to generate unit cell manually from user inputs

        Args:
                latt_params:         (list of floats) list of lattice parameters. For example, for cubic: latt_params = [a],
                                     for hexagonal: latt_params = [a, c], for monoclinic: latt_params = [a,b,c,beta],
                                     and in general: latt_params = [a,b,c,alpha,beta,gamma]
                elements:            (list of strings) list of elements, for example for SnS: elements = ["Sn", "S"]
                positions:           (list) list of (x,y,z) positions for each element present in the elements, default: fractional coord
                space_group:         (optional) (string or int) space group of the crystal system, if specified, unit cell will be created using
                                     pymatgen Structure.from_spacegroup function
                lattice_type:        (string) type of crystal family: cubic, hexagonal, triclinic etc; default: 'cubic'
                from_cartesian:      (bool) if True, positions will be considered as cartesian, default: False
                conventional_standard_structure: (bool) if True, conventional standard unit cell will be returned
                                     instead of the primitive unit cell pymatgen returns
        Returns:
                Crystal object

        """

        import pymatgen as mg

        if lattice_type == "cubic":
            assert (
                len(latt_params) == 1
            ), "Only 1 lattice parameter is expected for cubic: a, but given {}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.cubic(latt_params[0])
        elif lattice_type == "hexagonal":
            assert (
                len(latt_params) == 2
            ), "2 lattice parametere are expected for hexagonal: a, c, but given {len(latt_params)}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.hexagonal(latt_params[0], latt_params[1])
        elif lattice_type == "tetragonal":
            assert (
                len(latt_params) == 2
            ), "2 lattice parametere are expected for tetragonal: a, c, but given {len(latt_params)}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.tetragonal(latt_params[0], latt_params[1])
        elif lattice_type == "orthorhombic":
            assert (
                len(latt_params) == 3
            ), "3 lattice parametere are expected for orthorhombic: a, b, c, but given {len(latt_params)}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.orthorhombic(
                latt_params[0], latt_params[1], latt_params[2]
            )
        elif lattice_type == "monoclinic":
            assert (
                len(latt_params) == 4
            ), "4 lattice parametere are expected for monoclinic: a, b, c, beta,  but given {len(latt_params)}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.monoclinic(
                latt_params[0], latt_params[1], latt_params[2], latt_params[3]
            )
        else:
            assert (
                len(latt_params) == 6
            ), "all 6 lattice parametere are expected: a, b, c, alpha, beta, gamma, but given {len(latt_params)}".format(
                len(latt_params)
            )
            lattice = mg.core.Lattice.from_parameters(
                latt_params[0],
                latt_params[1],
                latt_params[2],
                latt_params[3],
                latt_params[4],
                latt_params[5],
            )

        if space_group:
            structure = mg.core.Structure.from_spacegroup(
                space_group,
                lattice,
                elements,
                positions,
                coords_are_cartesian=from_cartesian,
            )
        else:
            structure = mg.core.Structure(
                lattice, elements, positions, coords_are_cartesian=from_cartesian
            )

        return Crystal.from_pymatgen_structure(structure)

    def setup_diffraction(
        self, accelerating_voltage: float, cartesian_directions: bool = True
    ):
        """
        Set up attributes used for diffraction calculations without going
        through the full ACOM pipeline.
        """
        self.accel_voltage = accelerating_voltage
        self.wavelength = electron_wavelength_angstrom(self.accel_voltage)
        self.cartesian_directions = cartesian_directions

    def calculate_structure_factors(
        self,
        k_max: float = 2.0,
        tol_structure_factor: float = 1e-4,
        return_intensities: bool = False,
    ):
        """
        Calculate structure factors for all hkl indices up to max scattering vector k_max

        Args:
            k_max (numpy float):                max scattering vector to include (1/Angstroms)
            tol_structure_factor (numpy float): tolerance for removing low-valued structure factors
        """

        # Store k_max
        self.k_max = np.asarray(k_max)

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
        num_tile = np.ceil(self.k_max / k_leng_min)
        ya, xa, za = np.meshgrid(
            np.arange(-num_tile, num_tile + 1),
            np.arange(-num_tile, num_tile + 1),
            np.arange(-num_tile, num_tile + 1),
        )
        hkl = np.vstack([xa.ravel(), ya.ravel(), za.ravel()])
        g_vec_all = lat_inv @ hkl

        # Delete lattice vectors outside of k_max
        keep = np.linalg.norm(g_vec_all, axis=0) <= self.k_max
        self.hkl = hkl[:, keep]
        self.g_vec_all = g_vec_all[:, keep]
        self.g_vec_leng = np.linalg.norm(self.g_vec_all, axis=0)

        # Calculate single atom scattering factors
        # Note this can be sped up a lot, but we may want to generalize to allow non-1.0 occupancy in the future.
        f_all = np.zeros(
            (np.size(self.g_vec_leng, 0), self.positions.shape[0]), dtype="float_"
        )
        for a0 in range(self.positions.shape[0]):
            atom_sf = single_atom_scatter([self.numbers[a0]], [1], self.g_vec_leng, "A")
            atom_sf.get_scattering_factor([self.numbers[a0]], [1], self.g_vec_leng, "A")
            f_all[:, a0] = atom_sf.fe

        # Calculate structure factors
        self.struct_factors = np.zeros(np.size(self.g_vec_leng, 0), dtype="complex64")
        for a0 in range(self.positions.shape[0]):
            self.struct_factors += f_all[:, a0] * np.exp(
                (2j * np.pi)
                * np.sum(
                    self.hkl * np.expand_dims(self.positions[a0, :], axis=1), axis=0
                )
            )

        # Divide by unit cell volume
        unit_cell_volume = np.abs(np.linalg.det(self.lat_real))
        self.struct_factors /= unit_cell_volume

        # Remove structure factors below tolerance level
        keep = np.abs(self.struct_factors) > tol_structure_factor
        self.hkl = self.hkl[:, keep]

        self.g_vec_all = self.g_vec_all[:, keep]
        self.g_vec_leng = self.g_vec_leng[keep]
        self.struct_factors = self.struct_factors[keep]

        # Structure factor intensities
        self.struct_factors_int = np.abs(self.struct_factors) ** 2

        if return_intensities:
            q_SF = np.linspace(0, self.k_max, 250)
            I_SF = np.zeros_like(q_SF)
            for i in range(self.g_vec_leng.shape[0]):
                idx = np.argmin(np.abs(q_SF - self.g_vec_leng[i]))
                I_SF[idx] += self.struct_factors_int[i]
            I_SF = I_SF / np.max(I_SF)

            return (q_SF, I_SF)

    def generate_diffraction_pattern(
        self,
        zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
        foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
        proj_x_axis: Optional[Union[list, tuple, np.ndarray]] = None,
        sigma_excitation_error: float = 0.02,
        tol_excitation_error_mult: float = 3,
        tol_intensity: float = 0.001,
        k_max: float = None,
    ):
        """
        Generate a single diffraction pattern, return all peaks as a pointlist.

        Args:
            zone_axis (np float vector):     3 element projection direction for sim pattern
                                             Can also be a 3x3 orientation matrix (zone axis 3rd column)
            foil_normal:                     3 element foil normal - set to None to use zone_axis
            proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)
            sigma_excitation_error (float): sigma value for envelope applied to s_g (excitation errors) in units of inverse Angstroms
            tol_excitation_error_mult (float): tolerance in units of sigma for s_g inclusion
            tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots
            k_max (np float):                maximum scattering angle to keep in pattern

        Returns:
            bragg_peaks (PointList):         list of all Bragg peaks with fields [qx, qy, intensity, h, k, l]
        """

        zone_axis = np.asarray(zone_axis, dtype="float")

        if zone_axis.ndim == 1:
            zone_axis = np.asarray(zone_axis)
            zone_axis = zone_axis / np.linalg.norm(zone_axis)

            if not self.cartesian_directions:
                zone_axis = self.cartesian_to_crystal(zone_axis)

            if proj_x_axis is None:
                if np.all(np.abs(zone_axis) == np.array([1.0, 0.0, 0.0])):
                    v0 = np.array([0.0, -1.0, 0.0])
                else:
                    v0 = np.array([-1.0, 0.0, 0.0])
                proj_x_axis = np.cross(zone_axis, v0)
            else:
                proj_x_axis = np.asarray(proj_x_axis, dtype="float")
                if not self.cartesian_directions:
                    proj_x_axis = self.crystal_to_cartesian(proj_x_axis)

        elif zone_axis.shape == (3, 3):
            proj_x_axis = zone_axis[:, 0]
            zone_axis = zone_axis[:, 2]
        else:
            proj_x_axis = zone_axis[:, 0, 0]
            zone_axis = zone_axis[:, 2, 0]

        # Set x and y projection vectors
        ky_proj = np.cross(zone_axis, proj_x_axis)
        kx_proj = np.cross(ky_proj, zone_axis)

        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)

        # Foil normal vector
        if foil_normal is None:
            foil_normal = zone_axis
        else:
            foil_normal = np.asarray(foil_normal, dtype="float")
            if not self.cartesian_directions:
                foil_normal = self.crystal_to_cartesian(foil_normal)
            else:
                foil_normal = foil_normal / np.linalg.norm(foil_normal)

        # if proj_x_axis is None:
        #     if np.all(zone_axis == np.array([-1, 0, 0])):
        #         proj_x_axis = np.array([0, -1, 0])
        #     elif np.all(zone_axis == np.array([1, 0, 0])):
        #         proj_x_axis = np.array([0, 1, 0])
        #     else:
        #         proj_x_axis = np.array([-1, 0, 0])

        # # Logic to set x axis for projected images
        # Generate 2 zone_axis
        # if np.all(zone_axis  == np.array([1.0,0.0,0.0])):
        #     v0 = np.array([0.0,-1.0,0.0])
        # else:
        #     v0 = np.array([-1.0,0.0,0.0])
        # kx_proj = np.cross(zone_axis, v0)
        # ky_proj = np.cross(kx_proj, zone_axis)
        # kx_proj = kx_proj / np.linalg.norm(kx_proj)
        # ky_proj = ky_proj / np.linalg.norm(ky_proj)

        # wavevector
        zone_axis_norm = zone_axis / np.linalg.norm(zone_axis)
        k0 = zone_axis_norm / self.wavelength

        # Excitation errors
        cos_alpha = np.sum(
            (k0[:, None] + self.g_vec_all) * foil_normal[:, None], axis=0
        ) / np.linalg.norm(k0[:, None] + self.g_vec_all, axis=0)
        sg = (
            (-0.5)
            * np.sum((2 * k0[:, None] + self.g_vec_all) * self.g_vec_all, axis=0)
            / (np.linalg.norm(k0[:, None] + self.g_vec_all, axis=0))
            / cos_alpha
        )

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = np.abs(sg) <= sg_max
        g_diff = self.g_vec_all[:, keep]

        # Diffracted peak intensities and labels
        g_int = self.struct_factors_int[keep] * np.exp(
            sg[keep] ** 2 / (-2 * sigma_excitation_error ** 2)
        )
        hkl = self.hkl[:, keep]

        # Intensity tolerance
        keep_int = g_int > tol_intensity

        # Diffracted peak locations
        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)
        gx_proj = np.sum(g_diff * kx_proj[:, None], axis=0)
        gy_proj = np.sum(g_diff * ky_proj[:, None], axis=0)

        if k_max is not None:
            keep_kmax = np.hypot(gx_proj, gy_proj) < k_max
            keep_int = np.logical_and(keep_int, keep_kmax)

        gx_proj = gx_proj[keep_int]
        gy_proj = gy_proj[keep_int]

        # Diffracted peak labels
        h = hkl[0, keep_int]
        k = hkl[1, keep_int]
        l = hkl[2, keep_int]

        # Output as PointList
        bragg_peaks = PointList(
            [
                ("qx", "float64"),
                ("qy", "float64"),
                ("intensity", "float64"),
                ("h", "int"),
                ("k", "int"),
                ("l", "int"),
            ]
        )
        if np.any(keep_int):
            bragg_peaks.add_pointarray(
                np.vstack((gx_proj, gy_proj, g_int[keep_int], h, k, l)).T
            )

        return bragg_peaks

    def cartesian_to_crystal(self, zone_axis):
        vec_cart = zone_axis @ self.lat_real
        return vec_cart / np.linalg.norm(vec_cart)

    def crystal_to_cartesian(self, vec_cart):
        zone_axis = vec_cart @ np.linalg.inv(self.lat_real)
        return zone_axis / np.linalg.norm(zone_axis)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)
