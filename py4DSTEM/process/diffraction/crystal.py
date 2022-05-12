# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, single_atom_scatter, electron_wavelength_angstrom

from .crystal_viz import plot_diffraction_pattern
from .utils import Orientation


class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    """

    # Various methods for the Crystal class are implemented in a separate file. This
    # import statement inside the class declaration imports them as methods of the class!
    # (see https://stackoverflow.com/a/47562412)

    # Automated Crystal Orientation Mapping is implemented in crystal_ACOM.py
    from .crystal_ACOM import (
        orientation_plan, 
        match_orientations, 
        match_single_pattern,
        save_ang_file,
        symmetry_reduce_directions,
    )

    from .crystal_viz import (
        plot_structure,
        plot_structure_factors,
        plot_orientation_zones,
        plot_orientation_plan,
        plot_orientation_maps,
        plot_fiber_orientation_maps,
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
        alpha = np.deg2rad(self.cell[3])
        beta = np.deg2rad(self.cell[4])
        gamma = np.deg2rad(self.cell[5])
        f = np.cos(beta) * np.cos(gamma) - np.cos(alpha)
        vol = a*b*c*np.sqrt(1 \
            + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) \
            - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2)
        self.lat_real = np.array(
            [
                [a,               0,                 0],
                [b*np.cos(gamma), b*np.sin(gamma),   0],
                [c*np.cos(beta), -c*f/np.sin(gamma), vol/(a*b*np.sin(gamma))],
            ]
        )

        # Inverse lattice, metric tensors
        self.metric_real = self.lat_real @ self.lat_real.T
        self.metric_inv = np.linalg.inv(self.metric_real)
        self.lat_inv = self.metric_inv @ self.lat_real

        # pymatgen flag
        self.pymatgen_available = False
        

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
        MP_key=None,
        conventional_standard_structure=True,
    ):
        """
        Create a Crystal object from a pymatgen Structure object.
        If a Materials Project API key is installed, you may pass
        the Materials Project ID of a structure, which will be
        fetched through the MP API. For setup information see:
        https://pymatgen.org/usage.html#setting-the-pmg-mapi-key-in-the-config-file.
        Alternatively, Materials Porject API key can be pass as an argument through
        the function (MP_key). To get your API key, please visit Materials Project website
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
            MP_key:         (str) Materials Project API key
            conventional_standard_structure: (bool) if True, conventional standard unit cell will be returned
                            instead of the primitive unit cell pymatgen returns

        """
        import pymatgen as mg
        from pymatgen.ext.matproj import MPRester

        if structure is not None:
            if isinstance(structure, str):
                mpr = MPRester(MP_key)
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
            mpr = MPRester(MP_key)
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
        self, accelerating_voltage: float
    ):
        """
        Set up attributes used for diffraction calculations without going
        through the full ACOM pipeline.
        """
        self.accel_voltage = accelerating_voltage
        self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

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

        # Find shortest lattice vector direction
        k_test = np.vstack(
            [
                self.lat_inv[0, :],
                self.lat_inv[1, :],
                self.lat_inv[2, :],
                self.lat_inv[0, :] + self.lat_inv[1, :],
                self.lat_inv[0, :] + self.lat_inv[2, :],
                self.lat_inv[1, :] + self.lat_inv[2, :],
                self.lat_inv[0, :] + self.lat_inv[1, :] + self.lat_inv[2, :],
                self.lat_inv[0, :] - self.lat_inv[1, :] + self.lat_inv[2, :],
                self.lat_inv[0, :] + self.lat_inv[1, :] - self.lat_inv[2, :],
                self.lat_inv[0, :] - self.lat_inv[1, :] - self.lat_inv[2, :],
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
        # g_vec_all = self.lat_inv @ hkl
        g_vec_all =  (hkl.T @ self.lat_inv).T

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
        orientation: Optional[Orientation] = None,
        ind_orientation: Optional[int] = 0,
        orientation_matrix: Optional[np.ndarray] = None,
        zone_axis_lattice: Optional[np.ndarray] = None,
        proj_x_lattice: Optional[np.ndarray] = None,
        foil_normal_lattice: Optional[Union[list, tuple, np.ndarray]] = None,
        zone_axis_cartesian: Optional[np.ndarray] = None,
        proj_x_cartesian: Optional[np.ndarray] = None,
        foil_normal_cartesian: Optional[Union[list, tuple, np.ndarray]] = None,
        sigma_excitation_error: float = 0.02,
        tol_excitation_error_mult: float = 3,
        tol_intensity: float = 1e-4,
        k_max: Optional[float] = None,
        keep_qz = False,
        return_orientation_matrix=False,
    ):
        """
        Generate a single diffraction pattern, return all peaks as a pointlist.

        Args:
            orientation (Orientation):       an Orientation class object 
            ind_orientation                  If input is an Orientation class object with multiple orientations,
                                             this input can be used to select a specific orientation.
            
            orientation_matrix (array):      (3,3) orientation matrix, where columns represent projection directions.
            zone_axis_lattice (array):        (3,) projection direction in lattice indices
            proj_x_lattice (array):           (3,) x-axis direction in lattice indices
            zone_axis_cartesian (array):     (3,) cartesian projection direction
            proj_x_cartesian (array):        (3,) cartesian projection direction

            foil_normal:                     3 element foil normal - set to None to use zone_axis
            proj_x_axis (np float vector):   3 element vector defining image x axis (vertical)
            accel_voltage (float):           Accelerating voltage in Volts. If not specified,
                                             we check to see if crystal already has voltage specified.
            sigma_excitation_error (float):  sigma value for envelope applied to s_g (excitation errors) in units of inverse Angstroms
            tol_excitation_error_mult (float): tolerance in units of sigma for s_g inclusion
            tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots
            keep_qz (bool):                  Flag to return out-of-plane diffraction vectors
            return_orientation_matrix (bool): Return the orientation matrix

        Returns:
            bragg_peaks (PointList):         list of all Bragg peaks with fields [qx, qy, intensity, h, k, l]
            orientation_matrix (array):      3x3 orientation matrix (optional)
        """

        # Tolerance for angular tests
        tol = 1e-6

        # Parse orientation inputs
        if orientation is not None:
            if ind_orientation is None:
                orientation_matrix = orientation.matrix[0]
            else:
                orientation_matrix = orientation.matrix[ind_orientation]
        elif orientation_matrix is None:
            orientation_matrix = self.parse_orientation(
                zone_axis_lattice,
                proj_x_lattice,
                zone_axis_cartesian,
                proj_x_cartesian)

        # Get foil normal direction
        if foil_normal_lattice is not None:
            foil_normal = self.lattice_to_cartesian(np.array(foil_normal_lattice))
        elif foil_normal_cartesian is not None:
            foil_normal = np.array(foil_normal_cartesian)
        else:
            foil_normal = None
            # foil_normal = orientation_matrix[:,2]

        # Rotate crystal into desired projection
        g = orientation_matrix.T @ self.g_vec_all


        # Calculate excitation errors
        if foil_normal is None:
            sg = self.excitation_errors(g)
        else:
            foil_normal = (orientation_matrix.T \
                @ (-1*foil_normal[:,None]/np.linalg.norm(foil_normal))).ravel()
            sg = self.excitation_errors(g, foil_normal)

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = np.abs(sg) <= sg_max

        # Maximum scattering angle cutoff
        if k_max is not None:
            keep_kmax = np.linalg.norm(g,axis=0) <= k_max
            keep = np.logical_and(keep, keep_kmax)

        g_diff = g[:, keep]

        # Diffracted peak intensities and labels
        g_int = self.struct_factors_int[keep] * np.exp(
            (sg[keep] ** 2) / (-2 * sigma_excitation_error ** 2)
        )
        hkl = self.hkl[:, keep]

        # Intensity tolerance
        keep_int = g_int > tol_intensity

        # Output peaks
        gx_proj = g_diff[0,keep_int]
        gy_proj = g_diff[1,keep_int]

        # Diffracted peak labels
        h = hkl[0, keep_int]
        k = hkl[1, keep_int]
        l = hkl[2, keep_int]

        # Output as PointList
        if keep_qz:
            gz_proj = g_diff[2,keep_int]
            bragg_peaks = PointList(
                [
                    ("qx", "float64"),
                    ("qy", "float64"),
                    ("qz", "float64"),
                    ("intensity", "float64"),
                    ("h", "int"),
                    ("k", "int"),
                    ("l", "int"),
                ]
            )
            if np.any(keep_int):
                bragg_peaks.add_pointarray(
                    np.vstack((
                        gx_proj,
                        gy_proj,
                        gz_proj,
                        g_int[keep_int],
                        h,
                        k,
                        l)).T
                )
        else:
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
                    np.vstack((
                        gx_proj,
                        gy_proj,
                        g_int[keep_int],
                        h,
                        k,
                        l)).T
                )

        if return_orientation_matrix:
            return bragg_peaks, orientation_matrix
        else:
            return bragg_peaks



    # Vector conversions and other utilities for Crystal classes

    def cartesian_to_lattice(self, vec_cartesian):
        vec_lattice = self.lat_inv @ vec_cartesian
        return vec_lattice / np.linalg.norm(vec_lattice)

    def lattice_to_cartesian(self, vec_lattice):
        vec_cartesian = self.lat_real.T @ vec_lattice
        return vec_cartesian / np.linalg.norm(vec_cartesian)

    def hexagonal_to_lattice(self, vec_hexagonal):
        return np.array([
            2.0*vec_hexagonal[0] + vec_hexagonal[1],
            2.0*vec_hexagonal[1] + vec_hexagonal[0] ,
            vec_hexagonal[3]
            ])

    def lattice_to_hexagonal(self, vec_lattice):
        return np.array([
            (2.0*vec_lattice[0] - vec_lattice[1])/3.0,
            (2.0*vec_lattice[1] - vec_lattice[0])/3.0,
            (-vec_lattice[0] - vec_lattice[1])/3.0,
            vec_lattice[2]
            ])

    def cartesian_to_miller(self, vec_cartesian):
        vec_miller = self.lat_real.T @ self.metric_inv @ vec_cartesian 
        return vec_miller / np.linalg.norm(vec_miller)

    def miller_to_cartesian(self, vec_miller):
        vec_cartesian = self.lat_inv.T @ self.metric_real @ vec_miller
        return vec_cartesian / np.linalg.norm(vec_cartesian)

    def rational_ind(
        self, 
        vec,
        tol_den = 1000,
        ):
        # This function rationalizes the indices of a vector, up to 
        # some tolerance. Returns integers to prevent rounding errors.
        vec = np.array(vec,dtype='float64')
        sub = np.abs(vec) > 0
        if np.sum(sub) > 0:
            for ind in np.argwhere(sub):
                frac = Fraction(vec[ind[0]]).limit_denominator(tol_den)
                vec *= frac.denominator
            vec /= np.gcd.reduce(np.abs(vec[sub]).astype('int'))
        return vec.astype('int')

    def parse_orientation(
        self,
        zone_axis_lattice=None,
        proj_x_lattice=None,
        zone_axis_cartesian=None,
        proj_x_cartesian=None,
        ):
        # This helper function parse the various types of orientation inputs,
        # and returns the normalized, projected (x,y,z) cartesian vectors in
        # the form of an orientation matrix.

        if zone_axis_lattice is not None:
            proj_z = np.array(zone_axis_lattice)
            if proj_z.shape[0] == 4:
                proj_z = self.hexagonal_to_lattice(proj_z)
            proj_z = self.lattice_to_cartesian(proj_z)
        elif zone_axis_cartesian is not None:
            proj_z = np.array(zone_axis_cartesian)
        else:
            proj_z = np.array([0,0,1])

        if proj_x_lattice is not None:
            proj_x = np.array(proj_x_lattice)
            if proj_x.shape[0] == 4:
                proj_x = self.hexagonal_to_lattice(proj_x)
            proj_x = self.lattice_to_cartesian(proj_x)
        elif proj_x_cartesian is not None:
            proj_x = np.array(proj_x_cartesian)
        else:
            if np.abs(proj_z[2]) > 1-1e-6:
                proj_x = np.cross(np.array([0,1,0]),proj_z)
            else:
                proj_x = np.array([0,0,-1])

        # Generate orthogonal coordinate system, normalize
        proj_y = np.cross(proj_z, proj_x)
        proj_x = np.cross(proj_y, proj_z)
        proj_x = proj_x / np.linalg.norm(proj_x)
        proj_y = proj_y / np.linalg.norm(proj_y)
        proj_z = proj_z / np.linalg.norm(proj_z)

        return np.vstack((proj_x, proj_y, proj_z)).T

    def excitation_errors(
        self,
        g,
        foil_normal=None,
        ):
        '''
        Calculate the excitation errors, assuming k0 = [0, 0, -1/lambda].
        If foil normal is not specified, we assume it is [0,0,-1].
        '''
        if foil_normal is None:
            return (2*g[2,:] - self.wavelength*np.sum(g*g,axis=0)) \
                / (2 - 2*self.wavelength*g[2,:]) 
        else:
            return (2*g[2,:] - self.wavelength*np.sum(g*g,axis=0)) \
                / (2*self.wavelength*np.sum(g*foil_normal[:,None],axis=0) - 2*foil_normal[2])






