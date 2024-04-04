# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from fractions import Fraction
from typing import Union, Optional
import sys
import warnings

from emdfile import PointList
from py4DSTEM.process.utils import single_atom_scatter, electron_wavelength_angstrom

from py4DSTEM.process.diffraction.utils import Orientation


class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    """

    # Various methods for the Crystal class are implemented in a separate file. This
    # import statement inside the class declaration imports them as methods of the class!
    # (see https://stackoverflow.com/a/47562412)

    # Automated Crystal Orientation Mapping is implemented in crystal_ACOM.py
    from py4DSTEM.process.diffraction.crystal_ACOM import (
        orientation_plan,
        match_orientations,
        match_single_pattern,
        cluster_grains,
        cluster_orientation_map,
        calculate_strain,
        save_ang_file,
        symmetry_reduce_directions,
        orientation_map_to_orix_CrystalMap,
        save_ang_file,
    )

    from py4DSTEM.process.diffraction.crystal_viz import (
        plot_structure,
        plot_structure_factors,
        plot_scattering_intensity,
        plot_orientation_zones,
        plot_orientation_plan,
        plot_orientation_maps,
        plot_fiber_orientation_maps,
        plot_clusters,
        plot_cluster_size,
    )

    from py4DSTEM.process.diffraction.crystal_calibrate import (
        calibrate_pixel_size,
        calibrate_unit_cell,
    )

    # Dynamical diffraction calculations are implemented in crystal_bloch.py
    from py4DSTEM.process.diffraction.crystal_bloch import (
        generate_dynamical_diffraction_pattern,
        generate_CBED,
        calculate_dynamical_structure_factors,
    )

    def __init__(
        self,
        positions,
        numbers,
        cell,
        occupancy=None,
    ):
        """
        Args:
            positions (np.array): fractional coordinates of each atom in the cell
            numbers (np.array): Z number for each atom in the cell, if one number passed it is used for all atom positions
            cell (np.array): specify the unit cell, using a variable number of parameters
                1 number: the lattice parameter for a cubic cell
                3 numbers: the three lattice parameters for an orthorhombic cell
                6 numbers: the a,b,c lattice parameters and ɑ,β,ɣ angles for any cell
                3x3 array: row vectors containing the (u,v,w) lattice vectors.
            occupancy (np.array): Partial occupancy values for each atomic site. Must match the length of positions
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

        # unit cell, as one of:
        # [a a a 90 90 90]
        # [a b c 90 90 90]
        # [a b c alpha beta gamma]
        cell = np.asarray(cell, dtype="float_")
        if np.size(cell) == 1:
            self.cell = np.hstack([cell, cell, cell, 90, 90, 90])
        elif np.size(cell) == 3:
            self.cell = np.hstack([cell, 90, 90, 90])
        elif np.size(cell) == 6:
            self.cell = cell
        elif np.shape(cell)[0] == 3 and np.shape(cell)[1] == 3:
            self.lat_real = np.array(cell)
            a = np.linalg.norm(self.lat_real[0, :])
            b = np.linalg.norm(self.lat_real[1, :])
            c = np.linalg.norm(self.lat_real[2, :])
            alpha = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.sum(self.lat_real[1, :] * self.lat_real[2, :]) / b / c, -1, 1
                    )
                )
            )
            beta = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.sum(self.lat_real[0, :] * self.lat_real[2, :]) / a / c, -1, 1
                    )
                )
            )
            gamma = np.rad2deg(
                np.arccos(
                    np.clip(
                        np.sum(self.lat_real[0, :] * self.lat_real[1, :]) / a / b, -1, 1
                    )
                )
            )
            self.cell = (a, b, c, alpha, beta, gamma)
        else:
            raise Exception("Cell cannot contain " + np.size(cell) + " entries")

        # occupancy
        if occupancy is not None:
            self.occupancy = np.array(occupancy)
            # check the occupancy shape makes sense
            if self.occupancy.shape[0] != self.positions.shape[0]:
                raise Warning(
                    f"Number of occupancies ({self.occupancy.shape[0]}) and atomic positions ({self.positions.shape[0]}) do not match"
                )
        else:
            self.occupancy = np.ones(self.positions.shape[0], dtype=np.float32)

        # pymatgen flag
        if "pymatgen" in sys.modules:
            self.pymatgen_available = True
        else:
            self.pymatgen_available = False
        # Calculate lattice parameters
        self.calculate_lattice()

    def calculate_lattice(self):
        if not hasattr(self, "lat_real"):
            # calculate unit cell lattice vectors
            a = self.cell[0]
            b = self.cell[1]
            c = self.cell[2]
            alpha = np.deg2rad(self.cell[3])
            beta = np.deg2rad(self.cell[4])
            gamma = np.deg2rad(self.cell[5])
            f = np.cos(beta) * np.cos(gamma) - np.cos(alpha)
            vol = (
                a
                * b
                * c
                * np.sqrt(
                    1
                    + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                    - np.cos(alpha) ** 2
                    - np.cos(beta) ** 2
                    - np.cos(gamma) ** 2
                )
            )
            self.lat_real = np.array(
                [
                    [a, 0, 0],
                    [b * np.cos(gamma), b * np.sin(gamma), 0],
                    [
                        c * np.cos(beta),
                        -c * f / np.sin(gamma),
                        vol / (a * b * np.sin(gamma)),
                    ],
                ]
            )

        # Inverse lattice, metric tensors
        self.metric_real = self.lat_real @ self.lat_real.T
        self.metric_inv = np.linalg.inv(self.metric_real)
        self.lat_inv = self.metric_inv @ self.lat_real

    def get_strained_crystal(
        self,
        exx=0.0,
        eyy=0.0,
        ezz=0.0,
        exy=0.0,
        exz=0.0,
        eyz=0.0,
        deformation_matrix=None,
        return_deformation_matrix=False,
    ):
        """
        This method returns new Crystal class with strain applied. The directions of (x,y,z)
        are with respect to the default Crystal orientation, which can be checked with
        print(Crystal.lat_real) applied to the original Crystal.

        Strains are given in fractional values, so exx = 0.01 is 1% strain along the x direction.
        Deformation matrix should be of the form:
            deformation_matrix = np.array([
                [1.0+exx,   1.0*exy,    1.0*exz],
                [1.0*exy,   1.0+eyy,    1.0*eyz],
                [1.0*exz,   1.0*eyz,    1.0+ezz],
            ])

        Parameters
        --------

        exx (float):
            fractional strain along the xx direction
        eyy (float):
            fractional strain along the yy direction
        ezz (float):
            fractional strain along the zz direction
        exy (float):
            fractional strain along the xy direction
        exz (float):
            fractional strain along the xz direction
        eyz (float):
            fractional strain along the yz direction
        deformation_matrix (np.ndarray):
            3x3 array describing deformation matrix
        return_deformation_matrix (bool):
            boolean switch to return deformation matrix

        Returns
        --------
        return_deformation_matrix == False:
            strained_crystal (py4DSTEM.Crystal)
        return_deformation_matrix == True:
            (strained_crystal, deformation_matrix)
        """

        # deformation matrix
        if deformation_matrix is None:
            deformation_matrix = np.array(
                [
                    [1.0 + exx, 1.0 * exy, 1.0 * exz],
                    [1.0 * exy, 1.0 + eyy, 1.0 * eyz],
                    [1.0 * exz, 1.0 * eyz, 1.0 + ezz],
                ]
            )

        # new unit cell
        lat_new = self.lat_real @ deformation_matrix

        # make new crystal class
        from py4DSTEM.process.diffraction import Crystal

        crystal_strained = Crystal(
            positions=self.positions.copy(),
            numbers=self.numbers.copy(),
            cell=lat_new,
        )

        if return_deformation_matrix:
            return crystal_strained, deformation_matrix
        else:
            return crystal_strained

    @staticmethod
    def from_ase(
        atoms,
    ):
        """
        Create a py4DSTEM Crystal object from an ASE atoms object

        Args:
            atoms (ase.Atoms): an ASE atoms object

        """
        # get the occupancies from the atoms object
        occupancies = (
            atoms.arrays["occupancies"]
            if "occupancies" in atoms.arrays.keys()
            else None
        )

        if "occupancy" in atoms.info.keys():
            warnings.warn(
                "This Atoms object contains occupancy information but it will be ignored."
            )

        xtal = Crystal(
            positions=atoms.get_scaled_positions(),  # fractional coords
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            occupancy=occupancies,
        )
        return xtal

    @staticmethod
    def from_prismatic(filepath):
        """
        Create a py4DSTEM Crystal object from an prismatic style xyz co-ordinate file

        Args:
            filepath (str|Pathlib.Path): path to the prismatic format xyz file

        """

        from ase import io

        # read the atoms using ase
        atoms = io.read(filepath, format="prismatic")

        # get the occupancies from the atoms object
        occupancies = (
            atoms.arrays["occupancies"]
            if "occupancies" in atoms.arrays.keys()
            else None
        )
        xtal = Crystal(
            positions=atoms.get_scaled_positions(),  # fractional coords
            numbers=atoms.numbers,
            cell=atoms.cell.array,
            occupancy=occupancies,
        )
        return xtal

    @staticmethod
    def from_CIF(
        CIF, primitive: bool = True, conventional_standard_structure: bool = True
    ):
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

        structure = parser.get_structures(primitive=primitive)[0]

        return Crystal.from_pymatgen_structure(
            structure, conventional_standard_structure=conventional_standard_structure
        )

    @staticmethod
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

        if structure is not None:
            if isinstance(structure, str):
                from mp_api.client import MPRester

                with MPRester(MP_key) as mpr:
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
            from mp_api.client import MPRester

            with MPRester(MP_key) as mpr:
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
                        [
                            query[i]["structure"].lattice.volume
                            for i in range(len(query))
                        ]
                    )
                ]
                structure = (
                    mg.symmetry.analyzer.SpacegroupAnalyzer(
                        selected["structure"]
                    ).get_conventional_standard_structure()
                    if conventional_standard_structure
                    else selected["structure"]
                )

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

        site_data = np.array(
            [
                (*site.frac_coords, elem.number, comp)
                for site in structure
                for elem, comp in site.species.items()
            ]
        )
        positions = site_data[:, :3]
        numbers = site_data[:, 3]
        occupancies = site_data[:, 4]

        return Crystal(
            positions=positions, numbers=numbers, cell=cell, occupancy=occupancies
        )

    @staticmethod
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

    def setup_diffraction(self, accelerating_voltage: float):
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

        Parameters
        --------

        k_max: float
            max scattering vector to include (1/Angstroms)
        tol_structure_factor: float
            tolerance for removing low-valued structure factors
        return_intensities: bool
            return the intensities and positions of all structure factor peaks.

        Returns
        --------
        (q_SF, I_SF)
            Tuple of the q vectors and intensities of each structure factor.
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
        g_vec_all = (hkl.T @ self.lat_inv).T

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
            self.struct_factors += (
                f_all[:, a0]
                * self.occupancy[a0]
                * np.exp(
                    (2j * np.pi)
                    * np.sum(
                        self.hkl * np.expand_dims(self.positions[a0, :], axis=1), axis=0
                    )
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
        keep_qz=False,
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
            k_max (float):                   Maximum scattering vector
            keep_qz (bool):                  Flag to return out-of-plane diffraction vectors
            return_orientation_matrix (bool): Return the orientation matrix

        Returns:
            bragg_peaks (PointList):         list of all Bragg peaks with fields [qx, qy, intensity, h, k, l]
            orientation_matrix (array):      3x3 orientation matrix (optional)
        """

        if not (hasattr(self, "wavelength") and hasattr(self, "accel_voltage")):
            print("Accelerating voltage not set. Assuming 300 keV!")
            self.setup_diffraction(300e3)

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
                zone_axis_lattice, proj_x_lattice, zone_axis_cartesian, proj_x_cartesian
            )

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
            foil_normal = (
                orientation_matrix.T
                @ (-1 * foil_normal[:, None] / np.linalg.norm(foil_normal))
            ).ravel()
            sg = self.excitation_errors(g, foil_normal)

        # Threshold for inclusion in diffraction pattern
        sg_max = sigma_excitation_error * tol_excitation_error_mult
        keep = np.abs(sg) <= sg_max

        # Maximum scattering angle cutoff
        if k_max is not None:
            keep_kmax = np.linalg.norm(g, axis=0) <= k_max
            keep = np.logical_and(keep, keep_kmax)

        g_diff = g[:, keep]

        # Diffracted peak intensities and labels
        g_int = self.struct_factors_int[keep] * np.exp(
            (sg[keep] ** 2) / (-2 * sigma_excitation_error**2)
        )
        hkl = self.hkl[:, keep]

        # Intensity tolerance
        keep_int = g_int > tol_intensity

        # Output peaks
        gx_proj = g_diff[0, keep_int]
        gy_proj = g_diff[1, keep_int]

        # Diffracted peak labels
        h = hkl[0, keep_int]
        k = hkl[1, keep_int]
        l = hkl[2, keep_int]

        # Output as PointList
        if keep_qz:
            gz_proj = g_diff[2, keep_int]
            pl_dtype = np.dtype(
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
            bragg_peaks = PointList(np.array([], dtype=pl_dtype))
            if np.any(keep_int):
                bragg_peaks.add_data_by_field(
                    [gx_proj, gy_proj, gz_proj, g_int[keep_int], h, k, l]
                )
        else:
            pl_dtype = np.dtype(
                [
                    ("qx", "float64"),
                    ("qy", "float64"),
                    ("intensity", "float64"),
                    ("h", "int"),
                    ("k", "int"),
                    ("l", "int"),
                ]
            )
            bragg_peaks = PointList(np.array([], dtype=pl_dtype))
            if np.any(keep_int):
                bragg_peaks.add_data_by_field(
                    [gx_proj, gy_proj, g_int[keep_int], h, k, l]
                )

        if return_orientation_matrix:
            return bragg_peaks, orientation_matrix
        else:
            return bragg_peaks

    def generate_ring_pattern(
        self,
        k_max=2.0,
        use_bloch=False,
        thickness=None,
        bloch_params=None,
        orientation_plan_params=None,
        sigma_excitation_error=0.02,
        tol_intensity=1e-3,
        plot_rings=True,
        plot_params={},
        return_calc=True,
    ):
        """
        Calculate polycrystalline diffraction pattern from structure

        Args:
            k_max (float):                  Maximum scattering vector
            use_bloch (bool):               if true, use dynamic instead of kinematic approach
            thickness (float):              thickness in Ångström to evaluate diffraction patterns,
                                            only needed for dynamical calculations
            bloch_params (dict):            optional, parameters to calculate dynamical structure factor,
                                            see calculate_dynamical_structure_factors doc strings
            orientation_plan_params (dict): optional, parameters to calculate orientation plan,
                                            see orientation_plan doc strings
            sigma_excitation_error (float): sigma value for envelope applied to s_g (excitation errors)
                                            in units of inverse Angstroms
            tol_intensity (np float):       tolerance in intensity units for inclusion of diffraction spots
            plot_rings(bool):               if true, plot diffraction rings with plot_ring_pattern
            return_calc (bool):             return radii and intensities

        Returns:
            radii_unique (np array):        radii of ring pattern in units of scattering vector k
            intensity_unique (np array):    intensity of rings weighted by frequency of diffraciton spots
        """

        if use_bloch:
            assert (
                thickness is not None
            ), "provide thickness for dynamical diffraction calculation"
            assert hasattr(
                self, "Ug_dict"
            ), "run calculate_dynamical_structure_factors first"

        if not hasattr(self, "struct_factors"):
            self.calculate_structure_factors(
                k_max=k_max,
            )

        # check accelerating voltage
        if hasattr(self, "accel_voltage"):
            accelerating_voltage = self.accel_voltage
        else:
            self.accel_voltage = 300e3
            print("Accelerating voltage not set. Assuming 300 keV!")

        # check orientation plan
        if not hasattr(self, "orientation_vecs"):
            if orientation_plan_params is None:
                orientation_plan_params = {
                    "zone_axis_range": "auto",
                    "angle_step_zone_axis": 4,
                    "angle_step_in_plane": 4,
                }
            self.orientation_plan(
                **orientation_plan_params,
            )

        # calculate intensity and radius for rings
        radii = []
        intensity = []
        for a0 in range(self.orientation_vecs.shape[0]):
            if use_bloch:
                beams = self.generate_diffraction_pattern(
                    zone_axis_lattice=self.orientation_vecs[a0],
                    sigma_excitation_error=sigma_excitation_error,
                    tol_intensity=tol_intensity,
                    k_max=k_max,
                )
                pattern = self.generate_dynamical_diffraction_pattern(
                    beams=beams,
                    zone_axis_lattice=self.orientation_vecs[a0],
                    thickness=thickness,
                )
            else:
                pattern = self.generate_diffraction_pattern(
                    zone_axis_lattice=self.orientation_vecs[a0],
                    sigma_excitation_error=sigma_excitation_error,
                    tol_intensity=tol_intensity,
                    k_max=k_max,
                )

            intensity.append(pattern["intensity"])
            radii.append((pattern["qx"] ** 2 + pattern["qy"] ** 2) ** 0.5)

        intensity = np.concatenate(intensity)
        radii = np.concatenate(radii)

        radii_unique, idx, inv, cts = np.unique(
            radii, return_counts=True, return_index=True, return_inverse=True
        )
        intensity_unique = np.bincount(inv, weights=intensity)

        if plot_rings is True:
            from py4DSTEM.process.diffraction.crystal_viz import plot_ring_pattern

            plot_ring_pattern(radii_unique, intensity_unique, **plot_params)

        if return_calc is True:
            return radii_unique, intensity_unique

    def generate_projected_potential(
        self,
        im_size=(256, 256),
        pixel_size_angstroms=0.1,
        potential_radius_angstroms=3.0,
        sigma_image_blur_angstroms=0.1,
        thickness_angstroms=100,
        power_scale=1.0,
        plot_result=False,
        figsize=(6, 6),
        orientation: Optional[Orientation] = None,
        ind_orientation: Optional[int] = 0,
        orientation_matrix: Optional[np.ndarray] = None,
        zone_axis_lattice: Optional[np.ndarray] = None,
        proj_x_lattice: Optional[np.ndarray] = None,
        zone_axis_cartesian: Optional[np.ndarray] = None,
        proj_x_cartesian: Optional[np.ndarray] = None,
    ):
        """
        Generate an image of the projected potential of crystal in real space,
        using cell tiling, and a lookup table of the atomic potentials.
        Note that we round atomic positions to the nearest pixel for speed.

        TODO - fix scattering prefactor so that output units are sensible.

        Parameters
        ----------
        im_size: tuple, list, np.array
            (2,) vector specifying the output size in pixels.
        pixel_size_angstroms: float
            Pixel size in Angstroms.
        potential_radius_angstroms: float
            Radius in Angstroms for how far to integrate the atomic potentials
        sigma_image_blur_angstroms: float
            Image blurring in Angstroms.
        thickness_angstroms: float
            Thickness of the sample in Angstroms.
            Set thickness_thickness_angstroms = 0 to skip thickness projection.
        power_scale: float
            Power law scaling of potentials.  Set to 2.0 to approximate Z^2 images.
        plot_result: bool
            Plot the projected potential image.
        figsize:
            (2,) vector giving the size of the output.

        orientation: Orientation
            An Orientation class object
        ind_orientation: int
            If input is an Orientation class object with multiple orientations,
            this input can be used to select a specific orientation.
        orientation_matrix: array
            (3,3) orientation matrix, where columns represent projection directions.
        zone_axis_lattice: array
            (3,) projection direction in lattice indices
        proj_x_lattice: array)
            (3,) x-axis direction in lattice indices
        zone_axis_cartesian: array
            (3,) cartesian projection direction
        proj_x_cartesian: array
            (3,) cartesian projection direction

        Returns
        --------
        im_potential: (np.array)
            Output image of the projected potential.

        """

        # Determine image size in Angstroms
        im_size = np.array(im_size)
        im_size_Ang = im_size * pixel_size_angstroms

        # Parse orientation inputs
        if orientation is not None:
            if ind_orientation is None:
                orientation_matrix = orientation.matrix[0]
            else:
                orientation_matrix = orientation.matrix[ind_orientation]
        elif orientation_matrix is None:
            orientation_matrix = self.parse_orientation(
                zone_axis_lattice, proj_x_lattice, zone_axis_cartesian, proj_x_cartesian
            )

        # Rotate unit cell into projection direction
        lat_real = self.lat_real.copy() @ orientation_matrix

        # Determine unit cell axes to tile over, by selecting 2/3 with largest in-plane component
        inds_tile = np.argsort(np.linalg.norm(lat_real[:, 0:2], axis=1))[1:3]
        m_tile = lat_real[inds_tile, :]
        # Vector projected along optic axis
        m_proj = np.squeeze(np.delete(lat_real, inds_tile, axis=0))

        # Thickness
        if thickness_angstroms > 0:
            num_proj = np.round(thickness_angstroms / np.abs(m_proj[2])).astype("int")
            if num_proj > 1:
                vec_proj = m_proj[:2] / pixel_size_angstroms
                shifts = np.arange(num_proj).astype("float")
                shifts -= np.mean(shifts)
                x_proj = shifts * vec_proj[0]
                y_proj = shifts * vec_proj[1]
            else:
                num_proj = 1
        else:
            num_proj = 1

        # Determine tiling range
        if thickness_angstroms > 0:
            # include the cell height
            dz = m_proj[2] * num_proj * 0.5
            p_corners = np.array(
                [
                    [-im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, dz],
                    [im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, dz],
                    [-im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, dz],
                    [im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, dz],
                    [-im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, -dz],
                    [im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, -dz],
                    [-im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, -dz],
                    [im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, -dz],
                ]
            )
        else:
            p_corners = np.array(
                [
                    [-im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, 0.0],
                    [im_size_Ang[0] * 0.5, -im_size_Ang[1] * 0.5, 0.0],
                    [-im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, 0.0],
                    [im_size_Ang[0] * 0.5, im_size_Ang[1] * 0.5, 0.0],
                ]
            )

        ab = np.linalg.lstsq(m_tile[:, :2].T, p_corners[:, :2].T, rcond=None)[0]
        ab = np.floor(ab)
        a_range = np.array((np.min(ab[0]) - 1, np.max(ab[0]) + 2))
        b_range = np.array((np.min(ab[1]) - 1, np.max(ab[1]) + 2))

        # Tile unit cell
        a_ind, b_ind, atoms_ind = np.meshgrid(
            np.arange(a_range[0], a_range[1]),
            np.arange(b_range[0], b_range[1]),
            np.arange(self.positions.shape[0]),
        )
        abc_atoms = self.positions[atoms_ind.ravel(), :]
        abc_atoms[:, inds_tile[0]] += a_ind.ravel()
        abc_atoms[:, inds_tile[1]] += b_ind.ravel()
        xyz_atoms_ang = abc_atoms @ lat_real
        atoms_ID_all_0 = self.numbers[atoms_ind.ravel()]

        # Center atoms on image plane
        x0 = xyz_atoms_ang[:, 0] / pixel_size_angstroms + im_size[0] / 2.0
        y0 = xyz_atoms_ang[:, 1] / pixel_size_angstroms + im_size[1] / 2.0

        # if needed, tile atoms in the projection direction
        if num_proj > 1:
            x = (x0[:, None] + x_proj[None, :]).ravel()
            y = (y0[:, None] + y_proj[None, :]).ravel()
            atoms_ID_all = np.tile(atoms_ID_all_0, (num_proj, 1))
        else:
            x = x0
            y = y0
            atoms_ID_all = atoms_ID_all_0
        # print(x.shape, y.shape)

        # delete atoms outside the field of view
        bound = potential_radius_angstroms / pixel_size_angstroms
        atoms_del = np.logical_or.reduce(
            (
                x <= -bound,
                y <= -bound,
                x >= im_size[0] + bound,
                y >= im_size[1] + bound,
            )
        )
        x = np.delete(x, atoms_del)
        y = np.delete(y, atoms_del)
        atoms_ID_all = np.delete(atoms_ID_all, atoms_del)

        # Coordinate system for atomic projected potentials
        potential_radius = np.ceil(potential_radius_angstroms / pixel_size_angstroms)
        R = np.arange(0.5 - potential_radius, potential_radius + 0.5)
        R_ind = R.astype("int")
        R_2D = np.sqrt(R[:, None] ** 2 + R[None, :] ** 2)

        # Lookup table for atomic projected potentials
        atoms_ID = np.unique(self.numbers)
        atoms_lookup = np.zeros(
            (
                atoms_ID.shape[0],
                R_2D.shape[0],
                R_2D.shape[1],
            )
        )
        for a0 in range(atoms_ID.shape[0]):
            atom_sf = single_atom_scatter([atoms_ID[a0]])
            atoms_lookup[a0, :, :] = atom_sf.projected_potential(atoms_ID[a0], R_2D)

            # if needed, apply gaussian blurring to each atom
            if sigma_image_blur_angstroms > 0:
                atoms_lookup[a0, :, :] = gaussian_filter(
                    atoms_lookup[a0, :, :],
                    sigma_image_blur_angstroms / pixel_size_angstroms,
                    mode="nearest",
                )
        atoms_lookup **= power_scale

        # initialize potential
        im_potential = np.zeros(im_size)

        # Add atoms to potential image
        for a0 in range(atoms_ID_all.shape[0]):
            ind = np.argmin(np.abs(atoms_ID - atoms_ID_all[a0]))

            x_ind = np.round(x[a0]).astype("int") + R_ind
            y_ind = np.round(y[a0]).astype("int") + R_ind
            x_sub = np.logical_and(
                x_ind >= 0,
                x_ind < im_size[0],
            )
            y_sub = np.logical_and(
                y_ind >= 0,
                y_ind < im_size[1],
            )
            im_potential[x_ind[x_sub][:, None], y_ind[y_sub][None, :]] += atoms_lookup[
                ind
            ][x_sub][:, y_sub]

        if thickness_angstroms > 0:
            im_potential /= num_proj

        if plot_result:
            # quick plotting of the result
            int_vals = np.sort(im_potential.ravel())
            int_range = np.array(
                (
                    int_vals[np.round(0.02 * int_vals.size).astype("int")],
                    int_vals[np.round(0.999 * int_vals.size).astype("int")],
                )
            )

            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(
                im_potential,
                cmap="gray",
                vmin=int_range[0],
                vmax=int_range[1],
            )
            # ax.scatter(y,x,c='r')  # for testing
            ax.set_axis_off()
            ax.set_aspect("equal")

        return im_potential

    # Vector conversions and other utilities for Crystal classes
    def cartesian_to_lattice(self, vec_cartesian):
        vec_lattice = self.lat_inv @ vec_cartesian
        return vec_lattice / np.linalg.norm(vec_lattice)

    def lattice_to_cartesian(self, vec_lattice):
        vec_cartesian = self.lat_real.T @ vec_lattice
        return vec_cartesian / np.linalg.norm(vec_cartesian)

    def hexagonal_to_lattice(self, vec_hexagonal):
        return np.array(
            [
                2.0 * vec_hexagonal[0] + vec_hexagonal[1],
                2.0 * vec_hexagonal[1] + vec_hexagonal[0],
                vec_hexagonal[3],
            ]
        )

    def lattice_to_hexagonal(self, vec_lattice):
        return np.array(
            [
                (2.0 * vec_lattice[0] - vec_lattice[1]) / 3.0,
                (2.0 * vec_lattice[1] - vec_lattice[0]) / 3.0,
                (-vec_lattice[0] - vec_lattice[1]) / 3.0,
                vec_lattice[2],
            ]
        )

    def cartesian_to_miller(self, vec_cartesian):
        vec_miller = self.lat_real.T @ self.metric_inv @ vec_cartesian
        return vec_miller / np.linalg.norm(vec_miller)

    def miller_to_cartesian(self, vec_miller):
        vec_cartesian = self.lat_inv.T @ self.metric_real @ vec_miller
        return vec_cartesian / np.linalg.norm(vec_cartesian)

    def rational_ind(
        self,
        vec,
        tol_den=1000,
    ):
        # This function rationalizes the indices of a vector, up to
        # some tolerance. Returns integers to prevent rounding errors.
        vec = np.array(vec, dtype="float64")
        sub = np.abs(vec) > 0
        if np.sum(sub) > 0:
            for ind in np.argwhere(sub):
                frac = Fraction(vec[ind[0]]).limit_denominator(tol_den)
                vec *= np.round(frac.denominator)
            vec = np.round(
                vec / np.gcd.reduce(np.round(np.abs(vec[sub])).astype("int"))
            ).astype("int")

        return vec

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
            proj_z = np.array([0, 0, 1])

        if proj_x_lattice is not None:
            proj_x = np.array(proj_x_lattice)
            if proj_x.shape[0] == 4:
                proj_x = self.hexagonal_to_lattice(proj_x)
            proj_x = self.lattice_to_cartesian(proj_x)
        elif proj_x_cartesian is not None:
            proj_x = np.array(proj_x_cartesian)
        else:
            if np.abs(proj_z[2]) > 1 - 1e-6:
                proj_x = np.cross(np.array([0, 1, 0]), proj_z)
            else:
                proj_x = np.array([0, 0, -1])

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
        """
        Calculate the excitation errors, assuming k0 = [0, 0, -1/lambda].
        If foil normal is not specified, we assume it is [0,0,-1].
        """
        if foil_normal is None:
            return (2 * g[2, :] - self.wavelength * np.sum(g * g, axis=0)) / (
                2 - 2 * self.wavelength * g[2, :]
            )
        else:
            return (2 * g[2, :] - self.wavelength * np.sum(g * g, axis=0)) / (
                2 * self.wavelength * np.sum(g * foil_normal[:, None], axis=0)
                - 2 * foil_normal[2]
            )

    def calculate_bragg_peak_histogram(
        self,
        bragg_peaks,
        bragg_k_power=1.0,
        bragg_intensity_power=1.0,
        k_min=0.0,
        k_max=None,
        k_step=0.005,
    ):
        """
        Prepare experimental bragg peaks for lattice parameter or unit cell fitting.

        Args:
            bragg_peaks (BraggVectors):         Input Bragg vectors.
            bragg_k_power (float):              Input Bragg peak intensities are multiplied by k**bragg_k_power
                                                to change the weighting of longer scattering vectors
            bragg_intensity_power (float):      Input Bragg peak intensities are raised power **bragg_intensity_power.
            k_min (float):                      min k value for fitting range (Å^-1)
            k_max (float):                      max k value for fitting range (Å^-1)
            k_step (float):                     step size of k in fitting range (Å^-1)

        Returns:
            bragg_peaks_cali (BraggVectors):    Bragg vectors after calibration
            fig, ax (handles):                  Optional figure and axis handles, if returnfig=True.
        """

        # k coordinates
        if k_max is None:
            k_max = self.k_max
        k = np.arange(k_min, k_max + k_step, k_step)
        k_num = k.shape[0]

        # set rotate and ellipse based on their availability
        rotate = bragg_peaks.calibration.get_QR_rotation_degrees()
        ellipse = bragg_peaks.calibration.get_ellipse()
        rotate = False if rotate is None else True
        ellipse = False if ellipse is None else True

        # concatenate all peaks
        bigpl = np.concatenate(
            [
                bragg_peaks.get_vectors(
                    rx,
                    ry,
                    center=True,
                    ellipse=ellipse,
                    pixel=True,
                    rotate=rotate,
                ).data
                for rx in range(bragg_peaks.shape[0])
                for ry in range(bragg_peaks.shape[1])
            ]
        )
        qr = np.sqrt(bigpl["qx"] ** 2 + bigpl["qy"] ** 2)
        int_meas = bigpl["intensity"]

        # get discrete plot from structure factor amplitudes
        int_exp = np.zeros_like(k)
        k_px = (qr - k_min) / k_step
        kf = np.floor(k_px).astype("int")
        dk = k_px - kf

        sub = np.logical_and(kf >= 0, kf < k_num)
        int_exp = np.bincount(
            np.floor(k_px[sub]).astype("int"),
            weights=(1 - dk[sub]) * int_meas[sub],
            minlength=k_num,
        )
        sub = np.logical_and(k_px >= -1, k_px < k_num - 1)
        int_exp += np.bincount(
            np.floor(k_px[sub] + 1).astype("int"),
            weights=dk[sub] * int_meas[sub],
            minlength=k_num,
        )
        int_exp = (int_exp**bragg_intensity_power) * (k**bragg_k_power)
        int_exp /= np.max(int_exp)
        return k, int_exp


def generate_moire_diffraction_pattern(
    bragg_peaks_0,
    bragg_peaks_1,
    thresh_0=0.0002,
    thresh_1=0.0002,
    exx_1=0.0,
    eyy_1=0.0,
    exy_1=0.0,
    phi_1=0.0,
    power=2.0,
):
    """
    Calculate a Moire lattice from 2 parent diffraction patterns. The second lattice can be rotated
    and strained with respect to the original lattice. Note that this strain is applied in real space,
    and so the inverse of the calculated infinitestimal strain tensor is applied.

    Parameters
    --------
    bragg_peaks_0: BraggVector
        Bragg vectors for parent lattice 0.
    bragg_peaks_1: BraggVector
        Bragg vectors for parent lattice 1.
    thresh_0: float
        Intensity threshold for structure factors from lattice 0.
    thresh_1: float
        Intensity threshold for structure factors from lattice 1.
    exx_1: float
        Strain of lattice 1 in x direction (vertical) in real space.
    eyy_1: float
        Strain of lattice 1 in y direction (horizontal) in real space.
    exy_1: float
        Shear strain of lattice 1 in (x,y) direction (diagonal) in real space.
    phi_1: float
        Rotation of lattice 1 in real space.
    power: float
        Plotting power law (default is amplitude**2.0, i.e. intensity).

    Returns
    --------
    parent_peaks_0, parent_peaks_1, moire_peaks: BraggVectors
        Bragg vectors for the rotated & strained parent lattices
        and the moire lattice

    """

    # get intenties of all peaks
    int0 = bragg_peaks_0["intensity"] ** (power / 2.0)
    int1 = bragg_peaks_1["intensity"] ** (power / 2.0)

    # peaks above threshold
    sub0 = int0 >= thresh_0
    sub1 = int1 >= thresh_1

    # Remove origin (assuming brightest peak)
    ind0_or = np.argmax(bragg_peaks_0["intensity"])
    ind1_or = np.argmax(bragg_peaks_1["intensity"])
    sub0[ind0_or] = False
    sub1[ind1_or] = False
    int0_sub = int0[sub0]
    int1_sub = int1[sub1]

    # Get peaks
    qx0 = bragg_peaks_0["qx"][sub0]
    qy0 = bragg_peaks_0["qy"][sub0]
    qx1_init = bragg_peaks_1["qx"][sub1]
    qy1_init = bragg_peaks_1["qy"][sub1]

    # peak labels
    h0 = bragg_peaks_0["h"][sub0]
    k0 = bragg_peaks_0["k"][sub0]
    l0 = bragg_peaks_0["l"][sub0]
    h1 = bragg_peaks_1["h"][sub1]
    k1 = bragg_peaks_1["k"][sub1]
    l1 = bragg_peaks_1["l"][sub1]

    # apply strain tensor to lattice 1
    m = np.array(
        [
            [np.cos(phi_1), -np.sin(phi_1)],
            [np.sin(phi_1), np.cos(phi_1)],
        ]
    ) @ np.linalg.inv(
        np.array(
            [
                [1 + exx_1, exy_1 * 0.5],
                [exy_1 * 0.5, 1 + eyy_1],
            ]
        )
    )
    qx1 = m[0, 0] * qx1_init + m[0, 1] * qy1_init
    qy1 = m[1, 0] * qx1_init + m[1, 1] * qy1_init

    # Generate moire lattice
    ind0, ind1 = np.meshgrid(
        np.arange(np.sum(sub0)),
        np.arange(np.sum(sub1)),
        indexing="ij",
    )
    qx = qx0[ind0] + qx1[ind1]
    qy = qy0[ind0] + qy1[ind1]
    int_moire = (int0_sub[ind0] * int1_sub[ind1]) ** 0.5

    # moire labels
    m_h0 = h0[ind0]
    m_k0 = k0[ind0]
    m_l0 = l0[ind0]
    m_h1 = h1[ind1]
    m_k1 = k1[ind1]
    m_l1 = l1[ind1]

    # Convert thresholded and moire peaks to BraggVector class

    pl_dtype_parent = np.dtype(
        [
            ("qx", "float"),
            ("qy", "float"),
            ("intensity", "float"),
            ("h", "int"),
            ("k", "int"),
            ("l", "int"),
        ]
    )

    bragg_parent_0 = PointList(np.array([], dtype=pl_dtype_parent))
    bragg_parent_0.add_data_by_field(
        [
            qx0.ravel(),
            qy0.ravel(),
            int0_sub.ravel(),
            h0.ravel(),
            k0.ravel(),
            l0.ravel(),
        ]
    )

    bragg_parent_1 = PointList(np.array([], dtype=pl_dtype_parent))
    bragg_parent_1.add_data_by_field(
        [
            qx1.ravel(),
            qy1.ravel(),
            int1_sub.ravel(),
            h1.ravel(),
            k1.ravel(),
            l1.ravel(),
        ]
    )

    pl_dtype = np.dtype(
        [
            ("qx", "float"),
            ("qy", "float"),
            ("intensity", "float"),
            ("h0", "int"),
            ("k0", "int"),
            ("l0", "int"),
            ("h1", "int"),
            ("k1", "int"),
            ("l1", "int"),
        ]
    )
    bragg_moire = PointList(np.array([], dtype=pl_dtype))
    bragg_moire.add_data_by_field(
        [
            qx.ravel(),
            qy.ravel(),
            int_moire.ravel(),
            m_h0.ravel(),
            m_k0.ravel(),
            m_l0.ravel(),
            m_h1.ravel(),
            m_k1.ravel(),
            m_l1.ravel(),
        ]
    )

    return bragg_parent_0, bragg_parent_1, bragg_moire


def plot_moire_diffraction_pattern(
    bragg_parent_0,
    bragg_parent_1,
    bragg_moire,
    int_range=(0, 5e-3),
    k_max=1.0,
    plot_subpixel=True,
    labels=None,
    marker_size_parent=16,
    marker_size_moire=4,
    text_size_parent=10,
    text_size_moire=6,
    add_labels_parent=False,
    add_labels_moire=False,
    dist_labels=0.03,
    dist_check=0.06,
    sep_labels=0.03,
    figsize=(8, 6),
    returnfig=False,
):
    """
    Plot Moire lattice and parent lattices.

    Parameters
    --------
    bragg_peaks_0: BraggVector
        Bragg vectors for parent lattice 0.
    bragg_peaks_1: BraggVector
        Bragg vectors for parent lattice 1.
    bragg_moire: BraggVector
        Bragg vectors for moire lattice.
    int_range: (float, float)
        Plotting intensity range for the Moire peaks.
    k_max: float
        Max k value of the plotted Moire lattice.
    plot_subpixel: bool
        Apply subpixel corrections to the Bragg spot positions.
        Matplotlib default scatter plot rounds to the nearest pixel.
    labels: list
        List of text labels for parent lattices
    marker_size_parent: float
        Size of plot markers for the two parent lattices.
    marker_size_moire: float
        Size of plot markers for the Moire lattice.
    text_size_parent: float
        Label text size for parent lattice.
    text_size_moire: float
        Label text size for Moire lattice.
    add_labels_parent: bool
        Plot the parent lattice index labels.
    add_labels_moire: bool
        Plot the parent lattice index labels for the Moire spots.
    dist_labels: float
        Distance to move the labels off the spots.
    dist_check: float
        Set to some distance to "push" the labels away from each other if they are within this distance.
    sep_labels: float
        Separation distance for labels which are "pushed" apart.
    figsize: (float,float)
        Size of output figure.
    returnfig: bool
        Return the (fix,ax) handles of the plot.

    Returns
    --------
    fig, ax: matplotlib handles (optional)
        Figure and axes handles for the moire plot.
    """

    # peak labels

    if labels is None:
        labels = ("crystal 0", "crystal 1")

    def overline(x):
        return str(x) if x >= 0 else (r"\overline{" + str(np.abs(x)) + "}")

    # parent 1
    qx0 = bragg_parent_0["qx"]
    qy0 = bragg_parent_0["qy"]
    h0 = bragg_parent_0["h"]
    k0 = bragg_parent_0["k"]
    l0 = bragg_parent_0["l"]

    # parent 2
    qx1 = bragg_parent_1["qx"]
    qy1 = bragg_parent_1["qy"]
    h1 = bragg_parent_1["h"]
    k1 = bragg_parent_1["k"]
    l1 = bragg_parent_1["l"]

    # moire
    qx = bragg_moire["qx"]
    qy = bragg_moire["qy"]
    m_h0 = bragg_moire["h0"]
    m_k0 = bragg_moire["k0"]
    m_l0 = bragg_moire["l0"]
    m_h1 = bragg_moire["h1"]
    m_k1 = bragg_moire["k1"]
    m_l1 = bragg_moire["l1"]
    int_moire = bragg_moire["intensity"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.09, 0.09, 0.65, 0.9])
    ax_labels = fig.add_axes([0.75, 0, 0.25, 1])

    text_params_parent = {
        "ha": "center",
        "va": "center",
        "family": "sans-serif",
        "fontweight": "normal",
        "size": text_size_parent,
    }
    text_params_moire = {
        "ha": "center",
        "va": "center",
        "family": "sans-serif",
        "fontweight": "normal",
        "size": text_size_moire,
    }

    if plot_subpixel is False:
        # moire
        ax.scatter(
            qy,
            qx,
            # color = (0,0,0,1),
            c=int_moire,
            s=marker_size_moire,
            cmap="gray_r",
            vmin=int_range[0],
            vmax=int_range[1],
            antialiased=True,
        )

        # parent lattices
        ax.scatter(
            qy0,
            qx0,
            color=(1, 0, 0, 1),
            s=marker_size_parent,
            antialiased=True,
        )
        ax.scatter(
            qy1,
            qx1,
            color=(0, 0.7, 1, 1),
            s=marker_size_parent,
            antialiased=True,
        )

        # origin
        ax.scatter(
            0,
            0,
            color=(0, 0, 0, 1),
            s=marker_size_parent,
            antialiased=True,
        )

    else:
        # moire peaks
        int_all = np.clip(
            (int_moire - int_range[0]) / (int_range[1] - int_range[0]), 0, 1
        )
        keep = np.logical_and.reduce(
            (qx >= -k_max, qx <= k_max, qy >= -k_max, qy <= k_max)
        )
        for x, y, int_marker in zip(qx[keep], qy[keep], int_all[keep]):
            ax.add_artist(
                Circle(
                    xy=(y, x),
                    radius=np.sqrt(marker_size_moire) / 800.0,
                    color=(1 - int_marker, 1 - int_marker, 1 - int_marker),
                )
            )
        if add_labels_moire:
            for a0 in range(qx.size):
                if keep.ravel()[a0]:
                    x0 = qx.ravel()[a0]
                    y0 = qy.ravel()[a0]
                    d2 = (qx.ravel() - x0) ** 2 + (qy.ravel() - y0) ** 2
                    sub = d2 < dist_check**2
                    xc = np.mean(qx.ravel()[sub])
                    yc = np.mean(qy.ravel()[sub])
                    xp = x0 - xc
                    yp = y0 - yc
                    if xp == 0 and yp == 0.0:
                        xp = x0 - dist_labels
                        yp = y0
                    else:
                        leng = np.linalg.norm((xp, yp))
                        xp = x0 + xp * dist_labels / leng
                        yp = y0 + yp * dist_labels / leng

                    ax.text(
                        yp,
                        xp - sep_labels,
                        "$"
                        + overline(m_h0.ravel()[a0])
                        + overline(m_k0.ravel()[a0])
                        + overline(m_l0.ravel()[a0])
                        + "$",
                        c="r",
                        **text_params_moire,
                    )
                    ax.text(
                        yp,
                        xp,
                        "$"
                        + overline(m_h1.ravel()[a0])
                        + overline(m_k1.ravel()[a0])
                        + overline(m_l1.ravel()[a0])
                        + "$",
                        c=(0, 0.7, 1.0),
                        **text_params_moire,
                    )

        keep = np.logical_and.reduce(
            (qx0 >= -k_max, qx0 <= k_max, qy0 >= -k_max, qy0 <= k_max)
        )
        for x, y in zip(qx0[keep], qy0[keep]):
            ax.add_artist(
                Circle(
                    xy=(y, x),
                    radius=np.sqrt(marker_size_parent) / 800.0,
                    color=(1, 0, 0),
                )
            )
        if add_labels_parent:
            for a0 in range(qx0.size):
                if keep.ravel()[a0]:
                    xp = qx0.ravel()[a0] - dist_labels
                    yp = qy0.ravel()[a0]
                    ax.text(
                        yp,
                        xp,
                        "$"
                        + overline(h0.ravel()[a0])
                        + overline(k0.ravel()[a0])
                        + overline(l0.ravel()[a0])
                        + "$",
                        c="k",
                        **text_params_parent,
                    )

        keep = np.logical_and.reduce(
            (qx1 >= -k_max, qx1 <= k_max, qy1 >= -k_max, qy1 <= k_max)
        )
        for x, y in zip(qx1[keep], qy1[keep]):
            ax.add_artist(
                Circle(
                    xy=(y, x),
                    radius=np.sqrt(marker_size_parent) / 800.0,
                    color=(0, 0.7, 1),
                )
            )
        if add_labels_parent:
            for a0 in range(qx1.size):
                if keep.ravel()[a0]:
                    xp = qx1.ravel()[a0] - dist_labels
                    yp = qy1.ravel()[a0]
                    ax.text(
                        yp,
                        xp,
                        "$"
                        + overline(h1.ravel()[a0])
                        + overline(k1.ravel()[a0])
                        + overline(l1.ravel()[a0])
                        + "$",
                        c="k",
                        **text_params_parent,
                    )

        # origin
        ax.add_artist(
            Circle(
                xy=(0, 0),
                radius=np.sqrt(marker_size_parent) / 800.0,
                color=(0, 0, 0),
            )
        )

    ax.set_xlim((-k_max, k_max))
    ax.set_ylim((-k_max, k_max))
    ax.set_ylabel("$q_x$ (1/A)")
    ax.set_xlabel("$q_y$ (1/A)")
    ax.invert_yaxis()

    # labels
    ax_labels.scatter(
        0,
        0,
        color=(1, 0, 0, 1),
        s=marker_size_parent,
    )
    ax_labels.scatter(
        0,
        -1,
        color=(0, 0.7, 1, 1),
        s=marker_size_parent,
    )
    ax_labels.scatter(
        0,
        -2,
        color=(0, 0, 0, 1),
        s=marker_size_moire,
    )
    ax_labels.text(
        0.4,
        -0.2,
        labels[0],
        fontsize=14,
    )
    ax_labels.text(
        0.4,
        -1.2,
        labels[1],
        fontsize=14,
    )
    ax_labels.text(
        0.4,
        -2.2,
        "Moiré lattice",
        fontsize=14,
    )

    ax_labels.set_xlim((-1, 4))
    ax_labels.set_ylim((-21, 1))

    ax_labels.axis("off")

    if returnfig:
        return fig, ax
