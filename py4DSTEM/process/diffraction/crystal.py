# Functions for calculating diffraction patterns, matching them to experiments, and creating orientation and phase maps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D, art3d
import warnings
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, single_atom_scatter, electron_wavelength_angstrom


class Crystal:
    """
    A class storing a single crystal structure, and associated diffraction data.

    """

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
            self.numbers = np.ones(positions.shape[0], dtype="intp") * numbers
        elif np.size(numbers) == positions.shape[0]:
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

    def calculate_structure_factors(
        self,
        k_max: float = 2.0,
        tol_structure_factor: float = 1e-2,
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

    def plot_structure(
        self,
        proj_dir: Union[list, np.ndarray] = [3, 2, 1],
        size_marker: float = 400,
        tol_distance: float = 0.001,
        plot_limit: Optional[np.ndarray] = None,
        camera_dist: Optional[float] = None,
        show_axes: bool = False,
        figsize: Union[tuple, list, np.ndarray] = (8, 8),
        returnfig: bool = False,
    ):
        """
        Quick 3D plot of the untit cell /atomic structure.

        Args:
            proj_dir (float):           projection direction, either [elev azim] or normal vector
            scale_markers (float):      size scaling for markers
            tol_distance (float):       tolerance for repeating atoms on edges on cell boundaries
            plot_limit (float):         2x3 numpy array containing x y z plot min and max in columns.
                                        Default is 1.1* unit cell dimensions
            camera_dist (float):        Move camera closer to the plot (relative to matplotlib default of 10)
            show_axes (bool):           Whether to plot axes or not
            figsize (2 element float):  size scaling of figure axes
            returnfig (bool):           set to True to return figure and axes handles

        Returns:
            fig, ax                     (optional) figure and axes handles
        """

        # unit cell vectors
        u = self.lat_real[0, :]
        v = self.lat_real[1, :]
        w = self.lat_real[2, :]

        # atomic identities
        ID = self.numbers

        # Fractional atomic coordinates
        pos = self.positions
        # x tile
        sub = pos[:, 0] < tol_distance
        pos = np.vstack([pos, pos[sub, :] + np.array([1, 0, 0])])
        ID = np.hstack([ID, ID[sub]])
        # y tile
        sub = pos[:, 1] < tol_distance
        pos = np.vstack([pos, pos[sub, :] + np.array([0, 1, 0])])
        ID = np.hstack([ID, ID[sub]])
        # z tile
        sub = pos[:, 2] < tol_distance
        pos = np.vstack([pos, pos[sub, :] + np.array([0, 0, 1])])
        ID = np.hstack([ID, ID[sub]])

        # Cartesian atomic positions
        xyz = pos @ self.lat_real

        # projection direction of the plot
        if np.size(proj_dir) == 2:
            el = proj_dir[0]
            az = proj_dir[1]
        elif np.size(proj_dir) == 3:
            if proj_dir[0] == 0 and proj_dir[1] == 0:
                el = 90 * np.sign(proj_dir[2])
            else:
                el = (
                    np.arctan(
                        proj_dir[2] / np.sqrt(proj_dir[0] ** 2 + proj_dir[1] ** 2)
                    )
                    * 180
                    / np.pi
                )
            az = np.arctan2(proj_dir[1], proj_dir[0]) * 180 / np.pi
        else:
            raise Exception(
                "Projection direction cannot contain " + np.size(proj_dir) + " elements"
            )
        proj_dir = np.array(
            [
                np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180),
                np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180),
                np.sin(el * np.pi / 180),
            ]
        )

        # 3D plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d", elev=el, azim=az)

        # unit cell
        p = np.vstack([[0, 0, 0], u, u + v, v, w, u + w, u + v + w, v + w])
        p = p[:, [1, 0, 2]]  # Reorder cell boundaries

        f = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [2, 3, 7, 6],
                [0, 3, 7, 4],
                [1, 2, 6, 5],
            ]
        )

        # ax.plot3D(xline, yline, zline, 'gray')
        pc = art3d.Poly3DCollection(
            p[f],
            facecolors=[0, 0.7, 1],
            edgecolor=[0, 0, 0],
            linewidth=2,
            alpha=0.2,
        )
        ax.add_collection(pc)

        # atoms
        ID_all = np.unique(ID)
        for ID_plot in ID_all:
            sub = ID == ID_plot
            ax.scatter(
                xs=xyz[sub, 1],  # + d[0],
                ys=xyz[sub, 0],  # + d[1],
                zs=xyz[sub, 2],  # + d[2],
                s=size_marker,
                linewidth=2,
                color=atomic_colors(ID_plot),
                edgecolor=[0, 0, 0],
            )

        # plot limit
        if plot_limit is None:
            plot_limit = np.array(
                [
                    [np.min(p[:, 0]), np.min(p[:, 1]), np.min(p[:, 2])],
                    [np.max(p[:, 0]), np.max(p[:, 1]), np.max(p[:, 2])],
                ]
            )
            plot_limit = (plot_limit - np.mean(plot_limit, axis=0)) * 1.1 + np.mean(
                plot_limit, axis=0
            )

        ax.invert_yaxis()
        if show_axes is False:
            ax.set_axis_off()
        ax.axes.set_xlim3d(left=plot_limit[0, 1], right=plot_limit[1, 1])
        ax.axes.set_ylim3d(bottom=plot_limit[0, 0], top=plot_limit[1, 0])
        ax.axes.set_zlim3d(bottom=plot_limit[0, 2], top=plot_limit[1, 2])
        # ax.set_box_aspect((1, 1, 1))
        axisEqual3D(ax)

        if camera_dist is not None:
            ax.dist = camera_dist

        plt.show()

        if returnfig:
            return fig, ax

    def plot_structure_factors(
        self,
        proj_dir: Union[list, tuple, np.ndarray] = [10, 30],
        scale_markers: float = 1,
        plot_limit: Optional[Union[list, tuple, np.ndarray]] = None,
        figsize: Union[list, tuple, np.ndarray] = (8, 8),
        returnfig: bool = False,
    ):
        """
        3D scatter plot of the structure factors using magnitude^2, i.e. intensity.

        Args:
            dir_proj (float):           projection direction, either [elev azim] or normal vector
            scale_markers (float):      size scaling for markers
            plot_limit (float):         x y z plot limits, default is [-1 1]*self.k_max
            figsize (2 element float):  size scaling of figure axes
            returnfig (bool):           set to True to return figure and axes handles

        Returns:
            fig, ax                     (optional) figure and axes handles
        """

        if np.size(proj_dir) == 2:
            el = proj_dir[0]
            az = proj_dir[1]
        elif np.size(proj_dir) == 3:
            if hasattr(self, "cartesian_directions") and not self.cartesian_directions:
                proj_dir = self.crystal_to_cartesian(proj_dir)

            if proj_dir[0] == 0 and proj_dir[1] == 0:
                el = 90 * np.sign(proj_dir[2])
            else:
                el = (
                    np.arctan(
                        proj_dir[2] / np.sqrt(proj_dir[0] ** 2 + proj_dir[1] ** 2)
                    )
                    * 180
                    / np.pi
                )
            az = np.arctan2(proj_dir[1], proj_dir[0]) * 180 / np.pi
        else:
            raise Exception(
                "Projection direction cannot contain " + np.size(proj_dir) + " elements"
            )

        # 3D plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d", elev=el, azim=az)

        ax.scatter(
            xs=self.g_vec_all[0, :],
            ys=self.g_vec_all[1, :],
            zs=self.g_vec_all[2, :],
            s=scale_markers * self.struct_factors_int,
        )

        # axes limits
        if plot_limit is None:
            plot_limit = self.k_max * 1.05

        ax.axes.set_xlim3d(left=-plot_limit, right=plot_limit)
        ax.axes.set_ylim3d(bottom=-plot_limit, top=plot_limit)
        ax.axes.set_zlim3d(bottom=-plot_limit, top=plot_limit)
        ax.set_box_aspect((1, 1, 1))
        # ax.set_axis_off()
        # ax.setxticklabels([])
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        plt.show()

        if returnfig:
            return fig, ax

    def orientation_plan(
        self,
        zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
        angle_step_zone_axis: float = 2.0,
        angle_step_in_plane: float = 2.0,
        accel_voltage: float = 300e3,
        corr_kernel_size: float = 0.08,
        radial_power:float = 1.,
        intensity_power:float = 0.5,
        tol_peak_delete = None,
        tol_distance: float = 0.01,
        fiber_axis=None,
        fiber_angles=None,
        cartesian_directions=False,
        figsize: Union[list, tuple, np.ndarray] = (6, 6),
        progress_bar: bool = True
    ):
        # plot_corr_norm: bool = False,  # option removed due to new normalization

        """
        Calculate the rotation basis arrays for an SO(3) rotation correlogram.

        Args:
            zone_axis_range (float): Row vectors give the range for zone axis orientations.
                                     If user specifies 2 vectors (2x3 array), we start at [0,0,1]
                                        to make z-x-z rotation work.
                                     If user specifies 3 vectors (3x3 array), plan will span these vectors.
                                     Setting to 'full' as a string will use a hemispherical range.
                                     Setting to 'half' as a string will use a quarter sphere range.
                                     Setting to 'fiber' as a string will make a spherical cap around a given vector.
                                     Setting to 'auto' will use pymatgen to determine the point group symmetry
                                        of the structure and choose an appropriate zone_axis_range
            angle_step_zone_axis (float): Approximate angular step size for zone axis [degrees]
            angle_step_in_plane (float):  Approximate angular step size for in-plane rotation [degrees]
            accel_voltage (float):        Accelerating voltage for electrons [Volts]
            corr_kernel_size (float):        Correlation kernel size length in Angstroms
            radial_power (float):          Power for scaling the correlation intensity as a function of the peak radius
            intensity_power (float):       Power for scaling the correlation intensity as a function of the peak intensity
            tol_peak_delete (float):      Distance to delete peaks for multiple matches.
                                          Default is kernel_size * 0.5
            tol_distance (float):         Distance tolerance for radial shell assignment [1/Angstroms]
            fiber_axis (float):           (3,) vector specifying the fiber axis
            fiber_angles (float):         (2,) vector specifying angle range from fiber axis, and in-plane angular range [degrees]
            cartesian_directions (bool): When set to true, all zone axes and projection directions
                                         are specified in Cartesian directions.
            figsize (float):            (2,) vector giving the figure size
            progress_bar (bool):    If false no progress bar is displayed
        """

        # Store inputs
        self.accel_voltage = np.asarray(accel_voltage)
        self.orientation_kernel_size = np.asarray(corr_kernel_size)
        if tol_peak_delete is None:
            self.orientation_tol_peak_delete = self.orientation_kernel_size * 0.5
        else:
            self.orientation_tol_peak_delete = np.asarray(tol_peak_delete)
        self.orientation_fiber_axis = np.asarray(fiber_axis)
        self.orientation_fiber_angles = np.asarray(fiber_angles)
        self.cartesian_directions = cartesian_directions

        # Calculate wavelenth
        self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

        # store the radial and intensity scaling to use later for generating test patterns
        self.orientation_radial_power = radial_power
        self.orientation_intensity_power = intensity_power

        # Handle the "auto" case first, since it works by overriding zone_axis_range,
        #   fiber_axis, and fiber_angles then using the regular parser:
        if isinstance(zone_axis_range,str) and zone_axis_range == "auto":
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.core.structure import Structure

            structure = Structure(self.lat_real, self.numbers, self.positions, coords_are_cartesian=False)

            pointgroup = SpacegroupAnalyzer(structure).get_point_group_symbol()
            self.pointgroup = pointgroup

            assert pointgroup in orientation_ranges, "Unrecognized pointgroup returned by pymatgen!"

            zone_axis_range, fiber_axis, fiber_angles = orientation_ranges[pointgroup]
            if isinstance(zone_axis_range, list):
                zone_axis_range = np.array(zone_axis_range)
            elif zone_axis_range == "fiber":
                self.orientation_fiber_axis = np.asarray(fiber_axis)
                self.orientation_fiber_angles = np.asarray(fiber_angles)
            self.cartesian_directions = True # the entries in the orientation_ranges object assume cartesian zones

            print(f"Automatically detected point group {pointgroup}, using arguments: zone_axis_range={zone_axis_range}, fiber_axis={fiber_axis}, fiber_angles={fiber_angles}.")
            

        if isinstance(zone_axis_range, str):
            if (
                zone_axis_range == "fiber"
                and fiber_axis is not None
                and fiber_angles is not None
            ):

                # Determine vector ranges
                self.orientation_fiber_axis = np.array(
                    self.orientation_fiber_axis, dtype="float"
                )
                if self.cartesian_directions:
                    self.orientation_fiber_axis = (
                        self.orientation_fiber_axis
                        / np.linalg.norm(self.orientation_fiber_axis)
                    )
                else:
                    self.orientation_fiber_axis = self.crystal_to_cartesian(
                        self.orientation_fiber_axis
                    )

                # Generate 2 perpendicular vectors to self.orientation_fiber_axis
                if np.all(
                    np.abs(self.orientation_fiber_axis) == np.array([1.0, 0.0, 0.0])
                ):
                    v0 = np.array([0.0, 1.0, 0.0])
                else:
                    v0 = np.array([1.0, 0.0, 0.0])
                v2 = np.cross(self.orientation_fiber_axis, v0)
                v3 = np.cross(v2, self.orientation_fiber_axis)
                v2 = v2 / np.linalg.norm(v2)
                v3 = v3 / np.linalg.norm(v3)

                if self.orientation_fiber_angles[0] == 0:
                    self.orientation_zone_axis_range = np.vstack(
                        (self.orientation_fiber_axis, v2, v3)
                    ).astype("float")
                else:

                    if (
                        self.orientation_fiber_angles[0] == 180
                    ):
                        theta = np.pi / 2.0
                    else:
                        theta = self.orientation_fiber_angles[0] * np.pi / 180.0
                    if (
                        self.orientation_fiber_angles[1] == 180
                        or self.orientation_fiber_angles[1] == 360
                    ):
                        phi = np.pi / 2.0
                    else:
                        phi = self.orientation_fiber_angles[1] * np.pi / 180.0

                    v2output = self.orientation_fiber_axis * np.cos(
                        theta
                    ) + v2 * np.sin(theta)
                    v3output = (
                        self.orientation_fiber_axis * np.cos(theta)
                        + (v2 * np.sin(theta)) * np.cos(phi)
                        + (v3 * np.sin(theta)) * np.sin(phi)
                    )
                    self.orientation_zone_axis_range = np.vstack(
                        (self.orientation_fiber_axis, v2output, v3output)
                    ).astype("float")

                self.orientation_full = False
                self.orientation_half = False
                self.orientation_fiber = True
            else:
                self.orientation_zone_axis_range = np.array(
                    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
                )
                if zone_axis_range == "full":
                    self.orientation_full = True
                    self.orientation_half = False
                    self.orientation_fiber = False
                elif zone_axis_range == "half":
                    self.orientation_full = False
                    self.orientation_half = True
                    self.orientation_fiber = False
                else:
                    if zone_axis_range == "fiber" and fiber_axis is None:
                        raise ValueError(
                            "For fiber zone axes, you must specify the fiber axis and angular ranges"
                        )
                    else:
                        raise ValueError(
                            "Zone axis range must be a 2x3 array, 3x3 array, or full, half or fiber"
                        )

        else:
            self.orientation_zone_axis_range = np.array(zone_axis_range, dtype="float")

            if not self.cartesian_directions:
                for a0 in range(zone_axis_range.shape[0]):
                    self.orientation_zone_axis_range[a0, :] = self.crystal_to_cartesian(
                        self.orientation_zone_axis_range[a0, :]
                    )

            # Define 3 vectors which span zone axis orientation range, normalize
            if zone_axis_range.shape[0] == 3:
                self.orientation_zone_axis_range = np.array(
                    self.orientation_zone_axis_range, dtype="float"
                )
                self.orientation_zone_axis_range[0, :] /= np.linalg.norm(
                    self.orientation_zone_axis_range[0, :]
                )
                self.orientation_zone_axis_range[1, :] /= np.linalg.norm(
                    self.orientation_zone_axis_range[1, :]
                )
                self.orientation_zone_axis_range[2, :] /= np.linalg.norm(
                    self.orientation_zone_axis_range[2, :]
                )

            elif zone_axis_range.shape[0] == 2:
                self.orientation_zone_axis_range = np.vstack(
                    (
                        np.array([0, 0, 1]),
                        np.array(self.orientation_zone_axis_range, dtype="float"),
                    )
                ).astype("float")
                self.orientation_zone_axis_range[1, :] /= np.linalg.norm(
                    self.orientation_zone_axis_range[1, :]
                )
                self.orientation_zone_axis_range[2, :] /= np.linalg.norm(
                    self.orientation_zone_axis_range[2, :]
                )
            self.orientation_full = False
            self.orientation_half = False
            self.orientation_fiber = False

        # Solve for number of angular steps in zone axis (rads)
        angle_u_v = np.arccos(
            np.sum(
                self.orientation_zone_axis_range[0, :]
                * self.orientation_zone_axis_range[1, :]
            )
        )
        angle_u_w = np.arccos(
            np.sum(
                self.orientation_zone_axis_range[0, :]
                * self.orientation_zone_axis_range[2, :]
            )
        )
        self.orientation_zone_axis_steps = np.round(
            np.maximum(
                (180 / np.pi) * angle_u_v / angle_step_zone_axis,
                (180 / np.pi) * angle_u_w / angle_step_zone_axis,
            )
        ).astype(np.int)

        if self.orientation_fiber and self.orientation_fiber_angles[0] == 0:
            self.orientation_num_zones = int(1)
            self.orientation_vecs = np.zeros((1, 3))
            self.orientation_vecs[0, :] = self.orientation_zone_axis_range[0, :]
            self.orientation_inds = np.zeros((1, 3), dtype="int")

        else:

            # Generate points spanning the zone axis range
            # Calculate points along u and v using the SLERP formula
            # https://en.wikipedia.org/wiki/Slerp
            weights = np.linspace(0, 1, self.orientation_zone_axis_steps + 1)
            pv = self.orientation_zone_axis_range[0, :] * np.sin(
                (1 - weights[:, None]) * angle_u_v
            ) / np.sin(angle_u_v) + self.orientation_zone_axis_range[1, :] * np.sin(
                weights[:, None] * angle_u_v
            ) / np.sin(
                angle_u_v
            )

            # Calculate points along u and w using the SLERP formula
            pw = self.orientation_zone_axis_range[0, :] * np.sin(
                (1 - weights[:, None]) * angle_u_w
            ) / np.sin(angle_u_w) + self.orientation_zone_axis_range[2, :] * np.sin(
                weights[:, None] * angle_u_w
            ) / np.sin(
                angle_u_w
            )

            # Init array to hold all points
            self.orientation_num_zones = (
                (self.orientation_zone_axis_steps + 1)
                * (self.orientation_zone_axis_steps + 2)
                / 2
            ).astype(np.int)
            self.orientation_vecs = np.zeros((self.orientation_num_zones, 3))
            self.orientation_vecs[0, :] = self.orientation_zone_axis_range[0, :]
            self.orientation_inds = np.zeros(
                (self.orientation_num_zones, 3), dtype="int"
            )

            # Calculate zone axis points on the unit sphere with another application of SLERP,
            # or circular arc SLERP for fiber texture
            for a0 in np.arange(1, self.orientation_zone_axis_steps + 1):
                inds = np.arange(a0 * (a0 + 1) / 2, a0 * (a0 + 1) / 2 + a0 + 1).astype(
                    np.int
                )

                p0 = pv[a0, :]
                p1 = pw[a0, :]

                weights = np.linspace(0, 1, a0 + 1)

                if self.orientation_fiber:
                    # For fiber texture, place points on circular arc perpendicular to the fiber axis
                    self.orientation_vecs[inds, :] = p0[None, :]

                    p_proj = (
                        np.dot(p0, self.orientation_fiber_axis)
                        * self.orientation_fiber_axis
                    )
                    p0_sub = p0 - p_proj
                    p1_sub = p1 - p_proj

                    angle_p_sub = np.arccos(
                        np.sum(p0_sub * p1_sub) \
                        / np.linalg.norm(p0_sub) \
                        / np.linalg.norm(p1_sub))

                    self.orientation_vecs[inds, :] = (
                        p_proj
                        + p0_sub[None, :]
                        * np.sin((1 - weights[:, None]) * angle_p_sub)
                        / np.sin(angle_p_sub)
                        + p1_sub[None, :]
                        * np.sin(weights[:, None] * angle_p_sub)
                        / np.sin(angle_p_sub)
                    )
                else:
                    angle_p = np.arccos(np.sum(p0 * p1))

                    self.orientation_vecs[inds, :] = p0[None, :] * np.sin(
                        (1 - weights[:, None]) * angle_p
                    ) / np.sin(angle_p) + p1[None, :] * np.sin(
                        weights[:, None] * angle_p
                    ) / np.sin(
                        angle_p
                    )

                self.orientation_inds[inds, 0] = a0
                self.orientation_inds[inds, 1] = np.arange(a0 + 1)


        if (
            self.orientation_fiber
            and self.orientation_fiber_angles[0] == 180
        ):
            # Mirror about the equator of fiber_zone_axis
            m = np.identity(3) - 2 * (self.orientation_fiber_axis[:, None] @ self.orientation_fiber_axis[None, :])

            vec_new = np.copy(self.orientation_vecs) @ m
            orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

            keep = np.zeros(vec_new.shape[0], dtype="bool")
            for a0 in range(keep.size):
                if (
                    np.sqrt(
                        np.min(
                            np.sum(
                                (self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1
                            )
                        )
                    )
                    > tol_distance
                ):
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep, :]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]

            self.orientation_inds = np.vstack(
                (self.orientation_inds, self.orientation_inds[keep, :])
            ).astype("int")
            self.orientation_inds[:, 2] = np.hstack(
                (orientation_sector, np.ones(np.sum(keep), dtype="int"))
            )

        # Fiber texture angle 1 extend to 180 degree angular range if needed
        if (
            self.orientation_fiber
            and self.orientation_fiber_angles[0] != 0
            and (
                self.orientation_fiber_angles[1] == 180
                or self.orientation_fiber_angles[1] == 360
            )
        ):
            # Mirror about the axes 0 and 1
            n = np.cross(
                self.orientation_zone_axis_range[0, :],
                self.orientation_zone_axis_range[1, :],
            )
            n = n / np.linalg.norm(n)

            # n = self.orientation_zone_axis_range[2,:]
            m = np.identity(3) - 2 * (n[:, None] @ n[None, :])

            vec_new = np.copy(self.orientation_vecs) @ m
            orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

            keep = np.zeros(vec_new.shape[0], dtype="bool")
            for a0 in range(keep.size):
                if (
                    np.sqrt(
                        np.min(
                            np.sum(
                                (self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1
                            )
                        )
                    )
                    > tol_distance
                ):
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep, :]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]

            self.orientation_inds = np.vstack(
                (self.orientation_inds, self.orientation_inds[keep, :])
            ).astype("int")
            self.orientation_inds[:, 2] = np.hstack(
                (orientation_sector, np.ones(np.sum(keep), dtype="int"))
            )
        # Fiber texture extend to 360 angular range if needed
        if (
            self.orientation_fiber
            and self.orientation_fiber_angles[0] != 0
            and self.orientation_fiber_angles[1] == 360
        ):
            # Mirror about the axes 0 and 2
            n = np.cross(
                self.orientation_zone_axis_range[0, :],
                self.orientation_zone_axis_range[2, :],
            )
            n = n / np.linalg.norm(n)

            # n = self.orientation_zone_axis_range[2,:]
            m = np.identity(3) - 2 * (n[:, None] @ n[None, :])

            vec_new = np.copy(self.orientation_vecs) @ m
            orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

            keep = np.zeros(vec_new.shape[0], dtype="bool")
            for a0 in range(keep.size):
                if (
                    np.sqrt(
                        np.min(
                            np.sum(
                                (self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1
                            )
                        )
                    )
                    > tol_distance
                ):
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep, :]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]

            self.orientation_inds = np.vstack(
                (self.orientation_inds, self.orientation_inds[keep, :])
            ).astype("int")
            self.orientation_inds[:, 2] = np.hstack(
                (orientation_sector, np.ones(np.sum(keep), dtype="int"))
            )

        # expand to quarter sphere if needed
        if self.orientation_half or self.orientation_full:
            vec_new = np.copy(self.orientation_vecs) * np.array([-1, 1, 1])
            orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

            keep = np.zeros(vec_new.shape[0], dtype="bool")
            for a0 in range(keep.size):
                if (
                    np.sqrt(
                        np.min(
                            np.sum(
                                (self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1
                            )
                        )
                    )
                    > tol_distance
                ):
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep, :]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]

            self.orientation_inds = np.vstack(
                (self.orientation_inds, self.orientation_inds[keep, :])
            ).astype("int")
            self.orientation_inds[:, 2] = np.hstack(
                (orientation_sector, np.ones(np.sum(keep), dtype="int"))
            )

        # expand to hemisphere if needed
        if self.orientation_full:
            vec_new = np.copy(self.orientation_vecs) * np.array([1, -1, 1])

            keep = np.zeros(vec_new.shape[0], dtype="bool")
            for a0 in range(keep.size):
                if (
                    np.sqrt(
                        np.min(
                            np.sum(
                                (self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1
                            )
                        )
                    )
                    > tol_distance
                ):
                    keep[a0] = True

            self.orientation_vecs = np.vstack((self.orientation_vecs, vec_new[keep, :]))
            self.orientation_num_zones = self.orientation_vecs.shape[0]

            orientation_sector = np.hstack(
                (self.orientation_inds[:, 2], self.orientation_inds[keep, 2] + 2)
            )
            self.orientation_inds = np.vstack(
                (self.orientation_inds, self.orientation_inds[keep, :])
            ).astype("int")
            self.orientation_inds[:, 2] = orientation_sector

        # Convert to spherical coordinates
        elev = np.arctan2(
            np.hypot(self.orientation_vecs[:, 0], self.orientation_vecs[:, 1]),
            self.orientation_vecs[:, 2],
        )
        azim = -np.pi / 2 + np.arctan2(
            self.orientation_vecs[:, 1], self.orientation_vecs[:, 0]
        )

        # Solve for number of angular steps along in-plane rotation direction
        self.orientation_in_plane_steps = np.round(360 / angle_step_in_plane).astype(
            np.int
        )

        # Calculate -z angles (Euler angle 3)
        self.orientation_gamma = np.linspace(
            0, 2 * np.pi, self.orientation_in_plane_steps, endpoint=False
        )

        # Determine the radii of all spherical shells
        radii_test = np.round(self.g_vec_leng / tol_distance) * tol_distance
        radii = np.unique(radii_test)
        # Remove zero beam
        keep = np.abs(radii) > tol_distance
        self.orientation_shell_radii = radii[keep]

        # init
        self.orientation_shell_index = -1 * np.ones(
            self.g_vec_all.shape[1], dtype="int"
        )
        self.orientation_shell_count = np.zeros(self.orientation_shell_radii.size)

        # Assign each structure factor point to a radial shell
        for a0 in range(self.orientation_shell_radii.size):
            sub = (
                np.abs(self.orientation_shell_radii[a0] - radii_test)
                <= tol_distance / 2
            )

            self.orientation_shell_index[sub] = a0
            self.orientation_shell_count[a0] = np.sum(sub)
            self.orientation_shell_radii[a0] = np.mean(self.g_vec_leng[sub])

        # init storage arrays
        self.orientation_rotation_angles = np.zeros((self.orientation_num_zones, 2))
        self.orientation_rotation_matrices = np.zeros(
            (self.orientation_num_zones, 3, 3)
        )
        self.orientation_ref = np.zeros(
            (
                self.orientation_num_zones,
                np.size(self.orientation_shell_radii),
                self.orientation_in_plane_steps,
            ),
            dtype="complex64",
        )
        


        # Calculate rotation matrices for zone axes
        # for a0 in tqdmnd(np.arange(self.orientation_num_zones),desc='Computing orientation basis',unit=' terms',unit_scale=True):
        for a0 in np.arange(self.orientation_num_zones):
            m1z = np.array(
                [
                    [np.cos(azim[a0]), -np.sin(azim[a0]), 0],
                    [np.sin(azim[a0]), np.cos(azim[a0]), 0],
                    [0, 0, 1],
                ]
            )
            m2x = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(elev[a0]), np.sin(elev[a0])],
                    [0, -np.sin(elev[a0]), np.cos(elev[a0])],
                ]
            )
            self.orientation_rotation_matrices[a0, :, :] = m1z @ m2x
            self.orientation_rotation_angles[a0, :] = [azim[a0], elev[a0]]

        # init
        k0 = np.array([0, 0, 1]) / self.wavelength
        dphi = self.orientation_gamma[1] - self.orientation_gamma[0]

        # Calculate reference arrays for all orientations
        for a0 in tqdmnd(
            np.arange(self.orientation_num_zones),
            desc="Orientation plan",
            unit=" zone axes",
            disable=not progress_bar
        ):
            p = (
                np.linalg.inv(self.orientation_rotation_matrices[a0, :, :])
                @ self.g_vec_all
            )

            # Excitation errors
            cos_alpha = (k0[2, None] + p[2, :]) / np.linalg.norm(
                k0[:, None] + p, axis=0
            )
            sg = (
                (-0.5)
                * np.sum((2 * k0[:, None] + p) * p, axis=0)
                / (np.linalg.norm(k0[:, None] + p, axis=0))
                / cos_alpha
            )

            # in-plane rotation angle
            phi = np.arctan2(p[1, :], p[0, :])

            for a1 in np.arange(self.g_vec_all.shape[1]):
                ind_radial = self.orientation_shell_index[a1]

                if ind_radial >= 0:
                    self.orientation_ref[a0, ind_radial, :] += (
                        np.power(self.orientation_shell_radii[ind_radial], radial_power)
                        * np.power(self.struct_factors_int[a1], intensity_power)
                        * np.maximum(
                            1
                            - np.sqrt(
                                sg[a1] ** 2
                                + (
                                    (
                                        np.mod(
                                            self.orientation_gamma - phi[a1] + np.pi,
                                            2 * np.pi,
                                        )
                                        - np.pi
                                    )
                                    * self.orientation_shell_radii[ind_radial]
                                )
                                ** 2
                            )
                            / self.orientation_kernel_size,
                            0,
                        )
                    )

            # Normalization
            self.orientation_ref[a0, :, :] = self.orientation_ref[a0, :, :] / np.sqrt(
                np.sum(np.abs(self.orientation_ref[a0, :, :])**2)
            )

        # Maximum value
        self.orientation_ref_max = np.max(np.real(self.orientation_ref))

        # Fourier domain along angular axis
        self.orientation_ref = np.conj(np.fft.fft(self.orientation_ref))
       
        # # Init vectors for the 2D corr method
        # self.orientation_gamma_cos2 = np.cos(self.orientation_gamma)**2
        # self.orientation_gamma_cos2_fft = np.fft.fft(self.orientation_gamma_cos2)
        # self.orientation_gamma_shift = -2j*np.pi* \
        #     np.fft.fftfreq(self.orientation_in_plane_steps)

        # # Calculate perpendicular orientation reference if needed
        # if self.orientation_corr_2D_method:
        #     # self.orientation_ref_perp = np.real(np.fft.ifft(
        #     #     self.orientation_ref)
        #     # ).astype("complex64")
        #     # self.orientation_ref_perp = self.orientation_ref.copy()

        #     self.orientation_gamma_cos2 = np.cos(self.orientation_gamma)**2
        #     self.orientation_gamma_cos2_fft = np.fft.fft(self.orientation_gamma_cos2)
        #     self.orientation_gamma_shift = -2j*np.pi* \
        #         np.fft.fftfreq(self.orientation_in_plane_steps)

            # # if self.orientation_corr_2D_method:
            # for a0 in range(self.orientation_num_zones):
            #     cos2_corr = np.sum(np.real(np.fft.ifft(
            #         np.fft.fft(self.orientation_ref[a0,:,:]) * self.orientation_gamma_cos2_fft
            #     )), axis=0)
            #     ind_shift = np.argmax(cos2_corr)
            #     # self.orientation_ref_perp[a0,:,:] = self.orientation_ref_perp[a0,:,:] \
            #     #     * (1 - np.real(np.fft.ifft(self.orientation_gamma_cos2_fft \
            #     #     * np.exp(self.orientation_gamma_shift*ind_shift))))
            #     self.orientation_ref_perp[a0,:,:] = self.orientation_ref_perp[a0,:,:] \
            #         * (1-np.real(np.fft.ifft(self.orientation_gamma_cos2_fft \
            #         * np.exp(self.orientation_gamma_shift*ind_shift))))


            # self.orientation_ref_perp = np.conj(np.fft.fft(self.orientation_ref_perp))





    def plot_orientation_zones(
        self,
        proj_dir: Optional[Union[list, tuple, np.ndarray]] = None,
        marker_size: float = 20,
        plot_limit: Union[list, tuple, np.ndarray] = np.array([-1.1, 1.1]),
        figsize: Union[list, tuple, np.ndarray] = (8, 8),
        returnfig: bool = False,
    ):
        """
        3D scatter plot of the structure factors using magnitude^2, i.e. intensity.

        Args:
            dir_proj (float):           projection direction, either [elev azim] or normal vector
                                        Default is mean vector of self.orientation_zone_axis_range rows
            marker_size (float):        size of markers
            plot_limit (float):         x y z plot limits, default is [0, 1.05]
            figsize (2 element float):  size scaling of figure axes
            returnfig (bool):           set to True to return figure and axes handles

        Returns:
            fig, ax                     (optional) figure and axes handles
        """

        if proj_dir is None:
            proj_dir = np.mean(self.orientation_zone_axis_range, axis=0)
        elif not self.cartesian_directions:
            proj_dir = self.crystal_to_cartesian(proj_dir)

        if np.size(proj_dir) == 2:
            el = proj_dir[0]
            az = proj_dir[1]
        elif np.size(proj_dir) == 3:
            if proj_dir[0] == 0 and proj_dir[1] == 0:
                el = 90 * np.sign(proj_dir[2])
            else:
                el = (
                    np.arctan(
                        proj_dir[2] / np.sqrt(proj_dir[0] ** 2 + proj_dir[1] ** 2)
                    )
                    * 180
                    / np.pi
                )
            az = np.arctan2(proj_dir[1], proj_dir[0]) * 180 / np.pi
        else:
            raise Exception(
                "Projection direction cannot contain " + np.size(proj_dir) + " elements"
            )

        # 3D plotting
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d", elev=el, azim=90 - az)

        # Sphere
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        r = 0.95
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        ax.plot_surface(
            x,
            y,
            z,
            edgecolor=None,
            color=np.array([1.0, 0.8, 0.0]),
            alpha=0.4,
            antialiased=True,
        )

        # Lines
        r = 0.951
        t = np.linspace(0, 2 * np.pi, 181)
        t0 = np.zeros((181,))
        # z = np.linspace(-2, 2, 100)
        # r = z**2 + 1
        # x = r * np.sin(theta)
        # y = r * np.cos(theta)

        warnings.filterwarnings("ignore", module="matplotlib\..*")
        line_params = {"linewidth": 2, "alpha": 0.1, "c": "k"}
        for phi in np.arange(0, 180, 5):
            ax.plot3D(
                np.sin(phi * np.pi / 180) * np.cos(t) * r,
                np.sin(phi * np.pi / 180) * np.sin(t) * r,
                np.cos(phi * np.pi / 180) * r,
                **line_params,
            )

        # plot zone axes
        ax.scatter(
            xs=self.orientation_vecs[:, 1],
            ys=self.orientation_vecs[:, 0],
            zs=self.orientation_vecs[:, 2],
            s=marker_size,
        )

        # zone axis range labels
        # label_0 = self.cartesian_to_crystal(self.orientation_zone_axis_range[0, :])
        if self.cartesian_directions:
            label_0 = self.orientation_zone_axis_range[0, :]
        else:
            label_0 = self.cartesian_to_crystal(self.orientation_zone_axis_range[0, :])
        label_0 = np.round(label_0, decimals=3)
        label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))
        label_0 = np.round(label_0, decimals=3)

        if self.orientation_fiber is False \
            and self.orientation_full is False \
            and self.orientation_half is False:

            # label_1 = self.cartesian_to_crystal(
            #     self.orientation_zone_axis_range[1, :]
            #     )
            if self.cartesian_directions:
                label_1 = self.orientation_zone_axis_range[1, :]
            else:
                label_1 = self.cartesian_to_crystal(
                    self.orientation_zone_axis_range[1, :]
                )
            label_1 = np.round(label_1 * 1e3) * 1e-3
            label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))
            label_1 = np.round(label_1 * 1e3) * 1e-3

            # label_2 = self.cartesian_to_crystal(
            #     self.orientation_zone_axis_range[2, :]
            # )
            if self.cartesian_directions:
                label_2 = self.orientation_zone_axis_range[2, :]
            else:
                label_2 = self.cartesian_to_crystal(
                    self.orientation_zone_axis_range[2, :]
                )

            label_2 = np.round(label_2 * 1e3) * 1e-3
            label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))
            label_2 = np.round(label_2 * 1e3) * 1e-3

            inds = np.array(
                [
                    0,
                    self.orientation_num_zones - self.orientation_zone_axis_steps - 1,
                    self.orientation_num_zones - 1,
                ]
            )
        else:
            inds = np.array([0])

        ax.scatter(
            xs=self.orientation_vecs[inds, 1] * 1.02,
            ys=self.orientation_vecs[inds, 0] * 1.02,
            zs=self.orientation_vecs[inds, 2] * 1.02,
            s=marker_size * 8,
            linewidth=2,
            marker="o",
            edgecolors="r",
            alpha=1,
            zorder=10,
        )

        text_scale_pos = 1.2
        text_params = {
            "va": "center",
            "family": "sans-serif",
            "fontweight": "normal",
            "color": "k",
            "size": 20,
        }
        # 'ha': 'center',

        ax.text(
            self.orientation_vecs[inds[0], 1] * text_scale_pos,
            self.orientation_vecs[inds[0], 0] * text_scale_pos,
            self.orientation_vecs[inds[0], 2] * text_scale_pos,
            label_0,
            None,
            zorder=11,
            ha="center",
            **text_params,
        )
        if self.orientation_fiber is False \
            and self.orientation_full is False \
            and self.orientation_half is False:
            ax.text(
                self.orientation_vecs[inds[1], 1] * text_scale_pos,
                self.orientation_vecs[inds[1], 0] * text_scale_pos,
                self.orientation_vecs[inds[1], 2] * text_scale_pos,
                label_1,
                None,
                zorder=12,
                ha="right",
                **text_params,
            )
            ax.text(
                self.orientation_vecs[inds[2], 1] * text_scale_pos,
                self.orientation_vecs[inds[2], 0] * text_scale_pos,
                self.orientation_vecs[inds[2], 2] * text_scale_pos,
                label_2,
                None,
                zorder=13,
                ha="left",
                **text_params,
            )

        # ax.scatter(
        #     xs=self.g_vec_all[0,:],
        #     ys=self.g_vec_all[1,:],
        #     zs=self.g_vec_all[2,:],
        #     s=scale_markers*self.struct_factors_int)

        # axes limits
        ax.axes.set_xlim3d(left=plot_limit[0], right=plot_limit[1])
        ax.axes.set_ylim3d(bottom=plot_limit[0], top=plot_limit[1])
        ax.axes.set_zlim3d(bottom=plot_limit[0], top=plot_limit[1])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        # ax.setxticklabels([])
        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # plt.gca().invert_yaxis()
        ax.view_init(elev=el, azim=90 - az)

        plt.show()

        if returnfig:
            return fig, ax

    def plot_orientation_plan(
        self,
        index_plot: int = 0,
        zone_axis_plot = None,
        figsize: Union[list, tuple, np.ndarray] = (14, 6),
        returnfig: bool = False,
    ):
        """
        3D scatter plot of the structure factors using magnitude^2, 
        i.e. intensity.

        Args:
            index_plot (int):           which index slice to plot
            zone_axis_plot (3 element float): which zone axis slice to plot
            figsize (2 element float):  size scaling of figure axes
            returnfig (bool):           set to True to return figure and axes handles

        Returns:
            fig, ax                     (optional) figure and axes handles
        """

        # Determine which index to plot if zone_axis_plot is specified
        if zone_axis_plot is not None:
            zone_axis_plot = np.array(zone_axis_plot, dtype="float")
            zone_axis_plot = zone_axis_plot / np.linalg.norm(zone_axis_plot)
            index_plot = np.argmin(
                np.sum((self.orientation_vecs - zone_axis_plot) ** 2, axis=1)
            )
            print('Orientation plan index ' + str(index_plot))

        # initialize figure
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # Generate and plot diffraction pattern
        k_x_y_range = np.array([1, 1]) * self.k_max * 1.2
        bragg_peaks = self.generate_diffraction_pattern(
            zone_axis=self.orientation_vecs[index_plot, :],
            sigma_excitation_error=self.orientation_kernel_size / 3,
        )

        plot_diffraction_pattern(
            bragg_peaks,
            figsize=(figsize[1], figsize[1]),
            plot_range_kx_ky=k_x_y_range,
            scale_markers=10,
            shift_labels=0.10,
            input_fig_handle=[fig, ax],
        )

        # Plot orientation plan
        # if self.orientation_corr_2D_method:
        #     im_plot = np.vstack((
        #         np.real(np.fft.ifft(self.orientation_ref[index_plot, :, :], axis=1)
        #     ).astype("float"),
        #         np.real(np.fft.ifft(self.orientation_ref_perp[index_plot, :, :], axis=1)
        #     ).astype("float"))) / self.orientation_ref_max

        #     # im_plot = np.vstack((
        #     #     np.real(np.fft.ifft(self.orientation_ref[index_plot, :, :], axis=1)
        #     # ).astype("float"),
        #     #     self.orientation_ref_perp[index_plot, :, :])) / self.orientation_ref_max
        # else:
        im_plot = np.real(
            np.fft.ifft(self.orientation_ref[index_plot, :, :], axis=1)
        ).astype("float") / self.orientation_ref_max

        # coordinates
        x = self.orientation_gamma * 180 / np.pi
        # if self.orientation_corr_2D_method:
        #     y = np.arange(2*np.size(self.orientation_shell_radii))
        # else:
        y = np.arange(np.size(self.orientation_shell_radii))
        dx = (x[1] - x[0]) / 2.0
        dy = (y[1] - y[0]) / 2.0
        extent = [x[0] - dx, x[-1] + dx, y[-1] + dy, y[0] - dy]

        im = ax[1].imshow(
            im_plot,
            cmap="inferno",
            vmin=0.0,
            vmax=0.5,
            extent=extent,
            aspect="auto",
            interpolation="none",
        )
        fig.colorbar(im)
        ax[1].xaxis.tick_top()
        ax[1].set_xticks(np.arange(0, 360 + 90, 90))
        ax[1].set_ylabel("Radial Index", size=20)
        # if self.orientation_corr_2D_method:
        #     t0 = np.arange(0,np.size(self.orientation_shell_radii),10)
        #     t1 = t0 + np.size(self.orientation_shell_radii)
        #     ax[1].set_yticks(np.hstack((t0,t1)))
        #     ax[1].set_yticklabels(np.hstack((t0,t0)))

        # Add text label
        zone_axis_fit = self.orientation_vecs[index_plot, :]
        zone_axis_fit = zone_axis_fit / np.linalg.norm(zone_axis_fit)
        sub = np.abs(zone_axis_fit) > 0
        scale = np.min(np.abs(zone_axis_fit[sub]))
        if scale > 0.14:
            zone_axis_fit = zone_axis_fit / scale

        temp = np.round(zone_axis_fit, decimals=2)
        ax[0].text(
            -k_x_y_range[0] * 0.95,
            -k_x_y_range[1] * 0.95,
            "[" + str(temp[0]) + ", " + str(temp[1]) + ", " + str(temp[2]) + "]",
            size=18,
            va="top",
        )

        # plt.tight_layout()
        plt.show()

        if returnfig:
            return fig, ax

    def match_orientations(
        self,
        bragg_peaks_array: PointListArray,
        num_matches_return: int = 1,
        inversion_symmetry = True,
        multiple_corr_reset = True,
        return_corr: bool = False,
        subpixel_tilt: bool = False,
        progress_bar: bool = True
    ):

        if num_matches_return == 1:
            orientation_matrices = np.zeros(
                (*bragg_peaks_array.shape, 3, 3), dtype=np.float64
            )
            if return_corr:
                corr_all = np.zeros(bragg_peaks_array.shape, dtype=np.float64)
        else:
            orientation_matrices = np.zeros(
                (*bragg_peaks_array.shape, 3, 3, num_matches_return), dtype=np.float64
            )
            if return_corr:
                corr_all = np.zeros(
                    (*bragg_peaks_array.shape, num_matches_return), dtype=np.float64
                )

        for rx, ry in tqdmnd(
            *bragg_peaks_array.shape, desc="Matching Orientations", unit=" PointList", disable=not progress_bar
        ):
            bragg_peaks = bragg_peaks_array.get_pointlist(rx, ry)

            if return_corr:
                (
                    orientation_matrices[rx, ry],
                    corr_all[rx, ry],
                ) = self.match_single_pattern(
                    bragg_peaks,
                    subpixel_tilt=subpixel_tilt,
                    num_matches_return=num_matches_return,
                    inversion_symmetry = inversion_symmetry,
                    multiple_corr_reset = multiple_corr_reset,
                    plot_corr=False,
                    plot_corr_3D=False,
                    return_corr=True,
                    verbose=False,
                )
            else:
                orientation_matrices[rx, ry] = self.match_single_pattern(
                    bragg_peaks,
                    subpixel_tilt=subpixel_tilt,
                    num_matches_return=num_matches_return,
                    plot_corr=False,
                    plot_corr_3D=False,
                    return_corr=False,
                    verbose=False,
                )

        if return_corr:
            return orientation_matrices, corr_all
        else:
            return orientation_matrices

    def match_single_pattern(
        self,
        bragg_peaks: PointList,
        num_matches_return: int = 1,
        multiple_corr_reset = True,
        inversion_symmetry = True,
        subpixel_tilt: bool = False,
        plot_polar: bool = False,
        plot_corr: bool = False,
        plot_corr_3D: bool = False,
        return_corr: bool = False,
        returnfig: bool = False,
        figsize: Union[list, tuple, np.ndarray] = (12, 4),
        verbose: bool = False,
    ):
        """
        Solve for the best fit orientation of a single diffraction pattern.

        Args:
            bragg_peaks (PointList):      numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
            num_matches_return (int):     return these many matches as 3th dim of orient (matrix)
            multiple_corr_reset (bool):   keep original correlation score for multiple matches
            inversion_symmetry (bool):    check for inversion symmetry in the matches
            subpixel_tilt (bool):         set to false for faster matching, returning the nearest corr point
            plot_polar (bool):            set to true to plot the polar transform of the diffraction pattern
            plot_corr (bool):             set to true to plot the resulting correlogram

        Returns:
            orientation_output (3x3xN float)    orienation matrix where zone axis is the 3rd column, 3rd dim for multiple matches
            corr_value (float):                 (optional) return correlation values
        """

        # get bragg peak data
        qx = bragg_peaks.data["qx"]
        qy = bragg_peaks.data["qy"]
        intensity = bragg_peaks.data["intensity"]

        # init orientation output, delete distance threshold squared
        if num_matches_return == 1:
            orientation_output = np.zeros((3, 3))
        else:
            orientation_output = np.zeros((3, 3, num_matches_return))
            corr_output = np.zeros((num_matches_return))

        # loop over the number of matches to return
        for match_ind in range(num_matches_return):
            # Convert Bragg peaks to polar coordinates
            qr = np.sqrt(qx ** 2 + qy ** 2)
            qphi = np.arctan2(qy, qx)

            # Calculate polar Bragg peak image
            im_polar = np.zeros(
                (
                    np.size(self.orientation_shell_radii),
                    self.orientation_in_plane_steps,
                ),
                dtype="float",
            )

            for ind_radial, radius in enumerate(self.orientation_shell_radii):
                dqr = np.abs(qr - radius)
                sub = dqr < self.orientation_kernel_size

                if np.any(sub):
                    im_polar[ind_radial, :] = np.sum(
                        np.power(radius,self.orientation_radial_power)
                        * np.power(np.max(intensity[sub, None],0),self.orientation_intensity_power)
                        * np.maximum(
                            1
                            - np.sqrt(
                                dqr[sub, None] ** 2
                                + (
                                    (
                                        np.mod(
                                            self.orientation_gamma[None, :]
                                            - qphi[sub, None]
                                            + np.pi,
                                            2 * np.pi,
                                        )
                                        - np.pi
                                    )
                                    * radius
                                )
                                ** 2
                            )
                            / self.orientation_kernel_size,
                            0,
                        ),
                        axis=0,
                    )

            # FFT along theta
            im_polar_fft = np.fft.fft(im_polar)

            # # 2D correlation method
            # if corr_2D_method:
            #     cos2_corr = np.sum(np.real(np.fft.ifft(
            #         im_polar_fft * self.orientation_gamma_cos2_fft
            #     )), axis=0)
            #     ind_shift = np.argmax(cos2_corr)
            #     gamma_cos_shift = np.real(np.fft.ifft(self.orientation_gamma_cos2_fft \
            #         * np.exp(self.orientation_gamma_shift*-ind_shift)))

            #     im_polar_cos = im_polar * gamma_cos_shift
            #     im_polar_sin = im_polar * (1-gamma_cos_shift)
            #     im_polar = im_polar_cos + im_polar_sin * (np.sum(im_polar_cos) / np.sum(im_polar_sin))

            # init
            dphi = self.orientation_gamma[1] - self.orientation_gamma[0]
            corr_value = np.zeros(self.orientation_num_zones)
            corr_in_plane_angle = np.zeros(self.orientation_num_zones)

            # Calculate orientation correlogram
            corr_full = np.maximum(np.sum(
                np.real(
                    np.fft.ifft(self.orientation_ref * im_polar_fft[None, :, :])
                ),
                axis=1,
            ),0)
            ind_phi = np.argmax(corr_full, axis=1)

            # Calculate orientation correlogram for inverse pattern
            if inversion_symmetry:
                corr_full_inv = np.maximum(np.sum(
                    np.real(
                        np.fft.ifft(self.orientation_ref * np.conj(im_polar_fft)[None, :, :])
                    ),
                    axis=1,
                ),0)
                ind_phi_inv = np.argmax(corr_full_inv, axis=1)
                corr_inv = np.zeros(self.orientation_num_zones, dtype='bool')

            # Find best match for each zone axis
            for a0 in range(self.orientation_num_zones):
                # Correlation score
                if inversion_symmetry:
                    if corr_full_inv[a0,ind_phi_inv[a0]] > corr_full[a0,ind_phi[a0]]:
                        corr_value[a0] = corr_full_inv[a0,ind_phi_inv[a0]]
                        corr_inv[a0] = True
                    else:
                        corr_value[a0] = corr_full[a0,ind_phi[a0]]
                else:
                    corr_value[a0] = corr_full[a0,ind_phi[a0]]

                # Subpixel angular fit
                if inversion_symmetry and corr_inv[a0]:
                    inds = np.mod(
                        ind_phi_inv[a0] + np.arange(-1, 2), self.orientation_gamma.size
                    ).astype("int")
                    c = corr_full_inv[a0, inds]
                    if np.max(c) > 0:
                        dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                        corr_in_plane_angle[a0] = (
                            self.orientation_gamma[ind_phi_inv[a0]] + dc * dphi
                        )
                else:
                    inds = np.mod(
                        ind_phi[a0] + np.arange(-1, 2), self.orientation_gamma.size
                    ).astype("int")
                    c = corr_full[a0, inds]
                    if np.max(c) > 0:
                        dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])
                        corr_in_plane_angle[a0] = (
                            self.orientation_gamma[ind_phi[a0]] + dc * dphi
                        )

            # If needed, keep correlation values for additional matches
            if multiple_corr_reset and num_matches_return > 1 and match_ind == 0:
                corr_value_keep = corr_value.copy()
                corr_in_plane_angle_keep = corr_in_plane_angle.copy()

            # Determine the best fit orientation
            ind_best_fit = np.unravel_index(np.argmax(corr_value), corr_value.shape)[0]
            
            # Verify current match has a correlation > 0
            if corr_value[ind_best_fit] > 0:


                # Get orientation matrix
                if subpixel_tilt is False:
                    orientation_matrix = np.squeeze(
                        self.orientation_rotation_matrices[ind_best_fit, :, :]
                    )

                else:

                    def ind_to_sub(ind):
                        ind_x = np.floor(0.5 * np.sqrt(8.0 * ind + 1) - 0.5).astype("int")
                        ind_y = ind - np.floor(ind_x * (ind_x + 1) / 2).astype("int")
                        return ind_x, ind_y

                    def sub_to_ind(ind_x, ind_y):
                        return (np.floor(ind_x * (ind_x + 1) / 2) + ind_y).astype("int")

                    # Sub pixel refinement of zone axis orientation
                    if ind_best_fit == 0:
                        # Zone axis is (0,0,1)
                        orientation_matrix = np.squeeze(
                            self.orientation_rotation_matrices[ind_best_fit, :, :]
                        )

                    elif (
                        ind_best_fit
                        == self.orientation_num_zones - self.orientation_zone_axis_steps - 1
                    ):
                        # Zone axis is 1st user provided direction
                        orientation_matrix = np.squeeze(
                            self.orientation_rotation_matrices[ind_best_fit, :, :]
                        )

                    elif ind_best_fit == self.orientation_num_zones - 1:
                        # Zone axis is the 2nd user-provided direction
                        orientation_matrix = np.squeeze(
                            self.orientation_rotation_matrices[ind_best_fit, :, :]
                        )

                    else:
                        ind_x, ind_y = ind_to_sub(ind_best_fit)
                        max_x, max_y = ind_to_sub(self.orientation_num_zones - 1)

                        if ind_y == 0:
                            ind_x_prev = sub_to_ind(ind_x - 1, 0)
                            ind_x_post = sub_to_ind(ind_x + 1, 0)

                            c = np.array(
                                [
                                    corr_value[ind_x_prev],
                                    corr_value[ind_best_fit],
                                    corr_value[ind_x_post],
                                ]
                            )
                            dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

                            if dc > 0:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 - dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_post, :, :]
                                    )
                                    * dc
                                )
                            else:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 + dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_prev, :, :]
                                    )
                                    * -dc
                                )

                        elif ind_x == max_x:
                            ind_x_prev = sub_to_ind(max_x, ind_y - 1)
                            ind_x_post = sub_to_ind(max_x, ind_y + 1)

                            c = np.array(
                                [
                                    corr_value[ind_x_prev],
                                    corr_value[ind_best_fit],
                                    corr_value[ind_x_post],
                                ]
                            )
                            dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

                            if dc > 0:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 - dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_post, :, :]
                                    )
                                    * dc
                                )
                            else:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 + dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_prev, :, :]
                                    )
                                    * -dc
                                )

                        elif ind_x == ind_y:
                            ind_x_prev = sub_to_ind(ind_x - 1, ind_y - 1)
                            ind_x_post = sub_to_ind(ind_x + 1, ind_y + 1)

                            c = np.array(
                                [
                                    corr_value[ind_x_prev],
                                    corr_value[ind_best_fit],
                                    corr_value[ind_x_post],
                                ]
                            )
                            dc = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

                            if dc > 0:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 - dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_post, :, :]
                                    )
                                    * dc
                                )
                            else:
                                orientation_matrix = (
                                    np.squeeze(
                                        self.orientation_rotation_matrices[
                                            ind_best_fit, :, :
                                        ]
                                    )
                                    * (1 + dc)
                                    + np.squeeze(
                                        self.orientation_rotation_matrices[ind_x_prev, :, :]
                                    )
                                    * -dc
                                )

                        else:
                            # # best fit point is not on any of the corners or edges
                            ind_1 = sub_to_ind(ind_x - 1, ind_y - 1)
                            ind_2 = sub_to_ind(ind_x - 1, ind_y)
                            ind_3 = sub_to_ind(ind_x, ind_y - 1)
                            ind_4 = sub_to_ind(ind_x, ind_y + 1)
                            ind_5 = sub_to_ind(ind_x + 1, ind_y)
                            ind_6 = sub_to_ind(ind_x + 1, ind_y + 1)

                            c = np.array(
                                [
                                    (corr_value[ind_1] + corr_value[ind_2]) / 2,
                                    corr_value[ind_best_fit],
                                    (corr_value[ind_5] + corr_value[ind_6]) / 2,
                                ]
                            )
                            dx = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

                            c = np.array(
                                [
                                    corr_value[ind_3],
                                    corr_value[ind_best_fit],
                                    corr_value[ind_4],
                                ]
                            )
                            dy = (c[2] - c[0]) / (4 * c[1] - 2 * c[0] - 2 * c[2])

                            if dx > 0:
                                if dy > 0:
                                    orientation_matrix = (
                                        np.squeeze(
                                            self.orientation_rotation_matrices[
                                                ind_best_fit, :, :
                                            ]
                                        )
                                        * (1 - dx)
                                        * (1 - dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_4, :, :]
                                        )
                                        * (1 - dx)
                                        * (dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_6, :, :]
                                        )
                                        * dx
                                    )
                                else:
                                    orientation_matrix = (
                                        np.squeeze(
                                            self.orientation_rotation_matrices[
                                                ind_best_fit, :, :
                                            ]
                                        )
                                        * (1 - dx)
                                        * (1 + dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_3, :, :]
                                        )
                                        * (1 - dx)
                                        * (-dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_5, :, :]
                                        )
                                        * dx
                                    )
                            else:
                                if dy > 0:
                                    orientation_matrix = (
                                        np.squeeze(
                                            self.orientation_rotation_matrices[
                                                ind_best_fit, :, :
                                            ]
                                        )
                                        * (1 + dx)
                                        * (1 - dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_4, :, :]
                                        )
                                        * (1 + dx)
                                        * (dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_2, :, :]
                                        )
                                        * -dx
                                    )
                                else:
                                    orientation_matrix = (
                                        np.squeeze(
                                            self.orientation_rotation_matrices[
                                                ind_best_fit, :, :
                                            ]
                                        )
                                        * (1 + dx)
                                        * (1 + dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_3, :, :]
                                        )
                                        * (1 + dx)
                                        * (-dy)
                                        + np.squeeze(
                                            self.orientation_rotation_matrices[ind_1, :, :]
                                        )
                                        * -dx
                                    )

                # apply in-plane rotation, and inversion if needed
                if multiple_corr_reset and match_ind > 0:
                    phi = corr_in_plane_angle_keep[ind_best_fit]                     
                else:
                    phi = corr_in_plane_angle[ind_best_fit] 
                if inversion_symmetry and corr_inv[ind_best_fit]:
                    m3z = np.array(
                        [
                            [np.cos(phi), np.sin(phi), 0],
                            [-np.sin(phi), np.cos(phi), 0],
                            [0, 0, -1],
                        ]
                    )
                else:
                    m3z = np.array(
                        [
                            [np.cos(phi), np.sin(phi), 0],
                            [-np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1],
                        ]
                    )
                orientation_matrix = orientation_matrix @ m3z
                # if inversion_symmetry and corr_inv[ind_best_fit]:
                #     orientation_matrix = np.linalg.inv(np.linalg.inv(orientation_matrix) 
                #         @ np.array([
                #         [1,0,0],
                #         [0,-1,0],
                #         [0,0,-1],
                #     ])@ np.array([
                #         [-1,0,0],
                #         [0,-1,0],
                #         [0,0,1],
                #     ]))
                #     print(np.round(orientation_matrix,decimals=2))
       
                # if multiple_corr_reset:

                # else:
                #     corr_output = 


            else:
                # No more matches are detector, so output default orienation matrix and corr = 0
                orientation_matrix = np.squeeze(
                        self.orientation_rotation_matrices[0, :, :]
                    )
                if multiple_corr_reset and match_ind > 0:
                    corr_value_keep[ind_best_fit] = 0;

            # Output the orientation matrix
            if num_matches_return == 1:
                orientation_output = orientation_matrix
                if multiple_corr_reset and match_ind > 0:
                    corr_output = corr_value_keep[ind_best_fit]
                else:
                    corr_output = corr_value[ind_best_fit]

            else:
                orientation_output[:, :, match_ind] = orientation_matrix
                if multiple_corr_reset and match_ind > 0:
                    corr_output[match_ind] = corr_value_keep[ind_best_fit]
                else:
                    corr_output[match_ind] = corr_value[ind_best_fit]

            if verbose:
                zone_axis_fit = orientation_matrix[:, 2]

                if not self.cartesian_directions:
                    # TODO - I definitely think this should be cartesian--> zone,
                    # but that seems to return incorrect labels.  Not sure why! -CO

                    # zone_axis_fit = self.cartesian_to_crystal(zone_axis_fit)
                    zone_axis_fit = self.crystal_to_cartesian(zone_axis_fit)

                temp = zone_axis_fit / np.linalg.norm(zone_axis_fit)
                temp = np.round(temp, decimals=3)
                if multiple_corr_reset and match_ind > 0:
                    print(
                        "Best fit zone axis = ("
                        + str(temp)
                        + ")"
                        + " with corr value = "
                        + str(np.round(corr_value_keep[ind_best_fit],decimals=3))
                    )
                else:
                    print(
                        "Best fit zone axis = ("
                        + str(temp)
                        + ")"
                        + " with corr value = "
                        + str(np.round(corr_value[ind_best_fit],decimals=3))
                    )

            # if needed, delete peaks for next iteration
            if num_matches_return > 1 and corr_value[ind_best_fit] > 0:
                bragg_peaks_fit = self.generate_diffraction_pattern(
                    orientation_matrix,
                    sigma_excitation_error=self.orientation_kernel_size,
                )

                remove = np.zeros_like(qx, dtype="bool")
                scale_int = np.zeros_like(qx)
                for a0 in np.arange(qx.size):
                    d_2 = (bragg_peaks_fit.data["qx"] - qx[a0]) ** 2 + (
                        bragg_peaks_fit.data["qy"] - qy[a0]
                    ) ** 2

                    dist_min = np.sqrt(np.min(d_2))

                    if dist_min < self.orientation_tol_peak_delete:
                        remove[a0] = True
                    elif dist_min < self.orientation_kernel_size:
                        scale_int[a0] = (dist_min - self.orientation_tol_peak_delete) / (
                            self.orientation_kernel_size - self.orientation_tol_peak_delete
                        )

                intensity = intensity * scale_int
                qx = qx[~remove]
                qy = qy[~remove]
                intensity = intensity[~remove]




        # Plot polar space image
        if plot_polar is True:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_polar)

        # plotting correlation image
        if plot_corr is True:

            if self.orientation_full:
                fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 2]))
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros(
                    (
                        2 * self.orientation_zone_axis_steps + 1,
                        2 * self.orientation_zone_axis_steps + 1,
                    )
                )

                sub = self.orientation_inds[:, 2] == 0
                x_inds = (
                    self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
                ).astype("int") + self.orientation_zone_axis_steps
                y_inds = (
                    self.orientation_inds[sub, 1].astype("int")
                    + self.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:, 2] == 1
                x_inds = (
                    self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
                ).astype("int") + self.orientation_zone_axis_steps
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:, 2] == 2
                x_inds = (
                    self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
                ).astype("int") + self.orientation_zone_axis_steps
                y_inds = (
                    self.orientation_inds[sub, 1].astype("int")
                    + self.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:, 2] == 3
                x_inds = (
                    self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
                ).astype("int") + self.orientation_zone_axis_steps
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

            elif self.orientation_half:
                fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 1]))
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros(
                    (
                        self.orientation_zone_axis_steps + 1,
                        self.orientation_zone_axis_steps * 2 + 1,
                    )
                )

                sub = self.orientation_inds[:, 2] == 0
                x_inds = (
                    self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
                ).astype("int")
                y_inds = (
                    self.orientation_inds[sub, 1].astype("int")
                    + self.orientation_zone_axis_steps
                )
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                sub = self.orientation_inds[:, 2] == 1
                x_inds = (
                    self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
                ).astype("int")
                y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                    sub, 1
                ].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

                im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

            else:
                fig, ax = plt.subplots(1, 2, figsize=figsize)
                cmin = np.min(corr_value)
                cmax = np.max(corr_value)

                im_corr_zone_axis = np.zeros(
                    (
                        self.orientation_zone_axis_steps + 1,
                        self.orientation_zone_axis_steps + 1,
                    )
                )
                im_mask = np.ones(
                    (
                        self.orientation_zone_axis_steps + 1,
                        self.orientation_zone_axis_steps + 1,
                    ),
                    dtype="bool",
                )

                x_inds = (
                    self.orientation_inds[:, 0] - self.orientation_inds[:, 1]
                ).astype("int")
                y_inds = self.orientation_inds[:, 1].astype("int")
                inds_1D = np.ravel_multi_index(
                    [x_inds, y_inds], im_corr_zone_axis.shape
                )
                im_corr_zone_axis.ravel()[inds_1D] = corr_value
                im_mask.ravel()[inds_1D] = False

                im_plot = np.ma.masked_array(
                    (im_corr_zone_axis - cmin) / (cmax - cmin), mask=im_mask
                )

                ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)
                ax[0].spines["left"].set_color("none")
                ax[0].spines["right"].set_color("none")
                ax[0].spines["top"].set_color("none")
                ax[0].spines["bottom"].set_color("none")

                inds_plot = np.unravel_index(
                    np.argmax(im_plot, axis=None), im_plot.shape
                )
                ax[0].scatter(
                    inds_plot[1],
                    inds_plot[0],
                    s=120,
                    linewidth=2,
                    facecolors="none",
                    edgecolors="r",
                )

                label_0 = self.orientation_zone_axis_range[0, :]
                label_0 = np.round(label_0 * 1e3) * 1e-3
                label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))

                label_1 = self.orientation_zone_axis_range[1, :]
                label_1 = np.round(label_1 * 1e3) * 1e-3
                label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))

                label_2 = self.orientation_zone_axis_range[2, :]
                label_2 = np.round(label_2 * 1e3) * 1e-3
                label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))

                ax[0].set_xticks([0, self.orientation_zone_axis_steps])
                ax[0].set_xticklabels([str(label_0), str(label_2)], size=14)
                ax[0].xaxis.tick_top()

                ax[0].set_yticks([self.orientation_zone_axis_steps])
                ax[0].set_yticklabels([str(label_1)], size=14)

            # In-plane rotation
            # ax[1].plot(
            #     self.orientation_gamma * 180 / np.pi,
            #     (np.squeeze(corr_full[ind_best_fit, :]) - cmin) / (cmax - cmin),
            # )
            sig_in_plane = np.squeeze(corr_full[ind_best_fit, :])
            ax[1].plot(
                self.orientation_gamma * 180 / np.pi,
                sig_in_plane / np.max(sig_in_plane),
            )
            ax[1].set_xlabel("In-plane rotation angle [deg]", size=16)
            ax[1].set_ylabel("Corr. of Best Fit Zone Axis", size=16)
            ax[1].set_ylim([0, 1.01])

            plt.show()

        if plot_corr_3D is True:
            # 3D plotting

            fig = plt.figure(figsize=[figsize[0], figsize[0]])
            ax = fig.add_subplot(projection="3d", elev=90, azim=0)

            sig_zone_axis = np.max(corr, axis=1)

            el = self.orientation_rotation_angles[:, 0, 0]
            az = self.orientation_rotation_angles[:, 0, 1]
            x = np.cos(az) * np.sin(el)
            y = np.sin(az) * np.sin(el)
            z = np.cos(el)

            v = np.vstack((x.ravel(), y.ravel(), z.ravel()))

            v_order = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ]
            )
            d_sign = np.array(
                [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1],
                ]
            )

            for a1 in range(d_sign.shape[0]):
                for a0 in range(v_order.shape[0]):
                    ax.scatter(
                        xs=v[v_order[a0, 0]] * d_sign[a1, 0],
                        ys=v[v_order[a0, 1]] * d_sign[a1, 1],
                        zs=v[v_order[a0, 2]] * d_sign[a1, 2],
                        s=30,
                        c=sig_zone_axis.ravel(),
                        edgecolors=None,
                    )

            # axes limits
            r = 1.05
            ax.axes.set_xlim3d(left=-r, right=r)
            ax.axes.set_ylim3d(bottom=-r, top=r)
            ax.axes.set_zlim3d(bottom=-r, top=r)
            axisEqual3D(ax)

            plt.show()

        if return_corr:
            if returnfig:
                return orientation_output, corr_output, fig, ax
            else:
                return orientation_output, corr_output
        else:
            if returnfig:
                return orientation_output, fig, ax
            else:
                return orientation_output

    def generate_diffraction_pattern(
        self,
        zone_axis: Union[list, tuple, np.ndarray] = [0, 0, 1],
        foil_normal: Optional[Union[list, tuple, np.ndarray]] = None,
        proj_x_axis: Optional[Union[list, tuple, np.ndarray]] = None,
        sigma_excitation_error: float = 0.02,
        tol_excitation_error_mult: float = 3,
        tol_intensity: float = 0.1,
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
        # scale = 1/cos_alpha[keep]
        kx_proj = kx_proj / np.linalg.norm(kx_proj)
        ky_proj = ky_proj / np.linalg.norm(ky_proj)
        gx_proj = np.sum(g_diff[:, keep_int] * kx_proj[:, None], axis=0)
        gy_proj = np.sum(g_diff[:, keep_int] * ky_proj[:, None], axis=0)

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
        bragg_peaks.add_pointarray(
            np.vstack((gx_proj, gy_proj, g_int[keep_int], h, k, l)).T
        )

        return bragg_peaks

    def plot_orientation_maps(
        self,
        orientation_matrices: np.ndarray,
        corr_all: Optional[np.ndarray] = None,
        corr_range: np.ndarray = np.array([0, 5]),
        orientation_index_plot: int = 0,
        orientation_rotate_xy: bool = None,
        scale_legend: bool = None,
        corr_normalize: bool = True,
        figsize: Union[list, tuple, np.ndarray] = (20, 5),
        figlayout: Union[list, tuple, np.ndarray] = np.array([1, 4]),
        returnfig: bool = False,
    ):
        """
        Generate and plot the orientation maps

        Args:
            orientation_zone_axis_range(float):     numpy array (3,3) where the 3 rows are the basis vectors for the orientation triangle
            orientation_matrices (float):   numpy array containing orientations, with size (Rx, Ry, 3, 3) or (Rx, Ry, 3, 3, num_matches)
            corr_all(float):                numpy array containing the correlation values to use as a mask
            orientation_index_plot (int):   index of orientations to plot
            orientation_rotate_xy (float):  rotation in radians for the xy directions of plots
            scale_legend (float):           2 elements, x and y scaling of legend panel
            figlayout (int)                 2 elements giving the # of rows and columns for the figure.
                                            Must be [1, 4], [2, 2] or [4,1] currently.
            returnfig (bool):               set to True to return figure and axes handles

        Returns:
            images_orientation (int):       RGB images
            fig, axs (handles):             Figure and axes handes for the

        NOTE:
            Currently, no symmetry reduction.  Therefore the x and y orientations
            are going to be correct only for [001][011][111] orientation triangle.

        """

        # Inputs
        # Legend size
        leg_size = np.array([300, 300], dtype="int")

        # Color of the 3 corners
        color_basis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.3, 1.0],
            ]
        )

        # Basis for fitting
        A = self.orientation_zone_axis_range.T

        # Basis for fitting fiber texture
        if self.orientation_fiber:
            p1_proj = (
                np.dot(
                    self.orientation_zone_axis_range[1, :],
                    self.orientation_zone_axis_range[0, :],
                )
                * self.orientation_zone_axis_range[0, :]
            )
            p2_proj = (
                np.dot(
                    self.orientation_zone_axis_range[2, :],
                    self.orientation_zone_axis_range[0, :],
                )
                * self.orientation_zone_axis_range[0, :]
            )
            p1_sub = self.orientation_zone_axis_range[1, :] - p1_proj
            p2_sub = self.orientation_zone_axis_range[2, :] - p2_proj
            B = np.vstack((self.orientation_zone_axis_range[0, :], p1_sub, p2_sub)).T

        # initalize image arrays
        images_orientation = np.zeros(
            (orientation_matrices.shape[0], orientation_matrices.shape[1], 3, 3)
        )

        # in-plane rotation array if needed
        if orientation_rotate_xy is not None:
            m = np.array(
                [
                    [np.cos(orientation_rotate_xy), -np.sin(orientation_rotate_xy), 0],
                    [np.sin(orientation_rotate_xy), np.cos(orientation_rotate_xy), 0],
                    [0, 0, 1],
                ]
            )

        # loop over all pixels and calculate weights
        for ax in range(orientation_matrices.shape[0]):
            for ay in range(orientation_matrices.shape[1]):
                if orientation_matrices.ndim == 4:
                    orient = orientation_matrices[ax, ay, :, :]
                else:
                    orient = orientation_matrices[ax, ay, :, :, orientation_index_plot]

                # Rotate in-plane if needed
                if orientation_rotate_xy is not None:
                    orient = m @ orient

                if self.orientation_fiber:
                    # in-plane rotation
                    # w = np.linalg.solve(A, orient[:, 0])
                    w = np.linalg.solve(B, orient[:, 0])

                    h = np.mod(
                        np.arctan2(w[2], w[1])
                        * 180
                        / np.pi
                        / self.orientation_fiber_angles[1],
                        1,
                    )
                    w0 = np.maximum(1 - 3 * np.abs(np.mod(3 / 6 - h, 1) - 1 / 2), 0)
                    w1 = np.maximum(1 - 3 * np.abs(np.mod(5 / 6 - h, 1) - 1 / 2), 0)
                    w2 = np.maximum(1 - 3 * np.abs(np.mod(7 / 6 - h, 1) - 1 / 2), 0)
                    w_scale = 1 / (1 - np.exp(-np.max((w0, w1, w2))))

                    rgb = (
                        color_basis[0, :] * w0 * w_scale
                        + color_basis[1, :] * w1 * w_scale
                        + color_basis[2, :] * w2 * w_scale
                    )
                    images_orientation[ax, ay, :, 0] = rgb

                    # zone axis
                    w = np.linalg.solve(A, orient[:, 2])
                    w = w / (1 - np.exp(-np.max(w)))
                    rgb = (
                        color_basis[0, :] * w[0]
                        + color_basis[1, :] * w[1]
                        + color_basis[2, :] * w[2]
                    )
                    images_orientation[ax, ay, :, 2] = rgb

                else:
                    for a0 in range(3):
                        # Cubic sorting for now - needs to be updated with symmetries
                        # w = np.linalg.solve(A,orient[:,a0])
                        w = np.linalg.solve(A, np.sort(np.abs(orient[:, a0])))
                        # w = np.linalg.solve(A, orient[:, a0])
                        w = w / (1 - np.exp(-np.max(w)))

                        rgb = (
                            color_basis[0, :] * w[0]
                            + color_basis[1, :] * w[1]
                            + color_basis[2, :] * w[2]
                        )
                        images_orientation[ax, ay, :, a0] = rgb

        # clip range
        images_orientation = np.clip(images_orientation, 0, 1)

        # Masking
        if corr_all is not None:
            if orientation_matrices.ndim == 4:
                if corr_normalize:
                    mask = corr_all / np.mean(corr_all)
                else:
                    mask = corr_all
            else:
                if corr_normalize:
                    mask = corr_all[:, :, orientation_index_plot] / np.mean(
                        corr_all[:, :, orientation_index_plot]
                    )
                else:
                    mask = corr_all[:, :, orientation_index_plot]

            mask = (mask - corr_range[0]) / (corr_range[1] - corr_range[0])
            mask = np.clip(mask, 0, 1)

            for a0 in range(3):
                for a1 in range(3):
                    images_orientation[:, :, a0, a1] *= mask

        # Draw legend for zone axis
        x = np.linspace(0, 1, leg_size[0])
        y = np.linspace(0, 1, leg_size[1])
        ya, xa = np.meshgrid(y, x)
        mask_legend = np.logical_and(2 * xa > ya, 2 * xa < 2 - ya)
        w0 = 1 - xa - 0.5 * ya
        w1 = xa - 0.5 * ya
        w2 = ya

        w_scale = np.maximum(np.maximum(w0, w1), w2)
        # w_scale = w0 + w1 + w2
        # w_scale = (w0**4 + w1**4 + w2**4)**0.25
        w_scale = 1 - np.exp(-w_scale)
        w0 = w0 / w_scale  # * mask_legend
        w1 = w1 / w_scale  # * mask_legend
        w2 = w2 / w_scale  # * mask_legend

        im_legend = np.zeros((leg_size[0], leg_size[1], 3))
        for a0 in range(3):
            im_legend[:, :, a0] = (
                w0 * color_basis[0, a0]
                + w1 * color_basis[1, a0]
                + w2 * color_basis[2, a0]
            )
            im_legend[:, :, a0] *= mask_legend
            im_legend[:, :, a0] += 1 - mask_legend
        im_legend = np.clip(im_legend, 0, 1)

        if self.orientation_fiber:
            # Draw legend for in-plane rotation
            x = np.linspace(-1, 1, leg_size[0])
            y = np.linspace(-1, 1, leg_size[1])
            ya, xa = np.meshgrid(y, x)
            mask_legend = xa ** 2 + ya ** 2 <= 1

            h = np.mod(
                np.arctan2(ya, xa) * 180 / np.pi / self.orientation_fiber_angles[1], 1
            )
            w0 = np.maximum(1 - 3 * np.abs(np.mod(3 / 6 - h, 1) - 1 / 2), 0)
            w1 = np.maximum(1 - 3 * np.abs(np.mod(5 / 6 - h, 1) - 1 / 2), 0)
            w2 = np.maximum(1 - 3 * np.abs(np.mod(7 / 6 - h, 1) - 1 / 2), 0)

            w_scale = np.maximum(np.maximum(w0, w1), w2)
            # w_scale = w0 + w1 + w2
            # w_scale = (w0**4 + w1**4 + w2**4)**0.25
            w_scale = 1 - np.exp(-w_scale)
            w0 = w0 / w_scale  # * mask_legend
            w1 = w1 / w_scale  # * mask_legend
            w2 = w2 / w_scale  # * mask_legend

            inplane_legend = np.zeros((leg_size[0], leg_size[1], 3))
            for a0 in range(3):
                inplane_legend[:, :, a0] = (
                    w0 * color_basis[0, a0]
                    + w1 * color_basis[1, a0]
                    + w2 * color_basis[2, a0]
                )
                inplane_legend[:, :, a0] *= mask_legend
                inplane_legend[:, :, a0] += 1 - mask_legend
            inplane_legend = np.clip(inplane_legend, 0, 1)

        # plotting
        if figlayout[0] == 1 and figlayout[1] == 4:
            fig, ax = plt.subplots(1, 4, figsize=figsize)
        elif figlayout[0] == 2 and figlayout[1] == 2:
            fig, ax = plt.subplots(2, 2, figsize=figsize)
            ax = np.array(
                [
                    ax[0, 0],
                    ax[0, 1],
                    ax[1, 0],
                    ax[1, 1],
                ]
            )
        elif figlayout[0] == 4 and figlayout[1] == 1:
            fig, ax = plt.subplots(4, 1, figsize=figsize)

        ax[0].imshow(images_orientation[:, :, :, 0])
        if self.orientation_fiber:
            ax[1].imshow(inplane_legend, aspect="auto")
        else:
            ax[1].imshow(images_orientation[:, :, :, 1])
        ax[2].imshow(images_orientation[:, :, :, 2])

        if self.orientation_fiber:
            ax[0].set_title("In-Plane Rotation", size=20)
            # ax[1].imshow(im_legend, aspect="auto")

        else:
            ax[0].set_title("Orientation of x-axis", size=20)
            ax[1].set_title("Orientation of y-axis", size=20)
        ax[2].set_title("Zone Axis", size=20)
        ax[0].xaxis.tick_top()
        ax[1].xaxis.tick_top()
        ax[2].xaxis.tick_top()

        # Legend
        ax[3].imshow(im_legend, aspect="auto")

        label_0 = self.orientation_zone_axis_range[0, :]
        label_0 = np.round(label_0 * 1e3) * 1e-3
        label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))

        label_1 = self.orientation_zone_axis_range[1, :]
        label_1 = np.round(label_1 * 1e3) * 1e-3
        label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))

        label_2 = self.orientation_zone_axis_range[2, :]
        label_2 = np.round(label_2 * 1e3) * 1e-3
        label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))

        ax[3].yaxis.tick_right()
        ax[3].set_yticks([(leg_size[0] - 1) / 2])
        ax[3].set_yticklabels([str(label_2)])

        ax3a = ax[3].twiny()
        ax3b = ax[3].twiny()

        ax3a.set_xticks([0])
        ax3a.set_xticklabels([str(label_0)])
        ax3a.xaxis.tick_top()
        ax3b.set_xticks([0])
        ax3b.set_xticklabels([str(label_1)])
        ax3b.xaxis.tick_bottom()
        ax[3].set_xticks([])

        # ax[3].xaxis.label.set_color('none')
        ax[3].spines["left"].set_color("none")
        ax[3].spines["right"].set_color("none")
        ax[3].spines["top"].set_color("none")
        ax[3].spines["bottom"].set_color("none")

        ax3a.spines["left"].set_color("none")
        ax3a.spines["right"].set_color("none")
        ax3a.spines["top"].set_color("none")
        ax3a.spines["bottom"].set_color("none")

        ax3b.spines["left"].set_color("none")
        ax3b.spines["right"].set_color("none")
        ax3b.spines["top"].set_color("none")
        ax3b.spines["bottom"].set_color("none")

        ax[3].tick_params(labelsize=16)
        ax3a.tick_params(labelsize=16)
        ax3b.tick_params(labelsize=16)

        if self.orientation_fiber:
            ax[1].axis("off")

        if scale_legend is not None:
            pos = ax[3].get_position()
            pos_new = [
                pos.x0,
                pos.y0 + pos.height * (1 - scale_legend[1]) / 2,
                pos.width * scale_legend[0],
                pos.height * scale_legend[1],
            ]
            ax[3].set_position(pos_new)

            if self.orientation_fiber:
                pos = ax[1].get_position()
                if np.size(scale_legend) == 2:
                    pos_new = [
                        pos.x0,
                        pos.y0 + pos.height * (1 - scale_legend[1]) / 2,
                        pos.width * scale_legend[0],
                        pos.height * scale_legend[1],
                    ]
                elif np.size(scale_legend) == 4:
                    pos_new = [
                        pos.x0,
                        pos.y0 + pos.height * (1 - scale_legend[3]) / 2,
                        pos.width * scale_legend[2],
                        pos.height * scale_legend[3],
                    ]
                ax[1].set_position(pos_new)

        if returnfig:
            return images_orientation, fig, ax
        else:
            return images_orientation

    # def crystal_to_cartesian(self, zone_axis):
    #     vec_cart = zone_axis @ self.lat_real
    #     return vec_cart / np.linalg.norm(vec_cart)

    # def cartesian_to_crystal(self, vec_cart):
    #     zone_axis = vec_cart @ np.linalg.inv(self.lat_real)
    #     return zone_axis / np.linalg.norm(zone_axis)

    def cartesian_to_crystal(self, zone_axis):
        vec_cart = zone_axis @ self.lat_real
        return vec_cart / np.linalg.norm(vec_cart)

    def crystal_to_cartesian(self, vec_cart):
        zone_axis = vec_cart @ np.linalg.inv(self.lat_real)
        return zone_axis / np.linalg.norm(zone_axis)


def plot_diffraction_pattern(
    bragg_peaks: PointList,
    bragg_peaks_compare: PointList = None,
    scale_markers: float = 10,
    scale_markers_compare: Optional[float] = None,
    power_markers: float = 1,
    plot_range_kx_ky: Optional[Union[list, tuple, np.ndarray]] = None,
    add_labels: bool = True,
    shift_labels: float = 0.08,
    shift_marker: float = 0.005,
    min_marker_size: float = 1e-6,
    figsize: Union[list, tuple, np.ndarray] = (8, 8),
    returnfig: bool = False,
    input_fig_handle=None,
):
    """
    2D scatter plot of the Bragg peaks

    Args:
        bragg_peaks (PointList):        numpy array containing ('qx', 'qy', 'intensity', 'h', 'k', 'l')
        bragg_peaks_compare(PointList): numpy array containing ('qx', 'qy', 'intensity')
        scale_markers (float):          size scaling for markers
        scale_markers_compare (float):  size scaling for markers of comparison
        power_markers (float):          power law scaling for marks (default is 1, i.e. amplitude)
        plot_range_kx_ky (float):       2 element numpy vector giving the plot range
        add_labels (bool):              flag to add hkl labels to peaks
        min_marker_size (float):        minimum marker size for the comparison peaks
        figsize (2 element float):      size scaling of figure axes
        returnfig (bool):               set to True to return figure and axes handles
    """

    # 2D plotting
    if input_fig_handle is None:
        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = input_fig_handle[0]
        ax_parent = input_fig_handle[1]
        ax = ax_parent[0]

    if power_markers == 2:
        marker_size = scale_markers * bragg_peaks.data["intensity"]
    else:
        marker_size = scale_markers * (
            bragg_peaks.data["intensity"] ** (power_markers / 2)
        )

    if bragg_peaks_compare is None:
        ax.scatter(
            bragg_peaks.data["qy"], bragg_peaks.data["qx"], s=marker_size, facecolor="k"
        )
    else:
        if scale_markers_compare is None:
            scale_markers_compare = scale_markers

        if power_markers == 2:
            marker_size_compare = np.maximum(
                scale_markers_compare * bragg_peaks_compare.data["intensity"],
                min_marker_size,
            )
        else:
            marker_size_compare = np.maximum(
                scale_markers_compare
                * (bragg_peaks_compare.data["intensity"] ** (power_markers / 2)),
                min_marker_size,
            )

        ax.scatter(
            bragg_peaks_compare.data["qy"],
            bragg_peaks_compare.data["qx"],
            s=marker_size_compare,
            marker="o",
            facecolor=[0.0, 0.7, 1.0],
        )
        ax.scatter(
            bragg_peaks.data["qy"],
            bragg_peaks.data["qx"],
            s=marker_size,
            marker="+",
            facecolor="k",
        )

    if plot_range_kx_ky is not None:
        ax.set_xlim((-plot_range_kx_ky[0], plot_range_kx_ky[0]))
        ax.set_ylim((-plot_range_kx_ky[1], plot_range_kx_ky[1]))

    ax.invert_yaxis()
    # ax.set_box_aspect(1)
    ax.xaxis.tick_top()

    # Labels for all peaks
    if add_labels is True:
        text_params = {
            "ha": "center",
            "va": "center",
            "family": "sans-serif",
            "fontweight": "normal",
            "color": "r",
            "size": 10,
        }

        def overline(x):
            return str(x) if x >= 0 else (r"\overline{" + str(np.abs(x)) + "}")

        for a0 in range(bragg_peaks.data.shape[0]):
            h = bragg_peaks.data["h"][a0]
            k = bragg_peaks.data["k"][a0]
            l = bragg_peaks.data["l"][a0]

            ax.text(
                bragg_peaks.data["qy"][a0],
                bragg_peaks.data["qx"][a0]
                - shift_labels
                - shift_marker * np.sqrt(marker_size[a0]),
                "$" + overline(h) + overline(k) + overline(l) + "$",
                **text_params,
            )

    # Force plot to have 1:1 aspect ratio
    ax.set_aspect("equal")

    if input_fig_handle is None:
        plt.show()

    if returnfig:
        return fig, ax


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def atomic_colors(ID):
    return {
        1: np.array([0.8, 0.8, 0.8]),
        2: np.array([1.0, 0.7, 0.0]),
        3: np.array([1.0, 0.0, 1.0]),
        4: np.array([0.0, 0.5, 0.0]),
        5: np.array([0.5, 0.0, 0.0]),
        6: np.array([0.5, 0.5, 0.5]),
        7: np.array([0.0, 0.7, 1.0]),
        8: np.array([1.0, 0.0, 0.0]),
        13: np.array([0.6, 0.7, 0.8]),
        14: np.array([0.3, 0.3, 0.3]),
        15: np.array([1.0, 0.6, 0.0]),
        16: np.array([1.0, 0.9, 0.0]),
        17: np.array([0.0, 1.0, 0.0]),
        79: np.array([1.0, 0.7, 0.0]),
    }.get(ID, np.array([0.0, 0.0, 0.0]))

# zone axis range arguments for orientation_plan corresponding
# to the symmetric wedge for each pointgroup, in the order:
#   [zone_axis_range, fiber_axis, fiber_angles]
orientation_ranges = {
    '1':    ['fiber', [0,0,1], [180., 360.]],
    '-1':   ['full', None, None],
    '2':    ['fiber', [0,0,1], [180., 180.]],
    'm':    ['full', None, None],
    '2/m':  ['half', None, None],
    '222':  ['fiber', [0,0,1], [90., 180.]],
    'mm2':  ['fiber', [0,0,1], [180., 90.]],
    'mmm':  [[[1,0,0], [0,1,0]], None, None],
    '4':    ['fiber', [0,0,1], [90., 180.]],
    '-4':   ['half', None, None],
    '4/m':  [[[1,0,0], [0,1,0]], None, None],
    '422':  ['fiber', [0,0,1], [180., 45.]],
    '4mm':  ['fiber', [0,0,1], [180., 45.]],
    '-42m': ['fiber', [0,0,1], [180., 45.]],
    '4/mmm':[[[1,0,0], [1,1,0]], None, None],
    '3':    ['fiber', [0,0,1], [180., 120.]],
    '-3':   ['fiber', [0,0,1], [180., 60.]],
    '32':   ['fiber', [0,0,1], [90., 60.]],
    '3m':   ['fiber', [0,0,1], [180., 60.]],
    '-3m':  ['fiber', [0,0,1], [180., 30.]],
    '6':    ['fiber', [0,0,1], [180., 60.]],
    '-6':   ['fiber', [0,0,1], [180., 60.]],
    '6/m':  [[[1,0,0],[0.5,np.sqrt(3)/2.,0]], None, None],
    '622':  ['fiber', [0,0,1], [180., 30.]],
    '6mm':  ['fiber', [0,0,1], [180., 30.]],
    '-6m2': ['fiber', [0,0,1], [90., 60.]],
    '6/mmm':[[[np.sqrt(3)/2.,0.5,0.],[1,0,0]], None, None],
    '23':   [[[1,0,0], [1,1,1]], None, None], # this is probably wrong, it is half the needed range
    'm-3':  [[[1,0,0], [1,1,1]], None, None],
    '432':  [[[1,0,0], [1,1,1]], None, None],
    '-43m': [[[1,-1,1], [1,1,1]], None, None],
    'm-3m': [[[0,1,1], [1,1,1]], None, None],
}
