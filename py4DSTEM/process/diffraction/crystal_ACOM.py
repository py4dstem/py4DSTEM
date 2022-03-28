import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, electron_wavelength_angstrom
from .utils import Orientation, OrientationMap, axisEqual3D

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core.structure import Structure
except ImportError:
    pass

def orientation_plan(
    self,
    zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
    angle_step_zone_axis: float = 2.0,
    angle_step_in_plane: float = 2.0,
    accel_voltage: float = 300e3,
    corr_kernel_size: float = 0.08,
    radial_power: float = 1.0,
    intensity_power: float = 0.25,  # New default intensity power scaling
    tol_peak_delete=None,
    tol_distance: float = 0.01,
    fiber_axis=None,
    fiber_angles=None,
    figsize: Union[list, tuple, np.ndarray] = (6, 6),
    progress_bar: bool = True,
    ):

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
    if fiber_axis is None:
        self.orientation_fiber_axis = None
    else:
        self.orientation_fiber_axis = np.asarray(fiber_axis)
    if fiber_angles is None:
        self.orientation_fiber_angles = None
    else:
        self.orientation_fiber_angles = np.asarray(fiber_angles)

    

    # Calculate wavelenth
    self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

    # store the radial and intensity scaling to use later for generating test patterns
    self.orientation_radial_power = radial_power
    self.orientation_intensity_power = intensity_power

    # Handle the "auto" case first, since it works by overriding zone_axis_range,
    #   fiber_axis, and fiber_angles then using the regular parser:
    if isinstance(zone_axis_range, str) and zone_axis_range == "auto":
        # from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        # from pymatgen.core.structure import Structure

        # initialize structure
        structure = Structure(
            self.lat_real, self.numbers, self.positions, coords_are_cartesian=False
        )

        # pointgroup = SpacegroupAnalyzer(structure).get_point_group_symbol()
        # self.pointgroup = pointgroup
        self.pointgroup = SpacegroupAnalyzer(structure)

        assert (
            self.pointgroup.get_point_group_symbol() in orientation_ranges
        ), "Unrecognized pointgroup returned by pymatgen!"

        zone_axis_range, fiber_axis, fiber_angles = orientation_ranges[ \
            self.pointgroup.get_point_group_symbol()]
        if isinstance(zone_axis_range, list):
            zone_axis_range = np.array(zone_axis_range)
        elif zone_axis_range == "fiber":
            self.orientation_fiber_axis = np.asarray(fiber_axis)
            self.orientation_fiber_angles = np.asarray(fiber_angles)

        print(
            f"Automatically detected point group {self.pointgroup.get_point_group_symbol()},\n"
            f" using arguments: zone_axis_range = \n{zone_axis_range}, \n fiber_axis={fiber_axis}, fiber_angles={fiber_angles}."
        )

        # Set a flag so we know pymatgen is available
        self.pymatgen_available = True


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
            # if self.cartesian_directions:
            self.orientation_fiber_axis = (
                self.orientation_fiber_axis
                / np.linalg.norm(self.orientation_fiber_axis)
            )

            # update fiber axis to be centered on the 1st unit cell vector
            v3 = np.cross(self.orientation_fiber_axis, self.lat_real[0,:])
            v2 = np.cross(v3, self.orientation_fiber_axis,)
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)

            # v2 = self.lat_real[0,:]
            # v3 = np.cross(self.orientation_fiber_axis, v2)


            # # else:
            # #     self.orientation_fiber_axis = self.crystal_to_cartesian(
            # #         self.orientation_fiber_axis
            # #     )

            # # Generate 2 perpendicular vectors to self.orientation_fiber_axis
            # # TESTING - different defaults for the zone axis ranges
            # # if np.sum(np.abs(
            # #     self.orientation_fiber_axis - np.array([0.0, 1.0, 0.0])
            # #     )) < 1e-6:
            # #     v0 = np.array([1.0, 0.0, 0.0])
            # # else:
            # #     v0 = np.array([0.0, -1.0, 0.0])
            # if np.linalg.norm(np.abs(self.orientation_fiber_axis) \
            #     - np.array([1.0, 0.0, 0.0])) < 1e-6:
            #     v0 = np.array([0.0, -1.0, 0.0])
            # else:
            #     v0 = np.array([1.0, 0.0, 0.0])
            # if np.linalg.norm(np.abs(self.orientation_fiber_axis) \
            #     - np.array([0.0, 1.0, 0.0])) < 1e-6:
            #     v0 = np.array([1.0, 0.0, 0.0])
            # else:
            #     v0 = np.array([0.0, 1.0, 0.0])

            # # v2 = np.cross(self.orientation_fiber_axis, v0)
            # # v3 = np.cross(v2, self.orientation_fiber_axis)
            # # v2 = v2 / np.linalg.norm(v2)
            # # v3 = v3 / np.linalg.norm(v3)
            # v2 = np.cross(v0, self.orientation_fiber_axis,)
            # v3 = np.cross(self.orientation_fiber_axis, v2)
            # v2 = v2 / np.linalg.norm(v2)
            # v3 = v3 / np.linalg.norm(v3)

            if self.orientation_fiber_angles[0] == 0:
                self.orientation_zone_axis_range = np.vstack(
                    (self.orientation_fiber_axis, v2, v3)
                ).astype("float")
            else:

                if self.orientation_fiber_angles[0] == 180:
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

                # Generate zone axis range
                v2output = self.orientation_fiber_axis * np.cos(theta) \
                    + v2 * np.sin(theta)
                v3output = (
                    self.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi)
                    + (v3 * np.sin(theta)) * np.sin(phi)
                )
                v2output = (
                    self.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi/2)
                    - (v3 * np.sin(theta)) * np.sin(phi/2)
                )
                v3output = (
                    self.orientation_fiber_axis * np.cos(theta)
                    + (v2 * np.sin(theta)) * np.cos(phi/2)
                    + (v3 * np.sin(theta)) * np.sin(phi/2)
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
        # if not self.cartesian_directions:
        #     for a0 in range(zone_axis_range.shape[0]):
        #         self.orientation_zone_axis_range[a0, :] = self.crystal_to_cartesian(
        #             self.orientation_zone_axis_range[a0, :]
        #         )

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
        self.orientation_inds = np.zeros((self.orientation_num_zones, 3), dtype="int")

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
                    np.sum(p0_sub * p1_sub)
                    / np.linalg.norm(p0_sub)
                    / np.linalg.norm(p1_sub)
                )

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

    if self.orientation_fiber and self.orientation_fiber_angles[0] == 180:
        # Mirror about the equator of fiber_zone_axis
        m = np.identity(3) - 2 * (
            self.orientation_fiber_axis[:, None] @ self.orientation_fiber_axis[None, :]
        )

        vec_new = np.copy(self.orientation_vecs) @ m
        orientation_sector = np.zeros(vec_new.shape[0], dtype="int")

        keep = np.zeros(vec_new.shape[0], dtype="bool")
        for a0 in range(keep.size):
            if (
                np.sqrt(
                    np.min(
                        np.sum((self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
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
                        np.sum((self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
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
                        np.sum((self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
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
                        np.sum((self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
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
                        np.sum((self.orientation_vecs - vec_new[a0, :]) ** 2, axis=1)
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
    # azim = np.pi / 2 + np.arctan2(
    #     self.orientation_vecs[:, 1], self.orientation_vecs[:, 0]
    # )
    azim = np.arctan2(
        self.orientation_vecs[:, 0], self.orientation_vecs[:, 1]
    )

    # Solve for number of angular steps along in-plane rotation direction
    self.orientation_in_plane_steps = np.round(360 / angle_step_in_plane).astype(np.int)

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
    self.orientation_shell_index = -1 * np.ones(self.g_vec_all.shape[1], dtype="int")
    self.orientation_shell_count = np.zeros(self.orientation_shell_radii.size)

    # Assign each structure factor point to a radial shell
    for a0 in range(self.orientation_shell_radii.size):
        sub = np.abs(self.orientation_shell_radii[a0] - radii_test) <= tol_distance / 2

        self.orientation_shell_index[sub] = a0
        self.orientation_shell_count[a0] = np.sum(sub)
        self.orientation_shell_radii[a0] = np.mean(self.g_vec_leng[sub])

    # init storage arrays
    self.orientation_rotation_angles = np.zeros((self.orientation_num_zones, 3))
    self.orientation_rotation_matrices = np.zeros((self.orientation_num_zones, 3, 3))
    self.orientation_ref = np.zeros(
        (
            self.orientation_num_zones,
            np.size(self.orientation_shell_radii),
            self.orientation_in_plane_steps,
        ),
        dtype="float",
    )


    # If possible,  Get symmetry operations for this spacegroup, store in matrix form
    if self.pymatgen_available:
        # get operators
        ops = self.pointgroup.get_point_group_operations()

        # Inverse of lattice 
        zone_axis_range_inv = np.linalg.inv(self.orientation_zone_axis_range)

        # init
        num_sym = len(ops)
        self.symmetry_operators = np.zeros((num_sym,3,3)) 
        self.symmetry_reduction = np.zeros((num_sym,3,3))

        # calculate symmetry and reduction matrices
        for a0 in range(num_sym):
            self.symmetry_operators[a0] = \
                self.lat_inv.T @ ops[a0].rotation_matrix.T @ self.lat_real
            self.symmetry_reduction[a0] = \
                (zone_axis_range_inv.T @ self.symmetry_operators[a0]).T

        # Remove duplicates
        keep = np.ones(num_sym,dtype='bool')
        for a0 in range(num_sym):
            if keep[a0]:
                diff = np.sum(np.abs(
                    self.symmetry_operators - self.symmetry_operators[a0]),
                    axis=(1,2))
                sub = diff < 1e-3
                sub[:a0+1] = False
                keep[sub] = False
        self.symmetry_operators = self.symmetry_operators[keep]
        self.symmetry_reduction = self.symmetry_reduction[keep]
    
        if self.orientation_fiber_angles is not None \
            and np.abs(self.orientation_fiber_angles[0] - 180.0) < 1e-3:
            zone_axis_range_flip = self.orientation_zone_axis_range.copy()
            zone_axis_range_flip[0,:] = -1*zone_axis_range_flip[0,:]
            zone_axis_range_inv = np.linalg.inv(zone_axis_range_flip)

            num_sym = self.symmetry_operators.shape[0]
            self.symmetry_operators = np.tile(self.symmetry_operators,[2,1,1])
            self.symmetry_reduction = np.tile(self.symmetry_reduction,[2,1,1])

            for a0 in range(num_sym):
                self.symmetry_reduction[a0+num_sym] = \
                    (zone_axis_range_inv.T @ self.symmetry_operators[a0+num_sym]).T

    # Calculate rotation matrices for zone axes
    # for a0 in tqdmnd(np.arange(self.orientation_num_zones),desc='Computing orientation basis',unit=' terms',unit_scale=True):
    for a0 in np.arange(self.orientation_num_zones):
        m1z = np.array(
            [
                [np.cos(azim[a0]), np.sin(azim[a0]), 0],
                [-np.sin(azim[a0]), np.cos(azim[a0]), 0],
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
        m3z = np.array(
            [
                [np.cos(azim[a0]), -np.sin(azim[a0]), 0],
                [np.sin(azim[a0]), np.cos(azim[a0]), 0],
                [0, 0, 1],
            ]
        )
        self.orientation_rotation_matrices[a0, :, :] = m1z @ m2x @ m3z
        self.orientation_rotation_angles[a0, :] = [azim[a0], elev[a0], -azim[a0]]

    # Calculate reference arrays for all orientations
    k0 = np.array([0.0, 0.0, -1.0/self.wavelength])
    n = np.array([0.0, 0.0, -1.0])

    for a0 in tqdmnd(
        np.arange(self.orientation_num_zones),
        desc="Orientation plan",
        unit=" zone axes",
        disable=not progress_bar,
    ):
        # reciprocal lattice spots and excitation errors
        g = self.orientation_rotation_matrices[a0, :, :].T @ self.g_vec_all
        sg = self.excitation_errors(g)

        # Keep only points that will contribute to this orientation plan slice
        keep = np.abs(sg) < self.orientation_kernel_size

        # in-plane rotation angle
        phi = np.arctan2(g[1, :], g[0, :])

        # Loop over all peaks
        for a1 in np.arange(self.g_vec_all.shape[1]):
            ind_radial = self.orientation_shell_index[a1]

            if keep[a1] and ind_radial >= 0:
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
        self.orientation_ref[a0, :, :] /= np.sqrt(
            np.sum(self.orientation_ref[a0, :, :] ** 2)
        )

    # Maximum value
    self.orientation_ref_max = np.max(np.real(self.orientation_ref))

    # Fourier domain along angular axis
    self.orientation_ref = np.conj(np.fft.fft(self.orientation_ref))


def match_orientations(
    self,
    bragg_peaks_array: PointListArray,
    num_matches_return: int = 1,
    inversion_symmetry=True,
    multiple_corr_reset=True,
    progress_bar: bool = True,
):
    '''
    This function computes the orientation of any number of PointLists stored in a PointListArray, and returns an OrienationMap.

    '''
    orientation_map = OrientationMap(
        num_x=bragg_peaks_array.shape[0],
        num_y=bragg_peaks_array.shape[1],
        num_matches=num_matches_return) 

    for rx, ry in tqdmnd(
        *bragg_peaks_array.shape,
        desc="Matching Orientations",
        unit=" PointList",
        disable=not progress_bar,
    ):

        orientation = self.match_single_pattern(
            bragg_peaks_array.get_pointlist(rx, ry),
            num_matches_return=num_matches_return,
            inversion_symmetry=inversion_symmetry,
            multiple_corr_reset=multiple_corr_reset,
            plot_corr=False,
            verbose=False,
            )

        orientation_map.set_orientation(orientation,rx,ry)

    return orientation_map

def match_single_pattern(
    self,
    bragg_peaks: PointList,
    num_matches_return: int = 1,
    multiple_corr_reset=True,
    inversion_symmetry=True,
    plot_polar: bool = False,
    plot_corr: bool = False,
    returnfig: bool = False,
    figsize: Union[list, tuple, np.ndarray] = (12, 4),
    verbose: bool = False,
    # subpixel_tilt: bool = False,
    # plot_corr_3D: bool = False,
    # return_corr: bool = False,
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
        orientation (Orientation):    Orientation class containing all outputs
        fig, ax (handles):            Figure handles for the plotting output
    """

    # get bragg peak data
    qx = bragg_peaks.data["qx"]
    qy = bragg_peaks.data["qy"]
    intensity = bragg_peaks.data["intensity"]

    # init orientation output
    orientation = Orientation(num_matches=num_matches_return)

    # other init
    dphi = self.orientation_gamma[1] - self.orientation_gamma[0]    
    corr_value = np.zeros(self.orientation_num_zones)
    corr_in_plane_angle = np.zeros(self.orientation_num_zones)
    if inversion_symmetry:
        corr_inv = np.zeros(self.orientation_num_zones, dtype="bool")

    # if num_matches_return == 1:
    #     orientation_output = np.zeros((3, 3))
    # else:
    #     orientation_output = np.zeros((3, 3, num_matches_return))
    #     corr_output = np.zeros((num_matches_return))

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
                    np.power(radius, self.orientation_radial_power)
                    * np.power(
                        np.maximum(intensity[sub, None], 0.0),
                        self.orientation_intensity_power,
                    )
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

        # Plot polar space image if needed
        if plot_polar is True and match_ind==0:
            # print(match_ind)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_polar)

        # FFT along theta
        im_polar_fft = np.fft.fft(im_polar)

        # Calculate orientation correlogram
        corr_full = np.maximum(
            np.sum(
                np.real(np.fft.ifft(self.orientation_ref * im_polar_fft[None, :, :])),
                axis=1,
            ),
            0,
        )
        ind_phi = np.argmax(corr_full, axis=1)

        # Calculate orientation correlogram for inverse pattern
        if inversion_symmetry:
            corr_full_inv = np.maximum(
                np.sum(
                    np.real(
                        np.fft.ifft(
                            self.orientation_ref * np.conj(im_polar_fft)[None, :, :]
                        )
                    ),
                    axis=1,
                ),
                0,
            )
            ind_phi_inv = np.argmax(corr_full_inv, axis=1)
            corr_inv = np.zeros(self.orientation_num_zones, dtype="bool")

        
        # Testing of direct index correlogram computation
        # # init
        # corr_full = np.zeros((
        #     self.orientation_num_zones,
        #     self.orientation_in_plane_steps)) 
        # if inversion_symmetry:
        #     corr_full_inv = np.zeros((
        #         self.orientation_num_zones,
        #         self.orientation_in_plane_steps))


        # for ind_peak, radius in enumerate(qr):
        #     ip = qphi[ind_peak] / dphi
        #     fp = np.floor(ip).astype('int')
        #     dp = ip - fp

        #     dqr_all = np.abs(self.orientation_shell_radii - radius)
        #     sub = dqr_all < self.orientation_kernel_size

        #     # for ind_radial, dqr in enumerate(dqr_all):
        #         # if dqr < self.orientation_kernel_size:
        #     weight = (1 - dqr_all[sub] / self.orientation_kernel_size) \
        #         *  np.power(np.maximum(intensity[ind_peak], 0),self.orientation_intensity_power) \
        #         *  np.power(qr[ind_peak],self.orientation_radial_power)

        #     corr_full += np.sum(np.roll(self.orientation_ref[:,sub,:],-fp  ,axis=2) * (weight*(1-dp))[None,:,None],axis=1)
        #     corr_full += np.sum(np.roll(self.orientation_ref[:,sub,:],-fp-1,axis=2) * (weight*(  dp))[None,:,None],axis=1)

        #     if inversion_symmetry:
        #         corr_full += np.sum(np.roll(self.orientation_ref[:,sub,::-1],-fp  ,axis=2) * (weight*(1-dp))[None,:,None],axis=1)
        #         corr_full += np.sum(np.roll(self.orientation_ref[:,sub,::-1],-fp-1,axis=2) * (weight*(  dp))[None,:,None],axis=1)          



        #     # inds = np.argwhere(dqr < self.orientation_kernel_size)

        #     # ip = qphi[ind_peak] / dphi
        #     # fp = np.floor(ip)
        #     # dp = ip - fp

        #     # for ind_radial in inds:

        #     #     weight = (1 - dqr[ind_radial] / self.orientation_kernel_size) \
        #     #         *  np.power(np.max(intensity[ind_peak], 0),self.orientation_intensity_power)

        #     #     print(ind_radial)
        #     #     # print(corr_full.shape)
        #     #     # print(type(self.orientation_ref))
        #     #     corr_full += (weight*(1-dp))*np.roll(self.orientation_ref[:,ind_radial),:],-fp  ,axis=1)
        #     #     corr_full += (weight*(  dp))*np.roll(self.orientation_ref[:,int(ind_radial),:],-fp-1,axis=1)

        #     #     if inversion_symmetry:
        #     #         corr_full += (weight*(1-dp))*np.roll(self.orientation_ref[:,int(ind_radial),:],fp  ,axis=1)
        #     #         corr_full += (weight*(  dp))*np.roll(self.orientation_ref[:,int(ind_radial),:],fp+1,axis=1)                    

        # # Get best fit in-plane rotations
        # ind_phi = np.argmax(corr_full, axis=1)
        # if inversion_symmetry:
        #     ind_phi_inv = np.argmax(corr_full_inv, axis=1)



        # Find best match for each zone axis
        for a0 in range(self.orientation_num_zones):
            # Correlation score
            if inversion_symmetry:
                if corr_full_inv[a0, ind_phi_inv[a0]] > corr_full[a0, ind_phi[a0]]:
                    corr_value[a0] = corr_full_inv[a0, ind_phi_inv[a0]]
                    corr_inv[a0] = True
                else:
                    corr_value[a0] = corr_full[a0, ind_phi[a0]]
            else:
                corr_value[a0] = corr_full[a0, ind_phi[a0]]

            # Subpixel angular fit
            if inversion_symmetry and corr_inv[a0]:
                inds = np.mod(
                    ind_phi_inv[a0] + np.arange(-1, 2), self.orientation_gamma.size
                ).astype("int")
                c = corr_full_inv[a0, inds]
                if np.max(c) > 0:
                    dc = (c[2] - c[0]) / (4*c[1] - 2*c[0] - 2*c[2])
                    corr_in_plane_angle[a0] = (
                        self.orientation_gamma[ind_phi_inv[a0]] + dc * dphi
                    ) + np.pi
            else:
                inds = np.mod(
                    ind_phi[a0] + np.arange(-1, 2), self.orientation_gamma.size
                ).astype("int")
                c = corr_full[a0, inds]
                if np.max(c) > 0:
                    dc = (c[2] - c[0]) / (4*c[1] - 2*c[0] - 2*c[2])
                    corr_in_plane_angle[a0] = (
                        self.orientation_gamma[ind_phi[a0]] + dc * dphi
                    )

        # keep plotting image if needed 
        if plot_corr and match_ind == 0:
            corr_plot = corr_value

        # If needed, keep correlation values for additional matches
        if multiple_corr_reset and num_matches_return > 1 and match_ind == 0:
            corr_value_keep = corr_value.copy()
            corr_in_plane_angle_keep = corr_in_plane_angle.copy()

        # Determine the best fit orientation
        ind_best_fit = np.unravel_index(np.argmax(corr_value), corr_value.shape)[0]

        # Verify current match has a correlation > 0
        if corr_value[ind_best_fit] > 0:
            # Get orientation matrix
            orientation_matrix = np.squeeze(
                self.orientation_rotation_matrices[ind_best_fit, :, :]
            )

            # apply in-plane rotation, and inversion if needed
            if multiple_corr_reset and match_ind > 0:
                phi = corr_in_plane_angle_keep[ind_best_fit]
            else:
                phi = corr_in_plane_angle[ind_best_fit]
            if inversion_symmetry and corr_inv[ind_best_fit]:
                m3z = np.array(
                    [
                        [-np.cos(phi), np.sin(phi), 0],
                        [np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1],
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

            # Output best fit values into Orientation class
            orientation.matrix[match_ind] = orientation_matrix
            
            if multiple_corr_reset and match_ind > 0:
                orientation.corr[match_ind] = corr_value_keep[ind_best_fit]
            else:
                orientation.corr[match_ind] = corr_value[ind_best_fit]

            if inversion_symmetry and corr_inv[ind_best_fit]:
                ind_phi = ind_phi_inv[ind_best_fit]
            else:
                ind_phi = ind_phi[ind_best_fit]
            orientation.inds[match_ind,0] = ind_best_fit
            orientation.inds[match_ind,1] = ind_phi

            if inversion_symmetry:
                orientation.mirror[match_ind] = corr_inv[ind_best_fit]

            orientation.angles[match_ind,:] = self.orientation_rotation_angles[ind_best_fit,:]
            orientation.angles[match_ind,2] += phi
            
            # If point group is known, use pymatgen to caculate the symmetry-
            # reduced orientation matrix, producing the crystal direction family.
            if self.pymatgen_available:
                orientation = self.symmetry_reduce_directions(
                    orientation,
                    match_ind=match_ind,
                    )
                
        else:
            # No more matches are detected, so output default orientation matrix and leave corr = 0
            orientation.matrix[match_ind] = np.squeeze(self.orientation_rotation_matrices[0, :, :])


        if verbose:
            if self.pymatgen_available:
                if np.abs(self.cell[5]-120.0) < 1e-6:
                    x_proj_lattice = self.lattice_to_hexagonal(self.cartesian_to_lattice(orientation.family[match_ind][:, 0]))
                    x_proj_lattice = np.round(x_proj_lattice,decimals=3)
                    zone_axis_lattice = self.lattice_to_hexagonal(self.cartesian_to_lattice(orientation.family[match_ind][:, 2]))
                    zone_axis_lattice = np.round(zone_axis_lattice,decimals=3)
                else:
                    x_proj_lattice = self.cartesian_to_lattice(orientation.family[match_ind][:, 0])
                    x_proj_lattice = np.round(x_proj_lattice,decimals=3)
                    zone_axis_lattice = self.cartesian_to_lattice(orientation.family[match_ind][:, 2])
                    zone_axis_lattice = np.round(zone_axis_lattice,decimals=3)

                if orientation.corr[match_ind] > 0:
                    print(
                        "Best fit lattice directions: z axis = ("
                        + str(zone_axis_lattice)
                        + "),"
                        " x axis = ("
                        + str(x_proj_lattice)
                        + "),"
                        + " with corr value = "
                        + str(np.round(orientation.corr[match_ind], decimals=3))
                    )

            else:
                zone_axis_fit = orientation.matrix[match_ind][:, 2]
                zone_axis_lattice = self.cartesian_to_lattice(zone_axis_fit)
                zone_axis_lattice = np.round(zone_axis_lattice,decimals=3)
                print(
                    "Best fit zone axis (lattice) = ("
                    + str(zone_axis_lattice)
                    + "),"
                    + " with corr value = "
                    + str(np.round(orientation.corr[match_ind], decimals=3))
                )

        # if needed, delete peaks for next iteration
        if num_matches_return > 1 and corr_value[ind_best_fit] > 0:
            bragg_peaks_fit=self.generate_diffraction_pattern(
                orientation,
                ind_orientation=match_ind,
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


    # plotting correlation image
    if plot_corr is True:

        if self.orientation_full:
            fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 2]))
            cmin = np.min(corr_plot)
            cmax = np.max(corr_plot)

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
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            sub = self.orientation_inds[:, 2] == 1
            x_inds = (
                self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            sub = self.orientation_inds[:, 2] == 2
            x_inds = (
                self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = (
                self.orientation_inds[sub, 1].astype("int")
                + self.orientation_zone_axis_steps
            )
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            sub = self.orientation_inds[:, 2] == 3
            x_inds = (
                self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
            ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

        elif self.orientation_half:
            fig, ax = plt.subplots(1, 2, figsize=figsize * np.array([2, 1]))
            cmin = np.min(corr_plot)
            cmax = np.max(corr_plot)

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
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            sub = self.orientation_inds[:, 2] == 1
            x_inds = (
                self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
            ).astype("int")
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]

            im_plot = (im_corr_zone_axis - cmin) / (cmax - cmin)
            ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)

        else:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            cmin = np.min(corr_plot)
            cmax = np.max(corr_plot)

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

            # Image indices
            x_inds = (self.orientation_inds[:, 0] - self.orientation_inds[:, 1]).astype(
                "int"
            )
            y_inds = self.orientation_inds[:, 1].astype("int")

            # Check vertical range of the orientation triangle.
            if self.orientation_fiber_angles is not None \
                and np.abs(self.orientation_fiber_angles[0] - 180.0) > 1e-3:
                # Orientation covers only top of orientation sphere

                inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot
                im_mask.ravel()[inds_1D] = False

            else:
                # Orientation covers full vertical range of orientation sphere.
                # top half
                sub = self.orientation_inds[:,2] == 0
                inds_1D = np.ravel_multi_index([x_inds[sub], y_inds[sub]], im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]
                im_mask.ravel()[inds_1D] = False
                # bottom half
                sub = self.orientation_inds[:,2] == 1
                inds_1D = np.ravel_multi_index([
                    self.orientation_zone_axis_steps - y_inds[sub], 
                    self.orientation_zone_axis_steps - x_inds[sub]
                    ],im_corr_zone_axis.shape)
                im_corr_zone_axis.ravel()[inds_1D] = corr_plot[sub]
                im_mask.ravel()[inds_1D] = False



            # print(self.orientation_inds)
            # print(np.vstack((x_inds,y_inds)))



            im_plot = np.ma.masked_array(
                (im_corr_zone_axis - cmin) / (cmax - cmin), mask=im_mask
            )

            ax[0].imshow(im_plot, cmap="viridis", vmin=0.0, vmax=1.0)
            ax[0].spines["left"].set_color("none")
            ax[0].spines["right"].set_color("none")
            ax[0].spines["top"].set_color("none")
            ax[0].spines["bottom"].set_color("none")

            inds_plot = np.unravel_index(np.argmax(im_plot, axis=None), im_plot.shape)
            ax[0].scatter(
                inds_plot[1],
                inds_plot[0],
                s=120,
                linewidth=2,
                facecolors="none",
                edgecolors="r",
            )

            # label_0 = self.orientation_zone_axis_range[0, :]
            # label_1 = self.orientation_zone_axis_range[1, :]
            # label_2 = self.orientation_zone_axis_range[2, :]

            # label_0 = np.round(label_0, decimals=3)
            # label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))
            # label_0 = np.round(label_0, decimals=3)

            # label_1 = np.round(label_1, decimals=3)
            # label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))
            # label_1 = np.round(label_1, decimals=3)

            # label_2 = np.round(label_2, decimals=3)
            # label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))
            # label_2 = np.round(label_2, decimals=3)


            if np.abs(self.cell[5]-120.0) < 1e-6:
                label_0 = self.rational_ind(
                    self.lattice_to_hexagonal(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[0, :])))
                label_1 = self.rational_ind(
                    self.lattice_to_hexagonal(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[1, :])))
                label_2 = self.rational_ind(
                    self.lattice_to_hexagonal(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[2, :])))
            else:
                label_0 = self.rational_ind(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[0, :]))
                label_1 = self.rational_ind(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[1, :]))
                label_2 = self.rational_ind(
                    self.cartesian_to_lattice(
                    self.orientation_zone_axis_range[2, :]))
            # label_0 = np.round(label_0, decimals=2)
            # label_1 = np.round(label_1, decimals=2)
            # label_2 = np.round(label_2, decimals=2)


            ax[0].set_xticks([0, self.orientation_zone_axis_steps])
            ax[0].set_xticklabels([str(label_0), str(label_2)], size=14)
            ax[0].xaxis.tick_top()

            ax[0].set_yticks([self.orientation_zone_axis_steps])
            ax[0].set_yticklabels([str(label_1)], size=14)


        # In-plane rotation
        sig_in_plane = np.squeeze(corr_full[ind_best_fit, :])
        sig_in_plane = sig_in_plane / np.max(sig_in_plane)
        ax[1].plot(
            self.orientation_gamma * 180 / np.pi,
            sig_in_plane,
        )

        # Add markers for the best fit 
        tol = 0.01
        sub = sig_in_plane > 1 - tol
        ax[1].scatter(
                self.orientation_gamma[sub] * 180 / np.pi,
                sig_in_plane[sub],
                s=120,
                linewidth=2,
                facecolors="none",
                edgecolors="r",
            )

        ax[1].set_xlabel("In-plane rotation angle [deg]", size=16)
        ax[1].set_ylabel("Corr. of Best Fit Zone Axis", size=16)
        ax[1].set_ylim([0, 1.03])

        plt.show()


    if returnfig:
        return orientation, fig, ax
    else:
        return orientation




def save_ang_file(
    self,
    file_name,
    orientation_map,
    ind_orientation = 0,
    pixel_size: float = None,
    ):
    '''
    This function outputs an ascii text file in the .ang format, containing 
    the Euler angles of an orientation map. 

    Args:
        orientation_map (OrientationMap):   Class containing orientation matrices,
                                            correlation values, etc.
        ind_orientation (int):              Which orientation match to plot if num_matches > 1
        pixel_size (float):                 Pixel size, if known.

    Returns:
        nothing

    '''

    num_dec = 4  # decimals of output numbers

    if pixel_size is None:
        pixel_size = 1.0
    else:
        pixel_size = np.array(pixel_size)

    # Add orientation map index to file name
    file_base = os.path.splitext(file_name)[0]
    file_output = file_base + '_' + str(ind_orientation) + '.ang'

    if self.pymatgen_available:
        sym = self.pointgroup.get_point_group_symbol()
    else:
        sym = 'unknown'

    # zone axis range
    if np.abs(self.cell[5]-120.0) < 1e-6:
        label_0 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[0, :])))
        label_1 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[1, :])))
        label_2 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[2, :])))
    else:
        label_0 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[0, :]))
        label_1 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[1, :]))
        label_2 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[2, :]))

    # Generate header for file
    header = '# ang file created by the py4DSTEM ACOM module\n' + \
        '#     https://github.com/py4dstem/py4DSTEM\n' + \
        '#     https://doi.org/10.1017/S1431927621000477\n' + \
        '#\n' + \
        '# lattice constants ' + \
        str(np.round(self.cell,decimals=num_dec)) + \
        '\n#\n' + \
        '# pointgroup ' + \
        sym + \
        '\n#\n' + \
        '# zone axis lattice range:' + \
        '\n#     ' + str(label_0) + \
        '\n#     ' + str(label_1) + \
        '\n#     ' + str(label_2) + \
        '\n#\n' + \
        '# atomic species:\n' + \
        '#     ' + str(np.unique(self.numbers)) + \
        '\n#'

    # Generate output data

    # Angles 00
    theta_0 = orientation_map.angles[:,:,ind_orientation,0].ravel()
    phi___1 = orientation_map.angles[:,:,ind_orientation,1].ravel()
    theta_2 = orientation_map.angles[:,:,ind_orientation,2].ravel()

    # # Angles 01
    # theta_0 = -1*orientation_map.angles[:,:,ind_orientation,0].ravel()
    # phi___1 = -1*orientation_map.angles[:,:,ind_orientation,1].ravel()
    # theta_2 = -1*orientation_map.angles[:,:,ind_orientation,2].ravel()

    # # Angles 02
    # theta_2 = orientation_map.angles[:,:,ind_orientation,0].ravel()
    # phi___1 = orientation_map.angles[:,:,ind_orientation,1].ravel()
    # theta_0 = orientation_map.angles[:,:,ind_orientation,2].ravel()

    # # Angles 03
    # theta_2 = -1*orientation_map.angles[:,:,ind_orientation,0].ravel()
    # phi___1 = -1*orientation_map.angles[:,:,ind_orientation,1].ravel()
    # theta_0 = -1*orientation_map.angles[:,:,ind_orientation,2].ravel()


    if pixel_size.shape[0] == 2:
        x = np.arange(orientation_map.num_x)*pixel_size[0]
        y = np.arange(orientation_map.num_y)*pixel_size[1]
    else:
        x = np.arange(orientation_map.num_x)*pixel_size
        y = np.arange(orientation_map.num_y)*pixel_size
    ya,xa = np.meshgrid(y,x)
    xa = xa.ravel()
    ya = ya.ravel()

    # image quality - contrast - TODO: perhaps use dark field intensity sum?
    IQ = np.ones(orientation_map.num_x*orientation_map.num_y)

    # confidence
    CI = orientation_map.corr[:,:,ind_orientation].ravel()

    # phase id - TODO: add ability to combine multiple phases in the future
    ID = np.ones(orientation_map.num_x*orientation_map.num_y)

    # detector intensity - TO: perhaps use sum of all bragg peak intensities?
    DI = np.ones(orientation_map.num_x*orientation_map.num_y)

    # fit - not sure how this is difference from confidence. 
    # TODO: we could add an updated goodness of fit if we add tilt refine
    fit = orientation_map.corr[:,:,ind_orientation].ravel()

    data = np.vstack((
        theta_0,
        phi___1,
        theta_2,
        xa,
        ya,
        IQ,
        CI,
        ID,
        DI,
        fit,
        )).T

    # Write file
    np.savetxt(
        file_output,
        data,
        fmt='%.4g',
        header=header,
        )


def symmetry_reduce_directions(
    self,
    orientation,
    match_ind = 0,
    plot_output = False,
    figsize=(15,6),
    el_shift=0.0,
    az_shift=-30.0,
    ):
    '''
    This function calculates the symmetry-reduced cartesian directions from
    and orientation matrix stored in orientation.matrix, and outputs them
    into orientation.family. It optionally plots the 3D output.

    '''

    # optional plot
    if plot_output:
        bound = 1.05;
        cam_dir = np.mean(self.orientation_zone_axis_range,axis=0)
        cam_dir = cam_dir / np.linalg.norm(cam_dir)
        az = np.rad2deg(np.arctan2(cam_dir[0],cam_dir[1])) + az_shift
        # if np.abs(self.orientation_fiber_angles[0] - 180.0) < 1e-3:
        #     el = 10
        # else:
        el = np.rad2deg(np.arcsin(cam_dir[2])) + el_shift
        el = 0
        fig = plt.figure(figsize=figsize)

        num_points = 10;
        t = np.linspace(0,1,num=num_points+1,endpoint=True)
        d = np.array([[0,1],[0,2],[1,2]])
        orientation_zone_axis_range_flip = self.orientation_zone_axis_range.copy()
        orientation_zone_axis_range_flip[0,:] = -1*orientation_zone_axis_range_flip[0,:]        


    # loop over orientation matrix directions
    for a0 in range(3):
        in_range = np.all(np.sum(self.symmetry_reduction * \
            orientation.matrix[match_ind,:,a0][None,:,None], 
            axis=1) >= 0,
            axis=1)

        orientation.family[match_ind,:,a0] = \
            self.symmetry_operators[np.argmax(in_range)] @ \
            orientation.matrix[match_ind,:,a0]
        

        # in_range = np.all(np.sum(self.symmetry_reduction * \
        #     orientation.matrix[match_ind,:,a0][None,:,None], 
        #     axis=1) >= 0,
        #     axis=1)
        # if np.any(in_range):
        #     ind = np.argmax(in_range)
        #     orientation.family[match_ind,:,a0] = self.symmetry_operators[ind] \
        #         @ orientation.matrix[match_ind,:,a0]
        # else:
        #     # Note this is a quick fix for fiber_angles[0] = 180 degrees
        #     in_range = np.all(np.sum(self.symmetry_reduction * \
        #         (np.array([1,1,-1])*orientation.matrix[match_ind,:,a0][None,:,None]), 
        #         axis=1) >= 0,
        #         axis=1)
        #     ind = np.argmax(in_range)
        #     orientation.family[match_ind,:,a0] = self.symmetry_operators[ind] \
        #         @ (np.array([1,1,-1])*orientation.matrix[match_ind,:,a0])


        if plot_output:
            ax = fig.add_subplot(1, 3, a0+1, 
                projection='3d',
                elev=el, 
                azim=az)

            # draw orienation triangle
            for a1 in range(d.shape[0]):
                v = self.orientation_zone_axis_range[d[a1,0],:][None,:]*t[:,None] + \
                    self.orientation_zone_axis_range[d[a1,1],:][None,:]*(1-t[:,None])
                v = v / np.linalg.norm(v,axis=1)[:,None]
                ax.plot(
                    v[:,1],
                    v[:,0],
                    v[:,2],
                    c='k',
                    )
                v = self.orientation_zone_axis_range[a1,:][None,:]*t[:,None]
                ax.plot(
                    v[:,1],
                    v[:,0],
                    v[:,2],
                    c='k',
                    )
                

            # if needed, draw orientation diamond
            if self.orientation_fiber_angles is not None \
                and np.abs(self.orientation_fiber_angles[0] - 180.0) < 1e-3:
                for a1 in range(d.shape[0]-1):
                    v = orientation_zone_axis_range_flip[d[a1,0],:][None,:]*t[:,None] + \
                        orientation_zone_axis_range_flip[d[a1,1],:][None,:]*(1-t[:,None])
                    v = v / np.linalg.norm(v,axis=1)[:,None]
                    ax.plot(
                        v[:,1],
                        v[:,0],
                        v[:,2],
                        c='k',
                        )
                v = orientation_zone_axis_range_flip[0,:][None,:]*t[:,None]
                ax.plot(
                    v[:,1],
                    v[:,0],
                    v[:,2],
                    c='k',
                    )

            # add points
            p = self.symmetry_operators @ \
                orientation.matrix[match_ind,:,a0]
            ax.scatter(
                xs=p[:,1],
                ys=p[:,0],
                zs=p[:,2],
                s=10,
                marker='o',
                # c='k',
            )
            v = orientation.family[match_ind,:,a0][None,:]*t[:,None]
            ax.plot(
                v[:,1],
                v[:,0],
                v[:,2],
                c='k',
                )
            ax.scatter(
                xs=orientation.family[match_ind,1,a0],
                ys=orientation.family[match_ind,0,a0],
                zs=orientation.family[match_ind,2,a0],
                s=160,
                marker='o',
                facecolors="None",
                edgecolors='r',
            )
            ax.scatter(
                xs=orientation.matrix[match_ind,1,a0],
                ys=orientation.matrix[match_ind,0,a0],
                zs=orientation.matrix[match_ind,2,a0],
                s=80,
                marker='o',
                facecolors="None",
                edgecolors='c',
            )



            ax.invert_yaxis()
            ax.axes.set_xlim3d(left=-bound, right=bound)
            ax.axes.set_ylim3d(bottom=-bound, top=bound)
            ax.axes.set_zlim3d(bottom=-bound, top=bound)
            axisEqual3D(ax)



    if plot_output:
        plt.show()


    return orientation



# zone axis range arguments for orientation_plan corresponding
# to the symmetric wedge for each pointgroup, in the order:
#   [zone_axis_range, fiber_axis, fiber_angles]
orientation_ranges = {
    "1": ["fiber", [0, 0, 1], [180.0, 360.0]],
    "-1": ["full", None, None],
    "2": ["fiber", [0, 0, 1], [180.0, 180.0]],
    "m": ["full", None, None],
    "2/m": ["half", None, None],
    "222": ["fiber", [0, 0, 1], [90.0, 180.0]],
    "mm2": ["fiber", [0, 0, 1], [180.0, 90.0]],
    "mmm": [[[1, 0, 0], [0, 1, 0]], None, None],
    "4": ["fiber", [0, 0, 1], [90.0, 180.0]],
    "-4": ["half", None, None],
    "4/m": [[[1, 0, 0], [0, 1, 0]], None, None],
    "422": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "4mm": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "-42m": ["fiber", [0, 0, 1], [180.0, 45.0]],
    "4/mmm": [[[1, 0, 0], [1, 1, 0]], None, None],
    "3": ["fiber", [0, 0, 1], [180.0, 120.0]],
    "-3": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "32": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "3m": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "-3m": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "-6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "6/m": [[[1, 0, 0], [0.5, 0.5*np.sqrt(3), 0]], None, None],
    "622": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "6mm": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "-6m2": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "6/mmm": [[[0.5*np.sqrt(3), 0.5, 0.0], [1, 0, 0]], None, None],
    "23": [
        [[1, 0, 0], [1, 1, 1]],
        None,
        None,
    ],  # this is probably wrong, it is half the needed range
    "m-3": [[[1, 0, 0], [1, 1, 1]], None, None],
    "432": [[[1, 0, 0], [1, 1, 1]], None, None],
    "-43m": [[[1, -1, 1], [1, 1, 1]], None, None],
    "m-3m": [[[0, 1, 1], [1, 1, 1]], None, None],
}

    # "-3m": ["fiber", [0, 0, 1], [90.0, 60.0]],
    # "-3m": ["fiber", [0, 0, 1], [180.0, 30.0]],

