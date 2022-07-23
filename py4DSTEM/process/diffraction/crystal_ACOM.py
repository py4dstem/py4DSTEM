import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Optional

from .utils import Orientation, OrientationMap, axisEqual3D
from ..utils import electron_wavelength_angstrom
from ...utils.tqdmnd import tqdmnd
from ...io.datastructure import PointList, PointListArray, RealSlice

from numpy.linalg import lstsq
try:
    import cupy as cp
except:
    cp = None

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core.structure import Structure
except ImportError:
    pass

def orientation_plan(
    self,
    zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
    angle_step_zone_axis: float = 2.0,
    angle_coarse_zone_axis: float = None,
    angle_refine_range: float = None,
    angle_step_in_plane: float = 2.0,
    accel_voltage: float = 300e3,
    corr_kernel_size: float = 0.08,
    radial_power: float = 1.0,
    intensity_power: float = 0.25,  # New default intensity power scaling
    tol_peak_delete=None,
    tol_distance: float = 0.01,
    fiber_axis = None,
    fiber_angles = None,
    figsize: Union[list, tuple, np.ndarray] = (6, 6),
    CUDA: bool = False,
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
        angle_step_zone_axis (float): Approximate angular step size for zone axis search [degrees]
        angle_coarse_zone_axis (float): Coarse step size for zone axis search [degrees]. Setting to 
                                        None uses the same value as angle_step_zone_axis.
        angle_refine_range (float):   Range of angles to use for zone axis refinement. Setting to
                                      None uses same value as angle_coarse_zone_axis.

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
        CUDA (bool):             Use CUDA for the Fourier operations.
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
    self.CUDA = CUDA

    # Calculate wavelenth
    self.wavelength = electron_wavelength_angstrom(self.accel_voltage)

    # store the radial and intensity scaling to use later for generating test patterns
    self.orientation_radial_power = radial_power
    self.orientation_intensity_power = intensity_power

    # Calculate the ratio between coarse and fine refinement
    if angle_coarse_zone_axis is not None:
        self.orientation_refine = True
        self.orientation_refine_ratio = np.round(
            angle_coarse_zone_axis/angle_step_zone_axis).astype('int')
        self.orientation_angle_coarse = angle_coarse_zone_axis
        if angle_refine_range is None:
            self.orientation_refine_range = angle_coarse_zone_axis
        else:
            self.orientation_refine_range = angle_refine_range
    else:
        self.orientation_refine_ratio = 1.0
        self.orientation_refine = False

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
    step = np.maximum(
        (180 / np.pi) * angle_u_v / angle_step_zone_axis,
        (180 / np.pi) * angle_u_w / angle_step_zone_axis,
    )
    self.orientation_zone_axis_steps = (np.round(
        step / self.orientation_refine_ratio
        ) * self.orientation_refine_ratio).astype(np.int)

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

    # If needed, create coarse orientation sieve
    if self.orientation_refine:
        self.orientation_sieve = np.logical_and(
            np.mod(self.orientation_inds[:, 0], self.orientation_refine_ratio) == 0,
            np.mod(self.orientation_inds[:, 1], self.orientation_refine_ratio) == 0)
        if self.CUDA:
            self.orientation_sieve_CUDA = cp.asarray(self.orientation_sieve)

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
    # self.orientation_ref_1D = np.zeros(
    #     (
    #         self.orientation_num_zones,
    #         np.size(self.orientation_shell_radii),
    #     ),
    #     dtype="float",
    # )

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
                # 2D orientation plan
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

        orientation_ref_norm = np.sqrt(np.sum(
            self.orientation_ref[a0, :, :] ** 2))
        if orientation_ref_norm > 0:
            self.orientation_ref[a0, :, :] /= orientation_ref_norm

    # Maximum value
    self.orientation_ref_max = np.max(np.real(self.orientation_ref))

    # Fourier domain along angular axis
    if self.CUDA:
        self.orientation_ref = cp.asarray(self.orientation_ref)
        self.orientation_ref = cp.conj(cp.fft.fft(self.orientation_ref))
    else:
        self.orientation_ref = np.conj(np.fft.fft(self.orientation_ref))


def match_orientations(
    self,
    bragg_peaks_array: PointListArray,
    num_matches_return: int = 1,
    min_number_peaks = 3,
    inversion_symmetry = True,
    multiple_corr_reset = False,
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
            min_number_peaks=min_number_peaks,
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
    min_number_peaks = 3,
    inversion_symmetry = True,
    multiple_corr_reset = False,
    plot_polar: bool = False,
    plot_corr: bool = False,
    returnfig: bool = False,
    figsize: Union[list, tuple, np.ndarray] = (12, 4),
    verbose: bool = False,
    # plot_corr_3D: bool = False,
    ):
    """
    Solve for the best fit orientation of a single diffraction pattern.

    Args:
        bragg_peaks (PointList):      numpy array containing the Bragg positions and intensities ('qx', 'qy', 'intensity')
        num_matches_return (int):     return these many matches as 3th dim of orient (matrix)
        min_number_peaks (int):       Minimum number of peaks required to perform ACOM matching
        inversion_symmetry (bool):    check for inversion symmetry in the matches
        multiple_corr_reset (bool):   keep original correlation score for multiple matches
        subpixel_tilt (bool):         set to false for faster matching, returning the nearest corr point
        plot_polar (bool):            set to true to plot the polar transform of the diffraction pattern
        plot_corr (bool):             set to true to plot the resulting correlogram
        returnfig (bool):             Return figure handles
        figsize (list):               size of figure
        verbose (bool):               Print the fitted zone axes, correlation scores
        CUDA (bool):                  Enable CUDA for the FFT steps

    Returns:
        orientation (Orientation):    Orientation class containing all outputs
        fig, ax (handles):            Figure handles for the plotting output
    """


    # init orientation output
    orientation = Orientation(num_matches=num_matches_return)
    if bragg_peaks.data.shape[0] < min_number_peaks:
        return orientation

    # get bragg peak data
    qx = bragg_peaks.data["qx"]
    qy = bragg_peaks.data["qy"]
    intensity = bragg_peaks.data["intensity"]

    # other init
    dphi = self.orientation_gamma[1] - self.orientation_gamma[0]
    corr_value = np.zeros(self.orientation_num_zones)
    corr_in_plane_angle = np.zeros(self.orientation_num_zones)
    if inversion_symmetry:
        corr_inv = np.zeros(self.orientation_num_zones, dtype="bool")

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

        # Determine the RMS signal from im_polar for the first match.
        # Note that we use scaling slightly below RMS so that following matches 
        # don't have higher correlating scores than previous matches.
        if multiple_corr_reset is False and num_matches_return > 1:
            if match_ind == 0:
                im_polar_scale_0 = np.mean(im_polar**2)**0.4
            else:
                im_polar_scale = np.mean(im_polar**2)**0.4
                if im_polar_scale > 0:
                    im_polar *= im_polar_scale_0 / im_polar_scale
                # im_polar /= np.sqrt(np.mean(im_polar**2))
                # im_polar *= im_polar_0_rms

        # If later refinement is performed, we need to keep the original image's polar tranform if corr reset is enabled
        if self.orientation_refine:
            if multiple_corr_reset:
                if match_ind == 0:
                    if self.CUDA:
                        im_polar_refine = cp.asarray(im_polar.copy())
                    else:
                        im_polar_refine = im_polar.copy()
            else:
                if self.CUDA:
                    im_polar_refine = cp.asarray(im_polar.copy())
                else:
                    im_polar_refine = im_polar.copy()

        # Plot polar space image if needed
        if plot_polar is True: # and match_ind==0:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(im_polar)
            plt.show()

        # FFT along theta
        if self.CUDA:
            im_polar_fft = cp.fft.fft(cp.asarray(im_polar))
        else:
            im_polar_fft = np.fft.fft(im_polar)
        if self.orientation_refine:
            if self.CUDA:
                im_polar_refine_fft = cp.fft.fft(cp.asarray(im_polar_refine))
            else:
                im_polar_refine_fft = np.fft.fft(im_polar_refine)


        # Calculate full orientation correlogram
        if self.orientation_refine:
            corr_full = np.zeros((
                self.orientation_num_zones,
                self.orientation_in_plane_steps,
                ))
            if self.CUDA:
                corr_full[self.orientation_sieve,:] = cp.maximum(
                    cp.sum(
                        cp.real(cp.fft.ifft(
                            self.orientation_ref[self.orientation_sieve_CUDA,:,:] \
                            * im_polar_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full[self.orientation_sieve,:] = np.maximum(
                    np.sum(
                        np.real(np.fft.ifft(
                            self.orientation_ref[self.orientation_sieve,:,:] \
                            * im_polar_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                )

        else:
            if self.CUDA:
                corr_full = np.maximum(
                    np.sum(
                        np.real(cp.fft.ifft(self.orientation_ref * im_polar_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full = np.maximum(
                    np.sum(
                        np.real(np.fft.ifft(self.orientation_ref * im_polar_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                )

        # Get maximum (non inverted) correlation value
        ind_phi = np.argmax(corr_full, axis=1)

        # Calculate orientation correlogram for inverse pattern (in-plane mirror)
        if inversion_symmetry:
            if self.orientation_refine:
                corr_full_inv = np.zeros((
                    self.orientation_num_zones,
                    self.orientation_in_plane_steps,
                    ))
                if self.CUDA:
                    corr_full_inv[self.orientation_sieve,:] = cp.maximum(
                        cp.sum(
                            cp.real(
                                cp.fft.ifft(
                                    self.orientation_ref[self.orientation_sieve_CUDA,:,:] \
                                    * cp.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
                    corr_full_inv[self.orientation_sieve,:] = np.maximum(
                        np.sum(
                            np.real(
                                np.fft.ifft(
                                    self.orientation_ref[self.orientation_sieve,:,:] \
                                    * np.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    )
            else:
                if self.CUDA:
                    corr_full_inv = np.maximum(
                        np.sum(
                            np.real(
                                cp.fft.ifft(
                                    self.orientation_ref * cp.conj(im_polar_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
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

        # Find best match for each zone axis
        corr_value[:] = 0
        for a0 in range(self.orientation_num_zones):
            if (self.orientation_refine is False) or self.orientation_sieve[a0]:

                # Correlation score
                if inversion_symmetry:
                    if corr_full_inv[a0, ind_phi_inv[a0]] > corr_full[a0, ind_phi[a0]]:
                        corr_value[a0] = corr_full_inv[a0, ind_phi_inv[a0]]
                        corr_inv[a0] = True
                    else:
                        corr_value[a0] = corr_full[a0, ind_phi[a0]]
                else:
                    corr_value[a0] = corr_full[a0, ind_phi[a0]]

                # In-plane sub-pixel angular fit
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

        # If needed, keep original polar image to recompute the correlations
        if multiple_corr_reset and num_matches_return > 1 and match_ind == 0 and not self.orientation_refine:
            corr_value_keep = corr_value.copy()
            corr_in_plane_angle_keep = corr_in_plane_angle.copy()

        # Determine the best fit orientation
        ind_best_fit = np.unravel_index(np.argmax(corr_value), corr_value.shape)[0]


        ############################################################
        # If needed, perform fine step refinement of the zone axis #
        ############################################################
        if self.orientation_refine:
            mask_refine = np.arccos(np.clip(np.sum(self.orientation_vecs \
                * self.orientation_vecs[ind_best_fit,:],axis=1),-1,1)) \
                < np.deg2rad(self.orientation_refine_range)
            if self.CUDA:
                mask_refine_CUDA = cp.asarray(mask_refine)

            if self.CUDA:
                corr_full[mask_refine,:] = cp.maximum(
                    cp.sum(
                        cp.real(cp.fft.ifft(
                            self.orientation_ref[mask_refine_CUDA,:,:] \
                            * im_polar_refine_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                ).get()
            else:
                corr_full[mask_refine,:] = np.maximum(
                    np.sum(
                        np.real(np.fft.ifft(
                            self.orientation_ref[mask_refine,:,:] \
                            * im_polar_refine_fft[None, :, :])),
                        axis=1,
                    ),
                    0,
                )

            # Get maximum (non inverted) correlation value
            ind_phi = np.argmax(corr_full, axis=1)

            # Inversion symmetry
            if inversion_symmetry:
                if self.CUDA:
                    corr_full_inv[mask_refine,:] = cp.maximum(
                        cp.sum(
                            cp.real(
                                cp.fft.ifft(
                                    self.orientation_ref[mask_refine_CUDA,:,:] \
                                    * cp.conj(im_polar_refine_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    ).get()
                else:
                    corr_full_inv[mask_refine,:] = np.maximum(
                        np.sum(
                            np.real(
                                np.fft.ifft(
                                    self.orientation_ref[mask_refine,:,:] \
                                    * np.conj(im_polar_refine_fft)[None, :, :]
                                )
                            ),
                            axis=1,
                        ),
                        0,
                    )
                ind_phi_inv = np.argmax(corr_full_inv, axis=1)


            # Determine best in-plane correlation
            for a0 in np.argwhere(mask_refine):
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

            # Determine the new best fit orientation
            ind_best_fit = np.unravel_index(np.argmax(corr_value * mask_refine[None,:]), corr_value.shape)[0]

        # Verify current match has a correlation > 0
        if corr_value[ind_best_fit] > 0:
            # Get orientation matrix
            orientation_matrix = np.squeeze(
                self.orientation_rotation_matrices[ind_best_fit, :, :]
            )

            # apply in-plane rotation, and inversion if needed
            if multiple_corr_reset and match_ind > 0 and self.orientation_refine is False:
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

            if self.orientation_refine:
                orientation.corr[match_ind] = corr_value[ind_best_fit]
            else:
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
                    if np.max(np.abs(orientation.family)) > 0.1:
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
                    print('No good match found for index ' + str(match_ind))

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
            corr_plot = corr_value.copy()
            sig_in_plane = np.squeeze(corr_full[ind_best_fit, :]).copy()

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

                if cmax > cmin:
                    im_plot = np.ma.masked_array(
                        (im_corr_zone_axis - cmin) / (cmax - cmin), mask=im_mask
                    )
                else:
                    im_plot = im_corr_zone_axis


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

                ax[0].set_xticks([0, self.orientation_zone_axis_steps])
                ax[0].set_xticklabels([str(label_0), str(label_2)], size=14)
                ax[0].xaxis.tick_top()

                ax[0].set_yticks([self.orientation_zone_axis_steps])
                ax[0].set_yticklabels([str(label_1)], size=14)

            # In-plane rotation
            # sig_in_plane = np.squeeze(corr_full[ind_best_fit, :])
            sig_in_plane_max = np.max(sig_in_plane)
            if sig_in_plane_max > 0:
                sig_in_plane /= sig_in_plane_max
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


def orientation_map_to_orix_CrystalMap(self, 
                                       orientation_map, 
                                       ind_orientation=0, 
                                       pixel_size=1.0, 
                                       pixel_units='px', 
                                       return_color_key=False
                                       ):
    from orix.quaternion import Rotation, Orientation
    from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
    from orix.plot import IPFColorKeyTSL
    from diffpy.structure import Atom, Lattice, Structure
    
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core.structure import Structure as pgStructure
    
    from py4DSTEM.process.diffraction.utils import element_symbols

    # generate a list of Rotation objects from the Euler angles
    rotations = Rotation.from_euler(
                                    orientation_map.angles[:,:,ind_orientation].reshape(-1,3), direction='crystal2lab')

    # Generate x,y coordinates since orix uses flat data internally
    coords, _ = create_coordinate_arrays((orientation_map.num_x,orientation_map.num_y),(pixel_size,)*2)

    # Generate an orix structure from the Crystal
    atoms = [ Atom( element_symbols[Z-1], pos) for Z, pos in zip(self.numbers, self.positions)]

    structure = Structure(
        atoms=atoms,
        lattice=Lattice(*self.cell),
    )
    
    # Use pymatgen to get the symmetry
    pg_structure = pgStructure(self.lat_real, self.numbers, self.positions, coords_are_cartesian=False)
    pointgroup = SpacegroupAnalyzer(pg_structure).get_point_group_symbol()
    
    # If the structure has only one element, name the phase based on the element
    if np.unique(self.numbers).size == 1:
        name = element_symbols[self.numbers[0]-1]
    else:
        name = pg_structure.formula    
    
    # Generate an orix Phase to store symmetry
    phase = Phase(
        name=name,
        point_group=pointgroup,
        structure=structure,
    )

    xmap = CrystalMap(
        rotations=rotations,
        x=coords["x"],
        y=coords["y"],
        phase_list=PhaseList(phase),
        prop={
            "iq": orientation_map.corr[:, :, ind_orientation].ravel(),
            "ci": orientation_map.corr[:, :, ind_orientation].ravel(),
        },
        scan_unit=pixel_units,
    )

    ckey = IPFColorKeyTSL(phase.point_group)

    return (xmap, ckey) if return_color_key else xmap



def calculate_strain(
    self,
    bragg_peaks_array: PointListArray,
    orientation_map: OrientationMap,
    corr_kernel_size = None,
    sigma_excitation_error = 0.02,
    tol_excitation_error_mult: float = 3,
    tol_intensity: float = 1e-4,
    k_max: Optional[float] = None,
    min_num_peaks = 5,
    rotation_range = None,
    progress_bar = True,
    ):
    '''
    This function takes in both a PointListArray containing Bragg peaks, and a
    corresponding OrientationMap, and uses least squares to compute the
    deformation tensor which transforms the simulated diffraction pattern
    into the experimental pattern, for all probe positons.

    TODO: add robust fitting?

    Args:
        bragg_peaks_array (PointListArray):   All Bragg peaks
        orientation_map (OrientationMap):     Orientation map generated from ACOM
        corr_kernel_size (float):           Correlation kernel size - if user does
                                            not specify, uses self.corr_kernel_size.
        sigma_excitation_error (float):  sigma value for envelope applied to s_g (excitation errors) in units of inverse Angstroms
        tol_excitation_error_mult (float): tolerance in units of sigma for s_g inclusion
        tol_intensity (np float):        tolerance in intensity units for inclusion of diffraction spots
        k_max (float):                   Maximum scattering vector
        min_num_peaks (int):             Minimum number of peaks required.
        rotation_range (float):          Maximum rotation range in radians (for symmetry reduction).

    Returns:
        strain_map (RealSlice):  strain tensor

    '''

    # Initialize empty strain maps
    strain_map = RealSlice(
        data=np.zeros((
            bragg_peaks_array.shape[0],
            bragg_peaks_array.shape[1],
            5)),
        slicelabels=('e_xx','e_yy','e_xy','theta','mask'),
        name='strain_map')
    strain_map.get_slice('mask').data[:] = 1.0

    # init values
    if corr_kernel_size is None:
        corr_kernel_size = self.orientation_kernel_size
    radius_max_2 = corr_kernel_size**2

    # Loop over all probe positions
    for rx, ry in tqdmnd(
        *bragg_peaks_array.shape,
        desc="Calculating strains",
        unit=" PointList",
        disable=not progress_bar,
        ):
        # Get bragg peaks from experiment and reference
        p = bragg_peaks_array.get_pointlist(rx,ry)

        if p.data.shape[0] >= min_num_peaks:
            p_ref = self.generate_diffraction_pattern(
                orientation_map.get_orientation(rx,ry),
                sigma_excitation_error = sigma_excitation_error,
                tol_excitation_error_mult = tol_excitation_error_mult,
                tol_intensity = tol_intensity,
                k_max = k_max,
            )

            # init
            keep = np.zeros(p.data.shape[0],dtype='bool')
            inds_match = np.zeros(p.data.shape[0],dtype='int')

            # Pair off experimental Bragg peaks with reference peaks
            for a0 in range(p.data.shape[0]):
                dist_2 = (p.data['qx'][a0] - p_ref.data['qx'])**2 \
                    +   (p.data['qy'][a0] - p_ref.data['qy'])**2
                ind_min = np.argmin(dist_2)

                if dist_2[ind_min] <= radius_max_2:
                    inds_match[a0] = ind_min
                    keep[a0] = True

            # Get all paired peaks
            qxy = np.vstack((
                p.data['qx'][keep],
                p.data['qy'][keep])).T
            qxy_ref = np.vstack((
                p_ref.data['qx'][inds_match[keep]],
                p_ref.data['qy'][inds_match[keep]])).T

            # Apply intensity weighting from experimental measurements
            qxy *= p.data['intensity'][keep,None]
            qxy_ref *= p.data['intensity'][keep,None]

            # Fit transformation matrix
            # Note - not sure about transpose here 
            # (though it might not matter if rotation isn't included)
            m = lstsq(qxy_ref, qxy, rcond=None)[0].T

            # Get the infinitesimal strain matrix
            strain_map.get_slice('e_xx').data[rx,ry] = 1 - m[0,0]
            strain_map.get_slice('e_yy').data[rx,ry] = 1 - m[1,1]
            strain_map.get_slice('e_xy').data[rx,ry] = -(m[0,1]+m[1,0])/2.0
            strain_map.get_slice('theta').data[rx,ry] =  (m[0,1]-m[1,0])/2.0

            # Add finite rotation from ACOM orientation map.
            # I am not sure about the relative signs here.
            # Also, I need to add in the mirror operator.
            if orientation_map.mirror[rx,ry,0]:
                strain_map.get_slice('theta').data[rx,ry] \
                    += (orientation_map.angles[rx,ry,0,0] \
                    + orientation_map.angles[rx,ry,0,2])
            else:
                strain_map.get_slice('theta').data[rx,ry] \
                    -= (orientation_map.angles[rx,ry,0,0] \
                    + orientation_map.angles[rx,ry,0,2])

        else:
            strain_map.get_slice('mask').data[rx,ry] = 0.0

    if rotation_range is not None:
        strain_map.get_slice('theta').data[:] \
            = np.mod(strain_map.get_slice('theta').data[:], rotation_range)

    return strain_map


def save_ang_file(
    self,
    file_name,
    orientation_map,
    ind_orientation=0,
    pixel_size=1.0,
    pixel_units="px",
):
    """
    This function outputs an ascii text file in the .ang format, containing
    the Euler angles of an orientation map.

    Args:
        file_name (str):                    Path to save .ang file.
        orientation_map (OrientationMap):   Class containing orientation matrices,
                                            correlation values, etc.
        ind_orientation (int):              Which orientation match to plot if num_matches > 1
        pixel_size (float):                 Pixel size, if known.
        pixel_units (str):                  Units of the pixel size

    Returns:
        nothing

    """

    from orix.io.plugins.ang import file_writer

    xmap = self.orientation_map_to_orix_CrystalMap(
        orientation_map,
        ind_orientation=ind_orientation,
        pixel_size=pixel_size,
        pixel_units=pixel_units,
        return_color_key=False,
    )

    file_writer(file_name, xmap)


def symmetry_reduce_directions(
    self,
    orientation,
    match_ind = 0,
    plot_output = False,
    figsize = (15,6),
    el_shift = 0.0,
    az_shift = -30.0,
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

