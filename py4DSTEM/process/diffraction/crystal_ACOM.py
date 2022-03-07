import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd, electron_wavelength_angstrom


def orientation_plan(
    self,
    zone_axis_range: np.ndarray = np.array([[0, 1, 1], [1, 1, 1]]),
    angle_step_zone_axis: float = 2.0,
    angle_step_in_plane: float = 2.0,
    accel_voltage: float = 300e3,
    corr_kernel_size: float = 0.08,
    radial_power: float = 1.0,
    intensity_power: float = 0.5,
    tol_peak_delete=None,
    tol_distance: float = 0.01,
    fiber_axis=None,
    fiber_angles=None,
    cartesian_directions=False,
    figsize: Union[list, tuple, np.ndarray] = (6, 6),
    progress_bar: bool = True,
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
    if isinstance(zone_axis_range, str) and zone_axis_range == "auto":
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.core.structure import Structure

        structure = Structure(
            self.lat_real, self.numbers, self.positions, coords_are_cartesian=False
        )

        pointgroup = SpacegroupAnalyzer(structure).get_point_group_symbol()
        self.pointgroup = pointgroup

        assert (
            pointgroup in orientation_ranges
        ), "Unrecognized pointgroup returned by pymatgen!"

        zone_axis_range, fiber_axis, fiber_angles = orientation_ranges[pointgroup]
        if isinstance(zone_axis_range, list):
            zone_axis_range = np.array(zone_axis_range)
        elif zone_axis_range == "fiber":
            self.orientation_fiber_axis = np.asarray(fiber_axis)
            self.orientation_fiber_angles = np.asarray(fiber_angles)
        self.cartesian_directions = (
            True  # the entries in the orientation_ranges object assume cartesian zones
        )

        print(
            f"Automatically detected point group {pointgroup}, using arguments: zone_axis_range={zone_axis_range}, fiber_axis={fiber_axis}, fiber_angles={fiber_angles}."
        )

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
            if np.all(np.abs(self.orientation_fiber_axis) == np.array([1.0, 0.0, 0.0])):
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

                v2output = self.orientation_fiber_axis * np.cos(theta) + v2 * np.sin(
                    theta
                )
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
    azim = -np.pi / 2 + np.arctan2(
        self.orientation_vecs[:, 1], self.orientation_vecs[:, 0]
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
    self.orientation_rotation_angles = np.zeros((self.orientation_num_zones, 2))
    self.orientation_rotation_matrices = np.zeros((self.orientation_num_zones, 3, 3))
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
        disable=not progress_bar,
    ):
        p = np.linalg.inv(self.orientation_rotation_matrices[a0, :, :]) @ self.g_vec_all

        # Excitation errors
        cos_alpha = (k0[2, None] + p[2, :]) / np.linalg.norm(k0[:, None] + p, axis=0)
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
            np.sum(np.abs(self.orientation_ref[a0, :, :]) ** 2)
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


def match_orientations(
    self,
    bragg_peaks_array: PointListArray,
    num_matches_return: int = 1,
    inversion_symmetry=True,
    multiple_corr_reset=True,
    return_corr: bool = False,
    subpixel_tilt: bool = False,
    progress_bar: bool = True,
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
        *bragg_peaks_array.shape,
        desc="Matching Orientations",
        unit=" PointList",
        disable=not progress_bar,
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
                inversion_symmetry=inversion_symmetry,
                multiple_corr_reset=multiple_corr_reset,
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
    multiple_corr_reset=True,
    inversion_symmetry=True,
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
                    np.power(radius, self.orientation_radial_power)
                    * np.power(
                        np.max(intensity[sub, None], 0),
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
            orientation_matrix = np.squeeze(self.orientation_rotation_matrices[0, :, :])
            if multiple_corr_reset and match_ind > 0:
                corr_value_keep[ind_best_fit] = 0

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
                    + str(np.round(corr_value_keep[ind_best_fit], decimals=3))
                )
            else:
                print(
                    "Best fit zone axis = ("
                    + str(temp)
                    + ")"
                    + " with corr value = "
                    + str(np.round(corr_value[ind_best_fit], decimals=3))
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
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

            sub = self.orientation_inds[:, 2] == 1
            x_inds = (
                self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

            sub = self.orientation_inds[:, 2] == 2
            x_inds = (
                self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = (
                self.orientation_inds[sub, 1].astype("int")
                + self.orientation_zone_axis_steps
            )
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

            sub = self.orientation_inds[:, 2] == 3
            x_inds = (
                self.orientation_inds[sub, 1] - self.orientation_inds[sub, 0]
            ).astype("int") + self.orientation_zone_axis_steps
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
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
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
            im_corr_zone_axis.ravel()[inds_1D] = corr_value[sub]

            sub = self.orientation_inds[:, 2] == 1
            x_inds = (
                self.orientation_inds[sub, 0] - self.orientation_inds[sub, 1]
            ).astype("int")
            y_inds = self.orientation_zone_axis_steps - self.orientation_inds[
                sub, 1
            ].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
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

            x_inds = (self.orientation_inds[:, 0] - self.orientation_inds[:, 1]).astype(
                "int"
            )
            y_inds = self.orientation_inds[:, 1].astype("int")
            inds_1D = np.ravel_multi_index([x_inds, y_inds], im_corr_zone_axis.shape)
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

            inds_plot = np.unravel_index(np.argmax(im_plot, axis=None), im_plot.shape)
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


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


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
    "-3m": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "-6": ["fiber", [0, 0, 1], [180.0, 60.0]],
    "6/m": [[[1, 0, 0], [0.5, np.sqrt(3) / 2.0, 0]], None, None],
    "622": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "6mm": ["fiber", [0, 0, 1], [180.0, 30.0]],
    "-6m2": ["fiber", [0, 0, 1], [90.0, 60.0]],
    "6/mmm": [[[np.sqrt(3) / 2.0, 0.5, 0.0], [1, 0, 0]], None, None],
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
