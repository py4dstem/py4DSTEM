import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.signal import medfilt

import warnings
import numpy as np
from typing import Union, Optional

from ...io.datastructure import PointList, PointListArray
from ..utils import tqdmnd


def plot_structure(
    self,
    orientation_matrix: Optional[np.ndarray] = None,
    zone_axis_lattice: Optional[np.ndarray] = None,
    proj_x_lattice: Optional[np.ndarray] = None,
    zone_axis_cartesian: Optional[np.ndarray] = None,
    proj_x_cartesian: Optional[np.ndarray] = None,
    size_marker: float = 400,
    tol_distance: float = 0.001,
    plot_limit: Optional[np.ndarray] = None,
    camera_dist: Optional[float] = None,
    show_axes: bool = False,
    perspective_axes: bool = True,
    figsize: Union[tuple, list, np.ndarray] = (8, 8),
    returnfig: bool = False,
):
    """
    Quick 3D plot of the untit cell /atomic structure.

    Args:
        orientation_matrix (array):  (3,3) orientation matrix, where columns represent projection directions.
        zone_axis_lattice (array):    (3,) projection direction in lattice indices
        proj_x_lattice (array):       (3,) x-axis direction in lattice indices
        zone_axis_cartesian (array): (3,) cartesian projection direction
        proj_x_cartesian (array):    (3,) cartesian projection direction
        scale_markers (float):       Size scaling for markers
        tol_distance (float):        Tolerance for repeating atoms on edges on cell boundaries.
        plot_limit (float):          (2,3) numpy array containing x y z plot min and max in columns.
                                     Default is 1.1* unit cell dimensions.
        camera_dist (float):         Move camera closer to the plot (relative to matplotlib default of 10)
        show_axes (bool):            Whether to plot axes or not.
        perspective_axes (bool):     Select either perspective (true) or orthogonal (false) axes
        figsize (2 element float):   Size scaling of figure axes.
        returnfig (bool):            Return figure and axes handles.

    Returns:
        fig, ax                     (optional) figure and axes handles
    """

    # projection directions
    if orientation_matrix is None:
        orientation_matrix = self.parse_orientation(
            zone_axis_lattice,
            proj_x_lattice,
            zone_axis_cartesian,
            proj_x_cartesian)

    # matplotlib camera orientation
    if np.abs(abs(orientation_matrix[2,2])-1) < 1e-6:
        el = 90.0 * np.sign(orientation_matrix[2,2])
    else:
        el = (np.rad2deg(np.arctan(orientation_matrix[2,2] \
            / np.sqrt(orientation_matrix[0,2] ** 2 
            + orientation_matrix[1,2] ** 2)))
        )
    az = np.rad2deg(np.arctan2(orientation_matrix[0,2], orientation_matrix[1,2]))
    # TODO roll is not yet implemented in matplot version 3.4.3
    # matplotlib x projection direction (i.e. estimate the roll angle)
    # init_y = np.cross(proj_z,np.array([0,1e-6,0]))
    # init_x = np.cross(init_y,proj_z)
    # init_x = init_x / np.linalg.norm(init_x)
    # init_y = init_y / np.linalg.norm(init_y)
    # beta = np.vstack((init_x,init_y)) @ proj_x[:,None]
    # alpha = np.rad2deg(np.arctan2(beta[1], beta[0]))

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

    # 3D plotting
    fig = plt.figure(figsize=figsize)
    if perspective_axes:
        ax = fig.add_subplot(
            projection="3d", 
            elev=el, 
            azim=az)
    else:
        ax = fig.add_subplot(
            projection="3d", 
            elev=el, 
            azim=az,
            proj_type='ortho')

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

    # appearance
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
    orientation_matrix: Optional[np.ndarray] = None,
    zone_axis_lattice: Optional[np.ndarray] = None,
    proj_x_lattice: Optional[np.ndarray] = None,
    zone_axis_cartesian: Optional[np.ndarray] = None,
    proj_x_cartesian: Optional[np.ndarray] = None,
    scale_markers: float = 1e3,
    plot_limit: Optional[Union[list, tuple, np.ndarray]] = None,
    camera_dist: Optional[float] = None,
    show_axes: bool = True,
    perspective_axes: bool = True,
    figsize: Union[list, tuple, np.ndarray] = (8, 8),
    returnfig: bool = False,
):
    """
    3D scatter plot of the structure factors using magnitude^2, i.e. intensity.

    Args:
        orientation_matrix (array):  (3,3) orientation matrix, where columns represent projection directions.
        zone_axis_lattice (array):    (3,) projection direction in lattice indices
        proj_x_lattice (array):       (3,) x-axis direction in lattice indices
        zone_axis_cartesian (array): (3,) cartesian projection direction
        proj_x_cartesian (array):    (3,) cartesian projection direction
        scale_markers (float):       size scaling for markers
        plot_limit (float):          x y z plot limits, default is [-1 1]*self.k_max
        camera_dist (float):         Move camera closer to the plot (relative to matplotlib default of 10)
        show_axes (bool):            Whether to plot axes or not.
        perspective_axes (bool):     Select either perspective (true) or orthogonal (false) axes
        figsize (2 element float):   size scaling of figure axes
        returnfig (bool):            set to True to return figure and axes handles

    Returns:
        fig, ax                     (optional) figure and axes handles
    """

    # projection directions
    if orientation_matrix is None:
        orientation_matrix = self.parse_orientation(
            zone_axis_lattice,
            proj_x_lattice,
            zone_axis_cartesian,
            proj_x_cartesian)

    # matplotlib camera orientation
    if np.abs(abs(orientation_matrix[2,2])-1) < 1e-6:
        el = 90.0 * np.sign(orientation_matrix[2,2])
    else:
        el = (np.rad2deg(np.arctan(
            orientation_matrix[2,2] / np.sqrt(
            orientation_matrix[0,2] ** 2 +
            orientation_matrix[1,2] ** 2)))
        )
    az = np.rad2deg(np.arctan2(orientation_matrix[0,2], orientation_matrix[1,2]))

    # TODO roll is not yet implemented in matplot version 3.4.3
    # matplotlib x projection direction (i.e. estimate the roll angle)
    # init_y = np.cross(proj_z,np.array([0,1e-6,0]))
    # init_x = np.cross(init_y,proj_z)
    # init_x = init_x / np.linalg.norm(init_x)
    # init_y = init_y / np.linalg.norm(init_y)
    # beta = np.vstack((init_x,init_y)) @ proj_x[:,None]
    # alpha = np.rad2deg(np.arctan2(beta[1], beta[0]))

    # 3D plotting
    fig = plt.figure(figsize=figsize)
    if perspective_axes:
        ax = fig.add_subplot(
            projection="3d", 
            elev=el, 
            azim=az)
    else:
        ax = fig.add_subplot(
            projection="3d", 
            elev=el, 
            azim=az,
            proj_type='ortho')

    # plot all structure factor points
    ax.scatter(
        xs=self.g_vec_all[1, :],
        ys=self.g_vec_all[0, :],
        zs=self.g_vec_all[2, :],
        s=scale_markers * self.struct_factors_int,
    )

    # axes limits
    if plot_limit is None:
        plot_limit = self.k_max * 1.05

    # appearance
    ax.invert_yaxis()
    if show_axes is False:
        ax.set_axis_off()
    ax.axes.set_xlim3d(left=-plot_limit, right=plot_limit)
    ax.axes.set_ylim3d(bottom=-plot_limit, top=plot_limit)
    ax.axes.set_zlim3d(bottom=-plot_limit, top=plot_limit)
    axisEqual3D(ax)

    if camera_dist is not None:
        ax.dist = camera_dist

    plt.show()

    if returnfig:
        return fig, ax


def plot_orientation_zones(
    self,
    azim_elev: Optional[Union[list, tuple, np.ndarray]] = None,
    proj_dir_lattice: Optional[Union[list, tuple, np.ndarray]] = None,
    proj_dir_cartesian: Optional[Union[list, tuple, np.ndarray]] = None,
    tol_den = 10,
    marker_size: float = 20,
    plot_limit: Union[list, tuple, np.ndarray] = np.array([-1.1, 1.1]),
    figsize: Union[list, tuple, np.ndarray] = (8, 8),
    returnfig: bool = False,
):
    """
    3D scatter plot of the structure factors using magnitude^2, i.e. intensity.

    Args:
        azim_elev (array):           az and el angles for plot
        proj_dir_lattice (array):    (3,) projection direction in lattice
        proj_dir_cartesian: (array): (3,) projection direction in cartesian
        tol_den (int):               tolerance for rational index denominator
        dir_proj (float):            projection direction, either [elev azim] or normal vector
                                     Default is mean vector of self.orientation_zone_axis_range rows
        marker_size (float):         size of markers
        plot_limit (float):          x y z plot limits, default is [0, 1.05]
        figsize (2 element float):   size scaling of figure axes
        returnfig (bool):            set to True to return figure and axes handles

    Returns:
        fig, ax                     (optional) figure and axes handles
    """

    if azim_elev is not None:
        proj_dir =  azim_elev
    elif proj_dir_lattice is not None:
        proj_dir =  self.lattice_to_cartesian(proj_dir_lattice)
    elif proj_dir_cartesian is not None:
        proj_dir = proj_dir_cartesian
    else:
        proj_dir = np.mean(self.orientation_zone_axis_range,axis=0)

    if np.size(proj_dir) == 2:
        el = proj_dir[0]
        az = proj_dir[1]
    elif np.size(proj_dir) == 3:
        if proj_dir[0] == 0 and proj_dir[1] == 0:
            el = 90 * np.sign(proj_dir[2])
        else:
            el = (
                np.arctan(proj_dir[2] / np.sqrt(proj_dir[0] ** 2 + proj_dir[1] ** 2))
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
    if np.abs(self.cell[5]-120.0) < 1e-6:
        label_0 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[0, :])),
            tol_den=tol_den)
        label_1 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[1, :])),
            tol_den=tol_den)
        label_2 = self.rational_ind(
            self.lattice_to_hexagonal(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[2, :])),
            tol_den=tol_den)
    else:
        label_0 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[0, :]),
            tol_den=tol_den)
        label_1 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[1, :]),
            tol_den=tol_den)
        label_2 = self.rational_ind(
            self.cartesian_to_lattice(
            self.orientation_zone_axis_range[2, :]),
            tol_den=tol_den)


    # # label_0 = self.cartesian_to_crystal(self.orientation_zone_axis_range[0, :])
    # # if self.cartesian_directions:
    # label_0 = self.orientation_zone_axis_range[0, :]
    # # else:
    # #     label_0 = self.cartesian_to_crystal(self.orientation_zone_axis_range[0, :])
    # label_0 = np.round(label_0, decimals=3)
    # label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))
    # label_0 = np.round(label_0, decimals=3)

    if (
        self.orientation_full is False and
        self.orientation_half is False
        ):
        # # label_1 = self.cartesian_to_crystal(
        # #     self.orientation_zone_axis_range[1, :]
        # #     )
        # # if self.cartesian_directions:
        # label_1 = self.orientation_zone_axis_range[1, :]
        # # else:
        # #     label_1 = self.cartesian_to_crystal(self.orientation_zone_axis_range[1, :])
        # label_1 = np.round(label_1 * 1e3) * 1e-3
        # label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))
        # label_1 = np.round(label_1 * 1e3) * 1e-3

        # # label_2 = self.cartesian_to_crystal(
        # #     self.orientation_zone_axis_range[2, :]
        # # )
        # # if self.cartesian_directions:
        # label_2 = self.orientation_zone_axis_range[2, :]
        # # else:
        # #     label_2 = self.cartesian_to_crystal(self.orientation_zone_axis_range[2, :])

        # label_2 = np.round(label_2 * 1e3) * 1e-3
        # label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))
        # label_2 = np.round(label_2 * 1e3) * 1e-3

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
        "size": 16,
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
    if (
        self.orientation_full is False and
        self.orientation_half is False
    ):
        ax.text(
            self.orientation_vecs[inds[1], 1] * text_scale_pos,
            self.orientation_vecs[inds[1], 0] * text_scale_pos,
            self.orientation_vecs[inds[1], 2] * text_scale_pos,
            label_1,
            None,
            zorder=12,
            ha="center",
            **text_params,
        )
        ax.text(
            self.orientation_vecs[inds[2], 1] * text_scale_pos,
            self.orientation_vecs[inds[2], 0] * text_scale_pos,
            self.orientation_vecs[inds[2], 2] * text_scale_pos,
            label_2,
            None,
            zorder=13,
            ha="center",
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
    zone_axis_lattice: Optional[np.ndarray] = None,
    zone_axis_cartesian: Optional[np.ndarray] = None,
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
    if zone_axis_lattice is not None or zone_axis_cartesian is not None:
        orientation_matrix = self.parse_orientation(
            zone_axis_lattice=zone_axis_lattice,
            proj_x_lattice=None,
            zone_axis_cartesian=zone_axis_cartesian,
            proj_x_cartesian=None)
        index_plot = np.argmin(np.sum(np.abs(
            self.orientation_vecs - orientation_matrix[:,2]), axis=1)) 



    # if zone_axis_plot is not None:
    #     zone_axis_plot = np.array(zone_axis_plot, dtype="float")
    #     zone_axis_plot = zone_axis_plot / np.linalg.norm(zone_axis_plot)

    #     if not self.cartesian_directions:
    #         print(np.round(zone_axis_plot,decimals=6))
    #         zone_axis_plot = self.crystal_to_cartesian(zone_axis_plot)
    #         print(np.round(zone_axis_plot,decimals=6))

    #     index_plot = np.argmin(
    #         np.sum((self.orientation_vecs - zone_axis_plot) ** 2, axis=1)
    #     )
    #     print("Orientation plan index " + str(index_plot))

    # initialize figure
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Generate and plot diffraction pattern
    k_x_y_range = np.array([1, 1]) * self.k_max * 1.2
    bragg_peaks = self.generate_diffraction_pattern(
        orientation_matrix=self.orientation_rotation_matrices[index_plot, :],
        sigma_excitation_error=self.orientation_kernel_size / 3,
    )

    plot_diffraction_pattern(
        bragg_peaks,
        figsize=(figsize[1], figsize[1]),
        plot_range_kx_ky=k_x_y_range,
        # scale_markers=10,
        # shift_labels=0.10,
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
    # im_plot = self.orientation_ref[index_plot, :, :] / self.orientation_ref_max
    im_plot = (
        np.real(np.fft.ifft(self.orientation_ref[index_plot, :, :], axis=1)).astype(
            "float"
        )
        / self.orientation_ref_max
    )

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


def plot_diffraction_pattern(
    bragg_peaks: PointList,
    bragg_peaks_compare: PointList = None,
    scale_markers: float = 500,
    scale_markers_compare: Optional[float] = None,
    power_markers: float = 1,
    plot_range_kx_ky: Optional[Union[list, tuple, np.ndarray]] = None,
    add_labels: bool = True,
    shift_labels: float = 0.08,
    shift_marker: float = 0.005,
    min_marker_size: float = 1e-6,
    figsize: Union[list, tuple, np.ndarray] = (12, 6),
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
    else:
        k_range = 1.05*np.sqrt(np.max(
            bragg_peaks.data["qx"]**2 + 
            bragg_peaks.data["qy"]**2))
        ax.set_xlim((-k_range, k_range))
        ax.set_ylim((-k_range, k_range))

    ax.invert_yaxis()
    ax.set_box_aspect(1)
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


def plot_orientation_maps(
    self,
    orientation_map,
    orientation_ind: int = None,
    dir_in_plane_degrees: float = 0.0,
    corr_range: np.ndarray = np.array([0, 5]),
    corr_normalize: bool = True,
    scale_legend: bool = None,
    figsize: Union[list, tuple, np.ndarray] = (16, 5),
    figbound: Union[list, tuple, np.ndarray] = (0.01, 0.005),
    camera_dist = None,
    plot_limit = None,
    swap_axes_xy_limits = False,
    returnfig: bool = False,
    progress_bar = False,
    ):
    """
    Generate and plot the orientation maps

    Args:
        orientation_map (OrientationMap):   Class containing orientation matrices, correlation values, etc.
        orientation_ind (int):              Which orientation match to plot if num_matches > 1
        dir_in_plane_degrees (float):       In-plane angle to plot in degrees.  Default is 0 / x-axis / vertical down.
        corr_range (np.ndarray):            Correlation intensity range for the plot
        corr_normalize (bool):              If true, set mean correlation to 1.
        scale_legend (float):               2 elements, x and y scaling of legend panel
        figsize (array):                    2 elements defining figure size
        figbound (array):                   2 elements defining figure boundary
        camera_dist (float):                distance of camera from legend
        plot_limit (array):                 2x3 array defining plot boundaries of egend
        swap_axes_xy_limits (bool):         swap x and y boundaries for legend (not sure why we need this in some cases)
        returnfig (bool):                   set to True to return figure and axes handles
        progress_bar (bool):                Enable progressbar when calculating orientation images.

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

    if orientation_ind is None:
        orientation_ind = 0

    # Color of the 3 corners
    color_basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.3, 1.0],
        ]
    )

    # Generate reflection operators for symmetry reduction
    A_ref = np.zeros((3,3,3))
    # A_ref[0] = np.array([
    #     [-1, 0, 0],
    #     [ 0,-1. 0],
    #     [ 0, 0,-1]])
    for a0 in range(3):
        if a0 == 0:
            v = np.cross(
                self.orientation_zone_axis_range[1,:],
                self.orientation_zone_axis_range[0,:])
        elif a0 == 1:
            v = np.cross(
                self.orientation_zone_axis_range[2,:],
                self.orientation_zone_axis_range[1,:])
        elif a0 == 2:
            v = np.cross(
                self.orientation_zone_axis_range[0,:],
                self.orientation_zone_axis_range[2,:])
        v = v / np.linalg.norm(v)

        A_ref[a0] = np.array([
            [ 1-2*v[0]**2, -2*v[0]*v[1], -2*v[0]*v[2]],
            [-2*v[0]*v[1],  1-2*v[1]**2, -2*v[1]*v[2]],
            [-2*v[0]*v[2], -2*v[1]*v[2],  1-2*v[2]**2],
            ])

    # init
    dir_in_plane = np.deg2rad(dir_in_plane_degrees)
    ct = np.cos(dir_in_plane)
    st = np.sin(dir_in_plane)
    basis_x = np.zeros((orientation_map.num_x,orientation_map.num_y,3))
    basis_z = np.zeros((orientation_map.num_x,orientation_map.num_y,3))
    rgb_x = np.zeros((orientation_map.num_x,orientation_map.num_y,3))
    rgb_z = np.zeros((orientation_map.num_x,orientation_map.num_y,3))

    # Basis for fitting orientation projections 
    A = np.linalg.inv(self.orientation_zone_axis_range).T
    # A = self.orientation_zone_axis_range
    # print(np.round(A,decimals=3))
    # A = np.linalg.inv(self.orientation_zone_axis_range).T
    # A = self.orientation_zone_axis_range

    # Correlation masking
    corr = orientation_map.corr[:,:,orientation_ind]
    if corr_normalize:
        corr = corr / np.mean(corr)
    mask = (corr - corr_range[0]) / (corr_range[1] - corr_range[0])
    mask = np.clip(mask, 0, 1)

    # Generate images
    for rx, ry in tqdmnd(
        orientation_map.num_x,
        orientation_map.num_y,
        desc="Generating orientation maps",
        unit=" PointList",
        disable=not progress_bar,
        ):

        if self.pymatgen_available:
            basis_x[rx,ry,:] = A @ orientation_map.family[rx,ry,orientation_ind,:,0]
            basis_z[rx,ry,:] = A @ orientation_map.family[rx,ry,orientation_ind,:,2]
        else:
            basis_z[rx,ry,:] = A @ orientation_map.matrix[rx,ry,orientation_ind,:,2]
    basis_x = np.clip(basis_x,0,1)
    basis_z = np.clip(basis_z,0,1)

    # Convert to RGB images
    basis_x_scale = mask[:,:,None] * basis_x / np.max(basis_x,axis=2)[:,:,None]
    rgb_x = basis_x_scale[:,:,0][:,:,None]*color_basis[0,:][None,None,:] \
        + basis_x_scale[:,:,1][:,:,None]*color_basis[1,:][None,None,:] \
        + basis_x_scale[:,:,2][:,:,None]*color_basis[2,:][None,None,:]
    basis_z_scale = mask[:,:,None] * basis_z / np.max(basis_z,axis=2)[:,:,None]
    rgb_z = basis_z_scale[:,:,0][:,:,None]*color_basis[0,:][None,None,:] \
        + basis_z_scale[:,:,1][:,:,None]*color_basis[1,:][None,None,:] \
        + basis_z_scale[:,:,2][:,:,None]*color_basis[2,:][None,None,:]

    # Legend init
    # projection vector
    cam_dir = np.mean(self.orientation_zone_axis_range,axis=0)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    az = np.rad2deg(np.arctan2(cam_dir[0],cam_dir[1]))
    # el = np.rad2deg(np.arccos(cam_dir[2]))
    el = np.rad2deg(np.arcsin(cam_dir[2]))
    # coloring
    wx = self.orientation_inds[:,0] / self.orientation_zone_axis_steps
    wy = self.orientation_inds[:,1] / self.orientation_zone_axis_steps
    w0 = 1 - wx - 0.5*wy
    w1 = wx - wy
    w2 = wy
    # w0 = 1 - w1/2 - w2/2
    w_scale = np.maximum(np.maximum(w0, w1), w2)
    w_scale = 1 - np.exp(-w_scale)
    w0 = w0 / w_scale
    w1 = w1 / w_scale
    w2 = w2 / w_scale
    rgb_legend = np.clip( 
        w0[:,None]*color_basis[0,:] + \
        w1[:,None]*color_basis[1,:] + \
        w2[:,None]*color_basis[2,:],
        0,1)

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

    inds_legend = np.array(
        [
            0,
            self.orientation_num_zones - self.orientation_zone_axis_steps - 1,
            self.orientation_num_zones - 1,
        ]
    )

    # Determine if lattice direction labels should be left-right
    # or right-left aligned.
    v0 = self.orientation_vecs[inds_legend[0], :]
    v1 = self.orientation_vecs[inds_legend[1], :]
    v2 = self.orientation_vecs[inds_legend[2], :]
    n = np.cross(v0,cam_dir)
    if np.sum(v1 * n) < np.sum(v2 * n):
        ha_1 = 'left'
        ha_2 = 'right'
    else:
        ha_1 = 'right'
        ha_2 = 'left'


    # plotting frame
    # fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig = plt.figure(figsize=figsize)
    ax_x = fig.add_axes(
        [0.0+figbound[0], 0.0, 0.4-2*+figbound[0], 1.0])
    ax_z = fig.add_axes(
        [0.4+figbound[0], 0.0, 0.4-2*+figbound[0], 1.0])
    ax_l = fig.add_axes(
        [0.8+figbound[0], 0.0, 0.2-2*+figbound[0], 1.0],
        projection='3d',
        elev=el,
        azim=az)

    # orientation images
    if self.pymatgen_available:
        ax_x.imshow(rgb_x)
    else:
        ax_x.imshow(np.ones_like(rgb_z))
        ax_x.text(
            rgb_z.shape[1]/2,
            rgb_z.shape[0]/2-10,
            'in-plane orientation',
            fontsize=14,
            horizontalalignment='center')
        ax_x.text(
            rgb_z.shape[1]/2,
            rgb_z.shape[0]/2+0,
            'for this crystal system',
            fontsize=14,
            horizontalalignment='center')
        ax_x.text(
            rgb_z.shape[1]/2,
            rgb_z.shape[0]/2+10,
            'requires pymatgen',
            fontsize=14,
            horizontalalignment='center')
    ax_z.imshow(rgb_z)

    # Triangulate faces
    p = self.orientation_vecs[:,(1,0,2)]
    tri = mtri.Triangulation(
        self.orientation_inds[:,1]-self.orientation_inds[:,0]*1e-3,
        self.orientation_inds[:,0]-self.orientation_inds[:,1]*1e-3)
    # convert rgb values of pixels to faces
    rgb_faces = (
        rgb_legend[tri.triangles[:,0],:] + \
        rgb_legend[tri.triangles[:,1],:] + \
        rgb_legend[tri.triangles[:,2],:] \
        ) / 3
    # Add triangulated surface plot to axes
    pc = art3d.Poly3DCollection(
        p[tri.triangles],
        facecolors=rgb_faces,
        alpha=1,
    )
    pc.set_antialiased(False)
    ax_l.add_collection(pc)

    if plot_limit is None:
        plot_limit = np.array(
            [
                [np.min(p[:, 0]), np.min(p[:, 1]), np.min(p[:, 2])],
                [np.max(p[:, 0]), np.max(p[:, 1]), np.max(p[:, 2])],
            ]
        )
        # plot_limit = (plot_limit - np.mean(plot_limit, axis=0)) * 1.5 + np.mean(
        #     plot_limit, axis=0
        # )
        plot_limit[:,0] = (plot_limit[:,0] - np.mean(plot_limit[:,0]))*1.5 \
            + np.mean(plot_limit[:,0])
        plot_limit[:,1] = (plot_limit[:,2] - np.mean(plot_limit[:,1]))*1.5 \
            + np.mean(plot_limit[:,1])
        plot_limit[:,2] = (plot_limit[:,1] - np.mean(plot_limit[:,2]))*1.1 \
            + np.mean(plot_limit[:,2])

    # ax_l.view_init(elev=el, azim=az)
    # Appearance
    ax_l.invert_yaxis()
    if swap_axes_xy_limits:
        ax_l.axes.set_xlim3d(left=plot_limit[0, 0], right=plot_limit[1, 0])
        ax_l.axes.set_ylim3d(bottom=plot_limit[0, 1], top=plot_limit[1, 1])
        ax_l.axes.set_zlim3d(bottom=plot_limit[0, 2], top=plot_limit[1, 2])
    else:
        ax_l.axes.set_xlim3d(left=plot_limit[0, 1], right=plot_limit[1, 1])
        ax_l.axes.set_ylim3d(bottom=plot_limit[0, 0], top=plot_limit[1, 0])
        ax_l.axes.set_zlim3d(bottom=plot_limit[0, 2], top=plot_limit[1, 2])        
    axisEqual3D(ax_l)
    if camera_dist is not None:
        ax_l.dist = camera_dist
    ax_l.axis("off")

    # Add text labels
    text_scale_pos = 0.1
    text_params = {
        "va": "center",
        "family": "sans-serif",
        "fontweight": "normal",
        "color": "k",
        "size": 14,
    }
    format_labels = "{0:.2g}"
    vec = self.orientation_vecs[inds_legend[0], :] - cam_dir 
    vec = vec / np.linalg.norm(vec)
    if np.abs(self.cell[5]-120.0) > 1e-6:
        ax_l.text(
            self.orientation_vecs[inds_legend[0], 1] + vec[1] * text_scale_pos,
            self.orientation_vecs[inds_legend[0], 0] + vec[0] * text_scale_pos,
            self.orientation_vecs[inds_legend[0], 2] + vec[2] * text_scale_pos,
              '[' + format_labels.format(label_0[0])
            + ' ' + format_labels.format(label_0[1])
            + ' ' + format_labels.format(label_0[2]) + ']',
            None,
            zorder=11,
            ha="center",
            **text_params,
        )
    else:
        ax_l.text(
        self.orientation_vecs[inds_legend[0], 1] + vec[1] * text_scale_pos,
        self.orientation_vecs[inds_legend[0], 0] + vec[0] * text_scale_pos,
        self.orientation_vecs[inds_legend[0], 2] + vec[2] * text_scale_pos,
          '[' + format_labels.format(label_0[0])
        + ' ' + format_labels.format(label_0[1])
        + ' ' + format_labels.format(label_0[2])
        + ' ' + format_labels.format(label_0[3]) + ']',
        None,
        zorder=11,
        ha="center",
        **text_params,
        )
    vec = self.orientation_vecs[inds_legend[1], :] - cam_dir 
    vec = vec / np.linalg.norm(vec)
    if np.abs(self.cell[5]-120.0) > 1e-6:
        ax_l.text(
            self.orientation_vecs[inds_legend[1], 1] + vec[1] * text_scale_pos,
            self.orientation_vecs[inds_legend[1], 0] + vec[0] * text_scale_pos,
            self.orientation_vecs[inds_legend[1], 2] + vec[2] * text_scale_pos,
              '[' + format_labels.format(label_1[0])
            + ' ' + format_labels.format(label_1[1])
            + ' ' + format_labels.format(label_1[2]) + ']',
            None,
            zorder=12,
            ha=ha_1,
            **text_params,
        )
    else:
        ax_l.text(
            self.orientation_vecs[inds_legend[1], 1] + vec[1] * text_scale_pos,
            self.orientation_vecs[inds_legend[1], 0] + vec[0] * text_scale_pos,
            self.orientation_vecs[inds_legend[1], 2] + vec[2] * text_scale_pos,
              '[' + format_labels.format(label_1[0])
            + ' ' + format_labels.format(label_1[1])
            + ' ' + format_labels.format(label_1[2])
            + ' ' + format_labels.format(label_1[3]) + ']',
            None,
            zorder=12,
            ha=ha_1,
            **text_params,
        )
    vec = self.orientation_vecs[inds_legend[2], :] - cam_dir 
    vec = vec / np.linalg.norm(vec)
    if np.abs(self.cell[5]-120.0) > 1e-6:
        ax_l.text(
            self.orientation_vecs[inds_legend[2], 1] + vec[1] * text_scale_pos,
            self.orientation_vecs[inds_legend[2], 0] + vec[0] * text_scale_pos,
            self.orientation_vecs[inds_legend[2], 2] + vec[2] * text_scale_pos,
              '[' + format_labels.format(label_2[0])
            + ' ' + format_labels.format(label_2[1])
            + ' ' + format_labels.format(label_2[2]) + ']',
            None,
            zorder=13,
            ha=ha_2,
            **text_params,
        )
    else:
        ax_l.text(
            self.orientation_vecs[inds_legend[2], 1] + vec[1] * text_scale_pos,
            self.orientation_vecs[inds_legend[2], 0] + vec[0] * text_scale_pos,
            self.orientation_vecs[inds_legend[2], 2] + vec[2] * text_scale_pos,
              '[' + format_labels.format(label_2[0])
            + ' ' + format_labels.format(label_2[1])
            + ' ' + format_labels.format(label_2[2])
            + ' ' + format_labels.format(label_2[3]) + ']',
            None,
            zorder=13,
            ha=ha_2,
            **text_params,
        )

    plt.show()



    # collection = art3d.PolyCollection(self.orientation_vecs)
    # ax_l.add_collection(collection)

    # print(tri.)
    # print(type(tri))
    # ax_l.plot_trisurf(
    #     self.orientation_vecs[:,1],
    #     self.orientation_vecs[:,0],
    #     self.orientation_vecs[:,2],
    #     triangles=tri.triangles, 
    #     color=rgb_legend)
    # maskedTris = tri.get_masked_triangles()
    # verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
    # collection = art3d.PolyCollection(verts)
    # # collection.set_vertexcolor(rgb_legend)
    # ax_l.add_collection(collection)
    # shading='gouraud'
    # pc = art3d.Poly3DCollection(
    #     self.orientation_vecs,
    #     alpha=1,
    # )
    # ax_l.add_collection(pc)


    # art3d.Poly3DCollection

    # plt.gca().autoscale_view()
    # ax_l.plot_surface(
    #     self.orientation_vecs[:,1], 
    #     self.orientation_vecs[:,0], 
    #     self.orientation_vecs[:,2], 
    #     )



    # # Generate crystal basis images.
    # # Each channel represents one vector of self.orientation_zone_axis_range.
    # # for ax in range(1):
    # #     for ay in range(1):
    # count_max = 1
    # for rx, ry in tqdmnd(
    #     orientation_map.num_x,
    #     10,#orientation_map.num_y,
    #     desc="Generating orientation maps",
    #     unit=" PointList",
    #     disable=not progress_bar,
    #     ):
    #     if orientation_map.corr[rx,ry] > 0:

    #         # Reflect the in-plane direction into orientation triangle
    #         v = orientation_map.matrix[rx,ry,orientation_ind,:,0] * ct \
    #             + orientation_map.matrix[rx,ry,orientation_ind,:,1] * st
    #         w = np.linalg.solve(A, v)
    #         if np.min(w) < 0:        
    #             count = 0
    #             while count < count_max:
    #                 if np.min(w) < 0:
    #                     v0 = v @ A_ref[0]
    #                     v1 = v @ A_ref[1]
    #                     v2 = v @ A_ref[2]
    #                     w0 = np.linalg.solve(A, v0)
    #                     w1 = np.linalg.solve(A, v1)
    #                     w2 = np.linalg.solve(A, v2)
    #                     ind = np.argmax([
    #                         np.mean(w0),
    #                         np.mean(w1),
    #                         np.mean(w2)])
    #                     if ind == 0:
    #                         v = v0
    #                         w = w0
    #                     elif ind == 1:
    #                         v = v1
    #                         w = w1
    #                     else:
    #                         v = v2
    #                         w = w2
    #                     # for a0 in range(A_ref.shape[0])
    #                     #     v_test = v @ A_ref[a0]
    #                     #     w_test = p.linalg.solve(A, v_test)
    #                     #     if a0 == 0:
    #                     #         v = v_test
    #                     #         w = w_test
    #                     #     else:
    #                     #         if np.mean(w_test) > np.mean(w):
    #                     #             v = v_test
    #                     #             w = w_test
    #                 else:
    #                     count = count_max
    #         dir_x[rx,ry] = w

    #         # Reflect the out-of-plane direction into orientation triangle
    #         v = orientation_map.matrix[rx,ry,orientation_ind,:,2]
    #         w = np.linalg.solve(A, v)
    #         if np.min(w) < 0:        
    #             count = 0
    #             while count < count_max:
    #                 if np.min(w) < 0:
    #                     v0 = v @ A_ref[0]
    #                     v1 = v @ A_ref[1]
    #                     v2 = v @ A_ref[2]
    #                     w0 = np.linalg.solve(A, v0)
    #                     w1 = np.linalg.solve(A, v1)
    #                     w2 = np.linalg.solve(A, v2)
    #                     ind = np.argmax([
    #                         np.mean(w0),
    #                         np.mean(w1),
    #                         np.mean(w2)])
    #                     if ind == 0:
    #                         v = v0
    #                         w = w0
    #                     elif ind == 1:
    #                         v = v1
    #                         w = w1
    #                     else:
    #                         v = v2
    #                         w = w2
    #                 else:
    #                     count = count_max
    #         dir_z[rx,ry] = w


    # # Basis for fitting fiber texture
    # if self.orientation_fiber:
    #     p1_proj = (
    #         np.dot(
    #             self.orientation_zone_axis_range[1, :],
    #             self.orientation_zone_axis_range[0, :],
    #         )
    #         * self.orientation_zone_axis_range[0, :]
    #     )
    #     p2_proj = (
    #         np.dot(
    #             self.orientation_zone_axis_range[2, :],
    #             self.orientation_zone_axis_range[0, :],
    #         )
    #         * self.orientation_zone_axis_range[0, :]
    #     )
    #     p1_sub = self.orientation_zone_axis_range[1, :] - p1_proj
    #     p2_sub = self.orientation_zone_axis_range[2, :] - p2_proj
    #     B = np.vstack((self.orientation_zone_axis_range[0, :], p1_sub, p2_sub)).T

    # # initalize image arrays
    # images_orientation = np.zeros(
    #     (orientation_matrices.shape[0], orientation_matrices.shape[1], 3, 3)
    # )

    # # in-plane rotation array if needed
    # if orientation_rotate_xy is not None:
    #     m = np.array(
    #         [
    #             [np.cos(orientation_rotate_xy), -np.sin(orientation_rotate_xy), 0],
    #             [np.sin(orientation_rotate_xy), np.cos(orientation_rotate_xy), 0],
    #             [0, 0, 1],
    #         ]
    #     )

    # # loop over all pixels and calculate weights
    # for ax in range(orientation_matrices.shape[0]):
    #     for ay in range(orientation_matrices.shape[1]):
    #         if orientation_matrices.ndim == 4:
    #             orient = orientation_matrices[ax, ay, :, :]
    #         else:
    #             orient = orientation_matrices[ax, ay, :, :, orientation_index_plot]

    #         # Rotate in-plane if needed
    #         if orientation_rotate_xy is not None:
    #             orient = m @ orient

    #         if self.orientation_fiber:
    #             # in-plane rotation
    #             # w = np.linalg.solve(A, orient[:, 0])
    #             w = np.linalg.solve(B, orient[:, 0])

    #             h = np.mod(
    #                 np.arctan2(w[2], w[1])
    #                 * 180
    #                 / np.pi
    #                 / self.orientation_fiber_angles[1],
    #                 1,
    #             )
    #             w0 = np.maximum(1 - 3 * np.abs(np.mod(3 / 6 - h, 1) - 1 / 2), 0)
    #             w1 = np.maximum(1 - 3 * np.abs(np.mod(5 / 6 - h, 1) - 1 / 2), 0)
    #             w2 = np.maximum(1 - 3 * np.abs(np.mod(7 / 6 - h, 1) - 1 / 2), 0)
    #             w_scale = 1 / (1 - np.exp(-np.max((w0, w1, w2))))

    #             rgb = (
    #                 color_basis[0, :] * w0 * w_scale
    #                 + color_basis[1, :] * w1 * w_scale
    #                 + color_basis[2, :] * w2 * w_scale
    #             )
    #             images_orientation[ax, ay, :, 0] = rgb

    #             # zone axis
    #             w = np.linalg.solve(A, orient[:, 2])
    #             w = w / (1 - np.exp(-np.max(w)))
    #             rgb = (
    #                 color_basis[0, :] * w[0]
    #                 + color_basis[1, :] * w[1]
    #                 + color_basis[2, :] * w[2]
    #             )
    #             images_orientation[ax, ay, :, 2] = rgb

    #         else:
    #             for a0 in range(3):
    #                 # Cubic sorting for now - needs to be updated with symmetries
    #                 # w = np.linalg.solve(A,orient[:,a0])
    #                 w = np.linalg.solve(A, np.sort(np.abs(orient[:, a0])))
    #                 # w = np.linalg.solve(A, orient[:, a0])
    #                 w = w / (1 - np.exp(-np.max(w)))

    #                 rgb = (
    #                     color_basis[0, :] * w[0]
    #                     + color_basis[1, :] * w[1]
    #                     + color_basis[2, :] * w[2]
    #                 )
    #                 images_orientation[ax, ay, :, a0] = rgb

    # # clip range
    # images_orientation = np.clip(images_orientation, 0, 1)

    # # Masking
    # if corr_all is not None:
    #     if orientation_matrices.ndim == 4:
    #         if corr_normalize:
    #             mask = corr_all / np.mean(corr_all)
    #         else:
    #             mask = corr_all
    #     else:
    #         if corr_normalize:
    #             mask = corr_all[:, :, orientation_index_plot] / np.mean(
    #                 corr_all[:, :, orientation_index_plot]
    #             )
    #         else:
    #             mask = corr_all[:, :, orientation_index_plot]

    #     mask = (mask - corr_range[0]) / (corr_range[1] - corr_range[0])
    #     mask = np.clip(mask, 0, 1)

    #     for a0 in range(3):
    #         for a1 in range(3):
    #             images_orientation[:, :, a0, a1] *= mask

    # # Draw legend for zone axis
    # x = np.linspace(0, 1, leg_size[0])
    # y = np.linspace(0, 1, leg_size[1])
    # ya, xa = np.meshgrid(y, x)
    # mask_legend = np.logical_and(2 * xa > ya, 2 * xa < 2 - ya)
    # w0 = 1 - xa - 0.5 * ya
    # w1 = xa - 0.5 * ya
    # w2 = ya

    # w_scale = np.maximum(np.maximum(w0, w1), w2)
    # # w_scale = w0 + w1 + w2
    # # w_scale = (w0**4 + w1**4 + w2**4)**0.25
    # w_scale = 1 - np.exp(-w_scale)
    # w0 = w0 / w_scale  # * mask_legend
    # w1 = w1 / w_scale  # * mask_legend
    # w2 = w2 / w_scale  # * mask_legend

    # im_legend = np.zeros((leg_size[0], leg_size[1], 3))
    # for a0 in range(3):
    #     im_legend[:, :, a0] = (
    #         w0 * color_basis[0, a0] + w1 * color_basis[1, a0] + w2 * color_basis[2, a0]
    #     )
    #     im_legend[:, :, a0] *= mask_legend
    #     im_legend[:, :, a0] += 1 - mask_legend
    # im_legend = np.clip(im_legend, 0, 1)

    # if self.orientation_fiber:
    #     # Draw legend for in-plane rotation
    #     x = np.linspace(-1, 1, leg_size[0])
    #     y = np.linspace(-1, 1, leg_size[1])
    #     ya, xa = np.meshgrid(y, x)
    #     mask_legend = xa ** 2 + ya ** 2 <= 1

    #     h = np.mod(
    #         np.arctan2(ya, xa) * 180 / np.pi / self.orientation_fiber_angles[1], 1
    #     )
    #     w0 = np.maximum(1 - 3 * np.abs(np.mod(3 / 6 - h, 1) - 1 / 2), 0)
    #     w1 = np.maximum(1 - 3 * np.abs(np.mod(5 / 6 - h, 1) - 1 / 2), 0)
    #     w2 = np.maximum(1 - 3 * np.abs(np.mod(7 / 6 - h, 1) - 1 / 2), 0)

    #     w_scale = np.maximum(np.maximum(w0, w1), w2)
    #     # w_scale = w0 + w1 + w2
    #     # w_scale = (w0**4 + w1**4 + w2**4)**0.25
    #     w_scale = 1 - np.exp(-w_scale)
    #     w0 = w0 / w_scale  # * mask_legend
    #     w1 = w1 / w_scale  # * mask_legend
    #     w2 = w2 / w_scale  # * mask_legend

    #     inplane_legend = np.zeros((leg_size[0], leg_size[1], 3))
    #     for a0 in range(3):
    #         inplane_legend[:, :, a0] = (
    #             w0 * color_basis[0, a0]
    #             + w1 * color_basis[1, a0]
    #             + w2 * color_basis[2, a0]
    #         )
    #         inplane_legend[:, :, a0] *= mask_legend
    #         inplane_legend[:, :, a0] += 1 - mask_legend
    #     inplane_legend = np.clip(inplane_legend, 0, 1)

    # # plotting
    # if figlayout[0] == 1 and figlayout[1] == 4:
    #     fig, ax = plt.subplots(1, 4, figsize=figsize)
    # elif figlayout[0] == 2 and figlayout[1] == 2:
    #     fig, ax = plt.subplots(2, 2, figsize=figsize)
    #     ax = np.array(
    #         [
    #             ax[0, 0],
    #             ax[0, 1],
    #             ax[1, 0],
    #             ax[1, 1],
    #         ]
    #     )
    # elif figlayout[0] == 4 and figlayout[1] == 1:
    #     fig, ax = plt.subplots(4, 1, figsize=figsize)

    # ax[0].imshow(images_orientation[:, :, :, 0])
    # if self.orientation_fiber:
    #     ax[1].imshow(inplane_legend, aspect="auto")
    # else:
    #     ax[1].imshow(images_orientation[:, :, :, 1])
    # ax[2].imshow(images_orientation[:, :, :, 2])

    # if self.orientation_fiber:
    #     ax[0].set_title("In-Plane Rotation", size=20)
    #     # ax[1].imshow(im_legend, aspect="auto")

    # else:
    #     ax[0].set_title("Orientation of x-axis", size=20)
    #     ax[1].set_title("Orientation of y-axis", size=20)
    # ax[2].set_title("Zone Axis", size=20)
    # ax[0].xaxis.tick_top()
    # ax[1].xaxis.tick_top()
    # ax[2].xaxis.tick_top()

    # # Legend
    # ax[3].imshow(im_legend, aspect="auto")

    # label_0 = self.orientation_zone_axis_range[0, :]
    # label_0 = np.round(label_0 * 1e3) * 1e-3
    # label_0 = label_0 / np.min(np.abs(label_0[np.abs(label_0) > 0]))

    # label_1 = self.orientation_zone_axis_range[1, :]
    # label_1 = np.round(label_1 * 1e3) * 1e-3
    # label_1 = label_1 / np.min(np.abs(label_1[np.abs(label_1) > 0]))

    # label_2 = self.orientation_zone_axis_range[2, :]
    # label_2 = np.round(label_2 * 1e3) * 1e-3
    # label_2 = label_2 / np.min(np.abs(label_2[np.abs(label_2) > 0]))

    # ax[3].yaxis.tick_right()
    # ax[3].set_yticks([(leg_size[0] - 1) / 2])
    # ax[3].set_yticklabels([str(label_2)])

    # ax3a = ax[3].twiny()
    # ax3b = ax[3].twiny()

    # ax3a.set_xticks([0])
    # ax3a.set_xticklabels([str(label_0)])
    # ax3a.xaxis.tick_top()
    # ax3b.set_xticks([0])
    # ax3b.set_xticklabels([str(label_1)])
    # ax3b.xaxis.tick_bottom()
    # ax[3].set_xticks([])

    # # ax[3].xaxis.label.set_color('none')
    # ax[3].spines["left"].set_color("none")
    # ax[3].spines["right"].set_color("none")
    # ax[3].spines["top"].set_color("none")
    # ax[3].spines["bottom"].set_color("none")

    # ax3a.spines["left"].set_color("none")
    # ax3a.spines["right"].set_color("none")
    # ax3a.spines["top"].set_color("none")
    # ax3a.spines["bottom"].set_color("none")

    # ax3b.spines["left"].set_color("none")
    # ax3b.spines["right"].set_color("none")
    # ax3b.spines["top"].set_color("none")
    # ax3b.spines["bottom"].set_color("none")

    # ax[3].tick_params(labelsize=16)
    # ax3a.tick_params(labelsize=16)
    # ax3b.tick_params(labelsize=16)

    # if self.orientation_fiber:
    #     ax[1].axis("off")

    # if scale_legend is not None:
    #     pos = ax[3].get_position()
    #     pos_new = [
    #         pos.x0,
    #         pos.y0 + pos.height * (1 - scale_legend[1]) / 2,
    #         pos.width * scale_legend[0],
    #         pos.height * scale_legend[1],
    #     ]
    #     ax[3].set_position(pos_new)

    #     if self.orientation_fiber:
    #         pos = ax[1].get_position()
    #         if np.size(scale_legend) == 2:
    #             pos_new = [
    #                 pos.x0,
    #                 pos.y0 + pos.height * (1 - scale_legend[1]) / 2,
    #                 pos.width * scale_legend[0],
    #                 pos.height * scale_legend[1],
    #             ]
    #         elif np.size(scale_legend) == 4:
    #             pos_new = [
    #                 pos.x0,
    #                 pos.y0 + pos.height * (1 - scale_legend[3]) / 2,
    #                 pos.width * scale_legend[2],
    #                 pos.height * scale_legend[3],
    #             ]
    #         ax[1].set_position(pos_new)


    images_orientation = np.zeros((
        orientation_map.num_x,
        orientation_map.num_y,
        3,2))
    if self.pymatgen_available:
        images_orientation[:,:,:,0] = rgb_x
    images_orientation[:,:,:,0] = rgb_z

    if returnfig:
        ax = [ax_x,ax_z,ax_l]
        return images_orientation, fig, ax
    else:
        return images_orientation


# def crystal_to_cartesian(self, zone_axis):
#     vec_cart = zone_axis @ self.lat_real
#     return vec_cart / np.linalg.norm(vec_cart)

# def cartesian_to_crystal(self, vec_cart):
#     zone_axis = vec_cart @ np.linalg.inv(self.lat_real)
#     return zone_axis / np.linalg.norm(zone_axis)


# def cartesian_to_crystal(self, zone_axis):
#     vec_cart = zone_axis @ self.lat_real
#     return vec_cart / np.linalg.norm(vec_cart)


# def crystal_to_cartesian(self, vec_cart):
#     zone_axis = vec_cart @ np.linalg.inv(self.lat_real)
#     return zone_axis / np.linalg.norm(zone_axis)






def plot_fiber_orientation_maps(
    self,
    orientation_map,
    orientation_ind: int = None,
    symmetry_order: int = None,
    symmetry_mirror: bool = False,
    dir_in_plane_degrees: float = 0.0,
    corr_range: np.ndarray = np.array([0, 5]),
    corr_normalize: bool = True,
    medfilt_size: int = None,
    cmap_out_of_plane: 'string' = 'plasma',
    leg_size: int = 200,
    figsize: Union[list, tuple, np.ndarray] = (12, 8),
    figbound: Union[list, tuple, np.ndarray] = (0.005, 0.04),
    returnfig: bool = False,
    ):
    """
    Generate and plot the orientation maps from fiber texture plots.

    Args:
        orientation_map (OrientationMap):   Class containing orientation matrices, correlation values, etc.
        orientation_ind (int):              Which orientation match to plot if num_matches > 1
        dir_in_plane_degrees (float):       Reference in-plane angle (degrees).  Default is 0 / x-axis / vertical down.
        corr_range (np.ndarray):            Correlation intensity range for the plot
        corr_normalize (bool):              If true, set mean correlation to 1.
        figsize (array):                    2 elements defining figure size
        figbound (array):                   2 elements defining figure boundary
        returnfig (bool):                   set to True to return figure and axes handles

    Returns:
        images_orientation (int):       RGB images
        fig, axs (handles):             Figure and axes handes for the

    NOTE:
        Currently, no symmetry reduction.  Therefore the x and y orientations
        are going to be correct only for [001][011][111] orientation triangle.

    """

    # angular colormap
    basis = np.array([
        [1.0, 0.2, 0.2],
        [1.0, 0.7, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.8, 1.0],
        [0.2, 0.4, 1.0],
        [0.9, 0.2, 1.0],
    ])

    # Correlation masking
    corr = orientation_map.corr[:,:,orientation_ind]
    if corr_normalize:
        corr = corr / np.mean(corr)
    if medfilt_size is not None:
        corr = medfilt(corr,medfilt_size)
    mask = (corr - corr_range[0]) / (corr_range[1] - corr_range[0])
    mask = np.clip(mask, 0, 1)

    # Get symmetry
    if symmetry_order is None:
        symmetry_order = np.round(360.0 / self.orientation_fiber_angles[1])
    elif symmetry_mirror:
        symmetry_order = 2 * symmetry_order

    # Generate out-of-plane orientation signal
    ang_op = orientation_map.angles[:,:,orientation_ind,1]
    sig_op = ang_op / np.deg2rad(self.orientation_fiber_angles[0])
    if medfilt_size is not None:
        sig_op = medfilt(sig_op,medfilt_size)

    # Generate in-plane orientation signal
    ang_ip = orientation_map.angles[:,:,orientation_ind,0] \
        + orientation_map.angles[:,:,orientation_ind,2]
    sig_ip = np.mod((symmetry_order/(2*np.pi))*ang_ip,1.0)
    if symmetry_mirror:
        sub = np.sin((symmetry_order/2)*ang_ip) < 0
        sig_ip[sub] = np.mod(-sig_ip[sub],1)
    if medfilt_size is not None:
        sig_ip = medfilt(sig_ip,medfilt_size)

    # out-of-plane RGB images
    # im_op = plt.cm.blues(sig_op)
    cmap = plt.get_cmap(cmap_out_of_plane)
    im_op = cmap(sig_op)
    im_op = np.delete(im_op, 3, axis=2)
    im_op = im_op * mask[:,:,None]

    # in-plane image
    im_ip = np.zeros((
        sig_ip.shape[0],
        sig_ip.shape[1],
        3))
    for a0 in range(basis.shape[0]):
        weight = np.maximum(1-np.abs(np.mod(
            sig_ip - a0/basis.shape[0] + 0.5, 1.0) - 0.5) * basis.shape[0], 0)
        im_ip += basis[a0,:][None,None,:] * weight[:,:,None]
    im_ip = np.clip(im_ip, 0, 1)
    im_ip = im_ip * mask[:,:,None]

    # draw in-plane legends
    r = np.arange(leg_size) - leg_size/2 + 0.5
    ya,xa = np.meshgrid(r,r)
    ra = np.sqrt(xa**2 + ya**2)
    ta = np.arctan2(ya,xa)
    sig_leg = np.mod((symmetry_order/(2*np.pi))*ta,1.0)
    if symmetry_mirror:
        sub = np.sin((symmetry_order/2)*ta) < 0
        sig_leg[sub] = np.mod(-sig_leg[sub],1)
    # leg_ip = 
    im_ip_leg = np.zeros((leg_size,leg_size,3))
    for a0 in range(basis.shape[0]):
        weight = np.maximum(1-np.abs(np.mod(
            sig_leg - a0/basis.shape[0] + 0.5, 1.0) - 0.5) * basis.shape[0], 0)
        im_ip_leg += basis[a0,:][None,None,:] * weight[:,:,None]
    im_ip_leg = np.clip(im_ip_leg, 0, 1)
    mask = np.clip(leg_size/2 - ra + 0.5, 0, 1) \
        * np.clip(ra - leg_size/4 + 0.5, 0, 1)
    im_ip_leg = im_ip_leg*mask[:,:,None] + (1-mask)[:,:,None]

    # t = np.linspace(0,2*np.pi,1001)
    # y = np.mod((symmetry_order/(2*np.pi))*t,1.0)
    # if symmetry_mirror:
    #     sub = np.sin((symmetry_order/2)*t) < 0
    #     y[sub] = np.mod(-y[sub],1)

    # plotting frame
    # fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig = plt.figure(figsize=figsize)

    ax_ip = fig.add_axes(
        [0.0+figbound[0], 0.25+figbound[1], 0.5-2*+figbound[0], 0.75-figbound[1]])
    ax_op = fig.add_axes(
        [0.5+figbound[0], 0.25+figbound[1], 0.5-2*+figbound[0], 0.75-figbound[1]])

    ax_ip_l = fig.add_axes(
        [0.1+figbound[0], 0.0+figbound[1], 0.3-2*+figbound[0], 0.25-figbound[1]])
    ax_op_l = fig.add_axes(
        [0.6+figbound[0], 0.0+figbound[1], 0.3-2*+figbound[0], 0.25-figbound[1]])

    # in-plane
    ax_ip.imshow(im_ip)
    ax_ip.set_title("In-Plane Rotation", size=16)

    #out of plane
    ax_op.imshow(im_op)
    ax_op.set_title("Out-of-Plane Tilt", size=16)

    # in plane legend
    ax_ip_l.imshow(im_ip_leg)
    ax_ip_l.set_axis_off()

    # out of plane legend
    t = np.tile(np.linspace(0,1,leg_size,endpoint=True),(np.round(leg_size/10).astype('int'),1))
    im_op_leg = cmap(t)
    im_op_leg = np.delete(im_op_leg, 3, axis=2)
    ax_op_l.imshow(im_op_leg)
    ax_op_l.set_yticks([])

    ticks = [
        np.round(leg_size*0.0), 
        np.round(leg_size*0.25), 
        np.round(leg_size*0.5), 
        np.round(leg_size*0.75), 
        np.round(leg_size*1.0), 
        ]
    labels = [
        str(np.round(self.orientation_fiber_angles[0]*0.00)) + '$\degree$', 
        str(np.round(self.orientation_fiber_angles[0]*0.25)) + '$\degree$', 
        str(np.round(self.orientation_fiber_angles[0]*0.50)) + '$\degree$', 
        str(np.round(self.orientation_fiber_angles[0]*0.75)) + '$\degree$', 
        str(np.round(self.orientation_fiber_angles[0]*1.00)) + '$\degree$', 
        ]
    ax_op_l.set_xticks(ticks)
    ax_op_l.set_xticklabels(labels)

    images_orientation = np.zeros((
        orientation_map.num_x,
        orientation_map.num_y,
        3,2))
    images_orientation[:,:,:,0] = im_ip
    images_orientation[:,:,:,1] = im_op

    if returnfig:
        ax = [ax_ip, ax_op, ax_ip_l, ax_op_l]
        return images_orientation, fig, ax
    else:
        return images_orientation



def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)
    ax.set_box_aspect((1, 1, 1))



def atomic_colors(Z, scheme="jmol"):
    """
    Return atomic colors for Z.

    Modes are "colin" and "jmol".
    "colin" uses the handmade but incomplete scheme of Colin Ophus
    "jmol" uses the JMOL scheme, from http://jmol.sourceforge.net/jscolors
        which includes all elements up to 109
    """
    if scheme == "jmol":
        return np.array(jmol_colors.get(Z, (0.0, 0.0, 0.0)))
    else:
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
        }.get(Z, np.array([0.0, 0.0, 0.0]))


jmol_colors = {
    1: (1.000, 1.000, 1.000),
    2: (0.851, 1.000, 1.000),
    3: (0.800, 0.502, 1.000),
    4: (0.761, 1.000, 0.000),
    5: (1.000, 0.710, 0.710),
    6: (0.565, 0.565, 0.565),
    7: (0.188, 0.314, 0.973),
    8: (1.000, 0.051, 0.051),
    9: (0.565, 0.878, 0.314),
    10: (0.702, 0.890, 0.961),
    11: (0.671, 0.361, 0.949),
    12: (0.541, 1.000, 0.000),
    13: (0.749, 0.651, 0.651),
    14: (0.941, 0.784, 0.627),
    15: (1.000, 0.502, 0.000),
    16: (1.000, 1.000, 0.188),
    17: (0.122, 0.941, 0.122),
    18: (0.502, 0.820, 0.890),
    19: (0.561, 0.251, 0.831),
    20: (0.239, 1.000, 0.000),
    21: (0.902, 0.902, 0.902),
    22: (0.749, 0.761, 0.780),
    23: (0.651, 0.651, 0.671),
    24: (0.541, 0.600, 0.780),
    25: (0.612, 0.478, 0.780),
    26: (0.878, 0.400, 0.200),
    27: (0.941, 0.565, 0.627),
    28: (0.314, 0.816, 0.314),
    29: (0.784, 0.502, 0.200),
    30: (0.490, 0.502, 0.690),
    31: (0.761, 0.561, 0.561),
    32: (0.400, 0.561, 0.561),
    33: (0.741, 0.502, 0.890),
    34: (1.000, 0.631, 0.000),
    35: (0.651, 0.161, 0.161),
    36: (0.361, 0.722, 0.820),
    37: (0.439, 0.180, 0.690),
    38: (0.000, 1.000, 0.000),
    39: (0.580, 1.000, 1.000),
    40: (0.580, 0.878, 0.878),
    41: (0.451, 0.761, 0.788),
    42: (0.329, 0.710, 0.710),
    43: (0.231, 0.620, 0.620),
    44: (0.141, 0.561, 0.561),
    45: (0.039, 0.490, 0.549),
    46: (0.000, 0.412, 0.522),
    47: (0.753, 0.753, 0.753),
    48: (1.000, 0.851, 0.561),
    49: (0.651, 0.459, 0.451),
    50: (0.400, 0.502, 0.502),
    51: (0.620, 0.388, 0.710),
    52: (0.831, 0.478, 0.000),
    53: (0.580, 0.000, 0.580),
    54: (0.259, 0.620, 0.690),
    55: (0.341, 0.090, 0.561),
    56: (0.000, 0.788, 0.000),
    57: (0.439, 0.831, 1.000),
    58: (1.000, 1.000, 0.780),
    59: (0.851, 1.000, 0.780),
    60: (0.780, 1.000, 0.780),
    61: (0.639, 1.000, 0.780),
    62: (0.561, 1.000, 0.780),
    63: (0.380, 1.000, 0.780),
    64: (0.271, 1.000, 0.780),
    65: (0.188, 1.000, 0.780),
    66: (0.122, 1.000, 0.780),
    67: (0.000, 1.000, 0.612),
    68: (0.000, 0.902, 0.459),
    69: (0.000, 0.831, 0.322),
    70: (0.000, 0.749, 0.220),
    71: (0.000, 0.671, 0.141),
    72: (0.302, 0.761, 1.000),
    73: (0.302, 0.651, 1.000),
    74: (0.129, 0.580, 0.839),
    75: (0.149, 0.490, 0.671),
    76: (0.149, 0.400, 0.588),
    77: (0.090, 0.329, 0.529),
    78: (0.816, 0.816, 0.878),
    79: (1.000, 0.820, 0.137),
    80: (0.722, 0.722, 0.816),
    81: (0.651, 0.329, 0.302),
    82: (0.341, 0.349, 0.380),
    83: (0.620, 0.310, 0.710),
    84: (0.671, 0.361, 0.000),
    85: (0.459, 0.310, 0.271),
    86: (0.259, 0.510, 0.588),
    87: (0.259, 0.000, 0.400),
    88: (0.000, 0.490, 0.000),
    89: (0.439, 0.671, 0.980),
    90: (0.000, 0.729, 1.000),
    91: (0.000, 0.631, 1.000),
    92: (0.000, 0.561, 1.000),
    93: (0.000, 0.502, 1.000),
    94: (0.000, 0.420, 1.000),
    95: (0.329, 0.361, 0.949),
    96: (0.471, 0.361, 0.890),
    97: (0.541, 0.310, 0.890),
    98: (0.631, 0.212, 0.831),
    99: (0.702, 0.122, 0.831),
    100: (0.702, 0.122, 0.729),
    101: (0.702, 0.051, 0.651),
    102: (0.741, 0.051, 0.529),
    103: (0.780, 0.000, 0.400),
    104: (0.800, 0.000, 0.349),
    105: (0.820, 0.000, 0.310),
    106: (0.851, 0.000, 0.271),
    107: (0.878, 0.000, 0.220),
    108: (0.902, 0.000, 0.180),
    109: (0.922, 0.000, 0.149),
}

# def isPointWithinPolygon(point, polygonVertexCoords):
#      path = matplotlib.path.Path( polygonVertexCoords )
#      return path.contains_point(point[0], point[1]) 