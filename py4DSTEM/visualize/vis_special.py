import numpy as np
from emdfile import PointList
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi

from py4DSTEM.visualize import show
from py4DSTEM.visualize.overlay import (
    add_bragg_index_labels,
    add_ellipses,
    add_pointlabels,
    add_points,
    add_scalebar,
    add_vector,
)
from py4DSTEM.visualize.vis_grid import show_image_grid
from py4DSTEM.visualize.vis_RQ import ax_addaxes, ax_addaxes_QtoR


def show_elliptical_fit(
    ar,
    fitradii,
    p_ellipse,
    fill=True,
    color_ann="y",
    color_ell="r",
    alpha_ann=0.2,
    alpha_ell=0.7,
    linewidth_ann=2,
    linewidth_ell=2,
    returnfig=False,
    **kwargs
):
    """
    Plots an elliptical curve over its annular fit region.

    Args:
        center (2-tuple): the center
        fitradii (2-tuple of numbers): the annulus inner and outer fit radii
        p_ellipse (5-tuple): the parameters of the fit ellipse, (qx0,qy0,a,b,theta).
            See the module docstring for utils.elliptical_coords for more details.
        fill (bool): if True, fills in the annular fitting region,
          else shows only inner/outer edges
        color_ann (color): annulus color
        color_ell (color): ellipse color
        alpha_ann: transparency for the annulus
        alpha_ell: transparency forn the fit ellipse
        linewidth_ann:
        linewidth_ell:
    """
    Ri, Ro = fitradii
    qx0, qy0, a, b, theta = p_ellipse
    fig, ax = show(
        ar,
        annulus={
            "center": (qx0, qy0),
            "radii": (Ri, Ro),
            "fill": fill,
            "color": color_ann,
            "alpha": alpha_ann,
            "linewidth": linewidth_ann,
        },
        ellipse={
            "center": (qx0, qy0),
            "a": a,
            "b": b,
            "theta": theta,
            "color": color_ell,
            "alpha": alpha_ell,
            "linewidth": linewidth_ell,
        },
        returnfig=True,
        **kwargs,
    )

    if not returnfig:
        plt.show()
        return
    else:
        return fig, ax


def show_amorphous_ring_fit(
    dp,
    fitradii,
    p_dsg,
    N=12,
    cmap=("gray", "gray"),
    fitborder=True,
    fitbordercolor="k",
    fitborderlw=0.5,
    scaling="log",
    ellipse=False,
    ellipse_color="r",
    ellipse_alpha=0.7,
    ellipse_lw=2,
    returnfig=False,
    **kwargs
):
    """
    Display a diffraction pattern with a fit to its amorphous ring, interleaving
    the data and the fit in a pinwheel pattern.

    Args:
        dp (array): the diffraction pattern
        fitradii (2-tuple of numbers): the min/max distances of the fitting annulus
        p_dsg (11-tuple): the fit parameters to the double-sided gaussian
            function returned by fit_ellipse_amorphous_ring
        N (int): the number of pinwheel sections
        cmap (colormap or 2-tuple of colormaps): if passed a single cmap, uses this
            colormap for both the data and the fit; if passed a 2-tuple of cmaps, uses
            the first for the data and the second for the fit
        fitborder (bool): if True, plots a border line around the fit data
        fitbordercolor (color): color of the fitborder
        fitborderlw (number): linewidth of the fitborder
        scaling (str): the normal scaling param -- see docstring for visualize.show
        ellipse (bool): if True, overlay an ellipse
        returnfig (bool): if True, returns the figure
    """
    from py4DSTEM.process.calibration import double_sided_gaussian
    from py4DSTEM.process.utils import convert_ellipse_params

    assert len(p_dsg) == 11
    assert isinstance(N, (int, np.integer))
    if isinstance(cmap, tuple):
        cmap_data, cmap_fit = cmap[0], cmap[1]
    else:
        cmap_data, cmap_fit = cmap, cmap
    Q_Nx, Q_Ny = dp.shape
    qmin, qmax = fitradii

    # Make coords
    qx0, qy0 = p_dsg[6], p_dsg[7]
    qyy, qxx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
    qx, qy = qxx - qx0, qyy - qy0
    q = np.hypot(qx, qy)
    theta = np.arctan2(qy, qx)

    # Make mask
    thetas = np.linspace(-np.pi, np.pi, 2 * N + 1)
    pinwheel = np.zeros((Q_Nx, Q_Ny), dtype=bool)
    for i in range(N):
        pinwheel += (theta > thetas[2 * i]) * (theta <= thetas[2 * i + 1])
    mask = pinwheel * (q > qmin) * (q <= qmax)

    # Get fit data
    fit = double_sided_gaussian(p_dsg, qxx, qyy)

    # Show
    (fig, ax), (vmin, vmax) = show(
        dp,
        scaling=scaling,
        cmap=cmap_data,
        mask=np.logical_not(mask),
        mask_color="empty",
        returnfig=True,
        returnclipvals=True,
        **kwargs,
    )
    show(
        fit,
        scaling=scaling,
        figax=(fig, ax),
        clipvals="manual",
        min=vmin,
        max=vmax,
        cmap=cmap_fit,
        mask=mask,
        mask_color="empty",
        **kwargs,
    )
    if fitborder:
        if N % 2 == 1:
            thetas += (thetas[1] - thetas[0]) / 2
        if (N // 2 % 2) == 0:
            thetas = np.roll(thetas, -1)
        for i in range(N):
            ax.add_patch(
                Wedge(
                    (qy0, qx0),
                    qmax,
                    np.degrees(thetas[2 * i]),
                    np.degrees(thetas[2 * i + 1]),
                    width=qmax - qmin,
                    fill=None,
                    color=fitbordercolor,
                    lw=fitborderlw,
                )
            )

    # Add ellipse overlay
    if ellipse:
        A, B, C = p_dsg[8], p_dsg[9], p_dsg[10]
        a, b, theta = convert_ellipse_params(A, B, C)
        ellipse = {
            "center": (qx0, qy0),
            "a": a,
            "b": b,
            "theta": theta,
            "color": ellipse_color,
            "alpha": ellipse_alpha,
            "linewidth": ellipse_lw,
        }
        add_ellipses(ax, ellipse)

    if not returnfig:
        plt.show()
        return
    else:
        return fig, ax


def show_qprofile(
    q,
    intensity,
    ymax=None,
    figsize=(12, 4),
    returnfig=False,
    color="k",
    xlabel="q (pixels)",
    ylabel="Intensity (A.U.)",
    labelsize=16,
    ticklabelsize=14,
    grid=True,
    label=None,
    **kwargs
):
    """
    Plots a diffraction space radial profile.
    Params:
        q               (1D array) the diffraction coordinate / x-axis
        intensity       (1D array) the y-axis values
        ymax            (number) max value for the yaxis
        color           (matplotlib color) profile color
        xlabel          (str)
        ylabel
        labelsize       size of x and y labels
        ticklabelsize
        grid            True or False
        label           a legend label for the plotted curve
    """
    if ymax is None:
        ymax = np.max(intensity) * 1.05

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, intensity, color=color, label=label)
    ax.grid(grid)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="x", labelsize=ticklabelsize)
    ax.set_yticklabels([])
    ax.set_xlabel(xlabel, size=labelsize)
    ax.set_ylabel(ylabel, size=labelsize)
    if not returnfig:
        plt.show()
        return
    else:
        return fig, ax


def show_kernel(kernel, R, L, W, figsize=(12, 6), returnfig=False, **kwargs):
    """
    Plots, side by side, the probe kernel and its line profile.
    R is the kernel plot's window size.
    L and W are the length and width of the lineprofile.
    """
    lineprofile_1 = np.concatenate(
        [np.sum(kernel[-L:, :W], axis=1), np.sum(kernel[:L, :W], axis=1)]
    )
    lineprofile_2 = np.concatenate(
        [np.sum(kernel[:W, -L:], axis=0), np.sum(kernel[:W, :L], axis=0)]
    )

    im_kernel = np.vstack(
        [
            np.hstack([kernel[-int(R) :, -int(R) :], kernel[-int(R) :, : int(R)]]),
            np.hstack([kernel[: int(R), -int(R) :], kernel[: int(R), : int(R)]]),
        ]
    )

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].matshow(im_kernel, cmap="gray")
    axs[0].plot(np.ones(2 * R) * R, np.arange(2 * R), c="r")
    axs[0].plot(np.arange(2 * R), np.ones(2 * R) * R, c="c")

    axs[1].plot(np.arange(len(lineprofile_1)), lineprofile_1, c="r")
    axs[1].plot(np.arange(len(lineprofile_2)), lineprofile_2, c="c")

    if not returnfig:
        plt.show()
        return
    else:
        return fig, axs


def show_voronoi(
    ar,
    x,
    y,
    color_points="r",
    color_lines="w",
    max_dist=None,
    returnfig=False,
    **kwargs
):
    """
    words
    """
    from py4DSTEM.process.utils import get_voronoi_vertices

    Nx, Ny = ar.shape
    points = np.vstack((x, y)).T
    voronoi = Voronoi(points)
    vertices = get_voronoi_vertices(voronoi, Nx, Ny)

    if max_dist is None:
        fig, ax = show(ar, returnfig=True, **kwargs)
    else:
        centers = [(x[i], y[i]) for i in range(len(x))]
        fig, ax = show(
            ar,
            returnfig=True,
            **kwargs,
            circle={
                "center": centers,
                "R": max_dist,
                "fill": False,
                "color": color_points,
            },
        )

    ax.scatter(voronoi.points[:, 1], voronoi.points[:, 0], color=color_points)
    for region in range(len(vertices)):
        vertices_curr = vertices[region]
        for i in range(len(vertices_curr)):
            x0, y0 = vertices_curr[i, :]
            xf, yf = vertices_curr[(i + 1) % len(vertices_curr), :]
            ax.plot((y0, yf), (x0, xf), color=color_lines)
    ax.set_xlim([0, Ny])
    ax.set_ylim([0, Nx])
    plt.gca().invert_yaxis()
    if not returnfig:
        plt.show()
        return
    else:
        return fig, ax


def show_class_BPs(ar, x, y, s, s2, color="r", color2="y", **kwargs):
    """
    words
    """
    N = len(x)
    assert N == len(y) == len(s)

    fig, ax = show(ar, returnfig=True, **kwargs)
    ax.scatter(y, x, s=s2, color=color2)
    ax.scatter(y, x, s=s, color=color)
    plt.show()
    return


def show_class_BPs_grid(
    ar,
    H,
    W,
    x,
    y,
    get_s,
    s2,
    color="r",
    color2="y",
    returnfig=False,
    axsize=(6, 6),
    titlesize=0,
    get_bordercolor=None,
    **kwargs
):
    """
    words
    """
    fig, axs = show_image_grid(
        lambda i: ar,
        H,
        W,
        axsize=axsize,
        titlesize=titlesize,
        get_bordercolor=get_bordercolor,
        returnfig=True,
        **kwargs,
    )
    for i in range(H):
        for j in range(W):
            ax = axs[i, j]
            N = i * W + j
            s = get_s(N)
            ax.scatter(y, x, s=s2, color=color2)
            ax.scatter(y, x, s=s, color=color)
    if not returnfig:
        plt.show()
        return
    else:
        return fig, axs


def show_strain(
    strainmap,
    vrange_exx,
    vrange_theta,
    vrange_exy=None,
    vrange_eyy=None,
    flip_theta=False,
    bkgrd=True,
    show_cbars=("exx", "eyy", "exy", "theta"),
    bordercolor="k",
    borderwidth=1,
    titlesize=24,
    ticklabelsize=16,
    ticknumber=5,
    unitlabelsize=24,
    show_axes=True,
    axes_x0=0,
    axes_y0=0,
    xaxis_x=1,
    xaxis_y=0,
    axes_length=10,
    axes_width=1,
    axes_color="r",
    xaxis_space="Q",
    labelaxes=True,
    QR_rotation=0,
    axes_labelsize=12,
    axes_labelcolor="r",
    axes_plots=("exx"),
    cmap="RdBu_r",
    layout=0,
    figsize=(12, 12),
    returnfig=False,
):
    """
    Display a strain map, showing the 4 strain components (e_xx,e_yy,e_xy,theta), and
    masking each image with strainmap.get_slice('mask')

    Args:
        strainmap (RealSlice):
        vrange_exx (length 2 list or tuple):
        vrange_theta (length 2 list or tuple):
        vrange_exy (length 2 list or tuple):
        vrange_eyy (length 2 list or tuple):
        flip_theta (bool): if True, take negative of angle
        bkgrd (bool):
        show_cbars (tuple of strings): Show colorbars for the specified axes. Must be a
            tuple containing any, all, or none of ('exx','eyy','exy','theta').
        bordercolor (color):
        borderwidth (number):
        titlesize (number):
        ticklabelsize (number):
        ticknumber (number): number of ticks on colorbars
        unitlabelsize (number):
        show_axes (bool):
        axes_x0 (number):
        axes_y0 (number):
        xaxis_x (number):
        xaxis_y (number):
        axes_length (number):
        axes_width (number):
        axes_color (color):
        xaxis_space (string): must be 'Q' or 'R'
        labelaxes (bool):
        QR_rotation (number):
        axes_labelsize (number):
        axes_labelcolor (color):
        axes_plots (tuple of strings): controls if coordinate axes showing the
            orientation of the strain matrices are overlaid over any of the plots.
            Must be a tuple of strings containing any, all, or none of
            ('exx','eyy','exy','theta').
        cmap (colormap):
        layout=0 (int): determines the layout of the grid which the strain components
            will be plotted in.  Must be in (0,1,2).  0=(2x2), 1=(1x4), 2=(4x1).
        figsize (length 2 tuple of numbers):
        returnfig (bool):
    """
    # Lookup table for different layouts
    assert layout in (0, 1, 2)
    layout_lookup = {
        0: ["left", "right", "left", "right"],
        1: ["bottom", "bottom", "bottom", "bottom"],
        2: ["right", "right", "right", "right"],
    }
    layout_p = layout_lookup[layout]

    # Contrast limits
    if vrange_exy is None:
        vrange_exy = vrange_exx
    if vrange_eyy is None:
        vrange_eyy = vrange_exx
    for vrange in (vrange_exx, vrange_eyy, vrange_exy, vrange_theta):
        assert len(vrange) == 2, "vranges must have length 2"
    vmin_exx, vmax_exx = vrange_exx[0] / 100.0, vrange_exx[1] / 100.0
    vmin_eyy, vmax_eyy = vrange_eyy[0] / 100.0, vrange_eyy[1] / 100.0
    vmin_exy, vmax_exy = vrange_exy[0] / 100.0, vrange_exy[1] / 100.0
    # theta is plotted in units of degrees
    vmin_theta, vmax_theta = vrange_theta[0] / (180.0 / np.pi), vrange_theta[1] / (
        180.0 / np.pi
    )

    # Get images
    e_xx = np.ma.array(
        strainmap.get_slice("e_xx").data, mask=strainmap.get_slice("mask").data == False
    )
    e_yy = np.ma.array(
        strainmap.get_slice("e_yy").data, mask=strainmap.get_slice("mask").data == False
    )
    e_xy = np.ma.array(
        strainmap.get_slice("e_xy").data, mask=strainmap.get_slice("mask").data == False
    )
    theta = np.ma.array(
        strainmap.get_slice("theta").data,
        mask=strainmap.get_slice("mask").data == False,
    )
    if flip_theta == True:
        theta = -theta

    # Plot
    if layout == 0:
        fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize)
    elif layout == 1:
        fig, (ax11, ax12, ax21, ax22) = plt.subplots(1, 4, figsize=figsize)
    else:
        fig, (ax11, ax12, ax21, ax22) = plt.subplots(4, 1, figsize=figsize)
    cax11 = show(
        e_xx,
        figax=(fig, ax11),
        vmin=vmin_exx,
        vmax=vmax_exx,
        intensity_range="absolute",
        cmap=cmap,
        returncax=True,
    )
    cax12 = show(
        e_yy,
        figax=(fig, ax12),
        vmin=vmin_eyy,
        vmax=vmax_eyy,
        intensity_range="absolute",
        cmap=cmap,
        returncax=True,
    )
    cax21 = show(
        e_xy,
        figax=(fig, ax21),
        vmin=vmin_exy,
        vmax=vmax_exy,
        intensity_range="absolute",
        cmap=cmap,
        returncax=True,
    )
    cax22 = show(
        theta,
        figax=(fig, ax22),
        vmin=vmin_theta,
        vmax=vmax_theta,
        intensity_range="absolute",
        cmap=cmap,
        returncax=True,
    )
    ax11.set_title(r"$\epsilon_{xx}$", size=titlesize)
    ax12.set_title(r"$\epsilon_{yy}$", size=titlesize)
    ax21.set_title(r"$\epsilon_{xy}$", size=titlesize)
    ax22.set_title(r"$\theta$", size=titlesize)

    # Add black background
    if bkgrd:
        mask = np.ma.masked_where(
            strainmap.get_slice("mask").data.astype(bool),
            np.zeros_like(strainmap.get_slice("mask").data),
        )
        ax11.matshow(mask, cmap="gray")
        ax12.matshow(mask, cmap="gray")
        ax21.matshow(mask, cmap="gray")
        ax22.matshow(mask, cmap="gray")

    # Colorbars
    show_cbars = np.array(
        [
            "exx" in show_cbars,
            "eyy" in show_cbars,
            "exy" in show_cbars,
            "theta" in show_cbars,
        ]
    )
    if np.any(show_cbars):
        divider11 = make_axes_locatable(ax11)
        divider12 = make_axes_locatable(ax12)
        divider21 = make_axes_locatable(ax21)
        divider22 = make_axes_locatable(ax22)
        cbax11 = divider11.append_axes(layout_p[0], size="4%", pad=0.15)
        cbax12 = divider12.append_axes(layout_p[1], size="4%", pad=0.15)
        cbax21 = divider21.append_axes(layout_p[2], size="4%", pad=0.15)
        cbax22 = divider22.append_axes(layout_p[3], size="4%", pad=0.15)
        for (ind, show_cbar, cax, cbax, vmin, vmax, tickside, tickunits) in zip(
            range(4),
            show_cbars,
            (cax11, cax12, cax21, cax22),
            (cbax11, cbax12, cbax21, cbax22),
            (vmin_exx, vmin_eyy, vmin_exy, vmin_theta),
            (vmax_exx, vmax_eyy, vmax_exy, vmax_theta),
            (layout_p[0], layout_p[1], layout_p[2], layout_p[3]),
            ("% ", " %", "% ", r" $^\circ$"),
        ):
            if show_cbar:
                ticks = np.linspace(vmin, vmax, ticknumber, endpoint=True)
                if ind < 3:
                    ticklabels = np.round(
                        np.linspace(100 * vmin, 100 * vmax, ticknumber, endpoint=True),
                        decimals=2,
                    ).astype(str)
                else:
                    ticklabels = np.round(
                        np.linspace(
                            (180 / np.pi) * vmin,
                            (180 / np.pi) * vmax,
                            ticknumber,
                            endpoint=True,
                        ),
                        decimals=2,
                    ).astype(str)

                if tickside in ("left", "right"):
                    cb = plt.colorbar(
                        cax, cax=cbax, ticks=ticks, orientation="vertical"
                    )
                    cb.ax.set_yticklabels(ticklabels, size=ticklabelsize)
                    cbax.yaxis.set_ticks_position(tickside)
                    cbax.set_ylabel(tickunits, size=unitlabelsize, rotation=0)
                    cbax.yaxis.set_label_position(tickside)
                else:
                    cb = plt.colorbar(
                        cax, cax=cbax, ticks=ticks, orientation="horizontal"
                    )
                    cb.ax.set_xticklabels(ticklabels, size=ticklabelsize)
                    cbax.xaxis.set_ticks_position(tickside)
                    cbax.set_xlabel(tickunits, size=unitlabelsize, rotation=0)
                    cbax.xaxis.set_label_position(tickside)
            else:
                cbax.axis("off")

    # Add coordinate axes
    if show_axes:
        assert xaxis_space in ("R", "Q"), "xaxis_space must be 'R' or 'Q'"
        show_which_axes = np.array(
            [
                "exx" in axes_plots,
                "eyy" in axes_plots,
                "exy" in axes_plots,
                "theta" in axes_plots,
            ]
        )
        for _show, _ax in zip(show_which_axes, (ax11, ax12, ax21, ax22)):
            if _show:
                if xaxis_space == "R":
                    ax_addaxes(
                        _ax,
                        xaxis_x,
                        xaxis_y,
                        axes_length,
                        axes_x0,
                        axes_y0,
                        width=axes_width,
                        color=axes_color,
                        labelaxes=labelaxes,
                        labelsize=axes_labelsize,
                        labelcolor=axes_labelcolor,
                    )
                else:
                    ax_addaxes_QtoR(
                        _ax,
                        xaxis_x,
                        xaxis_y,
                        axes_length,
                        axes_x0,
                        axes_y0,
                        QR_rotation,
                        width=axes_width,
                        color=axes_color,
                        labelaxes=labelaxes,
                        labelsize=axes_labelsize,
                        labelcolor=axes_labelcolor,
                    )

    # Add borders
    if bordercolor is not None:
        for ax in (ax11, ax12, ax21, ax22):
            for s in ["bottom", "top", "left", "right"]:
                ax.spines[s].set_color(bordercolor)
                ax.spines[s].set_linewidth(borderwidth)
            ax.set_xticks([])
            ax.set_yticks([])

    if not returnfig:
        plt.show()
        return
    else:
        axs = ((ax11, ax12), (ax21, ax22))
        return fig, axs


def show_pointlabels(
    ar, x, y, color="lightblue", size=20, alpha=1, returnfig=False, **kwargs
):
    """
    Show enumerated index labels for a set of points
    """
    fig, ax = show(ar, returnfig=True, **kwargs)
    d = {"x": x, "y": y, "size": size, "color": color, "alpha": alpha}
    add_pointlabels(ax, d)

    if returnfig:
        return fig, ax
    else:
        plt.show()
        return


def select_point(
    ar,
    x,
    y,
    i,
    color="lightblue",
    color_selected="r",
    size=20,
    returnfig=False,
    **kwargs
):
    """
    Show enumerated index labels for a set of points, with one selected point highlighted
    """
    fig, ax = show(ar, returnfig=True, **kwargs)
    d1 = {"x": x, "y": y, "size": size, "color": color}
    d2 = {
        "x": x[i],
        "y": y[i],
        "size": size,
        "color": color_selected,
        "fontweight": "bold",
    }
    add_pointlabels(ax, d1)
    add_pointlabels(ax, d2)

    if returnfig:
        return fig, ax
    else:
        plt.show()
        return


def show_max_peak_spacing(
    ar, spacing, braggdirections, color="g", lw=2, returnfig=False, **kwargs
):
    """Show a circle of radius `spacing` about each Bragg direction"""
    centers = [
        (braggdirections.data["qx"][i], braggdirections.data["qy"][i])
        for i in range(braggdirections.length)
    ]
    fig, ax = show(
        ar,
        circle={
            "center": centers,
            "R": spacing,
            "color": color,
            "fill": False,
            "lw": lw,
        },
        returnfig=True,
        **kwargs,
    )
    if returnfig:
        return fig, ax
    else:
        plt.show()
        return


def show_origin_meas(data):
    """
    Show the measured positions of the origin.

    Args:
        data (DataCube or Calibration or 2-tuple of arrays (qx0,qy0))
    """
    from py4DSTEM.data import Calibration
    from py4DSTEM.datacube import DataCube

    if isinstance(data, tuple):
        assert len(data) == 2
        qx, qy = data
    elif isinstance(data, DataCube):
        qx, qy = data.calibration.get_origin_meas()
    elif isinstance(data, Calibration):
        qx, qy = data.get_origin_meas()
    else:
        raise Exception("data must be of type Datacube or Calibration or tuple")

    show_image_grid(get_ar=lambda i: [qx, qy][i], H=1, W=2, cmap="RdBu")


def show_origin_fit(data):
    """
    Show the measured, fit, and residuals of the origin positions.

    Args:
        data (DataCube or Calibration or (3,2)-tuple of arrays
            ((qx0_meas,qy0_meas),(qx0_fit,qy0_fit),(qx0_residuals,qy0_residuals))
    """
    from py4DSTEM.data import Calibration
    from py4DSTEM.datacube import DataCube

    if isinstance(data, tuple):
        assert len(data) == 3
        qx0_meas, qy_meas = data[0]
        qx0_fit, qy0_fit = data[1]
        qx0_residuals, qy0_residuals = data[2]
    elif isinstance(data, DataCube):
        qx0_meas, qy0_meas = data.calibration.get_origin_meas()
        qx0_fit, qy0_fit = data.calibration.get_origin()
        qx0_residuals, qy0_residuals = data.calibration.get_origin_residuals()
    elif isinstance(data, Calibration):
        qx0_meas, qy0_meas = data.get_origin_meas()
        qx0_fit, qy0_fit = data.get_origin()
        qx0_residuals, qy0_residuals = data.get_origin_residuals()
    else:
        raise Exception("data must be of type Datacube or Calibration or tuple")

    show_image_grid(
        get_ar=lambda i: [
            qx0_meas,
            qx0_fit,
            qx0_residuals,
            qy0_meas,
            qy0_fit,
            qy0_residuals,
        ][i],
        H=2,
        W=3,
        cmap="RdBu",
    )


def show_selected_dps(
    datacube,
    positions,
    im,
    bragg_pos=None,
    colors=None,
    HW=None,
    figsize_im=(6, 6),
    figsize_dp=(4, 4),
    **kwargs
):
    """
    Shows two plots: first, a real space image overlaid with colored dots
    at the specified positions; second, a grid of diffraction patterns
    corresponding to these scan positions.

    Args:
        datacube (DataCube):
        positions (len N list or tuple of 2-tuples): the scan positions
        im (2d array): a real space image
        bragg_pos (len N list of pointlistarrays): bragg disk positions
            for each position. if passed, overlays the disk positions,
            and supresses plot of the real space image
        colors (len N list of colors or None):
        HW (2-tuple of ints): diffraction pattern grid shape
        figsize_im (2-tuple): size of the image figure
        figsize_dp (2-tuple): size of each diffraction pattern panel
        **kwargs (dict): arguments passed to visualize.show for the
            *diffraction patterns*. Default is `scaling='log'`
    """
    from py4DSTEM.datacube import DataCube

    assert isinstance(datacube, DataCube)
    N = len(positions)
    assert all(
        [len(x) == 2 for x in positions]
    ), "Improperly formated argument `positions`"
    if bragg_pos is not None:
        show_disk_pos = True
        assert len(bragg_pos) == N
    else:
        show_disk_pos = False
    if colors is None:
        from matplotlib.cm import gist_ncar

        linsp = np.linspace(0, 1, N, endpoint=False)
        colors = [gist_ncar(i) for i in linsp]
    assert len(colors) == N, "Number of positions and colors don't match"
    from matplotlib.colors import is_color_like

    assert [is_color_like(i) for i in colors]
    if HW is None:
        W = int(np.ceil(np.sqrt(N)))
        if W < 3:
            W = 3
        H = int(np.ceil(N / W))
    else:
        H, W = HW
    assert all([isinstance(x, (int, np.integer)) for x in (H, W)])

    x = [i[0] for i in positions]
    y = [i[1] for i in positions]
    if "scaling" not in kwargs.keys():
        kwargs["scaling"] = "log"
    if not show_disk_pos:
        fig, ax = show(im, figsize=figsize_im, returnfig=True)
        add_points(ax, d={"x": x, "y": y, "pointcolor": colors})
        show_image_grid(
            get_ar=lambda i: datacube.data[x[i], y[i], :, :],
            H=H,
            W=W,
            get_bordercolor=lambda i: colors[i],
            axsize=figsize_dp,
            **kwargs,
        )
    else:
        show_image_grid(
            get_ar=lambda i: datacube.data[x[i], y[i], :, :],
            H=H,
            W=W,
            get_bordercolor=lambda i: colors[i],
            axsize=figsize_dp,
            get_x=lambda i: bragg_pos[i].data["qx"],
            get_y=lambda i: bragg_pos[i].data["qy"],
            get_pointcolors=lambda i: colors[i],
            **kwargs,
        )


def Complex2RGB(complex_data, vmin=None, vmax=None, hue_start=0, invert=False):
    """
    complex_data (array): complex array to plot
    vmin (float)        : minimum absolute value
    vmax (float)        : maximum absolute value
    hue_start (float)   : rotational offset for colormap (degrees)
    inverse (bool)      : if True, uses light color scheme
    """
    amp = np.abs(complex_data)
    if np.isclose(np.max(amp), np.min(amp)):
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.max(amp)
    else:
        if vmin is None:
            vmin = 0.02
        if vmax is None:
            vmax = 0.98
        vals = np.sort(amp[~np.isnan(amp)])
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin = vals[ind_vmin]
        vmax = vals[ind_vmax]

    amp = np.where(amp < vmin, vmin, amp)
    amp = np.where(amp > vmax, vmax, amp)

    phase = np.angle(complex_data) + np.deg2rad(hue_start)
    amp /= np.max(amp)
    rgb = np.zeros(phase.shape + (3,))
    rgb[..., 0] = 0.5 * (np.sin(phase) + 1) * amp
    rgb[..., 1] = 0.5 * (np.sin(phase + np.pi / 2) + 1) * amp
    rgb[..., 2] = 0.5 * (-np.sin(phase) + 1) * amp

    return 1 - rgb if invert else rgb


def add_colorbar_arg(cax, vmin=None, vmax=None, hue_start=0, invert=False):
    """
    cax                 : axis to add cbar too
    vmin (float)        : minimum absolute value
    vmax (float)        : maximum absolute value
    hue_start (float)   : rotational offset for colormap (degrees)
    inverse (bool)      : if True, uses light color scheme
    """
    z = np.exp(1j * np.linspace(-np.pi, np.pi, 200))
    rgb_vals = Complex2RGB(z, vmin=vmin, vmax=vmax, hue_start=hue_start, invert=invert)
    newcmp = mcolors.ListedColormap(rgb_vals)
    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    cb1.set_label("arg", rotation=0, ha="center", va="bottom")
    cb1.ax.yaxis.set_label_coords(0.5, 1.01)
    cb1.set_ticks(np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]))
    cb1.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )


def show_complex(
    ar_complex,
    vmin=None,
    vmax=None,
    cbar=True,
    scalebar=False,
    pixelunits="pixels",
    pixelsize=1,
    returnfig=False,
    hue_start=0,
    invert=False,
    **kwargs
):
    """
    Function to plot complex arrays

    Args:
        ar_complex (2D array)       : complex array to be plotted. If ar_complex is list of complex arrarys
            such as [array1, array2], then arrays are horizonally plotted in one figure
        vmin (float, optional)      : minimum absolute value
        vmax (float, optional)      : maximum absolute value
            if None, vmin/vmax are set to fractions of the distribution of pixel values in the array,
            e.g. vmin=0.02 will set the minumum display value to saturate the lower 2% of pixels
        cbar (bool, optional)       : if True, include color wheel
        scalebar (bool, optional)   : if True, adds scale bar
        pixelunits (str, optional)  : units for scalebar
        pixelsize (float, optional) : size of one pixel in pixelunits for scalebar
        returnfig (bool, optional)  : if True, the function returns the tuple (figure,axis)
        hue_start (float, optional) : rotational offset for colormap (degrees)
        inverse (bool)              : if True, uses light color scheme

    Returns:
        if returnfig==False (default), the figure is plotted and nothing is returned.
        if returnfig==True, return the figure and the axis.
    """
    # convert to complex colors
    ar_complex = (
        ar_complex[0]
        if (isinstance(ar_complex, list) and len(ar_complex) == 1)
        else ar_complex
    )
    if isinstance(ar_complex, list):
        if isinstance(ar_complex[0], list):
            rgb = [
                Complex2RGB(ar, vmin, vmax, hue_start=hue_start, invert=invert)
                for sublist in ar_complex
                for ar in sublist
            ]
            H = len(ar_complex)
            W = len(ar_complex[0])

        else:
            rgb = [
                Complex2RGB(ar, vmin, vmax, hue_start=hue_start, invert=invert)
                for ar in ar_complex
            ]
            if len(rgb[0].shape) == 4:
                H = len(ar_complex)
                W = rgb[0].shape[0]
            else:
                H = 1
                W = len(ar_complex)
        is_grid = True
    else:
        rgb = Complex2RGB(ar_complex, vmin, vmax, hue_start=hue_start, invert=invert)
        if len(rgb.shape) == 4:
            is_grid = True
            H = 1
            W = rgb.shape[0]
        elif len(rgb.shape) == 5:
            is_grid = True
            H = rgb.shape[0]
            W = rgb.shape[1]
            rgb = rgb.reshape((-1,) + rgb.shape[-3:])
        else:
            is_grid = False
    # plot
    if is_grid:
        from py4DSTEM.visualize import show_image_grid

        fig, ax = show_image_grid(
            get_ar=lambda i: rgb[i],
            H=H,
            W=W,
            vmin=0,
            vmax=1,
            intensity_range="absolute",
            returnfig=True,
            **kwargs,
        )
        if scalebar is True:
            scalebar = {
                "Nx": ar_complex[0].shape[0],
                "Ny": ar_complex[0].shape[1],
                "pixelsize": pixelsize,
                "pixelunits": pixelunits,
            }

            add_scalebar(ax[0, 0], scalebar)
    else:
        fig, ax = show(
            rgb, vmin=0, vmax=1, intensity_range="absolute", returnfig=True, **kwargs
        )

        if scalebar is True:
            scalebar = {
                "Nx": ar_complex.shape[0],
                "Ny": ar_complex.shape[1],
                "pixelsize": pixelsize,
                "pixelunits": pixelunits,
            }

            add_scalebar(ax, scalebar)

    # add color bar
    if cbar == True:
        ax0 = fig.add_axes([1, 0.35, 0.3, 0.3])

        # create wheel
        AA = 1000
        kx = np.fft.fftshift(np.fft.fftfreq(AA))
        ky = np.fft.fftshift(np.fft.fftfreq(AA))
        kya, kxa = np.meshgrid(ky, kx)
        kra = (kya**2 + kxa**2) ** 0.5
        ktheta = np.arctan2(-kxa, kya)
        ktheta = kra * np.exp(1j * ktheta)

        # convert to hsv
        rgb = Complex2RGB(ktheta, 0, 0.4, hue_start=hue_start, invert=invert)
        ind = kra > 0.4
        rgb[ind] = [1, 1, 1]

        # plot
        ax0.imshow(rgb)

        # add axes
        ax0.axhline(AA / 2, 0, AA, color="k")
        ax0.axvline(AA / 2, 0, AA, color="k")
        ax0.axis("off")

        label_size = 16

        ax0.text(AA, AA / 2, 1, fontsize=label_size)
        ax0.text(AA / 2, 0, "i", fontsize=label_size)
        ax0.text(AA / 2, AA, "-i", fontsize=label_size)
        ax0.text(0, AA / 2, -1, fontsize=label_size)

    if returnfig == True:
        return fig, ax
