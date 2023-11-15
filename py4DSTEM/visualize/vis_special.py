from matplotlib import cm, colors as mcolors, pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi

from emdfile import PointList
from py4DSTEM.visualize import show
from py4DSTEM.visualize.overlay import (
    add_pointlabels,
    add_vector,
    add_bragg_index_labels,
    add_ellipses,
    add_points,
    add_scalebar,
)
from py4DSTEM.visualize.vis_grid import show_image_grid
from py4DSTEM.visualize.vis_RQ import ax_addaxes, ax_addaxes_QtoR
from colorspacious import cspace_convert


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
    **kwargs,
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
    **kwargs,
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
        return_intensity_range=True,
        **kwargs,
    )
    show(
        fit,
        scaling=scaling,
        figax=(fig, ax),
        intensity_range="absolute",
        vmin=vmin,
        vmax=vmax,
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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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


def Complex2RGB(complex_data, vmin=None, vmax=None, power=None, chroma_boost=1):
    """
    complex_data (array): complex array to plot
    vmin (float)        : minimum absolute value
    vmax (float)        : maximum absolute value
    power (float)       : power to raise amplitude to
    chroma_boost (float): boosts chroma for higher-contrast (~1-2.5)
    """
    amp = np.abs(complex_data)
    phase = np.angle(complex_data)

    if power is not None:
        amp = amp**power

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
    amp = ((amp - vmin) / vmax).clip(1e-16, 1)

    J = amp * 61.5  # Note we restrict luminance to the monotonic chroma cutoff
    C = np.minimum(chroma_boost * 98 * J / 123, 110)
    h = np.rad2deg(phase) + 180

    JCh = np.stack((J, C, h), axis=-1)
    rgb = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)

    return rgb


def add_colorbar_arg(cax, chroma_boost=1, c=49, j=61.5):
    """
    cax                 : axis to add cbar to
    chroma_boost (float): boosts chroma for higher-contrast (~1-2.25)
    c (float)           : constant chroma value
    j (float)           : constant luminance value
    """

    h = np.linspace(0, 360, 256, endpoint=False)
    J = np.full_like(h, j)
    C = np.full_like(h, np.minimum(c * chroma_boost, 110))
    JCh = np.stack((J, C, h), axis=-1)
    rgb_vals = cspace_convert(JCh, "JCh", "sRGB1").clip(0, 1)
    newcmp = mcolors.ListedColormap(rgb_vals)
    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)

    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    cb.set_label("arg", rotation=0, ha="center", va="bottom")
    cb.ax.yaxis.set_label_coords(0.5, 1.01)
    cb.set_ticks(np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]))
    cb.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )


def show_complex(
    ar_complex,
    vmin=None,
    vmax=None,
    power=None,
    chroma_boost=1,
    cbar=True,
    scalebar=False,
    pixelunits="pixels",
    pixelsize=1,
    returnfig=False,
    **kwargs,
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
        power (float,optional)      : power to raise amplitude to
        chroma_boost (float)        : boosts chroma for higher-contrast (~1-2.25)
        cbar (bool, optional)       : if True, include color bar
        scalebar (bool, optional)   : if True, adds scale bar
        pixelunits (str, optional)  : units for scalebar
        pixelsize (float, optional) : size of one pixel in pixelunits for scalebar
        returnfig (bool, optional)  : if True, the function returns the tuple (figure,axis)

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
                Complex2RGB(ar, vmin, vmax, power=power, chroma_boost=chroma_boost)
                for sublist in ar_complex
                for ar in sublist
            ]
            H = len(ar_complex)
            W = len(ar_complex[0])

        else:
            rgb = [
                Complex2RGB(ar, vmin, vmax, power=power, chroma_boost=chroma_boost)
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
        rgb = Complex2RGB(
            ar_complex, vmin, vmax, power=power, chroma_boost=chroma_boost
        )
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
    if cbar:
        if is_grid:
            for ax_flat in ax.flatten():
                divider = make_axes_locatable(ax_flat)
                ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
                add_colorbar_arg(ax_cb, chroma_boost=chroma_boost)
        else:
            divider = make_axes_locatable(ax)
            ax_cb = divider.append_axes("right", size="5%", pad="2.5%")
            add_colorbar_arg(ax_cb, chroma_boost=chroma_boost)

        fig.tight_layout()

    if returnfig:
        return fig, ax


def return_scaled_histogram_ordering(array, vmin=None, vmax=None, normalize=False):
    """
    Utility function for calculating min and max values for plotting array
    based on distribution of pixel values

    Parameters
    ----------
    array: np.array
        array to be plotted
    vmin: float
        lower fraction cut off of pixel values
    vmax: float
        upper fraction cut off of pixel values
    normalize: bool
        if True, rescales from 0 to 1

    Returns
    ----------
    scaled_array: np.array
        array clipped outside vmin and vmax
    vmin: float
        lower value to be plotted
    vmax: float
        upper value to be plotted
    """

    if vmin is None:
        vmin = 0.02
    if vmax is None:
        vmax = 0.98

    vals = np.sort(array.ravel())
    ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
    ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
    ind_vmin = np.max([0, ind_vmin])
    ind_vmax = np.min([len(vals) - 1, ind_vmax])
    vmin = vals[ind_vmin]
    vmax = vals[ind_vmax]

    if vmax == vmin:
        vmin = vals[0]
        vmax = vals[-1]

    scaled_array = array.copy()
    scaled_array = np.where(scaled_array < vmin, vmin, scaled_array)
    scaled_array = np.where(scaled_array > vmax, vmax, scaled_array)

    if normalize:
        scaled_array -= scaled_array.min()
        scaled_array /= scaled_array.max()
        vmin = 0
        vmax = 1

    return scaled_array, vmin, vmax
