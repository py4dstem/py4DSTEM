from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
from matplotlib.gridspec import GridSpec

from py4DSTEM.process.wholepatternfit.wp_models import WPFModelType


def show_model_grid(self, x=None, **plot_kwargs):
    x = self.mean_CBED_fit.x if x is None else x

    model = [m for m in self.model if WPFModelType.DUMMY not in m.model_type]

    N = len(model)
    cols = int(np.ceil(np.sqrt(N)))
    rows = (N + 1) // cols

    kwargs = dict(constrained_layout=True)
    kwargs.update(plot_kwargs)
    fig, ax = plt.subplots(rows, cols, **kwargs)

    for a, m in zip(ax.flat, model):
        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))
        m.func(DP, x, **self.static_data)

        a.matshow(DP, cmap="turbo")

        # Determine if text color should be white or black
        int_range = np.array((np.min(DP), np.max(DP)))
        if int_range[0] != int_range[1]:
            r = (np.mean(DP[: DP.shape[0] // 10, :]) - int_range[0]) / (
                int_range[1] - int_range[0]
            )
            if r < 0.5:
                color = "w"
            else:
                color = "k"
        else:
            color = "w"

        a.text(
            0.5,
            0.92,
            m.name,
            transform=a.transAxes,
            ha="center",
            va="center",
            color=color,
        )
    for a in ax.flat:
        a.axis("off")

    plt.show()


def show_lattice_points(
    self,
    im=None,
    vmin=None,
    vmax=None,
    power=None,
    show_vectors=True,
    crop_to_pattern=False,
    returnfig=False,
    moire_origin_idx=[0, 0, 0, 0],
    *args,
    **kwargs,
):
    """
    Plotting utility to show the initial lattice points.

    Parameters
    ----------
    im: np.ndarray
        Optional: Image to show, defaults to mean CBED
    vmin, vmax: float
        Intensity ranges for plotting im
    power: float
        Gamma level for showing im
    show_vectors: bool
        Flag to plot the lattice vectors
    crop_to_pattern: bool
        Flag to limit the field of view to the pattern area. If False,
        spots outside the pattern are shown
    returnfig: bool
        If True, (fig,ax) are returned and plt.show() is not called
    moire_origin_idx: list of length 4
        Indices of peak on which to draw Moire vectors, written as
        [a_u, a_v, b_u, b_v]
    args, kwargs
        Passed to plt.subplots

    Returns
    -------
    fig,ax: If returnfig=True
    """

    if im is None:
        im = self.meanCBED
    if power is None:
        power = 0.5

    fig, ax = plt.subplots(*args, **kwargs)
    if vmin is None and vmax is None:
        ax.matshow(
            im**power,
            cmap="gray",
        )
    else:
        ax.matshow(
            im**power,
            vmin=vmin,
            vmax=vmax,
            cmap="gray",
        )

    lattices = [m for m in self.model if WPFModelType.LATTICE in m.model_type]

    for m in lattices:
        ux, uy = m.params["ux"].initial_value, m.params["uy"].initial_value
        vx, vy = m.params["vx"].initial_value, m.params["vy"].initial_value

        lat = np.array([[ux, uy], [vx, vy]])
        inds = np.stack([m.u_inds, m.v_inds], axis=1)

        spots = inds @ lat
        spots[:, 0] += m.params["x center"].initial_value
        spots[:, 1] += m.params["y center"].initial_value

        axpts = ax.scatter(
            spots[:, 1],
            spots[:, 0],
            s=100,
            marker="x",
            label=m.name,
        )

        if show_vectors:
            ax.arrow(
                m.params["y center"].initial_value,
                m.params["x center"].initial_value,
                m.params["uy"].initial_value,
                m.params["ux"].initial_value,
                length_includes_head=True,
                color=axpts.get_facecolor(),
                width=1.0,
            )

            ax.arrow(
                m.params["y center"].initial_value,
                m.params["x center"].initial_value,
                m.params["vy"].initial_value,
                m.params["vx"].initial_value,
                length_includes_head=True,
                color=axpts.get_facecolor(),
                width=1.0,
            )

    moires = [m for m in self.model if WPFModelType.MOIRE in m.model_type]

    for m in moires:
        lat_ab = m._get_parent_lattices(m.lattice_a, m.lattice_b)
        lat_abm = np.vstack((lat_ab, m.moire_matrix @ lat_ab))

        spots = m.moire_indices_uvm @ lat_abm
        spots[:, 0] += m.params["x center"].initial_value
        spots[:, 1] += m.params["y center"].initial_value

        axpts = ax.scatter(
            spots[:, 1],
            spots[:, 0],
            s=100,
            marker="+",
            label=m.name,
        )

        if show_vectors:
            arrow_origin = np.array(moire_origin_idx) @ lat_ab
            arrow_origin[0] += m.params["x center"].initial_value
            arrow_origin[1] += m.params["y center"].initial_value

            ax.arrow(
                arrow_origin[1],
                arrow_origin[0],
                lat_abm[4, 1],
                lat_abm[4, 0],
                length_includes_head=True,
                color=axpts.get_facecolor(),
                width=1.0,
            )

            ax.arrow(
                arrow_origin[1],
                arrow_origin[0],
                lat_abm[5, 1],
                lat_abm[5, 0],
                length_includes_head=True,
                color=axpts.get_facecolor(),
                width=1.0,
            )

    ax.legend()

    if crop_to_pattern:
        ax.set_xlim(0, im.shape[1] - 1)
        ax.set_ylim(im.shape[0] - 1, 0)

    return (fig, ax) if returnfig else plt.show()


def show_fit_metrics(self, returnfig=False, **subplots_kwargs):
    assert hasattr(self, "fit_metrics"), "Please run fitting first!"

    kwargs = dict(figsize=(14, 12), constrained_layout=True)
    kwargs.update(subplots_kwargs)
    fig, ax = plt.subplots(2, 2, **kwargs)
    im = ax[0, 0].matshow(self.fit_metrics["cost"].data, norm=mpl_c.LogNorm())
    ax[0, 0].set_title("Final Cost Function")
    fig.colorbar(im, ax=ax[0, 0])

    opt_cmap = mpl_c.ListedColormap(
        (
            (0.6, 0.05, 0.05),
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
            (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
            (1.0, 0.4980392156862745, 0.0),
            (1.0, 1.0, 0.2),
        )
    )
    im = ax[0, 1].matshow(
        self.fit_metrics["status"].data, cmap=opt_cmap, vmin=-2.5, vmax=4.5
    )
    cbar = fig.colorbar(im, ax=ax[0, 1], ticks=[-2, -1, 0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(
        [
            "Unknown Error",
            "MINPACK Error",
            "Max f evals exceeded",
            "$gtol$ satisfied",
            "$ftol$ satisfied",
            "$xtol$ satisfied",
            "$xtol$ & $ftol$ satisfied",
        ]
    )
    ax[0, 1].set_title("Optimizer Status")
    fig.set_facecolor("w")

    im = ax[1, 0].matshow(self.fit_metrics["optimality"].data, norm=mpl_c.LogNorm())
    ax[1, 0].set_title("First Order Optimality")
    fig.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].matshow(self.fit_metrics["nfev"].data)
    ax[1, 1].set_title("Number f evals")
    fig.colorbar(im, ax=ax[1, 1])

    fig.set_facecolor("w")

    return (fig, ax) if returnfig else plt.show()
