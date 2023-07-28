from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
from matplotlib.gridspec import GridSpec

def show_model_grid(self, x=None, **plot_kwargs):
    if x is None:
        x = self.mean_CBED_fit.x

    shared_data = self.static_data.copy()
    shared_data["global_x0"] = x[0]
    shared_data["global_y0"] = x[1]
    shared_data["global_r"] = np.hypot(
        (shared_data["xArray"] - x[0]),
        (shared_data["yArray"] - x[1]),
    )

    shared_data["global_x0"] = x[0]
    shared_data["global_y0"] = x[1]
    shared_data["global_r"] = np.hypot(
        (shared_data["xArray"] - x[0]),
        (shared_data["yArray"] - x[1]),
    )

    N = len(self.model)
    cols = int(np.ceil(np.sqrt(N)))
    rows = (N + 1) // cols

    kwargs = dict(constrained_layout=True)
    kwargs.update(plot_kwargs)
    fig, ax = plt.subplots(rows, cols, **kwargs)

    for i, (a, m) in enumerate(zip(ax.flat, self.model)):
        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))
        ind = self.model_param_inds[i] + 2
        m.func(DP, *x[ind : ind + m.nParams].tolist(), **shared_data)

        a.matshow(DP, cmap="turbo")

        # Determine if text color should be white or black
        int_range = np.array((np.min(DP), np.max(DP)))
        if int_range[0] != int_range[1]:
            r = (np.mean(DP[:DP.shape[0]//10,:]) - int_range[0]) / (int_range[1] - int_range[0])
            if r < 0.5:
                color = 'w'
            else:
                color = 'k'
        else:
            color = 'w'

        a.text(
            0.5, 
            0.92, 
            m.name, 
            transform = a.transAxes, 
            ha = "center", 
            va = "center",
            color = color)
    for a in ax.flat:
        a.axis("off")

    plt.show()

def show_lattice_points(
    self, 
    im = None,
    vmin = None,
    vmax = None,
    power = None,
    returnfig=False, 
    *args, 
    **kwargs
    ):
    """
    Plotting utility to show the initial lattice points.
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
            vmin = vmin,
            vmax = vmax, 
            cmap="gray",
            )

    for m in self.model:
        if "Lattice" in m.name:
            ux, uy = m.params["ux"].initial_value, m.params["uy"].initial_value
            vx, vy = m.params["vx"].initial_value, m.params["vy"].initial_value

            lat = np.array([[ux, uy], [vx, vy]])
            inds = np.stack([m.u_inds, m.v_inds], axis=1)

            spots = inds @ lat
            spots[:, 0] += self.static_data["global_x0"]
            spots[:, 1] += self.static_data["global_y0"]

            ax.scatter(
                spots[:, 1], 
                spots[:, 0], 
                s = 100,
                marker="x", 
                label=m.name,
                )

    ax.legend()

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
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
            (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
            (1.0, 0.4980392156862745, 0.0),
            (1.0, 1.0, 0.2),
        )
    )
    im = ax[0, 1].matshow(
        self.fit_metrics["status"].data, cmap=opt_cmap, vmin=-1.5, vmax=4.5
    )
    cbar = fig.colorbar(im, ax=ax[0, 1], ticks=[-1, 0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(
        [
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