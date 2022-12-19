from py4DSTEM.io import DataCube, RealSlice
from py4DSTEM import tqdmnd
from py4DSTEM.process.wholepatternfit.wp_models import WPFModelPrototype

from typing import Optional
import numpy as np

from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
from matplotlib.gridspec import GridSpec


class WholePatternFit:

    from py4DSTEM.process.wholepatternfit.wpf_viz import (
        show_model_grid,
        show_lattice_points,
        show_fit_metrics,
    )

    def __init__(
        self,
        datacube: DataCube,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
        use_jacobian: bool = True,
        meanCBED: Optional[np.ndarray] = None,
        fit_power: float = 1,
    ):
        """
        Perform pixelwise fits using composable models and numerical optimization.

        Instantiate components of the fit model using the objects in wp_models,
        and add them to the WPF object using ``add_model``.
        All fitting parameters, including ``x0`` and ``y0``, can be specified as
        floats or, if the parameter should be bounded, as a tuple with the format:
            (initial guess, lower bound, upper bound)
        Then, refine the initial guess using ``fit_to_mean_CBED``. If the initial
        refinement is good, save it using ``accept_mean_CBED_fit``, which updates
        the initial guesses in each model object.

        Then, refine the model to each diffraction pattern in the dataset using
        ``fit_all_patterns``. The fit results are returned in RealSlice objects
        with slice labels corresponding to the names of each model and their
        parameters.

        To map strain, use ``get_lattice_maps`` to extract RealSice object with
        the refined g vectors at each point, and then use the ordinary py4DSTEM
        strain mapping pipeline

        Parameters
        ----------
        datacube : (DataCube)
        x0, y0 : Optional float or np.ndarray to specify the initial guess for the origin
            of diffraction space, in pixels
        mask : Optional np.ndarray to specify which pixels in the diffraction pattern
            should be used for computing the loss function. Pixels occluded by a beamstop
            or fixed detector should be set to False so they do not contribte to the loss
        use_jacobian: bool, whether or not to use the analytic Jacobians for each model
            in the optimizer. When False, finite differences is used for all gradient evaluations
        meanCBED: Optional np.ndarray, used to specify the diffraction pattern used
            for initial refinement of the parameters. If not specified, the average across
            all scan positions is computed
        fit_power: float, diffraction patterns are raised to this power, sets the gamma
            level at which the patterns are compared

        """
        self.datacube = datacube
        self.meanCBED = (
            meanCBED if meanCBED is not None else np.mean(datacube.data, axis=(0, 1))
        )

        self.mask = mask if mask is not None else np.ones_like(self.meanCBED)

        self.model = []
        self.model_param_inds = []

        self.nParams = 0
        self.use_jacobian = use_jacobian

        if hasattr(x0, "__iter__") and hasattr(y0, "__iter__"):
            # the initial position was specified with bounds
            try:
                self.global_xy0_lb = np.array([x0[1], y0[1]])
                self.global_xy0_ub = np.array([x0[2], y0[2]])
            except:
                self.global_xy0_lb = np.array([0.0, 0.0])
                self.global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])
            x0 = x0[0]
            y0 = y0[0]
        else:
            self.global_xy0_lb = np.array([0.0, 0.0])
            self.global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])

        # set up the global arguments
        self._setup_static_data()

        self.fit_power = fit_power

        # for debugging: tracks all function evals
        self._track = False
        self._fevals = []
        self._xevals = []
        # self._cost_history = []

    def add_model(self, model: WPFModelPrototype):
        self.model.append(model)

        # keep track of where each model's parameter list begins
        self.model_param_inds.append(self.nParams)
        self.nParams += len(model.params.keys())

        self._scrape_model_params()

    def add_model_list(self, model_list):
        for m in model_list:
            self.add_model(m)

    def generate_initial_pattern(self):

        # update parameters:
        self._scrape_model_params()
        return self._pattern(self.x0, self.static_data.copy())

    def fit_to_mean_CBED(self, **fit_opts):

        # first make sure we have the latest parameters
        self._scrape_model_params()

        # set the current active pattern to the mean CBED:
        current_pattern = self.meanCBED
        shared_data = self.static_data.copy()

        self._fevals = []
        self._xevals = []
        self._cost_history = []

        default_opts = {
            "method": "trf",
            "verbose": 1,
        }
        default_opts.update(fit_opts)

        if self.hasJacobian & self.use_jacobian:
            opt = least_squares(
                self._pattern_error,
                self.x0,
                jac=self._jacobian,
                bounds=(self.lower_bound, self.upper_bound),
                args=(current_pattern, shared_data),
                **default_opts,
            )
        else:
            opt = least_squares(
                self._pattern_error,
                self.x0,
                bounds=(self.lower_bound, self.upper_bound),
                args=(current_pattern, shared_data),
                **default_opts,
            )

        self.mean_CBED_fit = opt

        # Plotting
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        gs = GridSpec(2, 2, figure=fig)

        ax = fig.add_subplot(gs[0, 0])
        err_hist = np.array(self._cost_history)
        ax.plot(err_hist)
        ax.set_ylabel("Sum Squared Error")
        ax.set_xlabel("Iterations")
        ax.set_yscale("log")

        DP = self._pattern(self.mean_CBED_fit.x)
        ax = fig.add_subplot(gs[0, 1])
        CyRd = mpl_c.LinearSegmentedColormap.from_list(
            "CyRd", ["#00ccff", "#ffffff", "#ff0000"]
        )
        im = ax.matshow(
            err_im := -(DP - self.meanCBED),
            cmap=CyRd,
            vmin=-np.abs(err_im).max() / 4,
            vmax=np.abs(err_im).max() / 4,
        )
        # fig.colorbar(im)
        ax.set_title("Error")
        ax.axis("off")

        ax = fig.add_subplot(gs[1, :])
        ax.matshow(np.hstack((DP, self.meanCBED)) ** 0.25, cmap="turbo")
        ax.axis("off")
        ax.text(0.25, 0.92, "Refined", transform=ax.transAxes, ha="center", va="center")
        ax.text(
            0.75, 0.92, "Mean CBED", transform=ax.transAxes, ha="center", va="center"
        )

        plt.show()

        return opt

    def fit_all_patterns(self, restart=False, **fit_opts):

        # make sure we have the latest parameters
        self._scrape_model_params()

        # set tracking off
        self._track = False
        self._fevals = []

        if restart:
            assert hasattr(self, "fit_data"), "No existing data for restart!"

        fit_data = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny, self.x0.shape[0]))
        fit_metrics = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny, 4))

        for rx, ry in tqdmnd(self.datacube.R_Nx, self.datacube.R_Ny):
            current_pattern = self.datacube.data[rx, ry, :, :]
            shared_data = self.static_data.copy()
            self._cost_history = (
                []
            )  # clear this so it doesn't grow: TODO make this not stupid

            x0 = self.fit_data.data[rx, ry] if restart else self.x0

            if self.hasJacobian & self.use_jacobian:
                opt = least_squares(
                    self._pattern_error,
                    x0,
                    jac=self._jacobian,
                    bounds=(self.lower_bound, self.upper_bound),
                    args=(current_pattern, shared_data),
                    **fit_opts,
                )
            else:
                opt = least_squares(
                    self._pattern_error,
                    x0,
                    bounds=(self.lower_bound, self.upper_bound),
                    args=(current_pattern, shared_data),
                    **fit_opts,
                )

            fit_data[rx, ry, :] = opt.x
            fit_metrics[rx, ry, :] = [
                opt.cost,
                opt.optimality,
                opt.nfev,
                opt.status,
            ]

        # Convert to RealSlices
        model_names = []
        for m in self.model:
            n = m.name
            if n in model_names:
                i = 1
                while n in model_names:
                    n = m.name + "_" + str(i)
                    i += 1
            model_names.append(n)

        param_names = ["global_x0", "global_y0"] + [
            n + "/" + k
            for m, n in zip(self.model, model_names)
            for k in m.params.keys()
        ]

        self.fit_data = RealSlice(fit_data, name="Fit Data", slicelabels=param_names)
        self.fit_metrics = RealSlice(
            fit_metrics,
            name="Fit Metrics",
            slicelabels=["cost", "optimality", "nfev", "status"],
        )

        self.show_fit_metrics()

        return self.fit_data, self.fit_metrics

    def accept_mean_CBED_fit(self):
        x = self.mean_CBED_fit.x
        self.static_data["global_x0"] = x[0]
        self.static_data["global_y0"] = x[1]

        self.static_data["global_r"] = np.hypot(
            (self.static_data["xArray"] - x[0]), (self.static_data["yArray"] - x[1])
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            for j, k in enumerate(m.params.keys()):
                m.params[k].initial_value = x[ind + j]

    def get_lattice_maps(self):
        assert hasattr(self, "fit_data"), "Please run fitting first!"

        lattices = [
            (i, m)
            for i, m in enumerate(self.model)
            if "lattice" in type(m).__name__.lower()
        ]

        g_maps = []
        for (i, l) in lattices:
            param_list = list(l.params.keys())
            lattice_offset = param_list.index("ux")
            data_offset = self.model_param_inds[i] + 2 + lattice_offset

            # TODO: Use proper RealSlice semantics for access
            data = self.fit_data.data[:, :, data_offset : data_offset + 4]

            g_map = RealSlice(
                np.dstack((data, np.ones(data.shape[:2], dtype=np.bool_))),
                slicelabels=["g1x", "g1y", "g2x", "g2y", "mask"],
                name=l.name,
            )
            g_maps.append(g_map)

        return g_maps

    def _setup_static_data(self):
        self.static_data = {}

        xArray, yArray = np.mgrid[0 : self.datacube.Q_Nx, 0 : self.datacube.Q_Ny]
        self.static_data["xArray"] = xArray
        self.static_data["yArray"] = yArray

        self.static_data["Q_Nx"] = self.datacube.Q_Nx
        self.static_data["Q_Ny"] = self.datacube.Q_Ny

    def _pattern_error(self, x, current_pattern, shared_data):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        shared_data["global_x0"] = x[0]
        shared_data["global_y0"] = x[1]
        shared_data["global_r"] = np.hypot(
            (shared_data["xArray"] - x[0]),
            (shared_data["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **shared_data)

        DP = (DP**self.fit_power - current_pattern**self.fit_power) * self.mask

        if self._track:
            self._fevals.append(DP)
            self._xevals.append(x)
        self._cost_history.append(np.sum(DP**2))

        return DP.ravel()

    def _pattern(self, x, shared_data):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        shared_data["global_x0"] = x[0]
        shared_data["global_y0"] = x[1]
        shared_data["global_r"] = np.hypot(
            (shared_data["xArray"] - x[0]),
            (shared_data["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **shared_data)

        return (DP**self.fit_power) * self.mask

    def _jacobian(self, x, current_pattern, shared_data):
        # TODO: automatic mixed analytic/finite difference

        J = np.zeros(((self.datacube.Q_Nx * self.datacube.Q_Ny), self.nParams + 2))

        shared_data["global_x0"] = x[0]
        shared_data["global_y0"] = x[1]
        shared_data["global_r"] = np.hypot(
            (shared_data["xArray"] - x[0]),
            (shared_data["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.jacobian(J, *x[ind : ind + m.nParams].tolist(), offset=ind, **shared_data)

        return J * self.mask.ravel()[:, np.newaxis]

    def _scrape_model_params(self):

        self.x0 = np.zeros((self.nParams + 2,))
        self.upper_bound = np.zeros_like(self.x0)
        self.lower_bound = np.zeros_like(self.x0)

        self.x0[0:2] = np.array(
            [self.static_data["global_x0"], self.static_data["global_y0"]]
        )
        self.upper_bound[0:2] = self.global_xy0_ub
        self.lower_bound[0:2] = self.global_xy0_lb

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2

            for j, v in enumerate(m.params.values()):
                self.x0[ind + j] = v.initial_value
                self.upper_bound[ind + j] = v.upper_bound
                self.lower_bound[ind + j] = v.lower_bound

        self.hasJacobian = all([m.hasJacobian for m in self.model])
