from py4DSTEM import DataCube, RealSlice
from emdfile import tqdmnd
from py4DSTEM.process.wholepatternfit.wp_models import (
    WPFModelPrototype,
    _BaseModel,
    WPFModelType,
    Parameter,
)

from typing import Optional
import numpy as np

from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
from matplotlib.gridspec import GridSpec
import warnings

__all__ = ["WholePatternFit"]


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
        # Global scaling parameter
        self.intensity_scale = 1 / np.mean(self.meanCBED)

        self.mask = mask if mask is not None else np.ones_like(self.meanCBED)

        if hasattr(x0, "__iter__") and hasattr(y0, "__iter__"):
            x0 = np.array(x0)
            y0 = np.array(y0)
            if x0.size == 2:
                global_xy0_lb = np.array([x0[0] - x0[1], y0[0] - y0[1]])
                global_xy0_ub = np.array([x0[0] + x0[1], y0[0] + y0[1]])
            elif x0.size == 3:
                global_xy0_lb = np.array([x0[1], y0[1]])
                global_xy0_ub = np.array([x0[2], y0[2]])
            else:
                global_xy0_lb = np.array([0.0, 0.0])
                global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])
            x0 = x0[0]
            y0 = y0[0]

        else:
            global_xy0_lb = np.array([0.0, 0.0])
            global_xy0_ub = np.array([datacube.Q_Nx, datacube.Q_Ny])

        # The WPF object holds a special Model that manages the shareable center coordinates
        self.coordinate_model = _BaseModel(
            x0=(x0, global_xy0_lb[0], global_xy0_ub[0]),
            y0=(y0, global_xy0_lb[1], global_xy0_ub[1]),
        )
        # TODO: remove special cases for global/local center in the Models
        # Needs an efficient way to handle calculation of q_r

        self.model = [
            self.coordinate_model,
        ]

        self.nParams = 0
        self.use_jacobian = use_jacobian

        # set up the global arguments
        self._setup_static_data(x0, y0)

        self.fit_power = fit_power

        # for debugging: tracks all function evals
        self._track = False
        self._fevals = []
        self._xevals = []
        # self._cost_history = []

    def add_model(self, model: WPFModelPrototype):
        self.model.append(model)

        self.nParams += len(model.params.keys())

        self._finalize_model()

    def add_model_list(self, model_list):
        for m in model_list:
            self.add_model(m)

    def link_parameters(self, parent_model, child_model, parameters):
        """
        Link parameters of separate models together. The parameters of
        the child_model are replaced with the parameters of the parent_model.
        Note, this does not add the models to the WPF object, that must
        be performed separately.
        """
        # Make sure child_model and parameters are iterable
        child_model = (
            [
                child_model,
            ]
            if not hasattr(child_model, "__iter__")
            else child_model
        )

        parameters = (
            [
                parameters,
            ]
            if not hasattr(parameters, "__iter__")
            else parameters
        )

        for child in child_model:
            for par in parameters:
                child.params[par] = parent_model.params[par]

    def generate_initial_pattern(self):
        # update parameters:
        self._finalize_model()
        return self._pattern(self.x0, self.static_data.copy()) / self.intensity_scale

    def fit_to_mean_CBED(self, **fit_opts):
        # first make sure we have the latest parameters
        self._finalize_model()

        # set the current active pattern to the mean CBED:
        current_pattern = self.meanCBED * self.intensity_scale
        shared_data = self.static_data.copy()

        self._fevals = []
        self._xevals = []
        self._cost_history = []

        default_opts = {
            "method": "trf",
            "verbose": 1,
            "x_scale": "jac",
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

        DP = self._pattern(self.mean_CBED_fit.x, shared_data) / self.intensity_scale
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

    def fit_all_patterns(
        self,
        resume=False,
        real_space_mask=None,
        show_fit_metrics=True,
        distributed=True,
        num_jobs=None,
        **fit_opts,
    ):
        """
        Apply model fitting to all patterns.

        Parameters
        ----------
        resume: bool (optional)
            Set to true to continue a previous fit with more iterations.
        real_space_mask: np.array() of bools (optional)
            Only perform the fitting on a subset of the probe positions,
            where real_space_mask[rx,ry] == True.
        distributed: bool (optional)
            Whether to evaluate using a pool of worker threads
        num_jobs: int (optional)
            Set to an integer value giving the number of jobs to parallelize over probe positions.
        fit_opts: args (optional)
            args passed to scipy.optimize.least_squares

        Returns
        --------
        fit_data: RealSlice
            Fitted coefficients for all probe positions
        fit_metrics: RealSlice
            Fitting metrixs for all probe positions

        """

        # make sure we have the latest parameters
        self._finalize_model()

        # set tracking off
        self._track = False
        self._fevals = []

        if resume:
            assert hasattr(self, "fit_data"), "No existing data resuming fit!"

        # init
        fit_data = np.zeros((self.x0.shape[0], self.datacube.R_Nx, self.datacube.R_Ny))
        fit_metrics = np.zeros((4, self.datacube.R_Nx, self.datacube.R_Ny))

        # Default fitting options
        default_opts = {
            "method": "trf",
            "verbose": 0,
            "x_scale": "jac",
        }
        default_opts.update(fit_opts)

        # Loop over probe positions
        if not distributed:
            for rx, ry in tqdmnd(self.datacube.R_Nx, self.datacube.R_Ny):
                if real_space_mask is not None and real_space_mask[rx, ry] == True:
                    current_pattern = (
                        self.datacube.data[rx, ry, :, :].copy() * self.intensity_scale
                    )
                    shared_data = self.static_data.copy()
                    x0 = self.fit_data.data[rx, ry].copy() if resume else self.x0.copy()

                    try:
                        if self.hasJacobian & self.use_jacobian:
                            opt = least_squares(
                                self._pattern_error,
                                x0,
                                jac=self._jacobian,
                                bounds=(self.lower_bound, self.upper_bound),
                                args=(current_pattern, shared_data),
                                **default_opts,
                                # **fit_opts,
                            )
                        else:
                            opt = least_squares(
                                self._pattern_error,
                                x0,
                                bounds=(self.lower_bound, self.upper_bound),
                                args=(current_pattern, shared_data),
                                **default_opts,
                                # **fit_opts,
                            )

                        fit_data_single = opt.x
                        fit_metrics_single = [
                            opt.cost,
                            opt.optimality,
                            opt.nfev,
                            opt.status,
                        ]
                    except:
                        fit_data_single = x0
                        fit_metrics_single = [0, 0, 0, 0]

                    fit_data[:, rx, ry] = fit_data_single
                    fit_metrics[:, rx, ry] = fit_metrics_single

        else:
            # distributed evaluation
            self._fit_distributed(
                resume=resume,
                num_jobs=num_jobs,
                fit_opts=default_opts,
                fit_data=fit_data,
                fit_metrics=fit_metrics,
            )

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

        # TODO: this produces duplicate entries for linked params- is this good or not?
        param_names = [
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

        if show_fit_metrics:
            self.show_fit_metrics()

        return self.fit_data, self.fit_metrics

    def accept_mean_CBED_fit(self):
        x = self.mean_CBED_fit.x

        for model in self.model:
            for param in model.params.values():
                param.initial_value = x[param.offset]

    def get_lattice_maps(self):
        assert hasattr(self, "fit_data"), "Please run fitting first!"

        lattices = [
            (i, m)
            for i, m in enumerate(self.model)
            if WPFModelType.LATTICE in m.model_type
        ]

        g_maps = []
        for i, l in lattices:
            # TODO: use parameter object offsets to get indices
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

    def _setup_static_data(self, x0, y0):
        self.static_data = {}

        xArray, yArray = np.mgrid[0 : self.datacube.Q_Nx, 0 : self.datacube.Q_Ny]
        self.static_data["xArray"] = xArray
        self.static_data["yArray"] = yArray

        self.static_data["Q_Nx"] = self.datacube.Q_Nx
        self.static_data["Q_Ny"] = self.datacube.Q_Ny

        self.static_data["parent"] = self

    def _get_distance(self, params: np.ndarray, x: Parameter, y: Parameter):
        """
        Return the distance from a point in pixel coordinates specified
        by two Parameter objects.
        This method caches the result from the _BaseModel for performance
        """
        if (
            x is self.model[0].params["x center"]
            and y is self.model[0].params["y center"]
        ):
            # TODO: actually implement caching
            pass

        return np.hypot(
            self.static_data["xArray"] - params[x.offset],
            self.static_data["yArray"] - params[y.offset],
        )

    def _pattern_error(self, x, current_pattern, shared_data):
        DP = self._pattern(x, shared_data)

        DP = (DP - current_pattern**self.fit_power) * self.mask

        if self._track:
            self._fevals.append(DP)
            self._xevals.append(x)
        self._cost_history.append(np.sum(DP**2))

        return DP.ravel()

    def _pattern(self, x, shared_data):
        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        for m in self.model:
            m.func(DP, x, **shared_data)

        return (DP**self.fit_power) * self.mask

    def _jacobian(self, x, current_pattern, shared_data):
        # TODO: automatic mixed analytic/finite difference

        J = np.zeros(((self.datacube.Q_Nx * self.datacube.Q_Ny), self.nParams))

        for m in self.model:
            m.jacobian(J, x, **shared_data)

        return J * self.mask.ravel()[:, np.newaxis]

    def _finalize_model(self):
        # iterate over all models and assign indices, accumulate list
        # of unique parameters. then, accumulate initial value and bounds vectors
        unique_params = []
        idx = 0
        for model in self.model:
            for param in model.params.values():
                if param not in unique_params:
                    unique_params.append(param)
                    param.offset = idx
                    idx += 1

        self.x0 = np.array([param.initial_value for param in unique_params])
        self.upper_bound = np.array([param.upper_bound for param in unique_params])
        self.lower_bound = np.array([param.lower_bound for param in unique_params])

        self.hasJacobian = all([m.hasJacobian for m in self.model])

        self.nParams = self.x0.shape[0]

    def _fit_single_pattern(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        fit_opts,
    ):
        """
        Apply model fitting to one pattern.

        Parameters
        ----------
        data: np.ndarray
            Diffraction pattern
        initial_guess: np.ndarray
            starting guess for fitting
        fit_opts:
            args passed to scipy.optimize.least_squares

        Returns
        --------
        fit_coefs: np.array
            Fitted coefficients
        fit_metrics: np.array
            Fitting metrics

        """

        try:
            if self.hasJacobian & self.use_jacobian:
                opt = least_squares(
                    self._pattern_error,
                    initial_guess,
                    jac=self._jacobian,
                    bounds=(self.lower_bound, self.upper_bound),
                    args=(data, self.static_data),
                    **fit_opts,
                )
            else:
                opt = least_squares(
                    self._pattern_error,
                    initial_guess,
                    bounds=(self.lower_bound, self.upper_bound),
                    args=(data, self.static_data),
                    **fit_opts,
                )

            fit_coefs = opt.x
            fit_metrics_single = [
                opt.cost,
                opt.optimality,
                opt.nfev,
                opt.status,
            ]
        except Exception as err:
            print(err)
            fit_coefs = initial_guess
            fit_metrics_single = [0, 0, 0, 0]

        return fit_coefs, fit_metrics_single

    def _fit_distributed(
        self,
        fit_opts: dict,
        fit_data:np.ndarray,
        fit_metrics:np.ndarray,
        resume=False,
        num_jobs=None,
    ):
        from mpire import WorkerPool
        from threadpoolctl import threadpool_limits

        def f(shared_data, args):
            with threadpool_limits(limits=1):
                return self._fit_single_pattern(**args, fit_opts=shared_data)

        fit_inputs = [
            (
                {
                    "data": self.datacube[rx, ry],
                    "initial_guess": self.fit_data[rx, ry] if resume else self.x0,
                },
            )
            for rx in range(self.datacube.R_Nx)
            for ry in range(self.datacube.R_Ny)
        ]

        with WorkerPool(
            n_jobs=num_jobs,
            shared_objects=fit_opts,
        ) as pool:
            results = pool.map(
                f,
                fit_inputs,
                progress_bar=True,
            )

        for (rx,ry), res in zip(np.ndindex((self.datacube.R_Nx, self.datacube.R_Ny)), results):
            fit_data[:,rx,ry] = res[0]
            fit_metrics[:,rx,ry] = res[1]


    def __getstate__(self):
        # Prevent pickling from copying the datacube, so that distributed
        # evaluation does not balloon memory usage.
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["datacube"]
        return state
