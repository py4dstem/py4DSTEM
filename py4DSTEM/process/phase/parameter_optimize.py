from functools import partial
from itertools import product
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from py4DSTEM.process.phase.phase_base_class import PhaseReconstruction
from py4DSTEM.process.phase.utils import AffineTransform
from skopt import gp_minimize
from skopt.plots import plot_convergence as skopt_plot_convergence
from skopt.plots import plot_evaluations as skopt_plot_evaluations
from skopt.plots import plot_gaussian_process as skopt_plot_gaussian_process
from skopt.plots import plot_objective as skopt_plot_objective
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from tqdm import tqdm


class PtychographyOptimizer:
    """
    Optimize ptychographic hyperparameters with Bayesian Optimization of a
    Gaussian process. Any of the scalar-valued real or integer,  boolean, or categorical
    arguments to the ptychographic init-preprocess-reconstruct pipeline can be optimized over.
    """

    def __init__(
        self,
        reconstruction_type: type[PhaseReconstruction],
        init_args: dict,
        preprocess_args: dict = {},
        reconstruction_args: dict = {},
        affine_args: dict = {},
    ):
        """
        Parameter optimization for ptychographic reconstruction based on Bayesian Optimization
        with Gaussian Process.

        Usage
        -----
        Dictionaries of the arguments to __init__, AffineTransform (for distorting the initial
        scan positions), preprocess, and reconstruct are required. For parameters not optimized
        over, the value in the dictionary is used. To optimize a parameter, instead pass an
        OptimizationParameter object inside the dictionary to specify the initial guess, bounds,
        and type of parameter, for example:
            >>> 'param':OptimizationParameter(initial guess, lower bound, upper bound)
        Calling optimize will then run the optimization simultaneously over all
        optimization parameters. To obtain the optimized parameters, call get_optimized_arguments
        to return a set of dictionaries where the OptimizationParameter objects have been replaced
        with the optimal values. These can then be modified for running a full reconstruction.

        Parameters
        ----------
        reconstruction_type: class
            Type of ptychographic reconstruction to perform
        init_args: dict
            Keyword arguments passed to the __init__ method of the reconstruction class
        preprocess_args: dict
            Keyword arguments passed to the preprocess method the reconstruction object
        reconstruction_args: dict
            Keyword arguments passed to the reconstruct method the the reconstruction object
        affine_args: dict
            Keyword arguments passed to AffineTransform. The transform is applied to the initial
            scan positions.
        """

        # loop over each argument dictionary and split into static and optimization variables
        (
            self._init_static_args,
            self._init_optimize_args,
        ) = self._split_static_and_optimization_vars(init_args)
        (
            self._affine_static_args,
            self._affine_optimize_args,
        ) = self._split_static_and_optimization_vars(affine_args)
        (
            self._preprocess_static_args,
            self._preprocess_optimize_args,
        ) = self._split_static_and_optimization_vars(preprocess_args)
        (
            self._reconstruction_static_args,
            self._reconstruction_optimize_args,
        ) = self._split_static_and_optimization_vars(reconstruction_args)

        # Save list of skopt parameter objects and inital guess
        self._parameter_list = []
        self._x0 = []
        for k, v in (
            self._init_optimize_args
            | self._affine_optimize_args
            | self._preprocess_optimize_args
            | self._reconstruction_optimize_args
        ).items():
            self._parameter_list.append(v._get(k))
            self._x0.append(v._initial_value)

        self._init_args = init_args
        self._affine_args = affine_args
        self._preprocess_args = preprocess_args
        self._reconstruction_args = reconstruction_args

        self._reconstruction_type = reconstruction_type

        self._set_optimizer_defaults()

    def _generate_inclusive_boundary_grid(
        self,
        parameter,
        n_points,
    ):
        """ """

        # Categorical
        if hasattr(parameter, "categories"):
            return np.array(parameter.categories)

        # Real or Integer
        else:
            return np.unique(
                np.linspace(parameter.low, parameter.high, n_points).astype(
                    parameter.dtype
                )
            )

    def grid_search(
        self,
        n_points: Union[tuple, int] = 3,
        error_metric: Union[Callable, str] = "log",
        plot_reconstructed_objects: bool = True,
        return_reconstructed_objects: bool = False,
        **kwargs: dict,
    ):
        """
        Run optimizer

        Parameters
        ----------
        n_initial_points: int
            Number of uniformly spaced trial points to run on a grid
        error_metric: Callable or str
            Function used to compute the reconstruction error.
            When passed as a string, may be one of:
                'log': log(NMSE) of final object
                'linear': NMSE of final object
                'log-converged': log(NMSE) of final object if
                    NMSE is decreasing, 0 if NMSE increasing
                'linear-converged': NMSE of final object if
                    NMSE is decreasing, 1 if NMSE increasing
                'TV': sum( abs( grad( object ) ) ) / sum( abs( object ) )
                'std': negative standard deviation of cropped object
                'std-phase': negative standard deviation of
                    phase of the cropped object
                'entropy-phase': entropy of the phase of the
                    cropped object
            When passed as a Callable, a function that takes the
                PhaseReconstruction object as its only argument
                and returns the error metric as a single float

        """

        num_params = len(self._parameter_list)

        if isinstance(n_points, int):
            n_points = [n_points] * num_params
        elif len(n_points) != num_params:
            raise ValueError()

        params_grid = [
            self._generate_inclusive_boundary_grid(param, n_pts)
            for param, n_pts in zip(self._parameter_list, n_points)
        ]
        params_grid = list(product(*params_grid))
        num_evals = len(params_grid)

        error_metric = self._get_error_metric(error_metric)
        pbar = tqdm(total=num_evals, desc="Searching parameters")

        def evaluation_callback(ptycho):
            if plot_reconstructed_objects or return_reconstructed_objects:
                pbar.update(1)
                return (
                    ptycho._return_projected_cropped_potential(),
                    error_metric(ptycho),
                )
            else:
                pbar.update(1)
                error_metric(ptycho)

        grid_search_function = self._get_optimization_function(
            self._reconstruction_type,
            self._parameter_list,
            self._init_static_args,
            self._affine_static_args,
            self._preprocess_static_args,
            self._reconstruction_static_args,
            self._init_optimize_args,
            self._affine_optimize_args,
            self._preprocess_optimize_args,
            self._reconstruction_optimize_args,
            evaluation_callback,
        )

        grid_search_res = list(map(grid_search_function, params_grid))
        pbar.close()

        if plot_reconstructed_objects:
            if len(n_points) == 2:
                nrows, ncols = n_points
            else:
                nrows = kwargs.pop("nrows", int(np.sqrt(num_evals)))
                ncols = kwargs.pop("ncols", int(np.ceil(num_evals / nrows)))
            if nrows * ncols < num_evals:
                raise ValueError()

            spec = GridSpec(
                ncols=ncols,
                nrows=nrows,
                hspace=0.15,
                wspace=0.15,
            )

            sx, sy = grid_search_res[0][0].shape

            separator = kwargs.pop("separator", "\n")
            cmap = kwargs.pop("cmap", "magma")
            figsize = kwargs.pop("figsize", (2.5 * ncols, 3 / sy * sx * nrows))
            fig = plt.figure(figsize=figsize)

            for index, (params, res) in enumerate(zip(params_grid, grid_search_res)):
                row_index, col_index = np.unravel_index(index, (nrows, ncols))

                ax = fig.add_subplot(spec[row_index, col_index])
                ax.imshow(res[0], cmap=cmap)

                title_substrings = [
                    f"{param.name}: {val}"
                    for param, val in zip(self._parameter_list, params)
                ]
                title_substrings.append(f"error: {res[1]:.3e}")
                title = separator.join(title_substrings)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)
            spec.tight_layout(fig)

            if return_reconstructed_objects:
                return grid_search_res
        else:
            return grid_search_res

    def optimize(
        self,
        n_calls: int = 50,
        n_initial_points: int = 20,
        error_metric: Union[Callable, str] = "log",
        **skopt_kwargs: dict,
    ):
        """
        Run optimizer

        Parameters
        ----------
        n_calls: int
            Number of times to run ptychographic reconstruction
        n_initial_points: int
            Number of uniformly spaced trial points to test before
            beginning Bayesian optimization (must be less than n_calls)
        error_metric: Callable or str
            Function used to compute the reconstruction error.
            When passed as a string, may be one of:
                'log': log(NMSE) of final object
                'linear': NMSE of final object
                'log-converged': log(NMSE) of final object if
                    NMSE is decreasing, 0 if NMSE increasing
                'linear-converged': NMSE of final object if
                    NMSE is decreasing, 1 if NMSE increasing
                'TV': sum( abs( grad( object ) ) ) / sum( abs( object ) )
                'std': negative standard deviation of cropped object
                'std-phase': negative standard deviation of
                    phase of the cropped object
                'entropy-phase': entropy of the phase of the
                    cropped object
            When passed as a Callable, a function that takes the
                PhaseReconstruction object as its only argument
                and returns the error metric as a single float
        skopt_kwargs: dict
            Additional arguments to be passed to skopt.gp_minimize

        """

        error_metric = self._get_error_metric(error_metric)

        optimization_function = self._get_optimization_function(
            self._reconstruction_type,
            self._parameter_list,
            self._init_static_args,
            self._affine_static_args,
            self._preprocess_static_args,
            self._reconstruction_static_args,
            self._init_optimize_args,
            self._affine_optimize_args,
            self._preprocess_optimize_args,
            self._reconstruction_optimize_args,
            error_metric,
        )

        # Make a progress bar
        pbar = tqdm(total=n_calls, desc="Optimizing parameters")

        # We need to wrap the callback because if it returns a value
        # the optimizer breaks its loop
        def callback(*args, **kwargs):
            pbar.update(1)

        try:
            self._skopt_result = gp_minimize(
                optimization_function,
                self._parameter_list,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                x0=self._x0,
                callback=callback,
                **skopt_kwargs,
            )

            # Remove the optimization result's reference to the function, as it potentially contains a
            # copy of the ptycho object
            del self._skopt_result["fun"]

            # If using the GPU, free some cached stuff
            if self._init_args.get("device", "cpu") == "gpu":
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                from cupy.fft.config import get_plan_cache

                get_plan_cache().clear()

            print("Optimized parameters:")
            for p, x in zip(self._parameter_list, self._skopt_result.x):
                print(f"{p.name}: {x}")
        finally:
            # close the pbar gracefully on interrupt
            pbar.close()

        return self

    def visualize(
        self,
        plot_gp_model=True,
        plot_convergence=False,
        plot_objective=True,
        plot_evaluations=False,
        **kwargs,
    ):
        """
        Visualize optimization results

        Parameters
        ----------
        plot_gp_model: bool
            Display fitted Gaussian process model (only available for 1-dimensional problem)
        plot_convergence: bool
            Display convergence history
        plot_objective: bool
            Display GP objective function and partial dependence plots
        plot_evaluations: bool
            Display histograms of sampled points
        kwargs:
            Passed directly to the skopt plot_gassian_process/plot_objective
        """
        ndims = len(self._parameter_list)
        if ndims == 1:
            if plot_convergence:
                figsize = kwargs.pop("figsize", (9, 9))
                spec = GridSpec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.15)
            else:
                figsize = kwargs.pop("figsize", (9, 6))
                spec = GridSpec(nrows=1, ncols=1)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(spec[0])
            skopt_plot_gaussian_process(
                self._skopt_result, ax=ax, show_title=False, **kwargs
            )

            if plot_convergence:
                ax = fig.add_subplot(spec[1])
                skopt_plot_convergence(self._skopt_result, ax=ax)

            spec.tight_layout(fig)

        else:
            if plot_evaluations:
                skopt_plot_evaluations(self._skopt_result)
            elif plot_objective:
                cmap = kwargs.pop("cmap", "magma")
                skopt_plot_objective(self._skopt_result, cmap=cmap, **kwargs)
            elif plot_convergence:
                skopt_plot_convergence(self._skopt_result)

        return self

    def get_optimized_arguments(self):
        """
        Get argument dictionaries containing optimized hyperparameters

        Returns
        -------
        init_opt, prep_opt, reco_opt: dicts
            Dictionaries of arguments to __init__, preprocess, and reconstruct
            where the OptimizationParameter items have been replaced with the optimal
            values obtained from the optimizer
        """
        optimized_dict = {
            p.name: v for p, v in zip(self._parameter_list, self._skopt_result.x)
        }

        filtered_dict = {
            k: v for k, v in optimized_dict.items() if k in self._init_args
        }
        init_opt = self._init_args | filtered_dict

        filtered_dict = {
            k: v for k, v in optimized_dict.items() if k in self._affine_args
        }
        affine_opt = self._affine_args | filtered_dict

        affine_transform = partial(AffineTransform, **self._affine_static_args)(
            **affine_opt
        )
        scan_positions = self._get_scan_positions(
            affine_transform, init_opt["datacube"]
        )
        init_opt["initial_scan_positions"] = scan_positions

        filtered_dict = {
            k: v for k, v in optimized_dict.items() if k in self._preprocess_args
        }
        prep_opt = self._preprocess_args | filtered_dict

        filtered_dict = {
            k: v for k, v in optimized_dict.items() if k in self._reconstruction_args
        }
        reco_opt = self._reconstruction_args | filtered_dict

        return init_opt, prep_opt, reco_opt

    def _split_static_and_optimization_vars(self, argdict):
        static_args = {}
        optimization_args = {}
        for k, v in argdict.items():
            if isinstance(v, OptimizationParameter):
                optimization_args[k] = v
            else:
                static_args[k] = v
        return static_args, optimization_args

    def _get_scan_positions(self, affine_transform, dataset):
        R_pixel_size = dataset.calibration.get_R_pixel_size()
        x, y = (
            np.arange(dataset.R_Nx) * R_pixel_size,
            np.arange(dataset.R_Ny) * R_pixel_size,
        )
        x, y = np.meshgrid(x, y, indexing="ij")
        scan_positions = np.stack((x.ravel(), y.ravel()), axis=1)
        scan_positions = scan_positions @ affine_transform.asarray()
        return scan_positions

    def _get_error_metric(self, error_metric: Union[Callable, str]) -> Callable:
        """
        Get error metric as a function, converting builtin method names
        to functions
        """

        if callable(error_metric):
            return error_metric

        assert error_metric in (
            "log",
            "linear",
            "log-converged",
            "linear-converged",
            "TV",
            "std",
            "std-phase",
            "entropy-phase",
        ), f"Error metric {error_metric} not recognized."

        if error_metric == "log":

            def f(ptycho):
                return np.log(ptycho.error)

        elif error_metric == "linear":

            def f(ptycho):
                return ptycho.error

        elif error_metric == "log-converged":

            def f(ptycho):
                converged = ptycho.error_iterations[-1] <= np.min(
                    ptycho.error_iterations
                )
                return np.log(ptycho.error) if converged else 0.0

        elif error_metric == "linear-converged":

            def f(ptycho):
                converged = ptycho.error_iterations[-1] <= np.min(
                    ptycho.error_iterations
                )
                return ptycho.error if converged else 1.0

        elif error_metric == "TV":

            def f(ptycho):
                gx, gy = np.gradient(ptycho.object_cropped, axis=(-2, -1))
                obj_mag = np.sum(np.abs(ptycho.object_cropped))
                tv = np.sum(np.abs(gx)) + np.sum(np.abs(gy))
                return tv / obj_mag

        elif error_metric == "std":

            def f(ptycho):
                return -np.std(ptycho.object_cropped)

        elif error_metric == "std-phase":

            def f(ptycho):
                return -np.std(np.angle(ptycho.object_cropped))

        elif error_metric == "entropy-phase":

            def f(ptycho):
                obj = np.angle(ptycho.object_cropped)
                gx, gy = np.gradient(obj)
                ghist, _, _ = np.histogram2d(
                    gx.ravel(), gy.ravel(), bins=obj.shape, density=True
                )
                nz = ghist > 0
                S = np.sum(ghist[nz] * np.log2(ghist[nz]))
                return S

        else:
            raise ValueError(f"Error metric {error_metric} not recognized.")

        return f

    def _get_optimization_function(
        self,
        cls: type[PhaseReconstruction],
        parameter_list: list,
        init_static_args: dict,
        affine_static_args: dict,
        preprocess_static_args: dict,
        reconstruct_static_args: dict,
        init_optimization_params: dict,
        affine_optimization_params: dict,
        preprocess_optimization_params: dict,
        reconstruct_optimization_params: dict,
        error_metric: Callable,
    ):
        """
        Wrap the ptychography pipeline into a single function that encapsulates all of the
        non-optimization arguments and accepts a concatenated set of keyword arguments. The
        wrapper function returns the final error value from the ptychography run.

        parameter_list is a list of skopt Dimension objects

        Both static and optimization args are passed in dictionaries. The values of the
        static dictionary are the fixed parameters, and only the keys of the optimization
        dictionary are used.
        """

        # Get lists of optimization parameters for each step
        init_params = list(init_optimization_params.keys())
        afft_params = list(affine_optimization_params.keys())
        prep_params = list(preprocess_optimization_params.keys())
        reco_params = list(reconstruct_optimization_params.keys())

        # Construct partial methods to encapsulate the static parameters.
        # If only ``reconstruct`` has optimization variables, perform
        # preprocessing now, store the ptycho object, and use dummy
        # functions instead of the partials
        if (len(init_params), len(afft_params), len(prep_params)) == (0, 0, 0):
            affine_preprocessed = AffineTransform(**affine_static_args)
            init_args = init_static_args.copy()
            init_args["initial_scan_positions"] = self._get_scan_positions(
                affine_preprocessed, init_static_args["datacube"]
            )

            ptycho_preprocessed = cls(**init_args).preprocess(**preprocess_static_args)

            def obj(**kwargs):
                return ptycho_preprocessed

            def prep(ptycho, **kwargs):
                return ptycho

        else:
            obj = partial(cls, **init_static_args)
            prep = partial(cls.preprocess, **preprocess_static_args)

        affine = partial(AffineTransform, **affine_static_args)
        recon = partial(cls.reconstruct, **reconstruct_static_args)

        # Target function for Gaussian process optimization that takes a single
        # dict of named parameters and returns the ptycho error metric
        @use_named_args(parameter_list)
        def f(**kwargs):
            init_args = {k: kwargs[k] for k in init_params}
            afft_args = {k: kwargs[k] for k in afft_params}
            prep_args = {k: kwargs[k] for k in prep_params}
            reco_args = {k: kwargs[k] for k in reco_params}

            # Create affine transform object
            tr = affine(**afft_args)
            # Apply affine transform to pixel grid, using the
            # calibrations lifted from the dataset
            dataset = init_static_args["datacube"]
            init_args["initial_scan_positions"] = self._get_scan_positions(tr, dataset)

            ptycho = obj(**init_args)
            prep(ptycho, **prep_args)
            recon(ptycho, **reco_args)

            return error_metric(ptycho)

        return f

    def _set_optimizer_defaults(
        self,
        verbose=False,
        clear_fft_cache=False,
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        progress_bar=False,
        store_iterations=False,
        reset=True,
    ):
        """
        Set all of the verbose and plotting to False, allowing for user-overwrite.
        """
        self._init_static_args["verbose"] = verbose
        self._init_static_args["clear_fft_cache"] = clear_fft_cache

        self._preprocess_static_args["plot_center_of_mass"] = plot_center_of_mass
        self._preprocess_static_args["plot_rotation"] = plot_rotation
        self._preprocess_static_args["plot_probe_overlaps"] = plot_probe_overlaps

        self._reconstruction_static_args["progress_bar"] = progress_bar
        self._reconstruction_static_args["store_iterations"] = store_iterations
        self._reconstruction_static_args["reset"] = reset


class OptimizationParameter:
    """
    Wrapper for scikit-optimize Space objects used for convenient calling in the PtyhochraphyOptimizer
    """

    def __init__(
        self,
        initial_value: Union[float, int, bool],
        lower_bound: Union[float, int, bool] = None,
        upper_bound: Union[float, int, bool] = None,
        scaling: str = "uniform",
        space: str = "real",
        categories: list = [],
    ):
        """
        Wrapper for scikit-optimize Space objects used as inputs to PtychographyOptimizer

        Parameters
        ----------
        initial_value:
            Initial value, used for first evaluation in optimizer
        lower_bound, upper_bound:
            Bounds on real or integer variables (not needed for bool or categorical)
        scaling: str
            Prior knowledge on sensitivity of the variable. Can be 'uniform' or 'log-uniform'
        space: str
            Type of variable. Can be 'real', 'integer', 'bool', or 'categorical'
        categories: list
            List of options for Categorical parameter
        """
        # Check input
        space = space.lower()
        if space not in ("real", "integer", "bool", "categorical"):
            raise ValueError(f"Unknown Parameter type: {space}")

        scaling = scaling.lower()
        if scaling not in ("uniform", "log-uniform"):
            raise ValueError(f"Unknown scaling: {scaling}")

        # Get the right scikit-optimize class
        space_map = {
            "real": Real,
            "integer": Integer,
            "bool": Categorical,
            "categorical": Categorical,
        }
        param = space_map[space]

        # If a boolean property, the categories are True/False
        if space == "bool":
            categories = [True, False]

        if categories == [] and space in ("categorical", "bool"):
            raise ValueError("Empty list of categories!")

        # store necessary information
        self._initial_value = initial_value
        self._categories = categories
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scaling = scaling
        self._param_type = param

    def _get(self, name):
        self._name = name
        if self._param_type is Categorical:
            self._skopt_param = self._param_type(
                name=self._name, categories=self._categories
            )
        else:
            self._skopt_param = self._param_type(
                name=self._name,
                low=self._lower_bound,
                high=self._upper_bound,
                prior=self._scaling,
            )
        return self._skopt_param
