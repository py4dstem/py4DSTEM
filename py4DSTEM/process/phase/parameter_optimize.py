from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_gaussian_process, plot_evaluations, plot_objective
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

from functools import partial
from typing import Union

from py4DSTEM.process.phase.iterative_base_class import PhaseReconstruction

class PtychographyOptimizer:
    """
    doot doot
    """

    def __init__(
        self,
        reconstruction_type: type[PhaseReconstruction],
        init_args:dict,
        preprocess_args:dict,
        reconstruction_args:dict,
        ):
        """
        Parameter optimization for ptychographic reconstruction based on Bayesian Optimization
        with Gaussian Process. 
        """
        
        # loop over each argument dictionary and split into static and optimization variables
        self._init_static_args, self._init_optimize_args = self._split_static_and_optimization_vars(init_args)
        self._preprocess_static_args, self._preprocess_optimize_args = self._split_static_and_optimization_vars(preprocess_args)
        self._reconstruction_static_args, self._reconstruction_optimize_args = self._split_static_and_optimization_vars(reconstruction_args)

        # Save list of skopt parameter objects and inital guess
        self._parameter_list = []
        self._x0 = []
        for k,v in (self._init_optimize_args|self._preprocess_optimize_args|self._reconstruction_optimize_args).items():
            self._parameter_list.append(v._get(k))
            self._x0.append(v._initial_value)

        self._init_args = init_args
        self._preprocess_args = preprocess_args
        self._reconstruction_args = reconstruction_args

        self._reconstruction_type = reconstruction_type

        self._set_optimizer_defaults()

    def optimize(
        self,
        n_calls:int,
        n_initial_points:int,
        ):

        self._optimization_function, pbar = self._get_optimization_function(
            self._reconstruction_type,
            self._parameter_list,
            n_calls,
            self._init_static_args,
            self._preprocess_static_args,
            self._reconstruction_static_args,
            self._init_optimize_args,
            self._preprocess_optimize_args,
            self._reconstruction_optimize_args
            )

        self._skopt_result = gp_minimize(
            self._optimization_function,
            self._parameter_list,
            n_calls = n_calls,
            n_initial_points=n_initial_points,
            x0 = self._x0
        )

        # Finish the tqdm progressbar so subsequent things behave nicely
        pbar.close()

        return self

    def visualize(
        self,
        gp_model=True,
        convergence=False,
        objective=True,
        evaluations=False,
    ):
        if len(self._parameter_list) == 1 and gp_model:
            plot_gaussian_process(self._skopt_result)
            plt.show()

        if convergence:
            plot_convergence(self._skopt_result)
            plt.show()
        if objective:
            plot_objective(self._skopt_result)
            plt.show()
        if evaluations:
            plot_evaluations(self._skopt_result)
            plt.show()
        return self

    def get_optimized_arguments(self):
        init_opt = self._replace_optimized_parameters(self._init_args, self._parameter_list)
        prep_opt = self._replace_optimized_parameters(self._preprocess_args, self._parameter_list)
        reco_opt = self._replace_optimized_parameters(self._reconstruction_args, self._parameter_list)

        return init_opt, prep_opt, reco_opt

    def _replace_optimized_parameters(self, arg_dict, parameters):
        opt_args = {}
        for k,v in arg_dict.items():
            # Check for optimization parameters
            if isinstance(v,OptimizationParameter):
                # Find the right parameter in the list
                # There is probably a better way to do this inverse mapping!
                for i,p in enumerate(parameters):
                    if p.name == k:
                        opt_args[k] = self._skopt_result.x[i]
            else:
                opt_args[k] = v
        return opt_args


    def _split_static_and_optimization_vars(self,argdict):
        static_args = {}
        optimization_args = {}
        for k,v in argdict.items():
            if isinstance(v,OptimizationParameter):
                # unwrap the OptimizationParameter object into a skopt object
                optimization_args[k] = v #._get(k)
            else:
                static_args[k] = v
        return static_args, optimization_args


    def _get_optimization_function(
        self,
        cls:type[PhaseReconstruction], 
        parameter_list:list,
        n_iterations:int,
        init_static_args:dict, 
        preprocess_static_args:dict, 
        reconstruct_static_args:dict, 
        init_optimization_param_names:dict, 
        preprocess_optimization_param_names:dict, 
        reconstruct_optimization_param_names:dict):
        """
        Wrap the ptychography pipeline into a single function that encapsulates all of the
        non-optimization arguments and accepts a concatenated set of keyword arguments. The
        wrapper function returns the final error value from the ptychography run.

        parameter_list is a list of skopt Dimension objects

        Both static and optimization args are passed in dictionaries. The values of the
        static dictionary are the fixed parameters, and only the keys of the optimization
        dictionary are used. 
        """

        # Construct partial methods to encapsulate the static parameters
        obj = partial(cls,**init_static_args)
        prep = partial(cls.preprocess, **preprocess_static_args)
        recon = partial(cls.reconstruct, **reconstruct_static_args)

        init_params = list(init_optimization_param_names.keys())
        prep_params = list(preprocess_optimization_param_names.keys())
        reco_params = list(reconstruct_optimization_param_names.keys())

        # Make a progress bar
        pbar = tqdm(total=n_iterations,desc="Optimizing parameters")

        # Target function for Gaussian process optimization that takes a single
        # dict of named parameters and returns the ptycho error metric
        @use_named_args(parameter_list)
        def f(**kwargs):
            init_args = {k:kwargs[k] for k in init_params}
            prep_args = {k:kwargs[k] for k in prep_params}
            reco_args = {k:kwargs[k] for k in reco_params}

            ptycho = obj(**init_args)
            prep(ptycho,**prep_args)
            recon(ptycho,**reco_args)

            pbar.update(1)

            return ptycho.error
                
        return f, pbar

    def _set_optimizer_defaults(self):
        """
        Set all of the verbose and plotting to False
        """
        self._init_static_args['verbose'] = False

        self._preprocess_static_args['plot_center_of_mass'] = False
        self._preprocess_static_args['plot_rotation'] = False
        self._preprocess_static_args['plot_probe_overlaps'] = False

        self._reconstruction_static_args['progress_bar'] = False
        self._reconstruction_static_args['store_iterations'] = False


class OptimizationParameter:
    """
    Wrapper for scikit-optimize Space objects used for convenient calling in the PtyhochraphyOptimizer
    """

    def __init__(
        self,
        lower_bound:Union[float,int,bool],
        upper_bound:Union[float,int,bool],
        initial_value:Union[float,int,bool],
        scaling:str='uniform',
        space:str="real",
        categories:list=[],
        ):

        # Check input
        space = space.lower()
        assert space in ("real","integer","bool","categorical"), f"Unknown Parameter type: {space}"

        scaling = scaling.lower()
        assert scaling in ("uniform", "log-uniform"), f"Unknown scaling: {scaling}"

        # Get the right scikit-optimize class
        space_map = {'real':Real, 'integer':Integer, 'bool':Categorical, 'categorical':Categorical}
        param = space_map[space]

        # If a boolean property, the categories are True/False
        if space == 'bool':
            categories = [True, False]

        # store necessary information
        self._initial_value = initial_value
        self._categories = categories
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._scaling = scaling
        self._param_type = param

    def _get(self,name):
        self._name = name
        if self._param_type is Categorical:
            assert categories is not None, "Empty list of categories!"
            self._skopt_param = self._param_type(name=self._name, categories=self._categories)
        else:
            self._skopt_param = self._param_type(name=self._name, low=self._lower_bound, high=self._upper_bound, prior=self._scaling,)
        return self._skopt_param


