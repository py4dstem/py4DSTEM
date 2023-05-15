from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_gaussian_process, plot_evaluations, plot_objective
from skopt.utils import use_named_args

from functools import partialmethod
from typing import Union

from py4DSTEM.process.phase import PtychographicReconstruction

class PtychographyOptimizer:
	"""
	doot doot
	"""

	def __init__(
		self,
		reconstruction_type: type[PtychographicReconstruction],
		):
		"""
		Parameter optimization for ptychographic reconstruction based on Bayesian Optimization
		with Gaussian Process. 
		"""
		pass

	def optimize(self,):
		pass

	def visualize(self,):
		pass

class OptimizationParameter:
	"""
	Wrapper for scikit-optimize Space objects used for convenient calling in the PtyhochraphyOptimizer
	"""

	def __init__(
		self,
		name:str,
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
		self.name = name
		self.initial_value = initial_value

		if param is Categorical:
			assert categories is not None, "Empty list of categories!"
			self._skopt_param = param(name=name, categories=categories)
		else:
			self._skopt_param = param(name=name, low=lower_bound, high=upper_bound, prior=scaling,)



