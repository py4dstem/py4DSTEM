from ...file.datastructure import DataCube
from . import WPFModelPrototype

from typing import Optional
import numpy as np


class WholePatternFit:

	def __init__(self, data: DataCube, x0:Optional[float]=None, y0:Optional[float]=None):
		self.data = data

		self.model = []
		self.model_param_inds = []

		self.nParams = 0

		# set up the global arguments
		self.global_args = {}

		self.global_args["global_x0"] = x0 if x0 else data.Q_Nx / 2.
		self.global_args["global_y0"] = y0 if y0 else data.Q_Ny / 2.

		xArray, yArray = np.mgrid[0:data.Q_Nx,0:data.Q_Ny]
		self.global_args["xArray"] = xArray
		self.global_args["yArray"] = yArray


	def add_model(self, model: WPFModelPrototype):
		self.model.append(model)

		# keep track of where each model's parameter list begins
		self.model_param_inds.append(self.nParams)
		self.nParams += len(model.params.keys())

		self._scrape_model_params()

	def generate_initial_pattern(self):
		DP = np.zeros((self.data.Q_Nx,self.data.Q_Ny))

		for i,m in enumerate(self.model):
			ind = self.model_param_inds[i]
			m.func(DP, *self.x[ind:ind+m.nParams].tolist(), **self.global_args)

		return DP

	def _scrape_model_params(self):

		self.x = np.zeros((self.nParams,))

		for i,m in enumerate(self.model):
			ind = self.model_param_inds[i]
			self.x[ind:ind+m.nParams] = np.fromiter(m.params.values(),dtype=np.float32)