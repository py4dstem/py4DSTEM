from ...file.datastructure import DataCube
from . import WPFModelPrototype

from typing import Optional
import numpy as np

from scipy.optimize import least_squares


class WholePatternFit:
    def __init__(
        self,
        datacube: DataCube,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.datacube = datacube
        self.meanCBED = np.mean(datacube.data, axis=(0, 1))

        self.mask = mask if mask else np.ones_like(self.meanCBED)

        self.model = []
        self.model_param_inds = []

        self.nParams = 0

        # set up the global arguments
        self.global_args = {}

        self.global_args["global_x0"] = x0 if x0 else datacube.Q_Nx / 2.0
        self.global_args["global_y0"] = y0 if y0 else datacube.Q_Ny / 2.0

        xArray, yArray = np.mgrid[0 : datacube.Q_Nx, 0 : datacube.Q_Ny]
        self.global_args["xArray"] = xArray
        self.global_args["yArray"] = yArray

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

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *self.x0[ind : ind + m.nParams].tolist(), **self.global_args)

        return DP * self.mask

    def fit_to_mean_CBED(self):

        # first make sure we have the latest parameters
        self._scrape_model_params()

        # set the current active pattern to the mean CBED:
        self.current_pattern = self.meanCBED
        self.current_glob = self.global_args.copy()

        opt = least_squares(self._pattern, self.x0)

        self.mean_CBED_fit = opt

        return opt

    def _pattern(self, x):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **self.current_glob)

        DP = (DP - self.current_pattern) * self.mask

        return DP.ravel()

    def _jacobian(self, x):

        J = np.zeros(((self.datacube.Q_Nx * self.datacube.Q_Ny), self.nParams + 2))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.jacobian(
                J, *x[ind : ind + m.nParams].tolist(), offset=ind, **self.current_glob
            )

        return J * self.mask.ravel()[:, np.newaxis]

    def _scrape_model_params(self):

        self.x0 = np.zeros((self.nParams + 2,))

        self.x0[0:2] = np.array(
            [self.global_args["global_x0"], self.global_args["global_y0"]]
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            self.x0[ind : ind + m.nParams] = np.fromiter(
                m.params.values(), dtype=np.float32
            )
