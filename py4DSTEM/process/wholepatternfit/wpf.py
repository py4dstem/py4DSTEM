from py4DSTEM.io import DataCube
from py4DSTEM import tqdmnd
from . import WPFModelPrototype

from typing import Optional
import numpy as np

from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
from matplotlib.gridspec import GridSpec


class WholePatternFit:
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
        self.global_args = {}

        self.global_args["global_x0"] = x0 if x0 else datacube.Q_Nx / 2.0
        self.global_args["global_y0"] = y0 if y0 else datacube.Q_Ny / 2.0

        xArray, yArray = np.mgrid[0 : datacube.Q_Nx, 0 : datacube.Q_Ny]
        self.global_args["xArray"] = xArray
        self.global_args["yArray"] = yArray

        self.global_args["global_r"] = np.hypot((xArray - x0) ** 2, (yArray - y0) ** 2)

        self.global_args["Q_Nx"] = datacube.Q_Nx
        self.global_args["Q_Ny"] = datacube.Q_Ny

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

        # set the current active pattern to the mean CBED:
        self.current_pattern = self.meanCBED
        self.current_glob = self.global_args.copy()

        return self._pattern(self.x0)

    def fit_to_mean_CBED(self, **fit_opts):

        # first make sure we have the latest parameters
        self._scrape_model_params()

        # set the current active pattern to the mean CBED:
        self.current_pattern = self.meanCBED
        self.current_glob = self.global_args.copy()

        self._fevals = []
        self._xevals = []
        self._cost_history = []

        if self.hasJacobian & self.use_jacobian:
            opt = least_squares(
                self._pattern_error,
                self.x0,
                jac=self._jacobian,
                bounds=(self.lower_bound, self.upper_bound),
                **fit_opts
            )
        else:
            opt = least_squares(
                self._pattern_error,
                self.x0,
                bounds=(self.lower_bound, self.upper_bound),
                **fit_opts
            )
            # opt = minimize(
            #     lambda x: np.sum(self._pattern_error(x)**2),
            #     self.x0,
            #     callback=lambda x: self._fevals.append(x),
            #     bounds=[(lb,ub) for lb,ub in zip(self.lower_bound,self.upper_bound)],
            #     **fit_opts
            # )

        self.mean_CBED_fit = opt

        # Plotting
        fig = plt.figure(constrained_layout=True,figsize=(7,7))
        gs = GridSpec(2,2,figure=fig)

        ax = fig.add_subplot(gs[0,0])
        err_hist = np.array(self._cost_history)
        # err_hist = np.array([np.sum((self._pattern(dp)-self.meanCBED)**2) for dp in self._fevals])
        ax.plot(err_hist)
        ax.set_ylabel("Sum Squared Error")
        ax.set_xlabel("Iterations")
        ax.set_yscale("log")

        DP = self._pattern(self.mean_CBED_fit.x)
        ax = fig.add_subplot(gs[0,1])
        CyRd = mpl_c.LinearSegmentedColormap.from_list("CyRd",["#00ccff","#ffffff","#ff0000"])
        im = ax.matshow(err_im := -(DP - self.meanCBED),
            cmap=CyRd,vmin=-np.abs(err_im).max()/4,vmax=np.abs(err_im).max()/4)
        # fig.colorbar(im)
        ax.axis('off')

        ax = fig.add_subplot(gs[1,:])
        ax.matshow(np.hstack((DP,self.meanCBED))**0.25,cmap='jet')
        ax.axis('off')
        ax.text(0.25,0.92,"Refined",transform=ax.transAxes,ha='center',va='center')
        ax.text(0.75,0.92,"Mean CBED",transform=ax.transAxes,ha='center',va='center')

        plt.show()

        return opt

    def fit_all_patterns(self, **fit_opts):

        # make sure we have the latest parameters
        self._scrape_model_params()

        # set tracking off
        self._track = False
        self._fevals = []

        self.fit_data = np.zeros((self.datacube.R_Nx, self.datacube.R_Ny, self.x0.shape[0]))

        for rx, ry in tqdmnd(self.datacube.R_Nx, self.datacube.R_Ny):
            self.current_pattern = self.datacube.data[rx, ry, :, :]
            self.current_glob = self.global_args.copy()

            if self.hasJacobian & self.use_jacobian:
                opt = least_squares(
                    self._pattern_error,
                    self.x0,
                    jac=self._jacobian,
                    bounds=(self.lower_bound, self.upper_bound),
                    **fit_opts
                )
            else:
                opt = least_squares(
                    self._pattern_error,
                    self.x0,
                    bounds=(self.lower_bound, self.upper_bound),
                    **fit_opts
                )
                # opt = minimize(
                #     lambda x: np.sum(self._pattern_error(x)**2),
                #     self.x0,
                #     callback=lambda x: self._fevals.append(x),
                #     bounds=[(lb,ub) for lb,ub in zip(self.lower_bound,self.upper_bound)]
                #     **fit_opts
                # )

            self.fit_data[rx, ry, :] = opt.x

    def accept_mean_CBED_fit(self):
        x = self.mean_CBED_fit.x
        self.global_args["global_x0"] = x[0]
        self.global_args["global_y0"] = x[1]

        self.global_args["global_r"] = np.hypot(
            (self.global_args["xArray"] - x[0]), (self.global_args["yArray"] - x[1])
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            for j, k in enumerate(m.params.keys()):
                m.params[k].initial_value = x[ind + j]

    def show_model_grid(self,x):
        if x is None:
            x = self.mean_CBED_fit.x

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        N = len(self.model)
        cols = int(np.ceil(np.sqrt(N)))
        rows = (N + 1) // cols

        fig, ax = plt.subplots(rows,cols, constrained_layout=True)

        for i,(a,m) in enumerate(zip(ax.flat, self.model)):
            DP = np.zeros((self.datacube.Q_Nx,self.datacube.Q_Ny))
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **self.current_glob)

            a.matshow(DP,cmap='turbo')
            a.axis('off')
            a.text(0.5,0.92,m.name,transform=a.transAxes,ha='center',va='center')

        plt.show()

    def _pattern_error(self, x):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **self.current_glob)

        DP = (DP ** self.fit_power - self.current_pattern ** self.fit_power) * self.mask

        if self._track:
            self._fevals.append(DP)
            self._xevals.append(x)
        self._cost_history.append(np.sum(DP**2))

        return DP.ravel()

    def _pattern(self, x):

        DP = np.zeros((self.datacube.Q_Nx, self.datacube.Q_Ny))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.func(DP, *x[ind : ind + m.nParams].tolist(), **self.current_glob)

        return (DP ** self.fit_power) * self.mask

    def _jacobian(self, x):

        J = np.zeros(((self.datacube.Q_Nx * self.datacube.Q_Ny), self.nParams + 2))

        self.current_glob["global_x0"] = x[0]
        self.current_glob["global_y0"] = x[1]
        self.current_glob["global_r"] = np.hypot(
            (self.current_glob["xArray"] - x[0]), (self.current_glob["yArray"] - x[1]),
        )

        for i, m in enumerate(self.model):
            ind = self.model_param_inds[i] + 2
            m.jacobian(
                J, *x[ind : ind + m.nParams].tolist(), offset=ind, **self.current_glob
            )

        return J * self.mask.ravel()[:, np.newaxis]

    def _scrape_model_params(self):

        self.x0 = np.zeros((self.nParams + 2,))
        self.upper_bound = np.zeros_like(self.x0)
        self.lower_bound = np.zeros_like(self.x0)

        self.x0[0:2] = np.array(
            [self.global_args["global_x0"], self.global_args["global_y0"]]
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
