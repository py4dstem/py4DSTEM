from inspect import signature

import numpy as np

from pdb import set_trace


class WPFModelPrototype:
    """
    Prototype class for a compent of a whole-pattern model.
    Holds the following:
        name:       human-readable name of the model
        params:     a dict of names and initial (or returned) values of the model parameters
        func:       a function that takes as arguments:
                        • the diffraction pattern being built up, which the function should modify in place
                        • positional arguments in the same order as the params dictionary
                        • keyword arguments. this is to provide some pre-computed information for convenience
                            kwargs will include:
                                • xArray, yArray    meshgrid of the x and y coordinates
                                • global_x0         global x-coordinate of the pattern center
                                • global_y0         global y-coordinate of the pattern center
        jacobian:   a function that takes as arguments:
                        • the diffraction pattern being built up, which the function should modify in place
                        • positional arguments in the same order as the params dictionary
                        • offset: the first index (j) that values should be written into
                            (the function should ONLY write into 0,1, and offset:offset+nParams)
                            0 and 1 are the entries for global_x0 and global_y0, respectively
                            **REMEMBER TO ADD TO 0 and 1 SINCE ALL MODELS CAN CONTRIBUTE TO THIS PARTIAL DERIVATIVE**
                        • keyword arguments. this is to provide some pre-computed information for convenience
    """

    def __init__(
        self, name: str, params: dict,
    ):
        self.name = name
        self.params = params

        self.nParams = len(params.keys())

        self.hasJacobian = getattr(self, "jacobian", None) is not None

    #     # check the function obeys the spec
    #     assert (
    #         len(signature(self.func).parameters) == len(params) + 2
    #     ), f"The model function has the wrong number of arguments in its signature. It must be written as func(DP, param1, param2, ..., **kwargs). The current signature is {str(signature(self.func))}"

    def func(self, DP: np.ndarray, *args, **kwargs) -> None:
        raise NotImplementedError()

    # Required signature for the Jacobian:
    #
    # def jacobian(self, J: np.ndarray, *args, offset: int, **kwargs) -> None:
    #     raise NotImplementedError()


class DCBackground(WPFModelPrototype):
    def __init__(self, background_value=0.0, name="DC Background"):
        params = {"DC Level": background_value}

        super().__init__(name, params)

    def func(self, DP: np.ndarray, level, **kwargs) -> None:
        DP += level

    def jacobian(self, J: np.ndarray, *args, offset: int, **kwargs):
        J[:, offset] = 1


class GaussianBackground(WPFModelPrototype):
    def __init__(self, sigma, intensity, global_center=True, x0=0.0, y0=0.0):
        name = "Gaussian Background"
        params = {"sigma": sigma, "intensity": intensity}
        if global_center:
            self.func = self.global_center_func
            self.jacobian = self.global_center_jacobian
        else:
            params["x center"] = x0
            params["y center"] = y0
            self.func = self.local_center_func
            self.jacobian = self.local_center_jacobian

        super().__init__(name, params)

    def global_center_func(self, DP: np.ndarray, sigma, level, **kwargs) -> None:
        DP += level * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2))

    def global_center_jacobian(
        self, J: np.ndarray, sigma, level, offset: int, **kwargs
    ) -> None:

        # dF/d(global_x0)
        J[:, 0] += (
            level
            * (kwargs["xArray"] - kwargs["global_x0"])
            * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2))
            / sigma ** 2
        ).ravel()

        # dF/d(global_y0)
        J[:, 1] += (
            level
            * (kwargs["yArray"] - kwargs["global_y0"])
            * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2))
            / sigma ** 2
        ).ravel()

        # dF/s(sigma)
        J[:, offset] = (
            level
            * kwargs["global_r"] ** 2
            * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2))
            / sigma ** 3
        ).ravel()

        # dF/d(level)
        J[:, offset + 1] = np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2)).ravel()

    def local_center_func(self, DP: np.ndarray, sigma, level, x0, y0, **kwargs) -> None:
        DP += level * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma ** 2))

    def local_center_jacobian(
        self, J: np.ndarray, sigma, level, x0, y0, offset: int, **kwargs
    ) -> None:

        # dF/s(sigma)
        J[:, offset] = (level
            * ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma ** 2)
            )
            / sigma ** 3
        ).ravel()

        # dF/d(level)
        J[:, offset + 1] = np.exp(
            ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            / (-2 * sigma ** 2)
        ).ravel()

        # dF/d(x0)
        J[:, offset + 2] = (level
            * (kwargs["xArray"] - x0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma ** 2)
            )
            / sigma ** 2
        ).ravel()

        # dF/d(y0)
        J[:, offset + 3] = (level
            * (kwargs["yArray"] - y0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma ** 2)
            )
            / sigma ** 2
        ).ravel()


class SyntheticDiskLattice(WPFModelPrototype):
    def __init__(
        self,
        ux: float,
        uy: float,
        vx: float,
        vy: float,
        disk_radius: float,
        disk_width: float,
        u_max: int,
        v_max: int,
        intensity_0: float,
        refine_radius: bool = False,
        refine_width: bool = False,
        global_center: bool = True,
        x0: float = 0.0,
        y0: float = 0.0,
    ):
        self.disk_radius = disk_radius
        self.disk_width = disk_width

        name = "Synthetic Disk Lattice"
        params = {}

        if global_center:
            self.func = self.global_center_func
            self.jacobian = self.global_center_jacobian
        else:
            params["x center"] = x0
            params["y center"] = y0
            self.func = self.local_center_func

        params["ux"] = ux
        params["uy"] = uy
        params["vx"] = vx
        params["vy"] = vy

        u_inds, v_inds = np.mgrid[-u_max : u_max + 1, -v_max : v_max + 1]
        self.u_inds = u_inds.ravel()
        self.v_inds = v_inds.ravel()

        for u, v in zip(u_inds.ravel(), v_inds.ravel()):
            params[f"[{u},{v}] Intensity"] = intensity_0

        self.refine_radius = refine_radius
        self.refine_width = refine_width
        if refine_radius:
            params["disk radius"] = disk_radius
        if refine_width:
            params["edge width"] = disk_width

        super().__init__(name, params)

    def global_center_func(self, DP: np.ndarray, *args, **kwargs) -> None:
        # copy the global centers in the right place for the local center generator
        self.local_center_func(
            DP, kwargs["global_x0"], kwargs["global_y0"], *args, **kwargs
        )

    def local_center_func(self, DP: np.ndarray, *args, **kwargs) -> None:

        x0 = args[0]
        y0 = args[1]
        ux = args[2]
        uy = args[3]
        vx = args[4]
        vy = args[5]

        if self.refine_radius & self.refine_width:
            disk_radius = args[-2]
        elif self.refine_radius:
            disk_radius = args[-1]
        else:
            disk_radius = self.disk_radius
        disk_width = args[-1] if self.refine_width else self.disk_width

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)
            if (x > 0) & (x < kwargs["Q_Nx"]) & (y > 0) & (y < kwargs["Q_Nx"]):
                DP += args[i + 6] / (
                    1.0
                    + np.exp(
                        4
                        * (
                            np.sqrt(
                                (kwargs["xArray"] - x) ** 2
                                + (kwargs["yArray"] - y) ** 2
                            )
                            - disk_radius
                        )
                        / disk_width
                    )
                )

    def global_center_jacobian(
        self, J: np.ndarray, *args, offset: int, **kwargs
    ) -> None:

        x0 = kwargs["global_x0"]
        y0 = kwargs["global_y0"]
        r = np.maximum(5e-1, kwargs["global_r"])
        ux = args[0]
        uy = args[1]
        vx = args[2]
        vy = args[3]

        if self.refine_radius & self.refine_width:
            disk_radius = args[-2]
            radius_ind = -2
        elif self.refine_radius:
            disk_radius = args[-1]
            radius_ind = -1
        else:
            disk_radius = self.disk_radius
        disk_width = args[-1] if self.refine_width else self.disk_width

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)

            if (x > 0) & (x < kwargs["Q_Nx"]) & (y > 0) & (y < kwargs["Q_Nx"]):
                r_disk = np.maximum(
                    5e-1,
                    np.sqrt((kwargs["xArray"] - x) ** 2 + (kwargs["yArray"] - y) ** 2),
                )

                # dF/d(global_x0)
                dx = (
                    4
                    * args[i + 4]
                    * (kwargs["xArray"] - x)
                    * np.exp(4 * (r_disk - disk_radius) / disk_width)
                    / (
                        (1.0 + np.exp(4 * (r_disk - disk_radius) / disk_width)) ** 2
                        * disk_width
                        * r
                    )
                ).ravel()

                # dF/d(global_y0)
                dy = (
                    4
                    * args[i + 4]
                    * (kwargs["yArray"] - y)
                    * np.exp(4 * (r_disk - disk_radius) / disk_width)
                    / (
                        (1.0 + np.exp(4 * (r_disk - disk_radius) / disk_width)) ** 2
                        * disk_width
                        * r
                    )
                ).ravel()

                # because... reasons, sometimes we get NaN
                # very far from the disk center. let's zero those:
                dx[np.isnan(dx)] = 0.0
                dy[np.isnan(dy)] = 0.0

                # insert global positional derivatives
                J[:, 0] += dx
                J[:, 1] += dy

                # insert lattice vector derivatives
                J[:, offset] += u * dx
                J[:, offset + 1] += u * dy
                J[:, offset + 2] += v * dx
                J[:, offset + 3] += v * dy

                # insert intensity derivative
                dI = (
                    1.0
                    / (1.0 + np.exp(4 * (r_disk - disk_radius) / disk_width))
                ).ravel()
                dI[np.isnan(dI)] = 0.0
                J[:, offset + i + 4] = dI

                # insert disk radius derivative
                if self.refine_radius:
                    dR = (
                        4.0
                        * args[i + 4]
                        * np.exp(4 * (r_disk - disk_radius) / disk_width)
                        / (
                            disk_width
                            * (1.0 + np.exp(4 * (r_disk - disk_radius) / disk_width))
                            ** 2
                        )
                    ).ravel()
                    dR[np.isnan(dR)] = 0.0
                    J[:, offset + len(args) + radius_ind] += dR

                if self.refine_width:
                    dW = (
                        4.0
                        * args[i + 4]
                        * np.exp(4 * (r_disk - disk_radius) / disk_width)
                        * (r_disk - disk_radius)
                        / (
                            disk_width ** 2
                            * (1.0 + np.exp(4 * (r_disk - disk_radius) / disk_width))
                            ** 2
                        )
                    ).ravel()
                    dW[np.isnan(dW)] = 0.0
                    J[:, offset + len(args) - 1] += dW

                # set_trace()
