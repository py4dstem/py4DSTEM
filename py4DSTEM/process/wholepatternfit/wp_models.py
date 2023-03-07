from inspect import signature
from typing import Optional
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
        self,
        name: str,
        params: dict,
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


class Parameter:
    def __init__(
        self,
        initial_value,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        ):

        if hasattr(initial_value, "__iter__"):
            if len(initial_value) == 2:
                initial_value = (
                    initial_value[0],
                    initial_value[0]-initial_value[1],
                    initial_value[0]+initial_value[1],
                    )
            self.set_params(*initial_value)
        else:
            self.set_params(initial_value, lower_bound, upper_bound)



    def set_params(
        self, 
        initial_value, 
        lower_bound, 
        upper_bound,
        ):
        self.initial_value = initial_value
        self.lower_bound = lower_bound if lower_bound is not None else -np.inf
        self.upper_bound = upper_bound if upper_bound is not None else  np.inf

    def __str__(self):
        return f"Value: {self.initial_value} (Range: {self.lower_bound},{self.upper_bound})"

    def __repr__(self):
        return f"Value: {self.initial_value} (Range: {self.lower_bound},{self.upper_bound})"


class DCBackground(WPFModelPrototype):
    def __init__(self, background_value=0.0, name="DC Background"):
        params = {"DC Level": Parameter(background_value)}

        super().__init__(name, params)

    def func(self, DP: np.ndarray, level, **kwargs) -> None:
        DP += level

    def jacobian(self, J: np.ndarray, *args, offset: int, **kwargs):
        J[:, offset] = 1


class GaussianBackground(WPFModelPrototype):
    def __init__(
        self,
        sigma,
        intensity,
        global_center=True,
        x0=0.0,
        y0=0.0,
        name="Gaussian Background",
    ):
        params = {"sigma": Parameter(sigma), "intensity": Parameter(intensity)}
        if global_center:
            self.func = self.global_center_func
            self.jacobian = self.global_center_jacobian
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)
            self.func = self.local_center_func
            self.jacobian = self.local_center_jacobian

        super().__init__(name, params)

    def global_center_func(self, DP: np.ndarray, sigma, level, **kwargs) -> None:
        DP += level * np.exp(kwargs["global_r"] ** 2 / (-2 * sigma**2))

    def global_center_jacobian(
        self, J: np.ndarray, sigma, level, offset: int, **kwargs
    ) -> None:

        exp_expr = np.exp(kwargs["global_r"] ** 2 / (-2 * sigma**2))

        # dF/d(global_x0)
        J[:, 0] += (
            level * (kwargs["xArray"] - kwargs["global_x0"]) * exp_expr / sigma**2
        ).ravel()

        # dF/d(global_y0)
        J[:, 1] += (
            level * (kwargs["yArray"] - kwargs["global_y0"]) * exp_expr / sigma**2
        ).ravel()

        # dF/s(sigma)
        J[:, offset] = (level * kwargs["global_r"] ** 2 * exp_expr / sigma**3).ravel()

        # dF/d(level)
        J[:, offset + 1] = exp_expr.ravel()

    def local_center_func(self, DP: np.ndarray, sigma, level, x0, y0, **kwargs) -> None:
        DP += level * np.exp(
            ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            / (-2 * sigma**2)
        )

    def local_center_jacobian(
        self, J: np.ndarray, sigma, level, x0, y0, offset: int, **kwargs
    ) -> None:

        # dF/s(sigma)
        J[:, offset] = (
            level
            * ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**3
        ).ravel()

        # dF/d(level)
        J[:, offset + 1] = np.exp(
            ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            / (-2 * sigma**2)
        ).ravel()

        # dF/d(x0)
        J[:, offset + 2] = (
            level
            * (kwargs["xArray"] - x0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**2
        ).ravel()

        # dF/d(y0)
        J[:, offset + 3] = (
            level
            * (kwargs["yArray"] - y0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**2
        ).ravel()


class GaussianRing(WPFModelPrototype):
    def __init__(
        self,
        radius,
        sigma,
        intensity,
        global_center=True,
        x0=0.0,
        y0=0.0,
        name="Gaussian Ring",
    ):
        params = {
            "radius": Parameter(radius),
            "sigma": Parameter(sigma),
            "intensity": Parameter(intensity),
        }
        if global_center:
            self.func = self.global_center_func
            self.jacobian = self.global_center_jacobian
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)
            self.func = self.local_center_func
            self.jacobian = self.local_center_jacobian

        super().__init__(name, params)

    def global_center_func(
        self, DP: np.ndarray, radius, sigma, level, **kwargs
    ) -> None:
        DP += level * np.exp((kwargs["global_r"] - radius) ** 2 / (-2 * sigma**2))

    def global_center_jacobian(
        self, J: np.ndarray, radius, sigma, level, offset: int, **kwargs
    ) -> None:

        local_r = radius - kwargs["global_r"]
        clipped_r = np.maximum(local_r, 0.1)

        exp_expr = np.exp(local_r**2 / (-2 * sigma**2))

        # dF/d(global_x0)
        J[:, 0] += (
            level
            * exp_expr
            * (kwargs["xArray"] - kwargs["global_x0"])
            * local_r
            / (sigma**2 * clipped_r)
        ).ravel()

        # dF/d(global_y0)
        J[:, 1] += (
            level
            * exp_expr
            * (kwargs["yArray"] - kwargs["global_y0"])
            * local_r
            / (sigma**2 * clipped_r)
        ).ravel()

        # dF/d(radius)
        J[:, offset] += (-1.0 * level * exp_expr * local_r / (sigma**2)).ravel()

        # dF/d(sigma)
        J[:, offset + 1] = (
            level * local_r ** 2 * exp_expr / sigma**3
        ).ravel()

        # dF/d(level)
        J[:, offset + 2] = exp_expr.ravel()

    def local_center_func(
        self, DP: np.ndarray, radius, sigma, level, x0, y0, **kwargs
    ) -> None:
        local_r = np.hypot(kwargs["xArray"] - x0, kwargs["yArray"] - y0)
        DP += level * np.exp((local_r - radius) ** 2 / (-2 * sigma**2))

    def local_center_jacobian(
        self, J: np.ndarray, radius, sigma, level, x0, y0, offset: int, **kwargs
    ) -> None:
        return NotImplementedError()
        # dF/d(radius)

        # dF/s(sigma)
        J[:, offset] = (
            level
            * ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**3
        ).ravel()

        # dF/d(level)
        J[:, offset + 1] = np.exp(
            ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
            / (-2 * sigma**2)
        ).ravel()

        # dF/d(x0)
        J[:, offset + 2] = (
            level
            * (kwargs["xArray"] - x0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**2
        ).ravel()

        # dF/d(y0)
        J[:, offset + 3] = (
            level
            * (kwargs["yArray"] - y0)
            * np.exp(
                ((kwargs["xArray"] - x0) ** 2 + (kwargs["yArray"] - y0) ** 2)
                / (-2 * sigma**2)
            )
            / sigma**2
        ).ravel()


class SyntheticDiskLattice(WPFModelPrototype):
    def __init__(
        self,
        WPF,
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
        exclude_indices: list = [],
        include_indices: list = None,
        name="Synthetic Disk Lattice",
        verbose=False,
    ):
        self.disk_radius = disk_radius
        self.disk_width = disk_width

        params = {}

        if global_center:
            self.func = self.global_center_func
            self.jacobian = self.global_center_jacobian

            x0 = WPF.static_data["global_x0"]
            y0 = WPF.static_data["global_y0"]
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)
            self.func = self.local_center_func

        params["ux"] = Parameter(ux)
        params["uy"] = Parameter(uy)
        params["vx"] = Parameter(vx)
        params["vy"] = Parameter(vy)

        Q_Nx = WPF.static_data["Q_Nx"]
        Q_Ny = WPF.static_data["Q_Ny"]

        if include_indices is None:
            u_inds, v_inds = np.mgrid[-u_max : u_max + 1, -v_max : v_max + 1]
            self.u_inds = u_inds.ravel()
            self.v_inds = v_inds.ravel()

            delete_mask = np.zeros_like(self.u_inds, dtype=bool)
            for i, (u, v) in enumerate(zip(u_inds.ravel(), v_inds.ravel())):
                x = (
                    x0
                    + (u * params["ux"].initial_value)
                    + (v * params["vx"].initial_value)
                )
                y = (
                    y0
                    + (u * params["uy"].initial_value)
                    + (v * params["vy"].initial_value)
                )
                if [u, v] in exclude_indices:
                    delete_mask[i] = True
                elif (x < 0) or (x > Q_Nx) or (y < 0) or (y > Q_Ny):
                    delete_mask[i] = True
                    if verbose:
                        print(
                            f"Excluding peak [{u},{v}] because it is outside the pattern..."
                        )
                else:
                    params[f"[{u},{v}] Intensity"] = Parameter(intensity_0)

            self.u_inds = self.u_inds[~delete_mask]
            self.v_inds = self.v_inds[~delete_mask]
        else:
            for ind in include_indices:
                params[f"[{ind[0]},{ind[1]}] Intensity"] = Parameter(intensity_0)
            inds = np.array(include_indices)
            self.u_inds = inds[:, 0]
            self.v_inds = inds[:, 1]

        self.refine_radius = refine_radius
        self.refine_width = refine_width
        if refine_radius:
            params["disk radius"] = Parameter(disk_radius)
        if refine_width:
            params["edge width"] = Parameter(disk_width)

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
            # if (x > 0) & (x < kwargs["Q_Nx"]) & (y > 0) & (y < kwargs["Q_Nx"]):
            DP += args[i + 6] / (
                1.0
                + np.exp(
                    4
                    * (
                        np.sqrt(
                            (kwargs["xArray"] - x) ** 2 + (kwargs["yArray"] - y) ** 2
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

            disk_intensity = args[i + 6]

            # if (x > 0) & (x < kwargs["Q_Nx"]) & (y > 0) & (y < kwargs["Q_Nx"]):
            r_disk = np.maximum(
                5e-1,
                np.sqrt((kwargs["xArray"] - x) ** 2 + (kwargs["yArray"] - y) ** 2),
            )

            mask = r_disk < (2 * disk_radius)

            top_exp = mask * np.exp(4 * ((mask * r_disk) - disk_radius) / disk_width)

            # dF/d(global_x0)
            dx = (
                4
                * args[i + 4]
                * (kwargs["xArray"] - x)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # dF/d(global_y0)
            dy = (
                4
                * args[i + 4]
                * (kwargs["yArray"] - y)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # because... reasons, sometimes we get NaN
            # very far from the disk center. let's zero those:
            # dx[np.isnan(dx)] = 0.0
            # dy[np.isnan(dy)] = 0.0

            # insert global positional derivatives
            J[:, 0] += disk_intensity * dx
            J[:, 1] += disk_intensity * dy

            # insert lattice vector derivatives
            J[:, offset] += disk_intensity * u * dx
            J[:, offset + 1] += disk_intensity * u * dy
            J[:, offset + 2] += disk_intensity * v * dx
            J[:, offset + 3] += disk_intensity * v * dy

            # insert intensity derivative
            dI = (mask * (1.0 / (1.0 + top_exp))).ravel()
            # dI[np.isnan(dI)] = 0.0
            J[:, offset + i + 4] = dI

            # insert disk radius derivative
            if self.refine_radius:
                dR = (
                    4.0 * args[i + 4] * top_exp / (disk_width * (1.0 + top_exp) ** 2)
                ).ravel()
                # dR[np.isnan(dR)] = 0.0
                J[:, offset + len(args) + radius_ind] += dR

            if self.refine_width:
                dW = (
                    4.0
                    * args[i + 4]
                    * top_exp
                    * (r_disk - disk_radius)
                    / (disk_width**2 * (1.0 + top_exp) ** 2)
                ).ravel()
                # dW[np.isnan(dW)] = 0.0
                J[:, offset + len(args) - 1] += dW

            # set_trace()


class ComplexOverlapKernelDiskLattice(WPFModelPrototype):
    def __init__(
        self,
        WPF,
        probe_kernel: np.ndarray,
        ux: float,
        uy: float,
        vx: float,
        vy: float,
        u_max: int,
        v_max: int,
        intensity_0: float,
        exclude_indices: list = [],
        name="Complex Overlapped Disk Lattice",
        verbose=False,
    ):

        params = {}

        # if global_center:
        #     self.func = self.global_center_func
        #     self.jacobian = self.global_center_jacobian

        #     x0 = WPF.static_data["global_x0"]
        #     y0 = WPF.static_data["global_y0"]
        # else:
        #     params["x center"] = Parameter(x0)
        #     params["y center"] = Parameter(y0)
        #     self.func = self.local_center_func

        self.probe_kernelFT = np.fft.fft2(probe_kernel)

        params["ux"] = Parameter(ux)
        params["uy"] = Parameter(uy)
        params["vx"] = Parameter(vx)
        params["vy"] = Parameter(vy)

        u_inds, v_inds = np.mgrid[-u_max : u_max + 1, -v_max : v_max + 1]
        self.u_inds = u_inds.ravel()
        self.v_inds = v_inds.ravel()

        delete_mask = np.zeros_like(self.u_inds, dtype=bool)
        Q_Nx = WPF.static_data["Q_Nx"]
        Q_Ny = WPF.static_data["Q_Ny"]

        self.yqArray = np.tile(np.fft.fftfreq(Q_Ny)[np.newaxis, :], (Q_Nx, 1))
        self.xqArray = np.tile(np.fft.fftfreq(Q_Nx)[:, np.newaxis], (1, Q_Ny))

        for i, (u, v) in enumerate(zip(u_inds.ravel(), v_inds.ravel())):
            x = (
                WPF.static_data["global_x0"]
                + (u * params["ux"].initial_value)
                + (v * params["vx"].initial_value)
            )
            y = (
                WPF.static_data["global_y0"]
                + (u * params["uy"].initial_value)
                + (v * params["vy"].initial_value)
            )
            if [u, v] in exclude_indices:
                delete_mask[i] = True
            elif (x < 0) or (x > Q_Nx) or (y < 0) or (y > Q_Ny):
                delete_mask[i] = True
                if verbose:
                    print(
                        f"Excluding peak [{u},{v}] because it is outside the pattern..."
                    )
            else:
                params[f"[{u},{v}] Intensity"] = Parameter(intensity_0)
                if u == 0 and v == 0:
                    params[f"[{u}, {v}] Phase"] = Parameter(
                        0.0, 0.0, 0.0
                    )  # direct beam clamped at zero phase
                else:
                    params[f"[{u}, {v}] Phase"] = Parameter(0.01, -np.pi, np.pi)

        self.u_inds = self.u_inds[~delete_mask]
        self.v_inds = self.v_inds[~delete_mask]

        self.func = self.global_center_func

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

        localDP = np.zeros_like(DP, dtype=np.complex64)

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)

            localDP += (
                args[2 * i + 6]
                * np.exp(1j * args[2 * i + 7])
                * np.abs(
                    np.fft.ifft2(
                        self.probe_kernelFT
                        * np.exp(-2j * np.pi * (self.xqArray * x + self.yqArray * y))
                    )
                )
            )

        DP += np.abs(localDP) ** 2


class KernelDiskLattice(WPFModelPrototype):
    def __init__(
        self,
        WPF,
        probe_kernel: np.ndarray,
        ux: float,
        uy: float,
        vx: float,
        vy: float,
        u_max: int,
        v_max: int,
        intensity_0: float,
        exclude_indices: list = [],
        name="Custom Kernel Disk Lattice",
        verbose=False,
    ):

        params = {}

        # if global_center:
        #     self.func = self.global_center_func
        #     self.jacobian = self.global_center_jacobian

        #     x0 = WPF.static_data["global_x0"]
        #     y0 = WPF.static_data["global_y0"]
        # else:
        #     params["x center"] = Parameter(x0)
        #     params["y center"] = Parameter(y0)
        #     self.func = self.local_center_func

        self.probe_kernelFT = np.fft.fft2(probe_kernel)

        params["ux"] = Parameter(ux)
        params["uy"] = Parameter(uy)
        params["vx"] = Parameter(vx)
        params["vy"] = Parameter(vy)

        u_inds, v_inds = np.mgrid[-u_max : u_max + 1, -v_max : v_max + 1]
        self.u_inds = u_inds.ravel()
        self.v_inds = v_inds.ravel()

        delete_mask = np.zeros_like(self.u_inds, dtype=bool)
        Q_Nx = WPF.static_data["Q_Nx"]
        Q_Ny = WPF.static_data["Q_Ny"]

        self.yqArray = np.tile(np.fft.fftfreq(Q_Ny)[np.newaxis, :], (Q_Nx, 1))
        self.xqArray = np.tile(np.fft.fftfreq(Q_Nx)[:, np.newaxis], (1, Q_Ny))

        for i, (u, v) in enumerate(zip(u_inds.ravel(), v_inds.ravel())):
            x = (
                WPF.static_data["global_x0"]
                + (u * params["ux"].initial_value)
                + (v * params["vx"].initial_value)
            )
            y = (
                WPF.static_data["global_y0"]
                + (u * params["uy"].initial_value)
                + (v * params["vy"].initial_value)
            )
            if [u, v] in exclude_indices:
                delete_mask[i] = True
            elif (x < 0) or (x > Q_Nx) or (y < 0) or (y > Q_Ny):
                delete_mask[i] = True
                if verbose:
                    print(
                        f"Excluding peak [{u},{v}] because it is outside the pattern..."
                    )
            else:
                params[f"[{u},{v}] Intensity"] = Parameter(intensity_0)

        self.u_inds = self.u_inds[~delete_mask]
        self.v_inds = self.v_inds[~delete_mask]

        self.func = self.global_center_func

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

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)

            DP += (
                args[i + 6]
                * np.abs(
                    np.fft.ifft2(
                        self.probe_kernelFT
                        * np.exp(-2j * np.pi * (self.xqArray * x + self.yqArray * y))
                    )
                )
            ) ** 2
