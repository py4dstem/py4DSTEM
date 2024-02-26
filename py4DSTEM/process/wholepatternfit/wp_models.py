from typing import Optional
from enum import Flag, auto
import numpy as np


class WPFModelType(Flag):
    """
    Flags to signify capabilities and other semantics of a Model
    """

    BACKGROUND = auto()

    AMORPHOUS = auto()
    LATTICE = auto()
    MOIRE = auto()

    DUMMY = auto()  # Model has no direct contribution to pattern
    META = auto()  # Model depends on multiple sub-Models


class WPFModel:
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

    def __init__(self, name: str, params: dict, model_type=WPFModelType.DUMMY):
        self.name = name
        self.params = params

        self.nParams = len(params.keys())

        self.hasJacobian = getattr(self, "jacobian", None) is not None

        self.model_type = model_type

    def func(self, DP: np.ndarray, x, **kwargs) -> None:
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
        """
        Object representing a fitting parameter with bounds.

        Can be specified three ways:
        Parameter(initial_value) - Unbounded, with an initial guess
        Parameter(initial_value, deviation) - Bounded within deviation of initial_guess
        Parameter(initial_value, lower_bound, upper_bound) - Both bounds specified
        """
        if hasattr(initial_value, "__iter__"):
            if len(initial_value) == 2:
                initial_value = (
                    initial_value[0],
                    initial_value[0] - initial_value[1],
                    initial_value[0] + initial_value[1],
                )
            self.set_params(*initial_value)
        else:
            self.set_params(initial_value, lower_bound, upper_bound)

        # Store a dummy offset. This must be set by WPF during setup
        # This stores the index in the master parameter and Jacobian arrays
        # corresponding to this parameter
        self.offset = np.nan

    def set_params(
        self,
        initial_value,
        lower_bound,
        upper_bound,
    ):
        self.initial_value = initial_value
        self.lower_bound = lower_bound if lower_bound is not None else -np.inf
        self.upper_bound = upper_bound if upper_bound is not None else np.inf

    def __str__(self):
        return f"Value: {self.initial_value} (Range: {self.lower_bound},{self.upper_bound})"

    def __repr__(self):
        return f"Value: {self.initial_value} (Range: {self.lower_bound},{self.upper_bound})"


class _BaseModel(WPFModel):
    """
    Model object used by the WPF class as a container for the global Parameters.

    **This object should not be instantiated directly.**
    """

    def __init__(self, x0, y0, name="Globals"):
        params = {"x center": Parameter(x0), "y center": Parameter(y0)}

        super().__init__(name, params, model_type=WPFModelType.DUMMY)

    def func(self, DP: np.ndarray, x, **kwargs) -> None:
        pass

    def jacobian(self, J: np.ndarray, *args, **kwargs) -> None:
        pass


class DCBackground(WPFModel):
    """
    Model representing constant background intensity.

    Parameters
    ----------
    background_value
        Background intensity value.
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    """

    def __init__(self, background_value=0.0, name="DC Background"):
        params = {"DC Level": Parameter(background_value)}

        super().__init__(name, params, model_type=WPFModelType.BACKGROUND)

    def func(self, DP: np.ndarray, x, **kwargs) -> None:
        DP += x[self.params["DC Level"].offset]

    def jacobian(self, J: np.ndarray, *args, **kwargs):
        J[:, self.params["DC Level"].offset] = 1


class GaussianBackground(WPFModel):
    """
    Model representing a 2D Gaussian intensity distribution

    Parameters
    ----------
    WPF: WholePatternFit
        Parent WPF object
    sigma
        parameter specifying width of the Gaussian
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    intensity
        parameter specifying intensity of the Gaussian
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    global_center: bool
        If True, uses same center coordinate as the global model
        If False, uses an independent center
    x0, y0:
        Center coordinates of model for local origin
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    """

    def __init__(
        self,
        WPF,
        sigma,
        intensity,
        global_center=True,
        x0=0.0,
        y0=0.0,
        name="Gaussian Background",
    ):
        params = {"sigma": Parameter(sigma), "intensity": Parameter(intensity)}
        if global_center:
            params["x center"] = WPF.coordinate_model.params["x center"]
            params["y center"] = WPF.coordinate_model.params["y center"]
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)

        super().__init__(name, params, model_type=WPFModelType.BACKGROUND)

    def func(self, DP: np.ndarray, x: np.ndarray, **kwargs) -> None:
        sigma = x[self.params["sigma"].offset]
        level = x[self.params["intensity"].offset]

        r = kwargs["parent"]._get_distance(
            x, self.params["x center"], self.params["y center"]
        )

        DP += level * np.exp(r**2 / (-2 * sigma**2))

    def jacobian(self, J: np.ndarray, x: np.ndarray, **kwargs) -> None:
        sigma = x[self.params["sigma"].offset]
        level = x[self.params["intensity"].offset]
        x0 = x[self.params["x center"].offset]
        y0 = x[self.params["y center"].offset]

        r = kwargs["parent"]._get_distance(
            x, self.params["x center"], self.params["y center"]
        )
        exp_expr = np.exp(r**2 / (-2 * sigma**2))

        # dF/d(x0)
        J[:, self.params["x center"].offset] += (
            level * (kwargs["xArray"] - x0) * exp_expr / sigma**2
        ).ravel()

        # dF/d(y0)
        J[:, self.params["y center"].offset] += (
            level * (kwargs["yArray"] - y0) * exp_expr / sigma**2
        ).ravel()

        # dF/s(sigma)
        J[:, self.params["sigma"].offset] += (
            level * r**2 * exp_expr / sigma**3
        ).ravel()

        # dF/d(level)
        J[:, self.params["intensity"].offset] += exp_expr.ravel()


class GaussianRing(WPFModel):
    """
    Model representing a halo with Gaussian falloff

    Parameters
    ----------
    WPF: WholePatternFit
        parent fitting object
    radius:
        radius of halo
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    sigma:
        width of Gaussian falloff
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    intensity:
        Intensity of the halo
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    global_center: bool
        If True, uses same center coordinate as the global model
        If False, uses an independent center
    x0, y0:
        Center coordinates of model for local origin
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    """

    def __init__(
        self,
        WPF,
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
            "x center": (
                WPF.coordinate_model.params["x center"]
                if global_center
                else Parameter(x0)
            ),
            "y center": (
                WPF.coordinate_model.params["y center"]
                if global_center
                else Parameter(y0)
            ),
        }

        super().__init__(name, params, model_type=WPFModelType.AMORPHOUS)

    def func(self, DP: np.ndarray, x: np.ndarray, **kwargs) -> None:
        radius = x[self.params["radius"].offset]
        sigma = x[self.params["sigma"].offset]
        level = x[self.params["level"].offset]

        r = kwargs["parent"]._get_distance(
            x, self.params["x center"], self.params["y center"]
        )

        DP += level * np.exp((r - radius) ** 2 / (-2 * sigma**2))

    def jacobian(self, J: np.ndarray, x: np.ndarray, **kwargs) -> None:
        radius = x[self.params["radius"].offset]
        sigma = x[self.params["sigma"].offset]
        level = x[self.params["level"].offset]

        x0 = x[self.params["x center"].offset]
        y0 = x[self.params["y center"].offset]
        r = kwargs["parent"]._get_distance(
            x, self.params["x center"], self.params["y center"]
        )

        local_r = radius - r
        clipped_r = np.maximum(local_r, 0.1)

        exp_expr = np.exp(local_r**2 / (-2 * sigma**2))

        # dF/d(x0)
        J[:, self.params["x center"].offset] += (
            level
            * exp_expr
            * (kwargs["xArray"] - x0)
            * local_r
            / (sigma**2 * clipped_r)
        ).ravel()

        # dF/d(y0)
        J[:, self.parans["y center"].offset] += (
            level
            * exp_expr
            * (kwargs["yArray"] - y0)
            * local_r
            / (sigma**2 * clipped_r)
        ).ravel()

        # dF/d(radius)
        J[:, self.params["radius"].offset] += (
            -1.0 * level * exp_expr * local_r / (sigma**2)
        ).ravel()

        # dF/d(sigma)
        J[:, self.params["sigma"].offset] += (
            level * local_r**2 * exp_expr / sigma**3
        ).ravel()

        # dF/d(intensity)
        J[:, self.params["intensity"].offset] += exp_expr.ravel()


class SyntheticDiskLattice(WPFModel):
    """
    Model representing a lattice of diffraction disks with a soft edge

    Parameters
    ----------

    WPF: WholePatternFit
        parent fitting object
    ux,uy,vx,vy
        x and y components of the lattice vectors u and v.
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    disk_radius
        Radius of each diffraction disk.
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    disk_width
        Width of the smooth falloff at the edge of the disk
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    u_max, v_max
        Maximum lattice indices to include in the pattern.
        Disks outside the pattern are automatically clipped.
    intensity_0
        Initial intensity for each diffraction disk.
        Each disk intensity is an independent fit variable in the final model
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    refine_radius: bool
        Flag whether disk radius is made a fitting parameter
    refine_width: bool
        Flag whether disk edge width is made a fitting parameter
    global_center: bool
        If True, uses same center coordinate as the global model
        If False, uses an independent center
    x0, y0:
        Center coordinates of model for local origin
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    exclude_indices: list
        Indices to exclude from the pattern
    include_indices: list
        If specified, only the indices in the list are added to the pattern
    """

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
            params["x center"] = WPF.coordinate_model.params["x center"]
            params["y center"] = WPF.coordinate_model.params["y center"]
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)

        x0 = params["x center"].initial_value
        y0 = params["y center"].initial_value

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

        super().__init__(name, params, model_type=WPFModelType.LATTICE)

    def func(self, DP: np.ndarray, x: np.ndarray, **static_data) -> None:
        x0 = x[self.params["x center"].offset]
        y0 = x[self.params["y center"].offset]
        ux = x[self.params["ux"].offset]
        uy = x[self.params["uy"].offset]
        vx = x[self.params["vx"].offset]
        vy = x[self.params["vy"].offset]

        disk_radius = (
            x[self.params["disk radius"].offset]
            if self.refine_radius
            else self.disk_radius
        )

        disk_width = (
            x[self.params["edge width"].offset]
            if self.refine_width
            else self.disk_width
        )

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x_pos = x0 + (u * ux) + (v * vx)
            y_pos = y0 + (u * uy) + (v * vy)

            DP += x[self.params[f"[{u},{v}] Intensity"].offset] / (
                1.0
                + np.exp(
                    np.minimum(
                        4
                        * (
                            np.sqrt(
                                (static_data["xArray"] - x_pos) ** 2
                                + (static_data["yArray"] - y_pos) ** 2
                            )
                            - disk_radius
                        )
                        / disk_width,
                        20,
                    )
                )
            )

    def jacobian(self, J: np.ndarray, x: np.ndarray, **static_data) -> None:
        x0 = x[self.params["x center"].offset]
        y0 = x[self.params["y center"].offset]
        ux = x[self.params["ux"].offset]
        uy = x[self.params["uy"].offset]
        vx = x[self.params["vx"].offset]
        vy = x[self.params["vy"].offset]
        WPF = static_data["parent"]

        r = np.maximum(
            5e-1, WPF._get_distance(x, self.params["x center"], self.params["y center"])
        )

        disk_radius = (
            x[self.params["disk radius"].offset]
            if self.refine_radius
            else self.disk_radius
        )

        disk_width = (
            x[self.params["edge width"].offset]
            if self.refine_width
            else self.disk_width
        )

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x_pos = x0 + (u * ux) + (v * vx)
            y_pos = y0 + (u * uy) + (v * vy)

            disk_intensity = x[self.params[f"[{u},{v}] Intensity"].offset]

            r_disk = np.maximum(
                5e-1,
                np.sqrt(
                    (static_data["xArray"] - x_pos) ** 2
                    + (static_data["yArray"] - y_pos) ** 2
                ),
            )

            mask = r_disk < (2 * disk_radius)

            top_exp = mask * np.exp(
                np.minimum(30, 4 * ((mask * r_disk) - disk_radius) / disk_width)
            )

            # dF/d(x0)
            dx = (
                4
                * disk_intensity
                * (static_data["xArray"] - x_pos)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # dF/d(y0)
            dy = (
                4
                * disk_intensity
                * (static_data["yArray"] - y_pos)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # insert center position derivatives
            J[:, self.params["x center"].offset] += disk_intensity * dx
            J[:, self.params["y center"].offset] += disk_intensity * dy

            # insert lattice vector derivatives
            J[:, self.params["ux"].offset] += disk_intensity * u * dx
            J[:, self.params["uy"].offset] += disk_intensity * u * dy
            J[:, self.params["vx"].offset] += disk_intensity * v * dx
            J[:, self.params["vy"].offset] += disk_intensity * v * dy

            # insert intensity derivative
            dI = (mask * (1.0 / (1.0 + top_exp))).ravel()
            J[:, self.params[f"[{u},{v}] Intensity"].offset] += dI

            # insert disk radius derivative
            if self.refine_radius:
                dR = (
                    4.0 * disk_intensity * top_exp / (disk_width * (1.0 + top_exp) ** 2)
                ).ravel()
                J[:, self.params["disk radius"].offset] += dR

            if self.refine_width:
                dW = (
                    4.0
                    * disk_intensity
                    * top_exp
                    * (r_disk - disk_radius)
                    / (disk_width**2 * (1.0 + top_exp) ** 2)
                ).ravel()
                J[:, self.params["edge width"].offset] += dW


class SyntheticDiskMoire(WPFModel):
    """
    Model of diffraction disks arising from interference between two lattices.

    The Moire unit cell is determined automatically using the two input lattices.

    Parameters
    ----------
    WPF: WholePatternFit
        parent fitting object
    lattice_a, lattice_b: SyntheticDiskLattice
        parent lattices for the Moire
    intensity_0
        Initial guess of Moire disk intensity
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    decorated_peaks: list
        When specified, only the reflections in the list are decorated with Moire spots
        If not specified, all peaks are decorated
    link_moire_disk_intensities: bool
        When False, each Moire disk has an independently fit intensity
        When True, Moire disks arising from the same order of parent reflection share
        the same intensity
    link_disk_parameters: bool
        When True, edge_width and disk_radius are inherited from lattice_a
    refine_width: bool
        Flag whether disk edge width is a fit variable
    edge_width
        Width of the soft edge of the diffraction disk.
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    refine_radius: bool
        Flag whether disk radius is a fit variable
    disk radius
        Radius of the diffraction disks
        Specified as initial_value, (initial_value, deviation), or
            (initial_value, lower_bound, upper_bound). See
            Parameter documentation for details.
    """

    def __init__(
        self,
        WPF,
        lattice_a: SyntheticDiskLattice,
        lattice_b: SyntheticDiskLattice,
        intensity_0: float,
        decorated_peaks: list = None,
        link_moire_disk_intensities: bool = False,
        link_disk_parameters: bool = True,
        refine_width: bool = True,
        edge_width: list = None,
        refine_radius: bool = True,
        disk_radius: list = None,
        name: str = "Moire Lattice",
    ):
        # ensure both models share the same center coordinate
        if (lattice_a.params["x center"] is not lattice_b.params["x center"]) or (
            lattice_a.params["y center"] is not lattice_b.params["y center"]
        ):
            raise ValueError(
                "The center coordinates for each model must be linked, "
                "either by passing global_center=True or linking after instantiation."
            )

        self.lattice_a = lattice_a
        self.lattice_b = lattice_b

        # construct a 2x4 matrix "M" that transforms the parent lattices into
        # the moire lattice vectors

        lat_ab = self._get_parent_lattices(lattice_a, lattice_b)

        # pick the pairing that gives the smallest unit cell
        test_peaks = np.stack((lattice_b.u_inds, lattice_b.v_inds), axis=1)
        tests = np.stack(
            [
                np.hstack((np.eye(2), np.vstack((b1, b2))))
                for b1 in test_peaks
                for b2 in test_peaks
                if not np.allclose(b1, b2)
            ],
            axis=0,
        )
        # choose only cells where the two unit vectors are not nearly parallel,
        # and penalize cells with large discrepancy in lattce vector length
        lat_m = tests @ lat_ab
        a_dot_b = (
            np.sum(lat_m[:, 0] * lat_m[:, 1], axis=1)
            / np.minimum(
                np.linalg.norm(lat_m[:, 0], axis=1), np.linalg.norm(lat_m[:, 1], axis=1)
            )
            ** 2
        )
        tests = tests[
            np.abs(a_dot_b) < 0.9
        ]  # this factor of 0.9 sets the parallel cutoff
        # with the parallel vectors filtered, pick the cell with the smallest volume
        lat_m = tests @ lat_ab
        V = np.sum(
            lat_m[:, 0]
            * np.cross(
                np.hstack((lat_m[:, 1], np.zeros((lat_m.shape[0],))[:, None])),
                [0, 0, 1],
            )[:, :2],
            axis=1,
        )
        M = tests[np.argmin(np.abs(V))]

        # ensure the moire vectors are less 90 deg apart
        if np.arccos(
            ((M @ lat_ab)[0] @ (M @ lat_ab)[1])
            / (np.linalg.norm((M @ lat_ab)[0]) * np.linalg.norm((M @ lat_ab)[1]))
        ) > np.radians(90):
            M[1] *= -1.0

        # ensure they are right-handed
        if np.cross(*(M @ lat_ab)) < 0.0:
            M = np.flipud(np.eye(2)) @ M

        # store moire construction
        self.moire_matrix = M

        # generate the indices of each peak, then find unique peaks
        if decorated_peaks is not None:
            decorated_peaks = np.array(decorated_peaks)
            parent_peaks = np.vstack(
                (
                    np.concatenate(
                        (decorated_peaks, np.zeros_like(decorated_peaks)), axis=1
                    ),
                    np.concatenate(
                        (np.zeros_like(decorated_peaks), decorated_peaks), axis=1
                    ),
                )
            )
        else:
            parent_peaks = np.vstack(
                (
                    np.concatenate(
                        (
                            np.stack((lattice_a.u_inds, lattice_a.v_inds), axis=1),
                            np.zeros((lattice_a.u_inds.shape[0], 2)),
                        ),
                        axis=1,
                    ),
                    np.concatenate(
                        (
                            np.zeros((lattice_b.u_inds.shape[0], 2)),
                            np.stack((lattice_b.u_inds, lattice_b.v_inds), axis=1),
                        ),
                        axis=1,
                    ),
                )
            )

        # trial indices for moire peaks
        mx, my = np.mgrid[-1:2, -1:2]
        moire_peaks = np.stack([mx.ravel(), my.ravel()], axis=1)[1:-1]

        # construct a giant index array with columns a_h a_k b_h b_k m_h m_k
        parent_expanded = np.zeros((parent_peaks.shape[0], 6))
        parent_expanded[:, :4] = parent_peaks
        moire_expanded = np.zeros((moire_peaks.shape[0], 6))
        moire_expanded[:, 4:] = moire_peaks

        all_indices = (
            parent_expanded[:, None, :] + moire_expanded[None, :, :]
        ).reshape(-1, 6)

        lat_abm = np.vstack((lat_ab, M @ lat_ab))

        all_peaks = all_indices @ lat_abm

        _, idx_unique = np.unique(all_peaks, axis=0, return_index=True)

        all_indices = all_indices[idx_unique]

        # remove peaks outside of pattern
        Q_Nx = WPF.static_data["Q_Nx"]
        Q_Ny = WPF.static_data["Q_Ny"]
        all_peaks = all_indices @ lat_abm
        all_peaks[:, 0] += lattice_a.params["x center"].initial_value
        all_peaks[:, 1] += lattice_a.params["y center"].initial_value
        delete_mask = np.logical_or.reduce(
            [
                all_peaks[:, 0] < 0.0,
                all_peaks[:, 0] >= Q_Nx,
                all_peaks[:, 1] < 0.0,
                all_peaks[:, 1] >= Q_Ny,
            ]
        )
        all_indices = all_indices[~delete_mask]

        # remove spots that coincide with primary peaks
        parent_spots = parent_peaks @ lat_ab
        self.moire_indices_uvm = np.array(
            [idx for idx in all_indices if (idx @ lat_abm) not in parent_spots]
        )

        self.link_moire_disk_intensities = link_moire_disk_intensities
        if link_moire_disk_intensities:
            # each order of parent reflection has a separate moire intensity
            max_order = int(np.max(np.abs(self.moire_indices_uvm[:, :4])))

            params = {
                f"Order {n} Moire Intensity": Parameter(intensity_0)
                for n in range(max_order + 1)
            }
        else:
            params = {
                f"a ({ax},{ay}), b ({bx},{by}), moire ({mx},{my}) Intensity": Parameter(
                    intensity_0
                )
                for ax, ay, bx, by, mx, my in self.moire_indices_uvm
            }

        params["x center"] = lattice_a.params["x center"]
        params["y center"] = lattice_a.params["y center"]

        # add disk edge and width parameters if needed
        if link_disk_parameters:
            if (lattice_a.refine_width) and (lattice_b.refine_width):
                self.refine_width = True
                params["edge width"] = lattice_a.params["edge width"]
            if (lattice_a.refine_radius) and (lattice_b.refine_radius):
                self.refine_radius = True
                params["disk radius"] = lattice_a.params["disk radius"]
        else:
            self.refine_width = refine_width
            if self.refine_width:
                params["edge width"] = Parameter(edge_width)

            self.refine_radius = refine_radius
            if self.refine_radius:
                params["disk radius"] = Parameter(disk_radius)

        # store some data that helps compute the derivatives
        selector_matrices = np.eye(8).reshape(-1, 4, 2)
        selector_parameters = [
            self.lattice_a.params["ux"],
            self.lattice_a.params["uy"],
            self.lattice_a.params["vx"],
            self.lattice_a.params["vy"],
            self.lattice_b.params["ux"],
            self.lattice_b.params["uy"],
            self.lattice_b.params["vx"],
            self.lattice_b.params["vy"],
        ]
        self.parent_vector_selectors = [
            (p, m) for p, m in zip(selector_parameters, selector_matrices)
        ]

        super().__init__(
            name,
            params,
            model_type=WPFModelType.META | WPFModelType.MOIRE,
        )

    def _get_parent_lattices(self, lattice_a, lattice_b):
        lat_a = np.array(
            [
                [
                    lattice_a.params["ux"].initial_value,
                    lattice_a.params["uy"].initial_value,
                ],
                [
                    lattice_a.params["vx"].initial_value,
                    lattice_a.params["vy"].initial_value,
                ],
            ]
        )

        lat_b = np.array(
            [
                [
                    lattice_b.params["ux"].initial_value,
                    lattice_b.params["uy"].initial_value,
                ],
                [
                    lattice_b.params["vx"].initial_value,
                    lattice_b.params["vy"].initial_value,
                ],
            ]
        )

        return np.vstack((lat_a, lat_b))

    def func(self, DP: np.ndarray, x: np.ndarray, **static_data):
        # construct the moire unit cell from the current vectors
        # of the two parent lattices

        lat_ab = self._get_parent_lattices(self.lattice_a, self.lattice_b)
        lat_abm = np.vstack((lat_ab, self.moire_matrix @ lat_ab))

        # grab shared parameters
        disk_radius = (
            x[self.params["disk radius"].offset]
            if self.refine_radius
            else self.disk_radius
        )

        disk_width = (
            x[self.params["edge width"].offset]
            if self.refine_width
            else self.disk_width
        )

        # compute positions of each moire peak
        positions = self.moire_indices_uvm @ lat_abm
        positions += np.array(
            [x[self.params["x center"].offset], x[self.params["y center"].offset]]
        )

        for (x_pos, y_pos), indices in zip(positions, self.moire_indices_uvm):
            # Each peak has an intensity based on the max index of parent lattice
            # which it decorates
            order = int(np.max(np.abs(indices[:4])))

            if self.link_moire_disk_intensities:
                intensity = x[self.params[f"Order {order} Moire Intensity"].offset]
            else:
                ax, ay, bx, by, mx, my = indices
                intensity = x[
                    self.params[
                        f"a ({ax},{ay}), b ({bx},{by}), moire ({mx},{my}) Intensity"
                    ].offset
                ]

            DP += intensity / (
                1.0
                + np.exp(
                    np.minimum(
                        4
                        * (
                            np.sqrt(
                                (static_data["xArray"] - x_pos) ** 2
                                + (static_data["yArray"] - y_pos) ** 2
                            )
                            - disk_radius
                        )
                        / disk_width,
                        20,
                    )
                )
            )

    def jacobian(self, J: np.ndarray, x: np.ndarray, **static_data):
        # construct the moire unit cell from the current vectors
        # of the two parent lattices
        lat_ab = self._get_parent_lattices(self.lattice_a, self.lattice_b)
        lat_abm = np.vstack((lat_ab, self.moire_matrix @ lat_ab))

        # grab shared parameters
        disk_radius = (
            x[self.params["disk radius"].offset]
            if self.refine_radius
            else self.disk_radius
        )

        disk_width = (
            x[self.params["edge width"].offset]
            if self.refine_width
            else self.disk_width
        )

        # distance from center coordinate
        r = np.maximum(
            5e-1,
            static_data["parent"]._get_distance(
                x, self.params["x center"], self.params["y center"]
            ),
        )

        # compute positions of each moire peak
        positions = self.moire_indices_uvm @ lat_abm
        positions += np.array(
            [x[self.params["x center"].offset], x[self.params["y center"].offset]]
        )

        for (x_pos, y_pos), indices in zip(positions, self.moire_indices_uvm):
            # Each peak has an intensity based on the max index of parent lattice
            # which it decorates
            if self.link_moire_disk_intensities:
                order = int(np.max(np.abs(indices[:4])))
                intensity_idx = self.params[f"Order {order} Moire Intensity"].offset
            else:
                ax, ay, bx, by, mx, my = indices
                intensity_idx = self.params[
                    f"a ({ax},{ay}), b ({bx},{by}), moire ({mx},{my}) Intensity"
                ].offset
            disk_intensity = x[intensity_idx]

            r_disk = np.maximum(
                5e-1,
                np.sqrt(
                    (static_data["xArray"] - x_pos) ** 2
                    + (static_data["yArray"] - y_pos) ** 2
                ),
            )

            mask = r_disk < (2 * disk_radius)

            # clamp the argument of the exponent at a very large finite value
            top_exp = mask * np.exp(
                np.minimum(30, 4 * ((mask * r_disk) - disk_radius) / disk_width)
            )

            # dF/d(x0)
            dx = (
                4
                * disk_intensity
                * (static_data["xArray"] - x_pos)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # dF/d(y0)
            dy = (
                4
                * disk_intensity
                * (static_data["yArray"] - y_pos)
                * top_exp
                / ((1.0 + top_exp) ** 2 * disk_width * r)
            ).ravel()

            # insert center position derivatives
            J[:, self.params["x center"].offset] += disk_intensity * dx
            J[:, self.params["y center"].offset] += disk_intensity * dy

            # insert lattice vector derivatives
            for par, mat in self.parent_vector_selectors:
                # find the x and y derivatives of the position of this
                # disk in terms of each of the parent lattice vectors
                d_abm = np.vstack((mat, self.moire_matrix @ mat))
                d_param = indices @ d_abm
                J[:, par.offset] += disk_intensity * (d_param[0] * dx + d_param[1] * dy)

            # insert intensity derivative
            dI = (mask * (1.0 / (1.0 + top_exp))).ravel()
            J[:, intensity_idx] += dI

            # insert disk radius derivative
            if self.refine_radius:
                dR = (
                    4.0 * disk_intensity * top_exp / (disk_width * (1.0 + top_exp) ** 2)
                ).ravel()
                J[:, self.params["disk radius"].offset] += dR

            if self.refine_width:
                dW = (
                    4.0
                    * disk_intensity
                    * top_exp
                    * (r_disk - disk_radius)
                    / (disk_width**2 * (1.0 + top_exp) ** 2)
                ).ravel()
                J[:, self.params["edge width"].offset] += dW


class ComplexOverlapKernelDiskLattice(WPFModel):
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
        global_center: bool = True,
        x0=0.0,
        y0=0.0,
        name="Complex Overlapped Disk Lattice",
        verbose=False,
    ):
        raise NotImplementedError(
            "This model type has not been updated for use with the new architecture."
        )

        params = {}

        self.probe_kernelFT = np.fft.fft2(probe_kernel)

        if global_center:
            params["x center"] = WPF.coordinate_model.params["x center"]
            params["y center"] = WPF.coordinate_model.params["y center"]
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)

        x0 = params["x center"].initial_value
        y0 = params["y center"].initial_value

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

        super().__init__(name, params, model_type=WPFModelType.LATTICE)

    def func(self, DP: np.ndarray, x_fit, **kwargs) -> None:
        x0 = x_fit[self.params["x center"].offset]
        y0 = x_fit[self.params["y center"].offset]
        ux = x_fit[self.params["ux"].offset]
        uy = x_fit[self.params["uy"].offset]
        vx = x_fit[self.params["vx"].offset]
        vy = x_fit[self.params["vy"].offset]

        localDP = np.zeros_like(DP, dtype=np.complex64)

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)

            localDP += (
                x_fit[self.params[f"[{u},{v}] Intensity"].offset]
                * np.exp(1j * x_fit[self.params[f"[{u},{v}] Phase"].offset])
                * np.abs(
                    np.fft.ifft2(
                        self.probe_kernelFT
                        * np.exp(-2j * np.pi * (self.xqArray * x + self.yqArray * y))
                    )
                )
            )

        DP += np.abs(localDP) ** 2


class KernelDiskLattice(WPFModel):
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
        global_center: bool = True,
        x0=0.0,
        y0=0.0,
        name="Custom Kernel Disk Lattice",
        verbose=False,
    ):
        params = {}

        self.probe_kernelFT = np.fft.fft2(probe_kernel)

        if global_center:
            params["x center"] = WPF.coordinate_model.params["x center"]
            params["y center"] = WPF.coordinate_model.params["y center"]
        else:
            params["x center"] = Parameter(x0)
            params["y center"] = Parameter(y0)

        x0 = params["x center"].initial_value
        y0 = params["y center"].initial_value

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
            x = x0 + (u * params["ux"].initial_value) + (v * params["vx"].initial_value)
            y = y0 + (u * params["uy"].initial_value) + (v * params["vy"].initial_value)
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

        super().__init__(name, params, model_type=WPFModelType.LATTICE)

    def func(self, DP: np.ndarray, x_fit: np.ndarray, **static_data) -> None:
        x0 = x_fit[self.params["x center"].offset]
        y0 = x_fit[self.params["y center"].offset]
        ux = x_fit[self.params["ux"].offset]
        uy = x_fit[self.params["uy"].offset]
        vx = x_fit[self.params["vx"].offset]
        vy = x_fit[self.params["vy"].offset]

        for i, (u, v) in enumerate(zip(self.u_inds, self.v_inds)):
            x = x0 + (u * ux) + (v * vx)
            y = y0 + (u * uy) + (v * vy)

            DP += (
                x_fit[self.params[f"[{u},{v}] Intensity"].offset]
                * np.abs(
                    np.fft.ifft2(
                        self.probe_kernelFT
                        * np.exp(-2j * np.pi * (self.xqArray * x + self.yqArray * y))
                    )
                )
            ) ** 2
