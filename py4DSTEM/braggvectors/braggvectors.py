# Defines the BraggVectors class

from py4DSTEM.data import Data
from emdfile import Custom, PointListArray, PointList, Metadata
from py4DSTEM.braggvectors.braggvector_methods import BraggVectorMethods
from os.path import basename
import numpy as np
from warnings import warn


class BraggVectors(Custom, BraggVectorMethods, Data):
    """
    Stores localized bragg scattering positions and intensities
    for a 4D-STEM datacube.

    Raw (detector coordinate) vectors are accessible as

        >>> braggvectors.raw[ scan_x, scan_y ]

    and calibrated vectors as

        >>> braggvectors.cal[ scan_x, scan_y ]

    To set which calibrations are being applied, call

        >>> braggvectors.setcal(
        >>>     center = bool,
        >>>     ellipse = bool,
        >>>     pixel = bool,
        >>>     rotate = bool
        >>> )

    If .setcal is not called, calibrations will be automatically selected based
    based on the contents of the instance's `calibrations` property. The
    calibrations performed in the last call to `braggvectors.cal` are exposed as

        >>> braggvectors.calstate

    After grabbing some vectors

        >>> vects = braggvectors.raw[ scan_x,scan_y ]

    the values themselves are accessible as

        >>> vects.qx,vects.qy,vects.I
        >>> vects['qx'],vects['qy'],vects['intensity']

    Alternatively, you can access the centered vectors in pixel units with

        >>> vects.get_vectors(
        >>>     scan_x,
        >>>     scan_y,
        >>>     center = bool,
        >>>     ellipse = bool,
        >>>     pixel = bool,
        >>>     rotate = bool
        >>> )

    which will return the vectors at scan position (scan_x,scan_y) with the
    requested calibrations applied.
    """

    def __init__(
        self, Rshape, Qshape, name="braggvectors", verbose=False, calibration=None
    ):
        Custom.__init__(self, name=name)
        Data.__init__(self, calibration=calibration)

        self.Rshape = Rshape
        self.Qshape = Qshape
        self.verbose = verbose

        self._v_uncal = PointListArray(
            dtype=[("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)],
            shape=Rshape,
            name="_v_uncal",
        )

        # initial calibration state
        self._calstate = {
            "center": False,
            "ellipse": False,
            "pixel": False,
            "rotate": False,
        }

        # register with calibrations
        self.calibration.register_target(self)

        # setup vector getters
        self._set_raw_vector_getter()
        self._set_cal_vector_getter()

    # set new raw vectors
    def set_raw_vectors(self, x):
        """Given some PointListArray x of the correct shape, sets this to the raw vectors"""
        assert isinstance(
            x, PointListArray
        ), f"Raw vectors must be set to a PointListArray, not type {type(x)}"
        assert x.shape == self.Rshape, "Shapes don't match!"
        self._v_uncal = x
        self._set_raw_vector_getter()
        self._set_cal_vector_getter()

    # calibration state, vector getters

    @property
    def calstate(self):
        return self._calstate

    def _set_raw_vector_getter(self):
        self._raw_vector_getter = RawVectorGetter(braggvects=self)

    def _set_cal_vector_getter(self):
        self._cal_vector_getter = CalibratedVectorGetter(braggvects=self)

    # shape
    @property
    def shape(self):
        return self.Rshape

    # raw vectors

    @property
    def raw(self):
        """
        Calling

            >>> raw[ scan_x, scan_y ]

        returns those bragg vectors.
        """
        # use the vector getter to grab the vector
        return self._raw_vector_getter

    # calibrated vectors

    @property
    def cal(self):
        """
        Calling

            >>> cal[ scan_x, scan_y ]

        retrieves data.  Use `.setcal` to set the calibrations to be applied, or
        `.calstate` to see which calibrations are currently set.  Calibrations
        are initially all set to False.  Call `.setcal()` (with no arguments)
        to automatically detect which calibrations are present and apply those.
        """
        # retrieve the getter and return
        return self._cal_vector_getter

    # set calibration state

    def setcal(
        self,
        center=None,
        ellipse=None,
        pixel=None,
        rotate=None,
    ):
        """
        Calling

            >>> braggvectors.setcal(
            >>>     center = bool,
            >>>     ellipse = bool,
            >>>     pixel = bool,
            >>>     rotate = bool,
            >>> )

        sets the calibrations that will be applied to vectors subsequently
        retrieved with

            >>> braggvectors.cal[ scan_x, scan_y ]

        Any arguments left as `None` will be automatically set based on
        the calibration measurements available.
        """

        # check for calibrations
        try:
            c = self.calibration
        # if no calibrations are found, print a warning and set all to False
        except Exception:
            warn("No calibrations found at .calibration; setting all cals to False")
            self._calstate = {
                "center": False,
                "ellipse": False,
                "pixel": False,
                "rotate": False,
            }
            return

        # autodetect
        if center is None:
            center = False if c.get_origin() is None else True
        if ellipse is None:
            ellipse = False if c.get_ellipse() is None else True
        if pixel is None:
            pixel = False if c.get_Q_pixel_size() == 1 else True
        if rotate is None:
            rotate = False if c.get_QR_rotation() is None else True

        # validate requested state
        if center:
            assert c.get_origin() is not None, "Requested calibration not found"
        if ellipse:
            assert c.get_ellipse() is not None, "Requested calibration not found"
        if pixel:
            assert c.get_Q_pixel_size() is not None, "Requested calibration not found"
        if rotate:
            assert c.get_QR_rotation() is not None, "Requested calibration not found"

        # set the calibrations
        self._calstate = {
            "center": center,
            "ellipse": ellipse,
            "pixel": pixel,
            "rotate": rotate,
        }
        if self.verbose:
            print("current calibration state: ", self.calstate)
        pass

    def calibrate(self):
        """
        Autoupdate the calstate when relevant calibrations are set
        """
        self.setcal()

    # vector getter method

    def get_vectors(self, scan_x, scan_y, center, ellipse, pixel, rotate):
        """
        Returns the bragg vectors at the specified scan position with
        the specified calibration state.

        Parameters
        ----------
        scan_x : int
        scan_y : int
        center : bool
        ellipse : bool
        pixel : bool
        rotate : bool

        Returns
        -------
        vectors : BVects
        """

        ans = self._v_uncal[scan_x, scan_y].data
        ans = self.cal._transform(
            data=ans,
            cal=self.calibration,
            scanxy=(scan_x, scan_y),
            center=center,
            ellipse=ellipse,
            pixel=pixel,
            rotate=rotate,
        )
        return BVects(ans)

    # copy
    def copy(self, name=None):
        name = name if name is not None else self.name + "_copy"
        braggvector_copy = BraggVectors(
            self.Rshape, self.Qshape, name=name, calibration=self.calibration.copy()
        )

        braggvector_copy.set_raw_vectors(self._v_uncal.copy())
        for k in self.metadata.keys():
            braggvector_copy.metadata = self.metadata[k].copy()
        braggvector_copy.setcal()
        return braggvector_copy

    # write

    def to_h5(self, group):
        """Constructs the group, adds the bragg vector pointlists,
        and adds metadata describing the shape
        """
        md = Metadata(name="_braggvectors_shape")
        md["Rshape"] = self.Rshape
        md["Qshape"] = self.Qshape
        self.metadata = md
        grp = Custom.to_h5(self, group)
        return grp

    # read

    @classmethod
    def _get_constructor_args(cls, group):
        """ """
        # Get shape metadata from the metadatabundle group
        assert (
            "metadatabundle" in group.keys()
        ), "No metadata found, can't get Rshape and Qshape"
        grp_metadata = group["metadatabundle"]
        assert (
            "_braggvectors_shape" in grp_metadata.keys()
        ), "No _braggvectors_shape metadata found"
        md = Metadata.from_h5(grp_metadata["_braggvectors_shape"])
        # Populate args and return
        kwargs = {
            "name": basename(group.name),
            "Rshape": md["Rshape"],
            "Qshape": md["Qshape"],
        }
        return kwargs

    def _populate_instance(self, group):
        """ """
        # Get the vectors
        dic = self._get_emd_attr_data(group)
        assert "_v_uncal" in dic.keys(), "Uncalibrated bragg vectors not found!"
        self._v_uncal = dic["_v_uncal"]
        # Point the vector getters to the vectors
        self._set_raw_vector_getter()
        self._set_cal_vector_getter()

    # standard output display

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string


# Vector access classes


class BVects:
    """
    Enables

        >>> v.qx,v.qy,v.I, optionally, v.h,v.k,v.l

    -like access to a collection of Bragg vector.
    """

    def __init__(self, data):
        """pointlist must have fields 'qx', 'qy', and 'intensity'"""
        self._data = data

    @property
    def qx(self):
        return self._data["qx"]

    @property
    def qy(self):
        return self._data["qy"]

    @property
    def I(self):
        return self._data["intensity"]

    @property
    def data(self):
        return self._data

    @property
    def h(self):
        try:
            return self._data["h"]
        except ValueError:
            raise AttributeError("h indicies not set")

    @property
    def k(self):
        try:
            return self._data["k"]
        except ValueError:
            raise AttributeError("k indicies not set")

    @property
    def l(self):
        try:
            return self._data["l"]
        except ValueError:
            raise AttributeError("l indicies not set")

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += f"A set of {len(self.data)} bragg vectors."
        string += " Access data with .qx, .qy, .I, or .data. Optionally .h, .k, and.l, if indexed)"
        return string

    def plot(
        self,
        returnfig: bool = False,
        **kwargs,
    ):
        """
        Plot the diffraction pattern.
        Calls `py4DSTEM.process.diffraction.plot_diffraction_pattern` and passes kwargs to it.

        Parameters
        ----------
        returnfig: bool
            If True the figure is returned, else its ploted but not returned. Defaults to False

        Returns
        -------
        figure : matplotlib object
            If `returnfig` is True, the figure is returned.

        """
        from py4DSTEM.process.diffraction import plot_diffraction_pattern

        if returnfig:
            return plot_diffraction_pattern(self, returnfig=returnfig, **kwargs)
        else:
            plot_diffraction_pattern(self, **kwargs)


class RawVectorGetter:
    def __init__(
        self,
        braggvects,
    ):
        self._bvects = braggvects
        self._data = braggvects._v_uncal

    def __getitem__(self, pos):
        x, y = pos
        ans = self._data[x, y].data
        return BVects(ans)

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += "Retrieves raw bragg vectors. Get vectors for scan position x,y with [x,y]. )"
        return string


class CalibratedVectorGetter:
    def __init__(
        self,
        braggvects,
    ):
        self._bvects = braggvects
        self._data = braggvects._v_uncal

    def __getitem__(self, pos):
        x, y = pos
        ans = self._data[x, y].data
        ans = self._transform(
            data=ans,
            cal=self._bvects.calibration,
            scanxy=(x, y),
            center=self._bvects.calstate["center"],
            ellipse=self._bvects.calstate["ellipse"],
            pixel=self._bvects.calstate["pixel"],
            rotate=self._bvects.calstate["rotate"],
        )
        return BVects(ans)

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += "Retrieves calibrated Bragg vectors. Get vectors for scan position x,y with [x,y]."
        string += (
            "\n"
            + space
            + "Set which calibrations to apply with braggvectors.setcal(...). )"
        )
        return string

    def _transform(
        self,
        data,
        cal,
        scanxy,
        center,
        ellipse,
        pixel,
        rotate,
    ):
        """
        Return a transformed copy of stractured data `data` with fields
        with fields 'qx','qy','intensity', applying calibrating transforms
        according to the values of center, ellipse, pixel, using the
        measurements found in Calibration instance cal for scan position scanxy.
        """

        ans = data.copy()
        x, y = scanxy

        # origin

        if center:
            origin = cal.get_origin(x, y)
            assert origin is not None, "Requested calibration was not found!"
            ans["qx"] -= origin[0]
            ans["qy"] -= origin[1]

        # ellipse
        if ellipse:
            ell = cal.get_ellipse(x, y)
            assert ell is not None, "Requested calibration was not found!"
            a, b, theta = ell
            # Get the transformation matrix
            e = b / a
            sint, cost = np.sin(theta - np.pi / 2.0), np.cos(theta - np.pi / 2.0)
            T = np.array(
                [
                    [e * sint**2 + cost**2, sint * cost * (1 - e)],
                    [sint * cost * (1 - e), sint**2 + e * cost**2],
                ]
            )
            # apply it
            xyar_i = np.vstack([ans["qx"], ans["qy"]])
            xyar_f = np.matmul(T, xyar_i)
            ans["qx"] = xyar_f[0, :]
            ans["qy"] = xyar_f[1, :]

        # pixel size
        if pixel:
            qpix = cal.get_Q_pixel_size()
            assert qpix is not None, "Requested calibration was not found!"
            ans["qx"] *= qpix
            ans["qy"] *= qpix

        # Q/R rotation
        if rotate:
            theta = cal.get_QR_rotation()
            assert theta is not None, "Requested calibration was not found!"
            flip = cal.get_QR_flip()
            flip = False if flip is None else flip
            # rotation matrix
            R = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            # rotate and flip
            if flip:
                positions = R @ np.vstack((ans["qy"], ans["qx"]))
            else:
                positions = R @ np.vstack((ans["qx"], ans["qy"]))
            # update
            ans["qx"] = positions[0, :]
            ans["qy"] = positions[1, :]

        # return
        return ans
