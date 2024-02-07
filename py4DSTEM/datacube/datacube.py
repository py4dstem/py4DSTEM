# Defines the DataCube class, which stores 4D-STEM datacubes

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import (
    binary_opening,
    binary_dilation,
    distance_transform_edt,
    binary_fill_holes,
    gaussian_filter1d,
    gaussian_filter,)
from typing import Optional, Union

from emdfile import Array, Metadata, Node, Root, tqdmnd
from py4DSTEM.data import Data, Calibration
from py4DSTEM.datacube.preprocess import Preprocessor
from py4DSTEM.datacube.virtualimage import VirtualImager
from py4DSTEM.datacube.virtualdiffraction import VirtualDiffractioner
from py4DSTEM.datacube.diskdetection import BraggFinder
from py4DSTEM.datacube.diskdetection import ProbeMaker




class DataCube(
    Array,
    Data,
    Preprocessor,
    VirtualImager,
    VirtualDiffractioner,
    BraggFinder,
    ProbeMaker,
):
    """
    Storage and processing methods for 4D-STEM datasets.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "datacube",
        slicelabels: Optional[Union[bool, list]] = None,
        calibration: Optional[Union[Calibration, None]] = None,
    ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            calibration (None or Calibration or 'pass'): default (None)
                creates and attaches a new Calibration instance to root
                metadata, or, passing a Calibration instance uses this instead.
            slicelabels (None or list): names for slices if this is a
                stack of datacubes

        Returns:
            A new DataCube instance.
        """
        # initialize as an Array
        Array.__init__(
            self,
            data=data,
            name=name,
            units="pixel intensity",
            dim_names=["Rx", "Ry", "Qx", "Qy"],
            slicelabels=slicelabels,
        )

        # initialize as Data
        Data.__init__(self, calibration)

        # register with calibration
        self.calibration.register_target(self)

        # cartesian coords
        self.calibrate()

        # polar coords
        self.polar = None

    def calibrate(self):
        """
        Calibrate the coordinate axes of the datacube. Using the calibrations
        at self.calibration, sets the 4 dim vectors (Qx,Qy,Rx,Ry) according
        to the pixel size, units and origin positions, then updates the
        meshgrids representing Q and R space.
        """
        assert self.calibration is not None, "No calibration found!"

        # Get calibration values
        rpixsize = self.calibration.get_R_pixel_size()
        rpixunits = self.calibration.get_R_pixel_units()
        qpixsize = self.calibration.get_Q_pixel_size()
        qpixunits = self.calibration.get_Q_pixel_units()
        origin = self.calibration.get_origin_mean()
        if origin is None or origin == (None, None):
            origin = (0, 0)

        # Calc dim vectors
        dim_rx = np.arange(self.R_Nx) * rpixsize
        dim_ry = np.arange(self.R_Ny) * rpixsize
        dim_qx = -origin[0] + np.arange(self.Q_Nx) * qpixsize
        dim_qy = -origin[1] + np.arange(self.Q_Ny) * qpixsize

        # Set dim vectors
        self.set_dim(0, dim_rx, units=rpixunits)
        self.set_dim(1, dim_ry, units=rpixunits)
        self.set_dim(2, dim_qx, units=qpixunits)
        self.set_dim(3, dim_qy, units=qpixunits)

        # Set meshgrids
        self._qxx, self._qyy = np.meshgrid(dim_qx, dim_qy)
        self._rxx, self._ryy = np.meshgrid(dim_rx, dim_ry)

        self._qyy_raw, self._qxx_raw = np.meshgrid(
            np.arange(self.Q_Ny), np.arange(self.Q_Nx)
        )
        self._ryy_raw, self._rxx_raw = np.meshgrid(
            np.arange(self.R_Ny), np.arange(self.R_Nx)
        )

    # coordinate meshgrids
    @property
    def rxx(self):
        return self._rxx

    @property
    def ryy(self):
        return self._ryy

    @property
    def qxx(self):
        return self._qxx

    @property
    def qyy(self):
        return self._qyy

    @property
    def rxx_raw(self):
        return self._rxx_raw

    @property
    def ryy_raw(self):
        return self._ryy_raw

    @property
    def qxx_raw(self):
        return self._qxx_raw

    @property
    def qyy_raw(self):
        return self._qyy_raw

    # coordinate meshgrids with shifted origin
    def qxxs(self, rx, ry):
        qx0_shift = self.calibration.get_qx0shift(rx, ry)
        if qx0_shift is None:
            raise Exception(
                "Can't compute shifted meshgrid - origin shift is not defined"
            )
        return self.qxx - qx0_shift

    def qyys(self, rx, ry):
        qy0_shift = self.calibration.get_qy0shift(rx, ry)
        if qy0_shift is None:
            raise Exception(
                "Can't compute shifted meshgrid - origin shift is not defined"
            )
        return self.qyy - qy0_shift

    # shape properties

    ## shape

    # FOV
    @property
    def R_Nx(self):
        return self.data.shape[0]

    @property
    def R_Ny(self):
        return self.data.shape[1]

    @property
    def Q_Nx(self):
        return self.data.shape[2]

    @property
    def Q_Ny(self):
        return self.data.shape[3]

    @property
    def Rshape(self):
        return (self.data.shape[0], self.data.shape[1])

    @property
    def Qshape(self):
        return (self.data.shape[2], self.data.shape[3])

    @property
    def R_N(self):
        return self.R_Nx * self.R_Ny

    # aliases
    qnx = Q_Nx
    qny = Q_Ny
    rnx = R_Nx
    rny = R_Ny
    rshape = Rshape
    qshape = Qshape
    rn = R_N

    ## pixel size / units

    # Q
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()

    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()

    # R
    @property
    def R_pixel_size(self):
        return self.calibration.get_R_pixel_size()

    @property
    def R_pixel_units(self):
        return self.calibration.get_R_pixel_units()

    # aliases
    qpixsize = Q_pixel_size
    qpixunit = Q_pixel_units
    rpixsize = R_pixel_size
    rpixunit = R_pixel_units

    def copy(self):
        """
        Copys datacube
        """
        from py4DSTEM import DataCube

        new_datacube = DataCube(
            data=self.data.copy(),
            name=self.name,
            calibration=self.calibration.copy(),
            slicelabels=self.slicelabels,
        )

        Qpixsize = new_datacube.calibration.get_Q_pixel_size()
        Qpixunits = new_datacube.calibration.get_Q_pixel_units()
        Rpixsize = new_datacube.calibration.get_R_pixel_size()
        Rpixunits = new_datacube.calibration.get_R_pixel_units()

        new_datacube.set_dim(0, [0, Rpixsize], units=Rpixunits, name="Rx")
        new_datacube.set_dim(1, [0, Rpixsize], units=Rpixunits, name="Ry")

        new_datacube.set_dim(2, [0, Qpixsize], units=Qpixunits, name="Qx")
        new_datacube.set_dim(3, [0, Qpixsize], units=Qpixunits, name="Qy")

        return new_datacube

    # I/O

    # to_h5 is inherited from Array

    # read
    @classmethod
    def _get_constructor_args(cls, group):
        """Construct a datacube with no calibration / metadata"""
        # We only need some of the Array constructors;
        # dim vector/units are passed through when Calibration
        # is loaded, and the runtim dim vectors are then set
        # in _add_root_links
        ar_args = Array._get_constructor_args(group)

        args = {
            "data": ar_args["data"],
            "name": ar_args["name"],
            "slicelabels": ar_args["slicelabels"],
            "calibration": None,
        }

        return args

    def _add_root_links(self, group):
        """When reading from file, link to calibration metadata,
        then use it to populate the datacube dim vectors
        """
        # Link to the datacube
        self.calibration._datacube = self

        # Populate dim vectors
        self.calibration.set_Q_pixel_size(self.calibration.get_Q_pixel_size())
        self.calibration.set_R_pixel_size(self.calibration.get_R_pixel_size())
        self.calibration.set_Q_pixel_units(self.calibration.get_Q_pixel_units())
        self.calibration.set_R_pixel_units(self.calibration.get_R_pixel_units())

        return

    # Class methods

    def add(self, data, name=""):
        """
        Adds a block of data to the DataCube's tree. If `data` is an instance of
        an EMD/py4DSTEM class, add it to the tree.  If it's a numpy array,
        turn it into an Array instance, then save to the tree.
        """
        if isinstance(data, np.ndarray):
            data = Array(data=data, name=name)
        self.attach(data)



