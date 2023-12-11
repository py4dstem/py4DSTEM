# Defines the DataCube class, which stores 4D-STEM datacubes

from py4DSTEM.io.legacy.legacy13.v13_emd_classes.array import Array
from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.calibration import Calibration
from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.parenttree import ParentTree

from typing import Optional, Union
import numpy as np
import h5py


class DataCube(Array):
    """
    Stores 4D-STEM datasets.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "datacube",
        R_pixel_size: Optional[Union[float, list]] = 1,
        R_pixel_units: Optional[Union[str, list]] = "pixels",
        Q_pixel_size: Optional[Union[float, list]] = 1,
        Q_pixel_units: Optional[Union[str, list]] = "pixels",
        slicelabels: Optional[Union[bool, list]] = None,
        calibration: Optional = None,
    ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            R_pixel_size (float or length 2 list of floats): the real space
                pixel size
            R_pixel_units (str or length 2 list of str): the real space
                pixel units
            Q_pixel_size (float or length 2 list of str): the diffraction space
                pixel size
            Q_pixel_units (str or length 2 list of str): the diffraction space
                pixel units. Must be 'pixels' or 'A^-1'.
            slicelabels (None or list): names for slices if this is a
                stack of datacubes
            calibration (Calibration):
        Returns:
            A new DataCube instance
        """

        # initialize as an Array
        Array.__init__(
            self,
            data=data,
            name=name,
            units="pixel intensity",
            dims=[R_pixel_size, R_pixel_size, Q_pixel_size, Q_pixel_size],
            dim_units=[R_pixel_units, R_pixel_units, Q_pixel_units, Q_pixel_units],
            dim_names=["Rx", "Ry", "Qx", "Qy"],
            slicelabels=slicelabels,
        )

        # make a tree
        # we're overwriting the emd Tree with the py4DSTEM Tree
        # which knows how to track the parent datacube
        # also adds calibration to the tree
        self.tree = ParentTree(self, Calibration())

        # set size/units
        self.tree["calibration"].set_R_pixel_size(R_pixel_size)
        self.tree["calibration"].set_R_pixel_units(R_pixel_units)
        self.tree["calibration"].set_Q_pixel_size(Q_pixel_size)
        self.tree["calibration"].set_Q_pixel_units(Q_pixel_units)

    ## properties

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

    # pixel sizes/units

    # R
    @property
    def R_pixel_size(self):
        return self.calibration.get_R_pixel_size()

    @R_pixel_size.setter
    def R_pixel_size(self, x):
        if type(x) is not list:
            x = [x, x]
        self.set_dim(0, [0, x[0]])
        self.set_dim(1, [0, x[1]])
        self.calibration.set_R_pixel_size(x)

    @property
    def R_pixel_units(self):
        return self.calibration.get_R_pixel_units()

    @R_pixel_units.setter
    def R_pixel_units(self, x):
        if type(x) is not list:
            x = [x, x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self.calibration.set_R_pixel_units(x)

    # Q
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()

    @Q_pixel_size.setter
    def Q_pixel_size(self, x):
        if type(x) is not list:
            x = [x, x]
        self.set_dim(2, [0, x[0]])
        self.set_dim(3, [0, x[1]])
        self.calibration.set_Q_pixel_size(x)

    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()

    @Q_pixel_units.setter
    def Q_pixel_units(self, x):
        if type(x) is not list:
            x = [x, x]
        self.dim_units[2] = x[0]
        self.dim_units[3] = x[1]
        self.calibration.set_Q_pixel_units(x)

    # calibration
    @property
    def calibration(self):
        return self.tree["calibration"]

    @calibration.setter
    def calibration(self, x):
        assert isinstance(x, Calibration)
        self.tree["calibration"] = x

    # for parent datacube tracking
    def track_parent(self, x):
        x._parent = self
        x.calibration = self.calibration

    # HDF5 read/write

    # write is inherited from Array

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import DataCube_from_h5

        return DataCube_from_h5(group)


############ END OF CLASS ###########
