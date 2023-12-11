# Defines the BraggVectors class


from typing import Optional, Union
import numpy as np
import h5py

from py4DSTEM.io.legacy.legacy13.v13_emd_classes import PointListArray
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.tree import Tree
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.metadata import Metadata


class BraggVectors:
    """
    Stores bragg scattering information for a 4D datacube.
        >>> braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )
    initializes an instance of the appropriate shape for a DataCube `datacube`.
        >>> braggvectors.v[rx,ry]
        >>> braggvectors.v_uncal[rx,ry]
    retrieve, respectively, the calibrated and uncalibrated bragg vectors at
    scan position [rx,ry], and
        >>> braggvectors.v[rx,ry]['qx']
        >>> braggvectors.v[rx,ry]['qy']
        >>> braggvectors.v[rx,ry]['intensity']
    retrieve the positiona and intensity of the scattering.
    """

    def __init__(self, Rshape, Qshape, name="braggvectors"):
        self.name = name
        self.Rshape = Rshape
        self.shape = self.Rshape
        self.Qshape = Qshape

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        if "braggvectors" not in self._metadata.keys():
            self.metadata = Metadata(name="braggvectors")
        self.metadata["braggvectors"]["Qshape"] = self.Qshape

        self._v_uncal = PointListArray(
            dtype=[("qx", np.float64), ("qy", np.float64), ("intensity", np.float64)],
            shape=Rshape,
            name="_v_uncal",
        )

    @property
    def vectors(self):
        try:
            return self._v_cal
        except AttributeError:
            er = "No calibrated bragg vectors found. Try running .calibrate()!"
            raise Exception(er)

    @property
    def vectors_uncal(self):
        return self._v_uncal

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, x):
        assert isinstance(x, Metadata)
        self._metadata[x.name] = x

    ## Representation to standard output

    def __repr__(self):
        space = " " * len(self.__class__.__name__) + "  "
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string

    # HDF5 read/write

    # write
    def to_h5(self, group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import (
            BraggVectors_to_h5,
        )

        BraggVectors_to_h5(self, group)

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import (
            BraggVectors_from_h5,
        )

        return BraggVectors_from_h5(group)


############ END OF CLASS ###########
