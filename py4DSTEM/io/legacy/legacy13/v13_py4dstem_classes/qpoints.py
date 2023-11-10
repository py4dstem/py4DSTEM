# Defines the QPoints class, which stores PointLists with fields 'qx','qy','intensity'

from py4DSTEM.io.legacy.legacy13.v13_emd_classes.pointlist import PointList

from typing import Optional
import numpy as np


class QPoints(PointList):
    """
    Stores a set of diffraction space points,
    with fields 'qx', 'qy' and 'intensity'
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = "qpoints",
    ):
        """
        Accepts:
            data (structured numpy ndarray): should have three fields, which
                will be renamed 'qx','qy','intensity'
            name (str): the name of the QPoints instance
        Returns:
            A new QPoints instance
        """

        # initialize as a PointList
        PointList.__init__(
            self,
            data=data,
            name=name,
        )

        # rename fields
        self.fields = "qx", "qy", "intensity"

    @property
    def qx(self):
        return self.data["qx"]

    @property
    def qy(self):
        return self.data["qy"]

    @property
    def intensity(self):
        return self.data["intensity"]

    # HDF5 read/write

    # write inherited from PointList

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import QPoints_from_h5

        return QPoints_from_h5(group)


############ END OF CLASS ###########
