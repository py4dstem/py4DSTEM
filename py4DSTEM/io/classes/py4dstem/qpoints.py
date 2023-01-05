# Defines the QPoints class, which stores PointLists with fields 'qx','qy','intensity'

from py4DSTEM.io.classes.pointlist import PointList

from typing import Optional,Union
import numpy as np
import h5py

class QPoints(PointList):
    """
    Stores a set of diffraction space points,
    with fields 'qx', 'qy' and 'intensity'
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'qpoints',
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
            data = data,
            name = name,
        )

        # rename fields
        self.fields = 'qx','qy','intensity'


    @property
    def qx(self):
        return self.data['qx']
    @property
    def qy(self):
        return self.data['qy']
    @property
    def intensity(self):
        return self.data['intensity']



    # HDF5 i/o

    # write inherited from PointList

    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in
        read mode. Determines if it's a valid QPoints instance, and if so
        loads and returns it. Otherwise, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A QPoints instance
        """
        # Load from H5 as a PointList
        qpoints = PointList.from_h5(group)

        # Convert to QPoints
        pointlist.__class__ = QPoints
        pointlist.__init__(
            data = pointlist.data,
            name = pointlist.name,
        )

        # Return
        return pointlist






############ END OF CLASS ###########





