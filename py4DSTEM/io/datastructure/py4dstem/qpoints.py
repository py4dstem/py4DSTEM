# Defines the QPoints class, which stores PointLists with fields 'qx','qy','intensity'

from ..emd.pointlist import PointList

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



    # HDF5 read/write

    # write inherited from PointList

    # read
    def from_h5(group):
        from .io import QPoints_from_h5
        return QPoints_from_h5(group)






############ END OF CLASS ###########





