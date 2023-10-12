# Defines the QPoints class, which stores PointLists with fields 'qx','qy','intensity'

from emdfile import PointList
from py4DSTEM.data import Data

from typing import Optional
import numpy as np


class QPoints(PointList, Data):
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

    # properties

    @property
    def qx(self):
        return self.data["qx"]

    @property
    def qy(self):
        return self.data["qy"]

    @property
    def intensity(self):
        return self.data["intensity"]

    # aliases
    I = intensity

    # read
    # this method is not necessary but is kept for consistency of structure!
    @classmethod
    def _get_constructor_args(cls, group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        pl_constr_args = PointList._get_constructor_args(group)
        args = {
            "data": pl_constr_args["data"],
            "name": pl_constr_args["name"],
        }
        return args
