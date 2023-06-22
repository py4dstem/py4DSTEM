
from typing import Optional
import numpy as np

from py4DSTEM.data import RealSlice,Data



class VirtualImage(RealSlice,Data):
    """
    Stores a real-space shaped 2D image with metadata
    indicating how this image was generated from a datacube.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'virtualimage',
        ):
        """
        Args:
            data (np.ndarray) : the 2D data
            name (str) : the name

        Returns:
            A new VirtualImage instance
        """
        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            data = data,
            name = name,
        )



    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = RealSlice._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
        }
        return args







