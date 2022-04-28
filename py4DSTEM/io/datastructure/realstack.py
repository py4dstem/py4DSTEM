# Defines the RealStack class, which stores a 3D stack of multiple
# 2D real-space shaped datasets

from .arraystack import ArrayStack
from .arrayio import Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py
import dask.array as da

class RealStack(ArrayStack):
    """
    Stores multiple real-space shaped 2D data arrays.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'realstack',
        pixelsize: Optional[Union[float,list]] = 1,
        pixelunits: Optional[Union[str,list]] = 'pixels',
        labels: Optional[list] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            pixelsize (float or length 2 list of floats): the pixel size
            pixelunits (str length 2 list of str): the pixel units
            labels (list): strings which label the final dimension of
                the data array

        Returns:
            A new RealStack instance
        """
        # expand pixel inputs to include 2 dimensions
        if type(pixelsize) is not list: pixelsize = [pixelsize,pixelsize]
        if type(pixelunits) is not list: pixelunits = [pixelunits,pixelunits]

        # initialize as an Array
        ArrayStack.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                pixelsize[0],
                pixelsize[1],
            ],
            dim_units = [
                pixelunits[0],
                pixelunits[1],
            ],
            dim_names = [
                'Rx',
                'Ry'
            ],
            labels = labels
        )

        # setup the size/units with getter/setters
        self._pixelsize = pixelsize
        self._pixelunits = pixelunits

    @property
    def pixelsize(self):
        return self._pixelsize
    @pixelsize.setter
    def pixelsize(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self._pixelsize = x
    @property
    def pixelunits(self):
        return self._pixelunits
    @pixelunits.setter
    def pixelunits(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self._pixelunits = x



############ END OF CLASS ###########




# Reading

def RealStack_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a RealStack. If it doesn't exist, or if
    it exists but does not have 3 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A RealStack instance
    """
    realstack = Array_from_h5(group, name)
    realstack = RealStack_from_Array(realstack)
    return realstack


def RealStack_from_Array(array):
    """
    Converts an Array to a RealStack.

    Accepts:
        array (Array)

    Returns:
        (RealStack)
    """
    assert(array.rank == 3), "Array must have 3 dimensions"
    array.__class__ = RealStack
    array.__init__(
        data = array.data,
        name = array.name,
        pixelsize = [array.dims[0][1]-array.dims[0][0],
                     array.dims[1][1]-array.dims[1][0]],
        pixelunits = [array.dim_units[0],
                      array.dim_units[1]],
        labels = array.dims[2]
    )
    return array



