# Defines the DataCubeStack class, which stores a stack of multiple 4D-STEM datacubes

from .arraystack import ArrayStack
from .arrayio import Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py
import dask.array as da

class DataCubeStack(ArrayStack):
    """
    Stores 4D-STEM datasets.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'datacube',
        rsize: Optional[Union[float,list]] = 1,
        runits: Optional[Union[str,list]] = 'pixels',
        qsize: Optional[Union[float,list]] = 1,
        qunits: Optional[Union[str,list]] = 'pixels',
        labels: Optional[list] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            rsize (float or length 2 list of floats): the real space pixel size
            runits (str length 2 list of str): the real space pixel units
            qsize (float or length 2 list of str): the diffraction space pixel size
            qunits (str or length 2 list of str): the diffraction space pixel units
            labels (list): strings which label the final dimension of
                the data array

        Returns:
            A new DataCubeStack instance
        """
        # expand r/q inputs to include 2 dimensions
        if type(rsize) is not list: rsize = [rsize,rsize]
        if type(runits) is not list: runits = [runits,runits]
        if type(qsize) is not list: qsize = [qsize,qsize]
        if type(qunits) is not list: qunits = [qunits,qunits]

        # initialize as an Array
        ArrayStack.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                rsize[0],
                rsize[1],
                qsize[0],
                qsize[1]
            ],
            dim_units = [
                runits[0],
                runits[1],
                qunits[0],
                qunits[1]
            ],
            dim_names = [
                'Rx',
                'Ry',
                'Qx',
                'Qy'
            ],
            labels = labels
        )

        # setup the size/units with getter/setters
        self._rsize = rsize
        self._runits = runits
        self._qsize = qsize
        self._qunits = qunits

    @property
    def rsize(self):
        return self._rsize
    @rsize.setter
    def rsize(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self._rsize = x
    @property
    def runits(self):
        return self._runits
    @runits.setter
    def runits(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self._runits = x

    @property
    def qsize(self):
        return self._qsize
    @qsize.setter
    def qsize(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(2,[0,x[0]])
        self.set_dim(3,[0,x[1]])
        self._qsize = x
    @property
    def qunits(self):
        return self._qunits
    @qunits.setter
    def qunits(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[2] = x[0]
        self.dim_units[3] = x[1]
        self._qunits = x


############ END OF CLASS ###########




# Reading

def DataCubeStack_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCubeStack. If it doesn't exist, or if
    it exists but does not have 5 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A DataCubeStack instance
    """
    datacubestack = Array_from_h5(group, name)
    datacubestack = DataCubeStack_from_Array(datacubestack)
    return datacubestack


def DataCubeStack_from_Array(array):
    """
    Converts an Array to a DataCubeStack.

    Accepts:
        array (Array)

    Returns:
        (DataCubeStack)
    """
    assert(array.rank == 5), "Array must have 5 dimensions"
    array.__class__ = DataCubeStack
    array.__init__(
        data = array.data,
        name = array.name,
        rsize = [array.dims[0][1]-array.dims[0][0],
                 array.dims[1][1]-array.dims[1][0]],
        runits = [array.dim_units[0],
                  array.dim_units[1]],
        qsize = [array.dims[2][1]-array.dims[2][0],
                 array.dims[3][1]-array.dims[3][0]],
        qunits = [array.dim_units[2],
                  array.dim_units[3]],
        labels = array.dims[4]
    )
    return array



