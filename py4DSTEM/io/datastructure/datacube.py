# Defines the DataCube class, which stores 4D-STEM datacubes

from .array import Array, Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py
import dask.array as da

class DataCube(Array):
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
        qunits: Optional[Union[str,list]] = 'pixels'
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            rsize (float or length 2 list of floats): the real space pixel size
            runits (str length 2 list of str): the real space pixel units
            qsize (float or length 2 list of str): the diffraction space pixel size
            qunits (str or length 2 list of str): the diffraction space pixel units

        Returns:
            A new DataCube instance
        """
        # expand r/q inputs to include 2 dimensions
        if type(rsize) is not list: rsize = [rsize,rsize]
        if type(runits) is not list: runits = [runits,runits]
        if type(qsize) is not list: qsize = [qsize,qsize]
        if type(qunits) is not list: qunits = [qunits,qunits]

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                [0,rsize[0]],
                [0,rsize[1]],
                [0,qsize[0]],
                [0,qsize[1]]
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
            ]
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

def DataCube_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't exist, or if
    it exists but does not have 4 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A DataCube instance
    """
    datacube = Array_from_h5(group, name)
    datacube = DataCube_from_Array(datacube)
    return datacube


def DataCube_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    assert(array.D == 4), "Array must have 4 dimensions"
    array.__class__ = DataCube
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
                  array.dim_units[3]]
    )
    return array



