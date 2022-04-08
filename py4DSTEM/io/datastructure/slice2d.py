# Defines the DataCube class, which stores 4D-STEM datacubes

from .array import Array, Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py

class Slice2D(Array):
    """
    Stores 2D arrays and 3D stacks of named 2D arrays.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'dataslice2d',
        size: Optional[Union[float,list]] = 1,
        units: Optional[Union[str,list]] = 'pixels',
        slicelabels: Optional[list] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            size (float or length 2 list of floats): the pixel size
            units (str length 2 list of str): the pixel units
            slicelabels (list): a list of strings assigning names to the 3rd dimension

        Returns:
            A new Slice2D instance
        """
        # expand size/units inputs to include 2 dimensions
        if type(size) is not list: size = [size,size]
        if type(units) is not list: units = [units,units]

        # set the dim vectors
        dims = [
            size[0],
            size[1],
        ]
        dim_units = [
            units[0],
            units[1]
        ]
        dim_names = [
            'x',
            'y',
        ]

        # check for a third dimension
        if len(data.shape)>2:
            # TODO: unclear how to handle strings in dim vectors
            # what needs to change?
            # maybe just the .h5 reader/writer
            # or should this be stuffed into units???? no...
            # into names???

            # append to dim vectors
            #dims.append(???) #TODO
            #dim_units.append(???) #TODO
            #dim_names.append(???) #TODO

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = dims,
            dim_units = dim_units,
            dim_names = dim_names
        )

        # setup the size/units with getter/setters
        self._size = size
        self._units = units
        self._size = size
        self._units = units

    @property
    def size(self):
        return self._size
    @size.setter
    def size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self._size = x
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self._units = x




############ END OF CLASS ###########




# Reading

def Slice2D_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't, exist, or if
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


def Slice2D_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    assert(array.D in (2,3)), "Array must have 2 or 3 dimensions"
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



