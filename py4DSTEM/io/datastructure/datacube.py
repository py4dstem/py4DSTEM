# Defines the DataCube class, which stores 4D-STEM datacubes

from .array import Array, Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py

class DataCube(Array):
    """
    Stores 4D-STEM datasets.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'datacube',
        R_pixel_size: Optional[Union[float,list]] = 1,
        R_pixel_units: Optional[Union[str,list]] = 'pixels',
        Q_pixel_size: Optional[Union[float,list]] = 1,
        Q_pixel_units: Optional[Union[str,list]] = 'pixels',
        slicelabels: Optional[Union[bool,list]] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            R_pixel_size (float or length 2 list of floats): the real space pixel size
            R_pixel_units (str length 2 list of str): the real space pixel units
            Q_pixel_size (float or length 2 list of str): the diffraction space pixel size
            Q_pixel_units (str or length 2 list of str): the diffraction space pixel units
            slicelabels(None or list): names for slices if this is a stack of datacubes

        Returns:
            A new DataCube instance
        """
        # expand r/q inputs to include 2 dimensions
        if type(R_pixel_size) is not list: R_pixel_size = [R_pixel_size,R_pixel_size]
        if type(R_pixel_units) is not list: R_pixel_units = [R_pixel_units,R_pixel_units]
        if type(Q_pixel_size) is not list: Q_pixel_size = [Q_pixel_size,Q_pixel_size]
        if type(Q_pixel_units) is not list: Q_pixel_units = [Q_pixel_units,Q_pixel_units]

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                R_pixel_size[0],
                R_pixel_size[1],
                Q_pixel_size[0],
                Q_pixel_size[1]
            ],
            dim_units = [
                R_pixel_units[0],
                R_pixel_units[1],
                Q_pixel_units[0],
                Q_pixel_units[1]
            ],
            dim_names = [
                'Rx',
                'Ry',
                'Qx',
                'Qy'
            ],
            slicelabels = slicelabels
        )

        # setup the size/units with getter/setters
        self._R_pixel_size = R_pixel_size
        self._R_pixel_units = R_pixel_units
        self._Q_pixel_size = Q_pixel_size
        self._Q_pixel_units = Q_pixel_units

    @property
    def R_pixel_size(self):
        return self._R_pixel_size
    @R_pixel_size.setter
    def R_pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self._R_pixel_size = x
    @property
    def R_pixel_units(self):
        return self._R_pixel_units
    @R_pixel_units.setter
    def R_pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self._R_pixel_units = x

    @property
    def Q_pixel_size(self):
        return self._Q_pixel_size
    @Q_pixel_size.setter
    def Q_pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(2,[0,x[0]])
        self.set_dim(3,[0,x[1]])
        self._Q_pixel_size = x
    @property
    def Q_pixel_units(self):
        return self._Q_pixel_units
    @Q_pixel_units.setter
    def Q_pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[2] = x[0]
        self.dim_units[3] = x[1]
        self._Q_pixel_units = x


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
    assert(array.rank == 4), "Array must have 4 dimensions"
    array.__class__ = DataCube
    array.__init__(
        data = array.data,
        name = array.name,
        R_pixel_size = [array.dims[0][1]-array.dims[0][0],
                        array.dims[1][1]-array.dims[1][0]],
        R_pixel_units = [array.dim_units[0],
                         array.dim_units[1]],
        Q_pixel_size = [array.dims[2][1]-array.dims[2][0],
                        array.dims[3][1]-array.dims[3][0]],
        Q_pixel_units = [array.dim_units[2],
                         array.dim_units[3]],
        slicelabels = array.slicelabels
    )
    return array



