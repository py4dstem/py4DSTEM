# Defines the DiffractionSlice class, which stores 2D, diffraction-shaped data

from .array import Array, Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py
import dask.array as da

class DiffractionSlice(Array):
    """
    Stores a diffraction-space shaped 2D data array.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'diffractionslice',
        pixelsize: Optional[Union[float,list]] = 1,
        pixelunits: Optional[Union[str,list]] = 'pixels',
        slicelabels: Optional[Union[bool,list]] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            pixelsize (float or length 2 list of floats): the pixel size
            pixelunits (str length 2 list of str): the pixel units
            slicelabels(None or list): names for slices if this is a stack of
                diffractionslices

        Returns:
            A new DiffractionSlice instance
        """
        # expand pixel inputs to include 2 dimensions
        if type(pixelsize) is not list: pixelsize = [pixelsize,pixelsize]
        if type(pixelunits) is not list: pixelunits = [pixelunits,pixelunits]

        # initialize as an Array
        Array.__init__(
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
                'Qx',
                'Qy'
            ],
            slicelabels = slicelabels
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

def DiffractionSlice_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DiffractionSlice. If it doesn't exist, or if
    it exists but does not have 2 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A DiffractionSlice instance
    """
    diffractionslice = Array_from_h5(group, name)
    diffractionslice = DiffractionSlice_from_Array(diffractionslice)
    return diffractionslice


def DiffractionSlice_from_Array(array):
    """
    Converts an Array to a DiffractionSlice.

    Accepts:
        array (Array)

    Returns:
        (DiffractionSlice)
    """
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = DiffractionSlice
    array.__init__(
        data = array.data,
        name = array.name,
        pixelsize = [array.dims[0][1]-array.dims[0][0],
                     array.dims[1][1]-array.dims[1][0]],
        pixelunits = [array.dim_units[0],
                      array.dim_units[1]],
        slicelabels = array.slicelabels
    )
    return array



