# Defines the RealSlice class, which stores 2(+1)D real-space shaped data

from ..emd.array import Array

from typing import Optional,Union
import numpy as np
import h5py

class RealSlice(Array):
    """
    Stores a real-space shaped 2D data array.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'realslice',
        pixel_size: Optional[Union[float,list]] = 1,
        pixel_units: Optional[Union[str,list]] = 'pixels',
        slicelabels: Optional[Union[bool,list]] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the realslice
            pixel_size (float or length 2 list of floats): the pixel size
            pixel_units (str length 2 list of str): the pixel units
            slicelabels(None or list): names for slices if this is a stack of
                realslices

        Returns:
            A new RealSlice instance
        """
        # expand pixel inputs to include 2 dimensions
        if type(pixel_size) is not list: pixel_size = [pixel_size,pixel_size]
        if type(pixel_units) is not list: pixel_units = [pixel_units,pixel_units]

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                pixel_size[0],
                pixel_size[1],
            ],
            dim_units = [
                pixel_units[0],
                pixel_units[1],
            ],
            dim_names = [
                'Rx',
                'Ry'
            ],
            slicelabels = slicelabels
        )

        # setup the size/units with getter/setters
        self._pixel_size = pixel_size
        self._pixel_units = pixel_units

    @property
    def pixel_size(self):
        return self._pixel_size
    @pixel_size.setter
    def pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self._pixel_size = x
    @property
    def pixel_units(self):
        return self._pixel_units
    @pixel_units.setter
    def pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self._pixel_units = x



    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from .io import RealSlice_from_h5
        return RealSlice_from_h5(group)






############ END OF CLASS ###########





