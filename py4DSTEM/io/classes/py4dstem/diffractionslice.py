# Defines the DiffractionSlice class, which stores 2(+1)D,
# diffraction-shaped data

from py4DSTEM.io.classes.array import Array

from typing import Optional,Union
import numpy as np
import h5py

class DiffractionSlice(Array):
    """
    Stores a diffraction-space shaped 2D data array.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'diffractionslice',
        slicelabels: Optional[Union[bool,list]] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the diffslice
            slicelabels(None or list): names for slices if this is a 3D stack

        Returns:
            (DiffractionSlice instance)
        """

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            slicelabels = slicelabels
        )



    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in
        read mode. Determines if it's a valid Array, and if so loads and
        returns it as a DiffractionSlice. Otherwise, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A DiffractionSlice instance
        """
        # Load from H5 as an Array
        diffractionslice = Array.from_h5(group)

        # Convert to a DiffractionSlice
        assert(array.rank == 2), "Array must have 2 dimensions"
        array.__class__ = DiffractionSlice
        array.__init__(
            data = array.data,
            name = array.name,
            slicelabels = array.slicelabels
        )

        # Return
        return array







############ END OF CLASS ###########





