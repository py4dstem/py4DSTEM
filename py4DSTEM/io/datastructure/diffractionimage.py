# Defines the DiffractionImage class, which stores 2D, diffraction-shaped data
# with metadata about how it was created

from .diffractionslice import DiffractionSlice, DiffractionSlice_from_h5

from typing import Optional,Union
import numpy as np
import h5py

class DiffractionImage(DiffractionSlice):
    """
    Stores a diffraction-space shaped 2D image.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'diffractionimage',
        pixelsize: Optional[Union[float,list]] = 1,
        pixelunits: Optional[Union[str,list]] = 'pixels',
        kind: Optional[str] = None,
        region: Optional[str] = None,
        region_geometry: Optional[Union[tuple,np.ndarray]] = None,
        center_corrected: bool = False
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            pixelsize (float or length 2 list of floats): the pixel size
            pixelunits (str length 2 list of str): the pixel units

        Returns:
            A new DiffractionImage instance
        """
        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self,
            data = data,
            name = name,
            pixelsize = pixelsize,
            pixelunits = pixelunits,
        )

        # Set metadata
        self.kind = kind
        self.region = region
        self.region_geometry = region_geometry
        self.center_corrected = center_corrected


############ END OF CLASS ###########




# Reading

def DiffractionImage_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DiffractionImage. If it doesn't exist, or if
    it exists but does not have 2 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A DiffractionImage instance
    """
    # TODO
    diffractionslice = Array_from_h5(group, name)
    diffractionslice = DiffractionSlice_from_Array(diffractionslice)
    return diffractionslice


# TODO
def DiffractionImage_from_Array(array):
    """
    Converts an Array to a DiffractionImage.

    Accepts:
        array (Array)

    Returns:
        (DiffractionImage)
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



