# Defines the DiffractionImage class, which stores 2D, diffraction-shaped data
# with metadata about how it was created

from .diffractionslice import DiffractionSlice
from ..emd.metadata import Metadata

from typing import Optional,Union
import numpy as np
import h5py

class DiffractionImage(DiffractionSlice):
    """
    Stores a diffraction-space shaped 2D image with metadata
    indicating how this image was generated from a datacube.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'diffractionimage',
        mode: Optional[str] = None,
        geometry: Optional[Union[tuple,np.ndarray]] = None,
        shift_corr: bool = False
        ):
        """
        Args:
            data (np.ndarray): the 2D data
            name (str): the name
            mode (str): must be in ('max','mean','median')
            geometry (variable): indicates the region the image will
                be computed over. Behavior depends on the argument type:
                    - None: uses the whole image
                    - 4-tuple: uses a subcube w/ (rxmin,rxmax,rymin,rymax)
                    - 2-tuple: (rx,ry), which are numbers or length L arrays.
                        Uses the specified scan positions.
                    - `mask`: boolean 2D array
                    - `mask_float`: floating point 2D array. Valid only for
                        `mean` mode
            shift_corr (bool): if True, correct for beam shift

        Returns:
            A new DiffractionImage instance
        """
        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self,
            data = data,
            name = name,
        )

        # Set metadata
        md = Metadata(name='diffractionimage')
        md['mode'] = mode
        md['geometry'] = geometry
        md['shift_corr'] = shift_corr
        self.metadata = md



    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from .io import DiffractionImage_from_h5
        return DiffractionImage_from_h5(group)






############ END OF CLASS ###########






