from py4DSTEM.io.classes.py4dstem.diffractionslice import DiffractionSlice
from py4DSTEM.io.classes.metadata import Metadata

from typing import Optional,Union
import numpy as np
import h5py

class VirtualDiffraction(DiffractionSlice):
    """
    Stores a diffraction-space shaped 2D image with metadata
    indicating how this image was generated from a datacube.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'diffractionimage',
        ):
        """
        Args:
            data (np.ndarray) : the 2D data
            name (str) : the name

        Returns:
            A new VirtualDiffraction instance
        """
        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self,
            data = data,
            name = name,
        )

        # Set metadata
        #md = Metadata(name='virtualdiffraction')
        #md['method'] = method
        #md['mode'] = mode
        #md['geometry'] = geometry
        #md['shift_center'] = shift_center
        #self.metadata = md


        



