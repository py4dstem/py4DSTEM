# Defines the VirtualImage class, which stores 2D, real-shaped data
# with metadata about how it was created

from .realslice import RealSlice
from ..emd.metadata import Metadata

from typing import Optional,Union
import numpy as np
import h5py

class VirtualImage(RealSlice):
    """
    Stores a real-space shaped 2D image with metadata
    indicating how this image was generated from a datacube.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'virtualimage',
        mode: Optional[str] = None,
        geometry: Optional[Union[tuple,np.ndarray]] = None,
        shift_corr: bool = False,
        eager_compute: bool = True
        ):
        """
        Args:
            data (np.ndarray): the 2D data
            name (str): the name
            mode (str): must be in
                ('point','circle','annulus','rectangle',
                 'cpoint','ccircle','cannulus','csquare',
                 'qpoint','qcircle','qannulus','qsquare',
                 'mask')
            geometry (variable): valid entries are determined by the `mode`
                argument
            shift_corr (bool):
            eager_compute (bool)

        Returns:
            A new VirtualImage instance
        """
        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            data = data,
            name = name,
        )


        # Set metadata
        md = Metadata(name='virtualimage')
        md['mode'] = mode
        md['geometry'] = geometry
        md['shift_corr'] = shift_corr
        md['eager_compute'] = eager_compute
        self.metadata = md



    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from .io import VirtualImage_from_h5
        return VirtualImage_from_h5(group)






############ END OF CLASS ###########






