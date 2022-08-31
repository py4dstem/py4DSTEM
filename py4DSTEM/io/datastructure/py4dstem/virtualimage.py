# Defines the VirtualImage class, which stores 2D, real-shaped data
# with metadata about how it was created

from py4DSTEM.io.datastructure.py4dstem.realslice import RealSlice
from py4DSTEM.io.datastructure.emd.metadata import Metadata

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
        centered: Optional[bool] = False,
        calibrated: Optional[bool] = False,
        shift_center: Optional[bool] = False,
        ):
        """
        Args:
            data (np.ndarray)   : the 2D data
            name (str)          : the name
            mode (str)          : defines geometry mode for calculating virtual image
                                    options:
                                    - 'point' uses singular point as detector
                                    - 'circle' or 'circular' uses round detector,like bright field
                                    - 'annular' or 'annulus' uses annular detector, like dark field
                                    - 'rectangle', 'square', 'rectangular', uses rectangular detector
                                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`, values in pixels
                                        argument, as follows:
                                    - 'point': 2-tuple, (qx,qy), 
                                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius), 
                                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                                    - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                                    - `mask`: flexible detector, any 2D array, same size as datacube.QShape
            centered (bool)     : if false (default), the origin is in the upper left corner.
                                  If True, the mean measured origin in the datacube calibrations 
                                  is set as center. In this case, for example, a centered bright field image 
                                  could be defined by geometry = ((0,0), R).
            calibrated (bool)   : if True, geometry is specified in units of 'A^-1' isntead of pixels. 
                                  The datacube must have updated calibration metadata.
            shift_center (bool) : if True, qx and qx are shifted for each position in real space
                                    supported for 'point', 'circle', and 'annular' geometry. 
                                    For the shifting center mode, the geometry argument shape
                                    should be modified so that qx and qy are the same size as Rshape
                                        - 'point': 2-tuple, (qx,qy) 
                                           where qx.shape and qy.shape == datacube.Rshape
                                        - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius) 
                                           where qx.shape and qx.shape == datacube.Rshape
                                        - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o))
                                           where qx.shape and qx.shape == datacube.Rshape
            dask (bool)         : if True, use dask arrays        
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
        md['centered'] = centered
        md['calibrated'] = calibrated
        md['shift_center'] = shift_center
        self.metadata = md


    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from py4DSTEM.io.datastructure.py4dstem.io import VirtualImage_from_h5
        return VirtualImage_from_h5(group)






############ END OF CLASS ###########






