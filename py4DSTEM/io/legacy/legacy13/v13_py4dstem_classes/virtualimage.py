# Defines the VirtualImage class, which stores 2D, real-shaped data
# with metadata about how it was created

from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.realslice import RealSlice
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.metadata import Metadata

from typing import Optional, Union
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
        name: Optional[str] = "virtualimage",
        mode: Optional[str] = None,
        geometry: Optional[Union[tuple, np.ndarray]] = None,
        centered: Optional[bool] = False,
        calibrated: Optional[bool] = False,
        shift_center: Optional[bool] = False,
        dask: Optional[bool] = False,
    ):
        """
        Args:
            data (np.ndarray)   : the 2D data
            name (str)          : the name
            mode (str)          : defines geometry mode for calculating virtual image.
                Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright field
                    - 'annular' or 'annulus' uses annular detector, like dark field
                    - 'rectangle', 'square', 'rectangular', uses rectangular detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows:
                    - 'point': 2-tuple, (qx,qy),
                       qx and qy are each single float or int to define center
                    - 'circle' or 'circular': nested 2-tuple, ((qx,qy),radius),
                       qx, qy and radius, are each single float or int
                    - 'annular' or 'annulus': nested 2-tuple, ((qx,qy),(radius_i,radius_o)),
                       qx, qy, radius_i, and radius_o are each single float or integer
                    - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                    - `mask`: flexible detector, any boolean or floating point 2D array with
                        the same shape as datacube.Qshape
            centered (bool)     : if False (default), the origin is in the upper left corner.
                 If True, the mean measured origin in the datacube calibrations
                 is set as center. The measured origin is set with datacube.calibration.set_origin()
                 In this case, for example, a centered bright field image could be defined
                 by geometry = ((0,0), R). For `mode="mask"`, has no effect.
            calibrated (bool)   : if True, geometry is specified in units of 'A^-1' instead of pixels.
                The datacube's calibrations must have its `"Q_pixel_units"` parameter set to "A^-1".
                For `mode="mask"`, has no effect.
            shift_center (bool) : if True, the mask is shifted at each real space position to
                account for any shifting of the origin of the diffraction images. The datacube's
                calibration['origin'] parameter must be set. The shift applied to each pattern is
                the difference between the local origin position and the mean origin position
                over all patterns, rounded to the nearest integer for speed.
            verbose (bool)      : if True, show progress bar
            dask (bool)         : if True, use dask arrays

        Returns:
            A new VirtualImage instance
        """
        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            data=data,
            name=name,
        )

        # Set metadata
        md = Metadata(name="virtualimage")
        md["mode"] = mode
        md["geometry"] = geometry
        md["centered"] = centered
        md["calibrated"] = calibrated
        md["shift_center"] = shift_center
        md["dask"] = dask
        self.metadata = md

    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import (
            VirtualImage_from_h5,
        )

        return VirtualImage_from_h5(group)


############ END OF CLASS ###########
