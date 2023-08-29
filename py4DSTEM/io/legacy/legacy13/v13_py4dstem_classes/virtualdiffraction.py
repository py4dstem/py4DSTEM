# Defines the VirtualDiffraction class, which stores 2D, diffraction-shaped data
# with metadata about how it was created

from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.diffractionslice import (
    DiffractionSlice,
)
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.metadata import Metadata

from typing import Optional, Union
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
        name: Optional[str] = "diffractionimage",
        method: Optional[str] = None,
        mode: Optional[str] = None,
        geometry: Optional[Union[tuple, np.ndarray]] = None,
        calibrated: Optional[bool] = False,
        shift_center: bool = False,
    ):
        """
        Args:
            data (np.ndarray) : the 2D data
            name (str) : the name
            method (str) : defines method used for diffraction pattern, options are
                'mean', 'median', and 'max'
            mode (str) : defines mode for selecting area in real space to use for
                virtual diffraction. The default is None, which means no
                geometry will be applied and the whole datacube will be used
                for the calculation. Options:
                    - 'point' uses singular point as detector
                    - 'circle' or 'circular' uses round detector, like bright field
                    - 'annular' or 'annulus' uses annular detector, like dark field
                    - 'rectangle', 'square', 'rectangular', uses rectangular detector
                    - 'mask' flexible detector, any 2D array
            geometry (variable) : valid entries are determined by the `mode`, values in pixels
                argument, as follows. The default is None, which means no geometry will be applied
                and the whole datacube will be used for the calculation. If mode is None the geometry
                will not be applied.
                    - 'point': 2-tuple, (rx,ry),
                       qx and qy are each single float or int to define center
                    - 'circle' or 'circular': nested 2-tuple, ((rx,ry),radius),
                       qx, qy and radius, are each single float or int
                    - 'annular' or 'annulus': nested 2-tuple, ((rx,ry),(radius_i,radius_o)),
                       qx, qy, radius_i, and radius_o are each single float or integer
                    - 'rectangle', 'square', 'rectangular': 4-tuple, (xmin,xmax,ymin,ymax)
                    - `mask`: flexible detector, any boolean or floating point 2D array with
                        the same shape as datacube.Rshape
            calibrated (bool)   : if True, geometry is specified in units of 'A' instead of pixels.
                The datacube's calibrations must have its `"R_pixel_units"` parameter set to "A".
                If mode is None the geometry and calibration will not be applied.
            shift_center (bool) : if True, the difraction pattern is shifted to account for beam shift
                or the changing of the origin through the scan. The datacube's calibration['origin']
                parameter must be set Only 'max' and 'mean' supported for this option.
        Returns:
            A new VirtualDiffraction instance
        """
        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self,
            data=data,
            name=name,
        )

        # Set metadata
        md = Metadata(name="virtualdiffraction")
        md["method"] = method
        md["mode"] = mode
        md["geometry"] = geometry
        md["shift_center"] = shift_center
        self.metadata = md

    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import (
            VirtualDiffraction_from_h5,
        )

        return VirtualDiffraction_from_h5(group)


############ END OF CLASS ###########
