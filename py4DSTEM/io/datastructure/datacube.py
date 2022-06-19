# Defines the DataCube class, which stores 4D-STEM datacubes

from .array import Array
from .calibration import Calibration

from typing import Optional,Union
import numpy as np
import h5py

class DataCube(Array):
    """
    Stores 4D-STEM datasets.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'datacube',
        R_pixel_size: Optional[Union[float,list]] = 1,
        R_pixel_units: Optional[Union[str,list]] = 'pixels',
        Q_pixel_size: Optional[Union[float,list]] = 1,
        Q_pixel_units: Optional[Union[str,list]] = 'pixels',
        slicelabels: Optional[Union[bool,list]] = None,
        calibration: Optional = None,
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            R_pixel_size (float or length 2 list of floats): the real space
                pixel size
            R_pixel_units (str or length 2 list of str): the real space
                pixel units
            Q_pixel_size (float or length 2 list of str): the diffraction space
                pixel size
            Q_pixel_units (str or length 2 list of str): the diffraction space
                pixel units
            slicelabels (None or list): names for slices if this is a
                stack of datacubes
            calibration (Calibration):

        Returns:
            A new DataCube instance
        """

        # expand r/q inputs to include 2 dimensions
        if type(R_pixel_size) is not list: R_pixel_size = [R_pixel_size,R_pixel_size]
        if type(R_pixel_units) is not list: R_pixel_units = [R_pixel_units,R_pixel_units]
        if type(Q_pixel_size) is not list: Q_pixel_size = [Q_pixel_size,Q_pixel_size]
        if type(Q_pixel_units) is not list: Q_pixel_units = [Q_pixel_units,Q_pixel_units]

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            dims = [
                R_pixel_size[0],
                R_pixel_size[1],
                Q_pixel_size[0],
                Q_pixel_size[1]
            ],
            dim_units = [
                R_pixel_units[0],
                R_pixel_units[1],
                Q_pixel_units[0],
                Q_pixel_units[1]
            ],
            dim_names = [
                'Rx',
                'Ry',
                'Qx',
                'Qy'
            ],
            slicelabels = slicelabels
        )

        # Make calibration container
        # set its size/units
        # put it in tree
        self.calibration = Calibration()
        self.calibration['R_pixel_size'], R_pixel_size
        self.calibration['R_pixel_units'], R_pixel_units
        self.calibration['Q_pixel_size'], Q_pixel_size
        self.calibration['Q_pixel_units'], Q_pixel_units
        self.tree = self.calibration, 'calibration'



    ## properties

    # R pixel sizes/units
    @property
    def R_pixel_size(self):
        return self.calibration.get_R_pixel_size()
    @R_pixel_size.setter
    def R_pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(0,[0,x[0]])
        self.set_dim(1,[0,x[1]])
        self.calibration.set_R_pixel_size(x)
    @property
    def R_pixel_units(self):
        return self.calibration.get_R_pixel_units()
    @R_pixel_units.setter
    def R_pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[0] = x[0]
        self.dim_units[1] = x[1]
        self.calibration.set_R_pixel_units(x)

    # R pixel sizes/units
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()
    @Q_pixel_size.setter
    def Q_pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(2,[0,x[0]])
        self.set_dim(3,[0,x[1]])
        self.calibration.set_Q_pixel_units(x)
    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()
    @Q_pixel_units.setter
    def Q_pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[2] = x[0]
        self.dim_units[3] = x[1]
        self.calibration.set_Q_pixel_units(x)

    # calibration
    #@property
    #def calibration(self):
    #    if self._calibration is not None:
    #        return self._calibration
    #    elif self._parent_calibration is not None:
    #        return self._parent_calibration
    #    else:
    #        return None
    #@calibration.setter
    #def calibration(self,x):
    #    self._calibration = x
    #    self.tree = x,'calibration'




    # HDF5 read/write

    # write is inherited from Array

    # read
    def from_h5(group):
        from .io_py4dstem import DataCube_from_h5
        return DataCube_from_h5(group)



############ END OF CLASS ###########






