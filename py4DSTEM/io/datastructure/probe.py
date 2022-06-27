# Defines the Probe class, which stores vacuum probes
# and cross-correlation kernels derived from them

from .array import Array

from typing import Optional,Union
import numpy as np
import h5py

class Probe(Array):
    """
    Stores 4D-STEM datasets.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'probe',
        ):
        """
        Accepts:
            data (np.ndarray): the 2D vacuum probe array
            name (str): a name

        Returns:
            A new Probe instance
        """

        # initialize as an Array
        Array.__init__(
            self,
            data = np.dstack([
                data,
                np.zeros_like(data)
            ]),
            name = name,
            units = 'intensity',
            dim_names = [
                'Qx',
                'Qy'
            ],
            slicelabels = [
                'probe',
                'kernel'
            ]
        )

        # Make calibration container
        # set its size/units
        # add to tree
        #self.calibration = Calibration()
        #self.calibration['R_pixel_size'] = R_pixel_size
        #self.calibration['R_pixel_units'] = R_pixel_units
        #self.calibration['Q_pixel_size'] = Q_pixel_size
        #self.calibration['Q_pixel_units'] = Q_pixel_units
        #self.tree['calibration'] = self.calibration



    ## properties

    # FOV
    # @property
    # def R_Nx(self):
    #     return self.data.shape[0]
    # @property
    # def R_Ny(self):
    #     return self.data.shape[1]
    # @property
    # def Q_Nx(self):
    #     return self.data.shape[2]
    # @property
    # def Q_Ny(self):
    #     return self.data.shape[3]

    # @property
    # def Rshape(self):
    #     return (self.data.shape[0],self.data.shape[1])
    # @property
    # def Qshape(self):
    #     return (self.data.shape[2],self.data.shape[3])

    # @property
    # def R_N(self):
    #     return self.R_Nx*self.R_Ny


    # # pixel sizes/units

    # # R
    # @property
    # def R_pixel_size(self):
    #     return self.calibration.get_R_pixel_size()
    # @R_pixel_size.setter
    # def R_pixel_size(self,x):
    #     if type(x) is not list: x = [x,x]
    #     self.set_dim(0,[0,x[0]])
    #     self.set_dim(1,[0,x[1]])
    #     self.calibration.set_R_pixel_size(x)
    # @property
    # def R_pixel_units(self):
    #     return self.calibration.get_R_pixel_units()
    # @R_pixel_units.setter
    # def R_pixel_units(self,x):
    #     if type(x) is not list: x = [x,x]
    #     self.dim_units[0] = x[0]
    #     self.dim_units[1] = x[1]
    #     self.calibration.set_R_pixel_units(x)

    # # Q 
    # @property
    # def Q_pixel_size(self):
    #     return self.calibration.get_Q_pixel_size()
    # @Q_pixel_size.setter
    # def Q_pixel_size(self,x):
    #     if type(x) is not list: x = [x,x]
    #     self.set_dim(2,[0,x[0]])
    #     self.set_dim(3,[0,x[1]])
    #     self.calibration.set_Q_pixel_size(x)
    # @property
    # def Q_pixel_units(self):
    #     return self.calibration.get_Q_pixel_units()
    # @Q_pixel_units.setter
    # def Q_pixel_units(self,x):
    #     if type(x) is not list: x = [x,x]
    #     self.dim_units[2] = x[0]
    #     self.dim_units[3] = x[1]
    #     self.calibration.set_Q_pixel_units(x)


    # # calibration
    # @property
    # def calibration(self):
    #     return self.tree['calibration']
    # @calibration.setter
    # def calibration(self, x):
    #     assert( isinstance( x, Calibration))
    #     self.tree['calibration'] = x



    # HDF5 read/write

    # write is inherited from Array

    # read
    #def from_h5(group):
    #    from .io_py4dstem import DataCube_from_h5
    #    return DataCube_from_h5(group)






