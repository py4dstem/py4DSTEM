# Defines the DataCube class, which stores 4D-STEM datacubes

from typing import Optional,Union
import numpy as np
import h5py

from py4DSTEM.io.classes.array import Array
from py4DSTEM.io.classes.py4dstem.datacube_fns import DataCubeMethods
from py4DSTEM.io.classes.py4dstem.calibration import Calibration
from py4DSTEM.io.classes import Branch
##;pfrom py4DSTEM.io.classes.tree import ParentBranch

class DataCube(Array,DataCubeMethods):
    """
    Storage and processing methods for 4D-STEM datasets.
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
                pixel units. Must be 'pixels' or 'A^-1'.
            slicelabels (None or list): names for slices if this is a
                stack of datacubes
            calibration (Calibration):

        Returns:
            A new DataCube instance
        """

        # initialize DataCubeMethods
        super(DataCubeMethods).__init__()

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'pixel intensity',
            dims = [
                R_pixel_size,
                R_pixel_size,
                Q_pixel_size,
                Q_pixel_size
            ],
            dim_units = [
                R_pixel_units,
                R_pixel_units,
                Q_pixel_units,
                Q_pixel_units
            ],
            dim_names = [
                'Rx',
                'Ry',
                'Qx',
                'Qy'
            ],
            slicelabels = slicelabels
        )


        # TODO: calibration
        # get the calibration...

        # set the units/sizes...
        # set size/units
        #self.tree['calibration'].set_R_pixel_size( R_pixel_size )
        #self.tree['calibration'].set_R_pixel_units( R_pixel_units )
        #self.tree['calibration'].set_Q_pixel_size( Q_pixel_size )
        #self.tree['calibration'].set_Q_pixel_units( Q_pixel_units )

    # TODO: calibration property
    # calibration
    @property
    def calibration(self):
        return self.tree['calibration']
    @calibration.setter
    def calibration(self, x):
        assert( isinstance( x, Calibration))
        self.tree['calibration'] = x








    ## shape properties

    # FOV
    @property
    def R_Nx(self):
        return self.data.shape[0]
    @property
    def R_Ny(self):
        return self.data.shape[1]
    @property
    def Q_Nx(self):
        return self.data.shape[2]
    @property
    def Q_Ny(self):
        return self.data.shape[3]

    @property
    def Rshape(self):
        return (self.data.shape[0],self.data.shape[1])
    @property
    def Qshape(self):
        return (self.data.shape[2],self.data.shape[3])

    @property
    def R_N(self):
        return self.R_Nx*self.R_Ny


    # pixel sizes/units

    # R
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

    # Q 
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()
    @Q_pixel_size.setter
    def Q_pixel_size(self,x):
        if type(x) is not list: x = [x,x]
        self.set_dim(2,[0,x[0]])
        self.set_dim(3,[0,x[1]])
        self.calibration.set_Q_pixel_size(x)
    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()
    @Q_pixel_units.setter
    def Q_pixel_units(self,x):
        if type(x) is not list: x = [x,x]
        self.dim_units[2] = x[0]
        self.dim_units[3] = x[1]
        self.calibration.set_Q_pixel_units(x)




    # HDF5 i/o

    # write is inherited from Array

    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in read
        mode.  Determines if an Array object of this name exists inside this
        group, and if it does, loads and returns it as a DataCube. If it doesn't
        exist, or if it exists but does not have a rank of 4, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A DataCube instance
        """
        # Load from H5 as an Array
        array = Array.from_h5(group)

        # Convert to a DataCube
        assert(array.rank == 4), "Array must have 4 dimensions"
        array.__class__ = DataCube
        array.__init__(
            data = array.data,
            name = array.name,
            R_pixel_size = array.dims[0][1]-array.dims[0][0],
            R_pixel_units = array.dim_units[0],
            Q_pixel_size = array.dims[2][1]-array.dims[2][0],
            Q_pixel_units = array.dim_units[2],
            slicelabels = array.slicelabels
        )

        # Return
        return array



    # TODO
    # add _get_constructor_args                 (2)
    # add _the_other_one


    # TODO:
    # write unit tests for datacube             (1)



# rough plan: do 1/2 above without calibration attached.
# then go to Calibration, write unit tests, execute
# then return here, add calibration to datacube

# TODO: we need a class, Py4dstemData, which all these hooplas inherit from
# which adds...
# - calibration
# - microscope


############ END OF CLASS ###########







