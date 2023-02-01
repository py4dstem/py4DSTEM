# Defines the DataCube class, which stores 4D-STEM datacubes

from typing import Optional,Union
import numpy as np

from py4DSTEM.emd import Array,RootedNode
from py4DSTEM.classes import Data, Calibration
from py4DSTEM.classes.methods import DataCubeMethods

class DataCube(Array,RootedNode,Data,DataCubeMethods):
    """
    Storage and processing methods for 4D-STEM datasets.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'datacube',
        calibration: Optional[Union[Calibration,None,'skip']] = None,
        slicelabels: Optional[Union[bool,list]] = None,
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the datacube
            calibration (None or Calibration or 'pass'): default (None)
                creates and attaches a new Calibration instance to root
                metadata, or, passing a Calibration instance uses this instead.
                'skip' is for internal use for the reader
            slicelabels (None or list): names for slices if this is a
                stack of datacubes

        Returns:
            A new DataCube instance.
        """

        # initialize DataCubeMethods
        super(DataCubeMethods).__init__()

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'pixel intensity',
            dim_names = [
                'Rx',
                'Ry',
                'Qx',
                'Qy'
            ],
            slicelabels = slicelabels
        )

        # Set up EMD tree
        RootedNode.__init__(self)

        # Check for / set up a calibration
        self._setup_calibration( calibration )



    def _setup_calibration(self, cal):
        """
        Ensures that a calibration instance exists. Passing None
        makes a new Calibration instance, puts it in root.calibration, and
        makes a two way link. Passing a Calibration instance attaches that
        instance. `'skip'` does nothing (used when reading from file).
        """
        if cal is None:
            self.calibration = Calibration( datacube = self )
        elif cal == 'skip':
            pass
        else:
            assert(isinstance(cal, Calibration)), "`calibration` must be a Calibration instance, not type f{type(cal)}"
            self.calibration = cal
            cal._datacube = self






    # properties


    ## pixel size / units

    # Q 
    @property
    def Q_pixel_size(self):
        return self.calibration.get_Q_pixel_size()
    @property
    def Q_pixel_units(self):
        return self.calibration.get_Q_pixel_units()

    # R
    @property
    def R_pixel_size(self):
        return self.calibration.get_R_pixel_size()
    @property
    def R_pixel_units(self):
        return self.calibration.get_R_pixel_units()

    # aliases
    qpixsize = Q_pixel_size
    qpixunit = Q_pixel_units
    rpixsize = R_pixel_size
    rpixunit = R_pixel_units


    ## shape

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

    # aliases
    qnx = Q_Nx
    qny = Q_Ny
    rnx = R_Nx
    rny = R_Ny
    rshape = Rshape
    qshape = Qshape
    rn = R_N





    # HDF5 i/o

    # to_h5 is inherited from Array

    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """ Construct a datacube with no calibration / metadata
        """
        # We only need some of the Array constructors;
        # dim vector/units are passed through when Calibration
        # is loaded, and the runtim dim vectors are then set
        # in _add_root_links
        ar_args = Array._get_constructor_args(group)

        args = {
            'data': ar_args['data'],
            'name': ar_args['name'],
            'slicelabels': ar_args['slicelabels'],
            'calibration': 'skip'
        }

        return args


    def _add_root_links(self,group):
        """ When reading from file, link to calibration metadata,
        then use it to populate the datacube dim vectors
        """
        # Link to the datacube
        self.calibration._datacube = self

        # Populate dim vectors
        self.calibration.set_Q_pixel_size( self.calibration.get_Q_pixel_size() )
        self.calibration.set_R_pixel_size( self.calibration.get_R_pixel_size() )
        self.calibration.set_Q_pixel_units( self.calibration.get_Q_pixel_units() )
        self.calibration.set_R_pixel_units( self.calibration.get_R_pixel_units() )

        return



