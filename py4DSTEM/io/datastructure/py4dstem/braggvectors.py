# Defines the BraggVectors class


from typing import Optional,Union
import numpy as np
import h5py

from ..emd import PointListArray
from ..emd.tree import Tree
from ..emd.metadata import Metadata



class BraggVectors:
    """
    Stores bragg scattering information for a 4D datacube.

        >>> braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )

    initializes an instance of the appropriate shape for a DataCube `datacube`.

        >>> braggvectors.v[rx,ry]
        >>> braggvectors.v_uncal[rx,ry]

    retrieve, respectively, the calibrated and uncalibrated bragg vectors at
    scan position [rx,ry], and

        >>> braggvectors.v[rx,ry]['qx']
        >>> braggvectors.v[rx,ry]['qy']
        >>> braggvectors.v[rx,ry]['intensity']

    retrieve the positiona and intensity of the scattering.
    """

    from .braggvectors_fns import (
        get_bvm,
        measure_origin,
        fit_origin,
        calibrate
    )

    def __init__(
        self,
        Rshape,
        Qshape,
        name = 'braggvectors'
        ):

        self.name = name
        self.shape = Rshape
        self.Qshape = Qshape

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        if 'braggvectors' not in self._metadata.keys():
            self.metadata = Metadata( name='braggvectors' )
        self.metadata['braggvectors']['Qshape'] = self.Qshape

        self._v_uncal = PointListArray(
            dtype = [
                ('qx',np.float),
                ('qy',np.float),
                ('intensity',np.float)
            ],
            shape = Rshape,
            name = 'v_uncal'
        )

        self._v_cal = PointListArray(
            dtype = [
                ('qx',np.float),
                ('qy',np.float),
                ('intensity',np.float)
            ],
            shape = Rshape,
            name = 'v_cal'
        )



    @property
    def vectors(self):
        return self._v_cal

    @property
    def vectors_uncal(self):
        return self._v_uncal


    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self,x):
        assert(isinstance(x,Metadata))
        self._metadata[x.name] = x



    ## Representation to standard output

    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string






    # HDF5 read/write

    # write
    def to_h5(self,group):
        from .io import BraggVectors_to_h5
        BraggVectors_to_h5(self,group)


    # read
    def from_h5(group):
        from .io import BraggVectors_from_h5
        return BraggVectors_from_h5(group)






############ END OF CLASS ###########











