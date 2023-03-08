# Defines the BraggVectors class


from typing import Optional,Union
import numpy as np
from os.path import basename

from emdfile import (
    Custom,
    PointListArray,
    Metadata
)

from py4DSTEM.classes.methods import BraggVectorMethods




class BraggVectors(Custom,BraggVectorMethods):
    """
    Stores bragg scattering information for a 4D datacube.

        >>> braggvectors = BraggVectors( datacube.Rshape, datacube.Qshape )

    initializes an instance of the appropriate shape for a DataCube `datacube`.

        >>> braggvectors.vectors[rx,ry]
        >>> braggvectors.vectors_uncal[rx,ry]

    or their aliases

        >>> braggvectors.v[rx,ry]
        >>> braggvectors.v_uncal[rx,ry]

    retrieve the calibrated and uncalibrated bragg vectors at
    scan position [rx,ry], and

        >>> braggvectors.v[rx,ry]['qx']
        >>> braggvectors.v[rx,ry]['qy']
        >>> braggvectors.v[rx,ry]['intensity']

    retrieve the position and intensity of the scattering.
    """

    def __init__(
        self,
        Rshape,
        Qshape,
        name = 'braggvectors'
        ):
        Custom.__init__(self,name=name)

        self.Rshape = Rshape
        self.shape = self.Rshape
        self.Qshape = Qshape

        self._v_uncal = PointListArray(
            dtype = [
                ('qx',np.float64),
                ('qy',np.float64),
                ('intensity',np.float64)
            ],
            shape = Rshape,
            name = '_v_uncal'
        )

    # vector properties

    @property
    def vectors(self):
        try:
            return self._v_cal
        except AttributeError:
            er = "No calibrated bragg vectors found. Try running .calibrate()!"
            raise Exception(er)
    @property
    def vectors_uncal(self):
        return self._v_uncal

    # aliases
    v = vectors
    v_uncal = vectors_uncal




    # methods

    def copy(self, name=None):
        name = name if name is not None else self.name+"_copy"
        braggvector_copy = BraggVectors(self.Rshape, self.Qshape, name=name)
        braggvector_copy._v_uncal = self._v_uncal.copy()
        try:
            braggvector_copy._v_cal = self._v_cal.copy()
        except AttributeError:
            pass
        for k in self.metadata.keys():
            braggvector_copy.metadata = self.metadata[k].copy()
        if hasattr(self,'calibration'):
            braggvector_copy.calibration = self.calibration

        return braggvector_copy




    # write

    def to_h5(self,group):
        """ Constructs the group, adds the bragg vector pointlists,
        and adds metadata describing the shape
        """
        md = Metadata( name = '_braggvectors_shape' )
        md['Rshape'] = self.Rshape
        md['Qshape'] = self.Qshape
        self.metadata = md
        grp = Custom.to_h5(self,group)




    # read

    @classmethod
    def _get_constructor_args(cls,group):
        """
        """
        # Get shape metadata from the metadatabundle group
        assert('metadatabundle' in group.keys()), "No metadata found, can't get Rshape and Qshape"
        grp_metadata = group['metadatabundle']
        assert('_braggvectors_shape' in grp_metadata.keys()), "No _braggvectors_shape metadata found"
        md = Metadata.from_h5(grp_metadata['_braggvectors_shape'])
        # Populate args and return
        kwargs = {
            'name' : basename(group.name),
            'Rshape' : md['Rshape'],
            'Qshape' : md['Qshape']
        }
        return kwargs

    def _populate_instance(self,group):
        """
        """
        dic = self._get_emd_attr_data(group)
        assert('_v_uncal' in dic.keys()), "Uncalibrated bragg vectors not found!"
        self._v_uncal = dic['_v_uncal']
        if '_v_cal' in dic.keys():
            self._v_cal = dic['_v_cal']



    # standard output display

    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string


