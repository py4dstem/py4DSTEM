# Defines the BraggVectors class


from typing import Optional,Union
import numpy as np
import h5py
from os.path import basename

from py4DSTEM.io.classes import (
    Custom,
    PointListArray,
    Metadata
)


class BraggVectors(Custom):
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

    from py4DSTEM.io.classes.py4dstem.braggvectors_fns import (
        get_bvm,
        measure_origin,
        fit_origin,
        calibrate,
        choose_lattice_vectors,
        index_bragg_directions,
        add_indices_to_braggpeaks,
        fit_lattice_vectors_all_DPs,
        get_strain_from_reference_region,
        get_rotated_strain_map,
        get_strain_from_reference_g1g2,
        get_masked_peaks,
    )

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




    ## Representation to standard output

    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( "
        string += f"A {self.shape}-shaped array of lists of bragg vectors )"
        return string


    # copy
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
            'Rshape' : Rshape,
            'Qshape' : Qshape
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






