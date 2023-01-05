# Defines the BraggVectors class


from typing import Optional,Union
import numpy as np
import h5py

from py4DSTEM.io.classes import PointListArray
from py4DSTEM.io.classes.tree import Tree
from py4DSTEM.io.classes.metadata import Metadata



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

        self.name = name
        self.Rshape = Rshape
        self.shape = self.Rshape
        self.Qshape = Qshape

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        if 'braggvectors' not in self._metadata.keys():
            self.metadata = Metadata( name='braggvectors' )
        self.metadata['braggvectors']['Qshape'] = self.Qshape

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


    # HDF5 i/o

    # write
    def to_h5(self,group):
        """
        Takes a valid group for an HDF5 file object which is open in
        write or append mode. Writes a new group with a name given by this
        BraggVectors .name field nested inside the passed group, and saves
        the data there.

        Accepts:
            group (HDF5 group)
        """
        ## Write the group
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type", 4) # this tag indicates a Custom type
        grp.attrs.create("py4dstem_class", self.__class__.__name__)

        # Add uncalibrated vectors
        self._v_uncal.name = "_v_uncal"
        self._v_uncal._to_h5( grp )

        # Add calibrated vectors, if present
        try:
            self._v_cal.name = "_v_cal"
            self._v_cal._to_h5( grp )
        except AttributeError:
            pass

        # Add metadata
        _write_metadata(self, grp)


    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in read mode,
        Determines if a valid BraggVectors object of this name exists inside
        this group, and if it does, loads and returns it. If it doesn't, raises
        an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A BraggVectors instance
        """
        # Confirm correct group type
        er = f"Group {group} is not a valid BraggVectors group"
        assert("emd_group_type" in group.attrs.keys()), er
        assert(group.attrs["emd_group_type"] == 4), er

        # Get uncalibrated peaks
        v_uncal = PointListArray.from_h5(group['_v_uncal'])

        # Get Qshape metadata
        try:
            grp_metadata = group['_metadata']
            Qshape = Metadata.from_h5(grp_metadata['braggvectors'])['Qshape']
        except KeyError:
            raise Exception("could not read Qshape")

        # Set up BraggVectors
        braggvectors = BraggVectors(
            v_uncal.shape,
            Qshape = Qshape,
            name = basename(group.name)
        )
        braggvectors._v_uncal = v_uncal

        # Add calibrated peaks, if they're there
        try:
            v_cal = PointListArray.from_h5(group['_v_cal'])
            braggvectors._v_cal = v_cal
        except KeyError:
            pass

        # Add remaining metadata
        _read_metadata(braggvectors, group)

        # Return
        return braggvectors






############ END OF CLASS ###########











