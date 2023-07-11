# Defines the Strain class

import numpy as np
from typing import Optional
from py4DSTEM.data import RealSlice, Data
from py4DSTEM.braggvectors import BraggVectors



class StrainMap(RealSlice,Data):
    """
    Stores strain map.

    TODO add docs

    """

    def __init__(
        self,
        braggvectors: BraggVectors,
        name: Optional[str] = 'strainmap'
        ):
        """
        TODO
        """
        # set up braggvectors
        self.braggvectors = braggvectors
        # TODO - how to handle changes to braggvectors
        # TODO - ellipse cal or no?

        # set up data
        data = np.empty((
            self.braggvectors.Rshape[0],
            self.braggvectors.Rshape[1],
            6
        ))

        # initialize as a RealSlice
        RealSlice.__init__(
            self,
            name = name,
            data = data,
            slicelabels = [
                'exx',
                'eyy',
                'exy',
                'theta',
                'mask',
                'error'
            ]
        )


    # braggvector properties

    @property
    def braggvectors(self):
        return self._braggvectors
    @braggvectors.setter
    def braggvectors(self,x):
        assert(isinstance(x,BraggVectors)), f".braggvectors must be BraggVectors, not type {type(x)}"
        assert(x.calibration.origin is not None), f"braggvectors must have a calibrated origin"
        self._braggvectors = x


    # TODO - copy method

    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = RealSlice._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
        }
        return args



