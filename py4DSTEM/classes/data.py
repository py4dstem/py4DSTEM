# Base class for all py4DSTEM data
# which adds a pointer to 'calibration' metadata

import warnings

from emdfile import Node, Root
from py4DSTEM.classes import Calibration


class Data:

    def __init__(
        self,
        setup_tree = True,
        calibration = None
        ):
        assert(isinstance(self,Node)), "Data instances must alse inherit from Node"
        pass

        if setup_tree:
            # set up EMD tree
            root = Root( name = self.name+"_root" )
            root.tree( self )

            # set up calibration
            calibration = Calibration() if calibration is None else calibration
            self._setup_calibration( calibration )


    # calibration

    @property
    def calibration(self):
        try:
            return self.root.metadata['calibration']
        except KeyError:
            warnings.warn("No calibration metadata found in root, returning None")
            return None
        except AttributeError:
            warnings.warn("No root or root metadata found, returning None")
            return None

    @calibration.setter
    def calibration(self, x):
        assert( isinstance( x, Calibration) )
        if 'calibration' in self.root.metadata.keys():
            warnings.warn("A 'calibration' key already exists in root.metadata - overwriting...")
        x.name = 'calibration'
        self.root.metadata['calibration'] = x





