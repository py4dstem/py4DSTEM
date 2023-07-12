# Base class for all py4DSTEM data
# which adds an EMD root and a pointer to 'calibration' metadata

import warnings

from emdfile import Node, Root
from py4DSTEM.data import Calibration


class Data:
    """
    Data in py4DSTEM is stored in filetree like representations, e.g.

    Root
      |--metadata
      |     |--calibration
      |
      |--some_object(e.g.datacube)
      |     |--another_object(e.g.max_dp)
      |             |--etc.
      |
      |--one_more_object(e.g.crystal)
      |     |--etc.
      :

    In a Python interpreter, do

        >>> data.tree(True)

    to display the data tree of Data instance `data`, and

        >>> data.tree()

    to display the tree of from the current node on, i.e. the branch
    downstream of `data`.

    Every object can access the calibrations which live in the root metadata
    of its tree with

        >>> data.calibration

    which returns the calibrations, or, if none are found, raises a warning
    and returns None.

    Some objects should be modified when the calibrations change - these
    objects must have .calibrate() method, which is called any time relevant
    calibration parameters change if the object has been registered with
    the calibrations.

    To transfer `data` from it's current tree to another existing tree, use

        >>> data.attach(some_other_data)

    which will move the data to the new tree. If the data was registered with
    it's old calibrations, this will also de-register it there and register
    it with the new calibrations such that .calibrate() is called when it
    should be.

    See also the Calibration docstring.
    """

    def __init__(
        self,
        calibration = None
        ):
        assert(isinstance(self,Node)), "Data instances must inherit from Node"
        pass

        # set up calibration + EMD tree
        if calibration is None:
            if self.root is not None and 'calibration' in self.root.metadata:
                pass
            else:
                root = Root( name = self.name+"_root" )
                root.tree( self )
                self.calibration = Calibration()
        else:
            assert(isinstance(calibration,Calibration)), f"`calibration` must be a Calibration, not type {type(calibration)}"
            if calibration.root is None:
                calibration._root = Root( name = self.name+"_root" )
                calibration.root.tree( self )
            else:
                calibration.root.tree( self )
            self.calibration = calibration


    # calibration property

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


    # transfer trees

    def attach(self,node):
        """
        Attach `node` to the current object's tree, attaching calibration and detaching
        calibrations as needed.
        """
        assert(isinstance(node,Node)), f"node must be a Node, not type {type(node)}"
        register = False
        if hasattr(node,'calibration'):
            if node.calibration is not None:
                if node in node.calibration._targets:
                    register = True
                    node.calibration.unregister_target(node)
        if node.root is None:
            self.tree(node)
        else:
            self.graft(node)
        if register:
            self.calibration.register_target(node)



