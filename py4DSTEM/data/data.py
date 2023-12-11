# Base class for all py4DSTEM data
# which adds an EMD root and a pointer to 'calibration' metadata

import warnings

from emdfile import Node, Root
from py4DSTEM.data import Calibration


class Data:
    """
    The purpose of the `Data` class is to ensure calibrations are linked
    to data containing class instances, while allowing multiple objects
    to share a single Calibration. The calibrations of a Data instance
    `data` is accessible as

        >>> data.calibration

    In py4DSTEM, Data containing objects are stored internally in filetree
    like representations, defined by the EMD1.0 and `emdfile` specifications,
    e.g.

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

    Calibrations are metadata which always live in the root of such a tree.
    Running `data.calibration` returns the calibrations from the tree root,
    and therefore the same calibration instance is referred to be all objects
    in the same tree.  The root itself is accessible from any Data instance
    as

        >>> data.root

    To examine the tree of a Data instance, in a Python interpreter do

        >>> data.tree(True)

    to display the whole data tree, and

        >>> data.tree()

    to display the tree of from the current node on, i.e. the branch
    downstream of `data`.

    Calling

        >>> data.calibration

    will raise a warning and return None if no root calibrations are found.

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

    def __init__(self, calibration=None):
        assert isinstance(self, Node), "Data instances must inherit from Node"
        assert calibration is None or isinstance(
            calibration, Calibration
        ), f"calibration must be None or a Calibration instance, not type {type(calibration)}"

        # set up calibration + EMD tree
        if calibration is None:
            if self.root is None:
                root = Root(name=self.name + "_root")
                root.tree(self)
                self.calibration = Calibration()
            elif "calibration" not in self.root.metadata:
                self.calibration = Calibration()
            else:
                pass
        elif calibration.root is None:
            if self.root is None:
                root = Root(name=self.name + "_root")
                root.tree(self)
                self.calibration = calibration
            elif "calibration" not in self.root.metadata:
                self.calibration = calibration
            else:
                warnings.warn(
                    "A calibration was passed to instantiate a new Data instance, but the instance already has a calibration. The passed calibration *WAS NOT* attached.  To attach the new calibration and overwrite the existing calibration, use `data.calibration = new_calibration`"
                )
                pass
        else:
            if self.root is None:
                calibration.root.tree(self)
                self.calibration = calibration
            elif "calibration" not in self.root.metadata:
                self.calibration = calibration
                warnings.warn(
                    "A calibration was passed to instantiate a new Data instance.  The Data already had a root but no calibration, and the calibration already exists in a different root.  The calibration has been added and now lives in both roots, and can therefore be modified from either place!"
                )
            else:
                warnings.warn(
                    "A calibration was passed to instantiate a new Data instance, however the Data already has a root and calibration, and the calibration already has a root!! The passed calibration *WAS NOT* attached. To attach the new calibration and overwrite the existing calibration, use `data.calibration = new_calibration."
                )

    # calibration property

    @property
    def calibration(self):
        try:
            return self.root.metadata["calibration"]
        except KeyError:
            warnings.warn("No calibration metadata found in root, returning None")
            return None
        except AttributeError:
            warnings.warn("No root or root metadata found, returning None")
            return None

    @calibration.setter
    def calibration(self, x):
        assert isinstance(x, Calibration)
        if "calibration" in self.root.metadata.keys():
            warnings.warn(
                "A 'calibration' key already exists in root.metadata - overwriting..."
            )
        x.name = "calibration"
        self.root.metadata["calibration"] = x

    # transfer trees

    def attach(self, node):
        """
        Attach `node` to the current object's tree, attaching calibration and detaching
        calibrations as needed.
        """
        assert isinstance(node, Node), f"node must be a Node, not type {type(node)}"
        register = False
        if hasattr(node, "calibration"):
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
