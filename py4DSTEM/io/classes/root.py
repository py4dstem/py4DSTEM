import numpy as np
from numbers import Number
from typing import Optional
import h5py
from os.path import basename

from py4DSTEM.io.classes.tree import Tree


class Root:
    """
    A class serving as a container for Trees
    """
    def __init__(
        self,
        name: Optional[str] ='root'
        ):
        """
         Args:
            name (Optional, string):
        """
        self.name = name
        self.tree = Tree()


    ### __get/setitem__

    def __getitem__(self,x):
        return self.tree[x]
    def __setitem__(self,k,v):
        self.tree[k] = v


    @property
    def keys(self):
        return self.tree.keys()



    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A Root instance called '{self.name}', containing the following top-level object instances:"
        string += "\n"
        for k,v in self.tree._tree.items():
            string += "\n"+space+f"    {k} \t\t ({v.__class__.__name__})"
        string += "\n)"
        return string



    # HDF5 i/o

    # write
    def to_h5(self,group):
        """
        Takes a valid group for an HDF5 file object which is open
        in write or append mode. Writes a new group with this object's
        name and saves this object's data in it.

        Accepts:
            group (HDF5 group)
        """
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",EMD_group_types['Root'])
        grp.attrs.create("py4dstem_class",self.metadata.__class__.__name__)

    # read
    def from_h5(group):
        """
        Takes a valid HDF5 group for an HDF5 file object
        which is open in read mode.  Determines if a valid
        Root object of this name exists inside this group, and if it does,
        loads and returns it. If it doesn't, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A Root instance
        """
        er = f"Group {group} is not a valid EMD Root group"
        assert("emd_group_type" in group.attrs.keys()), er
        assert(group.attrs["emd_group_type"] == EMD_group_types['Root']), er

        root = Root(basename(group.name))
        return root






