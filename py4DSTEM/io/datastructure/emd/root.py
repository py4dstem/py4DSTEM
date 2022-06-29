import numpy as np
from numbers import Number
from typing import Optional
import h5py

from .tree import Tree


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



    # HDF5 read/write

    def to_h5(self,group):
        from .io import Root_to_h5
        Root_to_h5(self,group)

    def from_h5(group):
        from .io import Root_from_h5
        return Root_from_h5(group)




