import numpy as np
from numbers import Number
from typing import Optional
import h5py

from .tree import Tree


class Metadata:
    """
    Stores metadata in the form of a flat (non-nested) dictionary.
    Keys are arbitrary strings.  Values may be strings, numbers, arrays,
    or lists of the above types.

    Usage:

        >>> meta = Metadata()
        >>> meta['param'] = value
        >>> val = meta['param']

    If the parameter has not been set, the getter methods return None.
    """
    def __init__(
        self,
        name: Optional[str] ='metadata'
        ):
        """
         Args:
            name (Optional, string):
        """
        self.name = name
        self.tree = Tree()

        # create parameter dictionary
        self._params = {}



    ### __get/setitem__

    def __getitem__(self,x):
        return self._params[x]
    def __setitem__(self,k,v):
        self._params[k] = v


    @property
    def keys(self):
        return self._params.keys()


    def copy(self,name=None):
        """
        """
        if name is None: name = self.name+"_copy"
        md = Metadata(name=name)
        md._params.update(self._params)
        return md



    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A Metadata instance called '{self.name}', containing the following fields:"
        string += "\n"

        maxlen = 0
        for k in self._params.keys():
            if len(k)>maxlen: maxlen=len(k)

        for k,v in self._params.items():
            if isinstance(v,np.ndarray):
                v = f"{v.ndim}D-array"
            string += "\n"+space+f"{k}:{(maxlen-len(k)+3)*' '}{str(v)}"
        string += "\n)"

        return string



    # HDF5 read/write

    def to_h5(self,group):
        from .io import Metadata_to_h5
        Metadata_to_h5(self,group)

    def from_h5(group):
        from .io import Metadata_from_h5
        return Metadata_from_h5(group)




