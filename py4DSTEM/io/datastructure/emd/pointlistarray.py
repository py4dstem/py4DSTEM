import numpy as np
from copy import copy
from typing import Optional
import h5py

from .tree import Tree
from .metadata import Metadata
from .pointlist import PointList

class PointListArray:
    """
    An 2D array of PointLists which share common coordinates.
    """
    def __init__(
        self,
        dtype,
        shape,
        name: Optional[str] = 'pointlistarray',
        ):
        """
		Creates an empty PointListArray.

        Args:
            dtype: the dtype of the numpy structured arrays which will comprise
                the data of each PointList
            shape (2-tuple of ints): the shape of the array of PointLists
            name (str): a name for the PointListArray

        Returns:
            a PointListArray instance
        """
        assert len(shape) == 2, "Shape must have length 2."

        self.name = name
        self.shape = shape

        self.dtype = np.dtype(dtype)
        self.fields = self.dtype.names
        self.types = tuple([self.dtype.fields[f][0] for f in self.fields])

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}

        # Populate with empty PointLists
        self._pointlists = [[PointList(data=np.zeros(0,dtype=self.dtype), name=f"{i},{j}")
                             for j in range(self.shape[1])] for i in range(self.shape[0])]


    ## get/set pointlists

    def get_pointlist(self, i, j, name=None):
        """
        Returns the pointlist at i,j
        """
        pl = self._pointlists[i][j]
        if name is not None:
            pl = pl.copy(name=name)
        return pl

    def __getitem__(self, tup):
        l = len(tup) if isinstance(tup,tuple) else 1
        assert(l==2), f"Expected 2 slice values, recieved {l}"
        return self.get_pointlist(tup[0],tup[1])

    def __setitem__(self, tup, pointlist):
        l = len(tup) if isinstance(tup,tuple) else 1
        assert(l==2), f"Expected 2 slice values, recieved {l}"
        assert(pointlist.fields == self.fields), "fields must match"
        self._pointlists[tup[0]][tup[1]] = pointlist



    ## Make copies

    def copy(self, name=''):
        """
        Returns a copy of itself.
        """
        new_pla = PointListArray(
            dtype=self.dtype,
            shape=self.shape,
            name=name)

        for i in range(new_pla.shape[0]):
            for j in range(new_pla.shape[1]):
                pl = new_pla.get_pointlist(i,j)
                pl.add(np.copy(self.get_pointlist(i,j).data))

        for k,v in self.metadata.items():
            new_pla.metadata = v.copy(name=k)

        return new_pla

    def add_fields(self, new_fields, name=''):
        """
        Creates a copy of the PointListArray, but with additional fields given
        by new_fields.

        Args:
            new_fields: a list of 2-tuples, ('name', dtype)
            name: a name for the new pointlist
        """
        dtype = []
        for f,t in zip(self.fields,self.types):
            dtype.append((f,t))
        for f,t in new_fields:
            dtype.append((f,t))

        new_pla = PointListArray(
            dtype=dtype,
            shape=self.shape,
            name=name)

        for i in range(new_pla.shape[0]):
            for j in range(new_pla.shape[1]):
                # Copy old data into a new structured array
                pl_old = self.get_pointlist(i,j)
                data = np.zeros(pl_old.length, np.dtype(dtype))
                for f in self.fields:
                    data[f] = np.copy(pl_old.data[f])

                # Write into new pointlist
                pl_new = new_pla.get_pointlist(i,j)
                pl_new.add(data)

        return new_pla


    # set up metadata property

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
        string = f"{self.__class__.__name__}( A shape {self.shape} PointListArray called '{self.name}',"
        string += "\n"+space+f"with {len(self.fields)} fields:"
        string += "\n"
        space2 = max([len(field) for field in self.fields])+3
        for f,t in zip(self.fields,self.types):
            string += "\n"+space+f"{f}{(space2-len(f))*' '}({str(t)})"
        string += "\n)"

        return string



    # HDF5 read/write

    def to_h5(self,group):
        from .io import PointListArray_to_h5
        PointListArray_to_h5(self,group)

    def from_h5(group):
        from .io import PointListArray_from_h5
        return PointListArray_from_h5(group)




