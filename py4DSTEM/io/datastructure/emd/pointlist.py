# Defines a class, PointList, for storing / accessing / manipulating data
# in the form of lists of vectors in named dimensions.  Wraps numpy
# structured arrays.

import numpy as np
import h5py
from copy import copy
from typing import Optional

from .tree import Tree
from .metadata import Metadata


class PointList:
    """
    A wrapper around structured numpy arrays, with read/write functionality in/out of
    py4DSTEM formatted HDF5 files.

    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'pointlist',
        ):
        """
		Instantiate a PointList.

        Args:
            data (structured numpy ndarray): the data; the dtype of this array will
                specify the fields of the PointList.
            name (str): name for the PointList

        Returns:
            a PointList instance
        """
        self.data = data
        self.name = name
        self.length = len(self.data)

        self._dtype = self.data.dtype
        self._fields = self.data.dtype.names
        self._types = tuple([self.data.dtype.fields[f][0] for f in self.fields])

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}


    # properties
    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, x):
        self._dtype = data.dtype

    @property
    def fields(self):
        return self._fields
    @fields.setter
    def fields(self, x):
        self.data.dtype.names = x
        self._fields = x

    @property
    def types(self):
        return self._types



    ## Add, remove, sort data

    def add(self, data):
        """
        Appends a numpy structured array. Its dtypes must agree with the existing data.
        """
        assert self.dtype==data.dtype, "Error: dtypes must agree"
        self.data = np.append(self.data, data)
        self.length += np.atleast_1d(data).shape[0]

    def remove(self, mask):
        """ Removes points wherever mask==True
        """
        assert np.atleast_1d(mask).shape[0] == self.length, "deletemask must be same length as the data"
        inds = mask.nonzero()[0]
        self.data = np.delete(self.data, inds)
        self.length -= len(inds)

    def sort(self, field, order='descending'):
        """
        Sorts the point list according to field,
        which must be a field in self.dtype.
        order should be 'descending' or 'ascending'.
        """
        assert field in self.fields
        assert (order=='descending') or (order=='ascending')
        if order=='ascending':
            self.data = np.sort(self.data, order=field)
        else:
            self.data = np.sort(self.data, order=field)[::-1]


    ## Copy, copy+modify PointList

    def copy(self, name=None):
        """ Returns a copy of the PointList. If name=None, sets to `{name}_copy`
        """
        name = name if name is not None else self.name+"_copy"

        pl = PointList(
            data = np.copy(self.data),
            name = name)

        for k,v in self.metadata.items():
            pl.metadata = v.copy(name=k)

        return pl

    def add_fields(self, new_fields, name=''):
        """
        Creates a copy of the PointList, but with additional fields given by new_fields.

        Args:
            new_fields: a list of 2-tuples, ('name', dtype)
            name: a name for the new pointlist
        """
        dtype = []
        for f,t in zip(self.fields,self.types):
            dtype.append((f,t))
        for f,t in new_fields:
            dtype.append((f,t))

        data = np.zeros(self.length, dtype=dtype)
        for f in self.fields:
            data[f] = np.copy(self.data[f])

        return PointList(data=data, name=name)


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
        string = f"{self.__class__.__name__}( A length {self.length} PointList called '{self.name}',"
        string += "\n"+space+f"with {len(self.fields)} fields:"
        string += "\n"
        space2 = max([len(field) for field in self.fields])+3
        for f,t in zip(self.fields,self.types):
            string += "\n"+space+f"{f}{(space2-len(f))*' '}({str(t)})"
        string += "\n)"

        return string


    # Slicing
    def __getitem__(self, v):
        return self.data[v]



    # HDF5 read/write

    def to_h5(self,group):
        from .io import PointList_to_h5
        PointList_to_h5(self,group)

    def from_h5(group):
        from .io import PointList_from_h5
        return PointList_from_h5(group)




