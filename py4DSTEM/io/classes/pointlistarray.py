import numpy as np
from copy import copy
from typing import Optional
import h5py
from os.path import basename

from py4DSTEM.io.classes.tree import Tree
from py4DSTEM.io.classes.metadata import Metadata
from py4DSTEM.io.classes.pointlist import PointList
from py4DSTEM.io.classes.class_io_utils import _read_metadata, _write_metadata

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



    # HDF5 i/o

    # write
    def to_h5(self,group):
        """
        Takes a valid group for an HDF5 file object which is open in
        write or append mode. Writes a new group with this object's name
        and saves this object's data in it.

        Accepts:
            group (HDF5 group)
        """
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",3) # this tag indicates a PointListArray
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # Add metadata
        dtype = h5py.special_dtype(vlen=self.dtype)
        dset = grp.create_dataset(
            "data",
            self.shape,
            dtype
        )

        # Add data
        for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
            dset[i,j] = self[i,j].data

        # Add additional metadata
        _write_metadata(self, grp)


    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in read mode,
        Determines if a valid PointListArray object of this name exists
        inside this group, and if it does, loads and returns it. If it doesn't,
        raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A PointListArray instance
        """
        er = f"Group {group} is not a valid EMD PointListArray group"
        assert("emd_group_type" in group.attrs.keys()), er
        assert(group.attrs["emd_group_type"] == EMD_group_types['PointListArray']), er

       # Get the DataSet
        dset = group['data']
        dtype = h5py.check_vlen_dtype( dset.dtype )
        shape = dset.shape

        # Initialize a PointListArray
        pla = PointListArray(
            dtype=dtype,
            shape=shape,
            name=basename(group.name)
        )

        # Add data
        for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
            try:
                pla[i,j].add(dset[i,j])
            except ValueError:
                pass

        # Add metadata
        _read_metadata(pla, group)

        return pla




