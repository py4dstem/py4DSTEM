import numpy as np
from copy import copy
import h5py

from .ioutils import determine_group_name
from .ioutils import EMD_group_exists, EMD_group_types
from .metadata import Metadata_from_h5
from .pointlist import PointList
from ...tqdmnd import tqdmnd

class PointListArray:
    """
    An 2D array of PointLists which share common coordinates.
    """
    def __init__(
        self,
        dtype,
        shape,
        name,
        calibration = None,
        ):
        """
		Creates an empty PointListArray.

        Args:
            dtype: the dtype of the numpy structured arrays which will comprise
                the data of each PointList
            shape (2-tuple of ints): the shape of the array of PointLists
            name (str): a name for the PointListArray
            calibration (Calibration): a Calibration instance

        Returns:
            a PointListArray instance
        """
        assert len(shape) == 2, "Shape must have length 2."

        self.name = name
        self.shape = shape
        self.calibration = calibration
        self._metadata = None

        self.dtype = np.dtype(dtype)
        self.fields = self.dtype.names
        self.types = tuple([self.dtype.fields[f][0] for f in self.fields])


        # Populate with empty PointLists
        self._pointlists = [[PointList(data=np.zeros(0,dtype=self.dtype), name=f"{i},{j}")
                             for j in range(self.shape[1])] for i in range(self.shape[0])]


    ## Retrieve pointlists

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
                pl.append(np.copy(self.get_pointlist(i,j).data))

        if self._metadata is not None:
            new_pla._metadata = self._metadata.copy()

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
                pl_new.append(data)

        return new_pla


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


    ## Writing to an HDF5 file

    def to_h5(self,group):
        """
        Takes a valid HDF5 group for an HDF5 file object which is open in write or append
        mode. Writes a new group with a name given by this PointList's .name field nested
        inside the passed group, and saves the data there.

        If the PointList has no name, it will be assigned the name "PointList#" where #
        is the lowest available integer.  If the PointList's name already exists here in
        this file, raises and exception.

        TODO: add overwite option.

        Accepts:
            group (HDF5 group)
        """

        # Detemine the name of the group
        # if current name is invalid, raises and exception
        # TODO: add overwrite option
        determine_group_name(self, group)

        ## Write
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",3) # this tag indicates a PointListArray
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # Add data
        dtype = h5py.special_dtype(vlen=self.dtype)
        dset = grp.create_dataset(
            "data",
            self.shape,
            dtype
        )
        for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
            dset[i,j] = self.get_pointlist(i,j).data

        # Add metadata
        if self._metadata is not None:
            self._metadata.name = 'metadata'
            self._metadata.to_h5(grp)


## Read PointList objects

def PointListArray_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid PointListArray object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A PointListArray instance
    """
    er = f"No PointlistArray called {name} could be found in group {group} of this HDF5 file."
    assert(EMD_group_exists(
            group,
            EMD_group_types['PointListArray'],
            name)), er
    grp = group[name]


    # Get data
    dset = grp['data']
    shape = grp['data'].shape
    dtype = grp['data'][0,0].dtype
    pla = PointListArray(dtype=dtype,shape=shape,name=name)
    for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
        try:
            pla.get_pointlist(i,j).append(dset[i,j])
        except ValueError:
            pass

    # Add metadata
    if 'metadata' in grp.keys():
        pla._metadata = Metadata_from_h5(
            grp,
            name='metadata')

    return pla


