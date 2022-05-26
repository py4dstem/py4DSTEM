# Defines a class - PointList - for storing / accessing / manipulating data in the form of
# lists of vectors.

import numpy as np
import h5py
from copy import copy
from typing import Optional
from .ioutils import determine_group_name


class PointList(object):
    """
    A wrapper around structured numpy arrays, with read/write functionality in/out of
    py4DSTEM formatted HDF5 files.

    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'pointlist'
        ):
        """
		Instantiate a PointList.

        Args:
            data (structured numpy ndarray): the data; the dtype of this array will
                specify the coordinates of the PointList.
            name (str): name for the PointList

        """
        self.data = data
        self.dtype = data.dtype
        self.fields = self.dtype.names
        self.name = name
        self.length = len(self.data)


    ## Add, remove, sort data

    def append(self, data):
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

    def sort(self, coordinate, order='descending'):
        """
        Sorts the point list according to coordinate.
        coordinate must be a field in self.dtype.
        order should be 'descending' or 'ascending'.
        """
        assert coordinate in self.dtype.names
        assert (order=='descending') or (order=='ascending')
        if order=='ascending':
            self.data = np.sort(self.data, order=coordinate)
        else:
            self.data = np.sort(self.data, order=coordinate)[::-1]


    ## Copy, copy+modify PointList

    def copy(self, name=None):
        """ Returns a copy of the PointList. If name=None, sets to `{name}_copy`
        """
        name = name if name is not None else self.name+"_copy"
        return PointList(
            data = np.copy(self.data),
            name = name)

    def add_coordinates(self, new_coords):
        """
        Creates a copy of the PointList, but with additional coordinates given by new_coords.
        new_coords must be a string of 2-tuples, ('name', dtype)
        """
        dtype = []
        for key in self.dtype.fields.keys():
            dtype.append((key,self.dtype.fields[key][0]))
        for coord in new_coords:
            dtype.append((coord[0],coord[1]))

        data = np.zeros(self.length, dtype=dtype)
        for key in self.dtype.fields.keys():
            data[key] = np.copy(self.data[key])

        return PointList(data=data)


    ## Representation to standard output

    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A length {self.length} PointList with {len(self.fields)} coordinates called '{self.name}',"
        string += "\n"+space+"with coordinates:"
        string += "\n"
        space2 = max([len(field) for field in self.fields])+3
        for field in self.fields:
            string += "\n"+space+f"{field}{(space2-len(field))*' '}({str(self.dtype.fields[field][0])})"
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
        grp.attrs.create("emd_group_type",2) # this tag indicates an Array type object
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # Add data
        for field in self.fields:
            group_current_field = grp.create_group(field)
            group_current_field.attrs.create("dtype", np.string_(self.dtype[field]))
            group_current_field.create_dataset("data", data=self.data[field])



## Read PointList objects

def PointList_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid PointList object of this name exists inside
    this group, and if it does, loads and returns it. If it doesn't, raises
    an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A PointList instance
    """
    assert(PointList_exists(group,name)), f"No PointList called {name} could be found in group {group} of this HDF5 file."
    grp = group[name]

    # Get metadata
    name = grp.name.split('/')[-1]
    fields = list(grp.keys())
    dtype = []
    for field in fields:
        curr_dtype = grp[field].attrs["dtype"].decode('utf-8')
        dtype.append((field,curr_dtype))
    length = len(grp[fields[0]+'/data'])

    # Get data
    data = np.zeros(length,dtype=dtype)
    if length > 0:
        for field in fields:
            data[field] = np.array(grp[field+'/data'])

    return PointList(
        data=data,
        name=name)



def find_PointLists(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and finds all PointList groups inside this group at its top level. Does not do a
    search for nested PointList groups. Returns the names of all PointList groups found.

    Accepts:
        group (HDF5 group)
    """
    keys = [k for k in group.keys() if "emd_group_type" in group[k].attrs.keys()]
    return [k for k in keys if group[k].attrs["emd_group_type"] == 2]


def PointList_exists(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a PointList object of this name exists inside this group,
    and returns a boolean.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        bool
    """
    if name in group.keys():
        if "emd_group_type" in group[name].attrs.keys():
            if group[name].attrs["emd_group_type"] == 2:
                return True
            return False
        return False
    return False


