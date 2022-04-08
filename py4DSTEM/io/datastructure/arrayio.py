# Functions for reading the Array and ArrayStack classes
# from valid, open HDF5 groups

import h5py

from .array import Array
from .arraystack import ArrayStack



def Array_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array or ArrayStack object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't, raises
    an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        An Array or ArrayStack instance
    """
    assert(Array_exists(group,name)), f"No Array called {name} could be found in group {group} of this HDF5 file."
    grp = group[name]

    # get data
    dset = grp['data']
    data = dset[:]
    units = dset.attrs['units']
    rank = len(data.shape)

    # determine if this is an ArrayStack
    last_dim = grp[f"dim{rank-1}"]
    if last_dim.attrs['name'] == '_labels_':
        is_arraystack = True
        normal_dims = rank-1
    else:
        is_arraystack = False
        normal_dims = rank

    # get dim vectors
    dims = []
    dim_units = []
    dim_names = []
    for n in range(normal_dims):
        dim_dset = grp[f"dim{n}"]
        dims.append(dim_dset[:])
        dim_units.append(dim_dset.attrs['units'])
        dim_names.append(dim_dset.attrs['name'])

    # if it's an ArrayStack, get the labels
    if is_arraystack:
        labels = last_dim[:]
        labels = [s.decode('utf-8') for s in labels]

    # make Array
    if is_arraystack:
        ar = ArrayStack(
            data = data,
            name = name,
            units = units,
            dims = dims,
            dim_names = dim_names,
            dim_units = dim_units,
            labels = labels
        )
    else:
        ar = Array(
            data = data,
            name = name,
            units = units,
            dims = dims,
            dim_units = dim_units,
            dim_names = dim_names
        )

    return ar


def find_Arrays(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and finds all Array groups inside this group at its top level. Does not do a search
    for nested Array groups. Returns the names of all Array groups found.

    Accepts:
        group (HDF5 group)
    """
    keys = [k for k in group.keys() if "emd_group_type" in group[k].attrs.keys()]
    return [k for k in keys if group[k].attrs["emd_group_type"] == 1]


def Array_exists(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and returns a boolean.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        bool
    """
    if name in group.keys():
        if "emd_group_type" in group[name].attrs.keys():
            if group[name].attrs["emd_group_type"] == 1:
                return True
            return False
        return False
    return False



