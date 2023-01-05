# I/O utility functions for reading/writing the base EMD types
# between HDF5 groups and python classes

import numpy as np
import h5py
from numbers import Number

from py4DSTEM.utils.tqdmnd import tqdmnd



# Define the EMD group types

EMD_group_types = {
    'Root' : 'root',
    'Metadata' : 0,
    'Array' : 1,
    'PointList' : 2,
    'PointListArray': 3,
    'Custom': 4
}



# Finding and validating EMD groups

def find_EMD_groups(group:h5py.Group, emd_group_type):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and finds all groups inside this group at its top level matching
    `emd_group_type`. Does not do a nested search. Returns the names of all
    groups found.

    Accepts:
        group (HDF5 group):
        emd_group_type (int)
    """
    keys = [k for k in group.keys() if "emd_group_type" in group[k].attrs.keys()]
    return [k for k in keys if group[k].attrs["emd_group_type"] == emd_group_type]


def EMD_group_exists(group:h5py.Group, emd_group_type, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an object of this `emd_group_type` and name exists
    inside this group, and returns a boolean.

    Accepts:
        group (HDF5 group):
        emd_group_type (int):
        name (string):

    Returns:
        bool
    """
    if name in group.keys():
        if "emd_group_type" in group[name].attrs.keys():
            if group[name].attrs["emd_group_type"] == emd_group_type:
                return True
            return False
        return False
    return False




def _get_class(grp):
    """
    Function returning Class signatures from corresponding strings
    """

    import inspect
    from py4DSTEM.io import classes

    # Build lookup table for classes
    lookup = {}
    for name, obj in inspect.getmembers(classes):
        if inspect.isclass(obj):
            lookup[name] = obj

    # Get the class from the group tags and return
    try:
        classname = grp.attrs['py4dstem_class']
        __class__ = lookup[classname]
        return __class__
    except KeyError:
        return None
        #raise Exception(f"Unknown classname {classname}")







# Metadata i/o helper functions

def _write_metadata(obj,grp):
    items = obj._metadata.items()
    if len(items)>0:
        grp_metadata = grp.create_group('_metadata')
        for name,md in items:
            obj._metadata[name].name = name
            obj._metadata[name].to_h5(grp_metadata)

def _read_metadata(obj, grp):
    try:
        grp_metadata = grp['_metadata']
        for key in grp_metadata.keys():
            obj.metadata = Metadata_from_h5(grp_metadata[key])
    except KeyError:
        pass


