# Functions for reading and writing the base EMD types between
# HDF5 and python classes

import numpy as np
import h5py
from numbers import Number
from emdfile import tqdmnd


# Define the EMD group types

EMD_group_types = {
    "Root": "root",
    "Metadata": 0,
    "Array": 1,
    "PointList": 2,
    "PointListArray": 3,
    "Custom": 4,
}


# Utility functions for finding and validating EMD groups


def find_EMD_groups(group: h5py.Group, emd_group_type):
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


def EMD_group_exists(group: h5py.Group, emd_group_type, name: str):
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


# Read and write for base EMD types


## ROOT


# write
def Root_to_h5(root, group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open
    in write or append mode. Writes a new group with a name given by
    this Root instance's .name field nested inside the passed
    group, and saves the data there.
    Accepts:
        group (HDF5 group)
    """

    ## Write
    grp = group.create_group(root.name)
    grp.attrs.create("emd_group_type", EMD_group_types["Root"])
    grp.attrs.create("py4dstem_class", root.metadata.__class__.__name__)


# read
def Root_from_h5(group: h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid Root object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.
    Accepts:
        group (HDF5 group)
    Returns:
        A Root instance
    """
    from py4DSTEM.io.legacy.legacy13.v13_emd_classes.root import Root
    from os.path import basename

    er = f"Group {group} is not a valid EMD Metadata group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == EMD_group_types["Root"], er

    root = Root(basename(group.name))
    return root


## METADATA


# write
def Metadata_to_h5(metadata, group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open
    in write or append mode. Writes a new group with a name given by
    this Metadata instance's .name field nested inside the passed
    group, and saves the data there.
    Accepts:
        group (HDF5 group)
    """

    ## Write
    grp = group.create_group(metadata.name)
    grp.attrs.create("emd_group_type", EMD_group_types["Metadata"])
    grp.attrs.create("py4dstem_class", metadata.__class__.__name__)

    # Save data
    for k, v in metadata._params.items():
        # None
        if v is None:
            v = "_None"
            v = np.string_(v)  # convert to byte string
            dset = grp.create_dataset(k, data=v)
            dset.attrs["type"] = np.string_("None")

        # strings
        elif isinstance(v, str):
            v = np.string_(v)  # convert to byte string
            dset = grp.create_dataset(k, data=v)
            dset.attrs["type"] = np.string_("string")

        # bools
        elif isinstance(v, bool):
            dset = grp.create_dataset(k, data=v, dtype=bool)
            dset.attrs["type"] = np.string_("bool")

        # numbers
        elif isinstance(v, Number):
            dset = grp.create_dataset(k, data=v, dtype=type(v))
            dset.attrs["type"] = np.string_("number")

        # arrays
        elif isinstance(v, np.ndarray):
            dset = grp.create_dataset(k, data=v, dtype=v.dtype)
            dset.attrs["type"] = np.string_("array")

        # tuples
        elif isinstance(v, tuple):
            # of numbers
            if isinstance(v[0], Number):
                dset = grp.create_dataset(k, data=v)
                dset.attrs["type"] = np.string_("tuple")

            # of tuples
            elif any([isinstance(v[i], tuple) for i in range(len(v))]):
                dset_grp = grp.create_group(k)
                dset_grp.attrs["type"] = np.string_("tuple_of_tuples")
                dset_grp.attrs["length"] = len(v)
                for i, x in enumerate(v):
                    dset_grp.create_dataset(str(i), data=x)

            # of arrays
            elif isinstance(v[0], np.ndarray):
                dset_grp = grp.create_group(k)
                dset_grp.attrs["type"] = np.string_("tuple_of_arrays")
                dset_grp.attrs["length"] = len(v)
                for i, ar in enumerate(v):
                    dset_grp.create_dataset(str(i), data=ar, dtype=ar.dtype)

            # of strings
            elif isinstance(v[0], str):
                dset_grp = grp.create_group(k)
                dset_grp.attrs["type"] = np.string_("tuple_of_strings")
                dset_grp.attrs["length"] = len(v)
                for i, s in enumerate(v):
                    dset_grp.create_dataset(str(i), data=np.string_(s))

            else:
                er = f"Metadata only supports writing tuples with numeric and array-like arguments; found type {type(v[0])}"
                raise Exception(er)

        # lists
        elif isinstance(v, list):
            # of numbers
            if isinstance(v[0], Number):
                dset = grp.create_dataset(k, data=v)
                dset.attrs["type"] = np.string_("list")

            # of arrays
            elif isinstance(v[0], np.ndarray):
                dset_grp = grp.create_group(k)
                dset_grp.attrs["type"] = np.string_("list_of_arrays")
                dset_grp.attrs["length"] = len(v)
                for i, ar in enumerate(v):
                    dset_grp.create_dataset(str(i), data=ar, dtype=ar.dtype)

            # of strings
            elif isinstance(v[0], str):
                dset_grp = grp.create_group(k)
                dset_grp.attrs["type"] = np.string_("list_of_strings")
                dset_grp.attrs["length"] = len(v)
                for i, s in enumerate(v):
                    dset_grp.create_dataset(str(i), data=np.string_(s))

            else:
                er = f"Metadata only supports writing lists with numeric and array-like arguments; found type {type(v[0])}"
                raise Exception(er)

        else:
            er = f"Metadata supports writing numbers, bools, strings, arrays, tuples of numbers or arrays, and lists of numbers or arrays. Found an unsupported type {type(v[0])}"
            raise Exception(er)


# read
def Metadata_from_h5(group: h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid Metadata object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.
    Accepts:
        group (HDF5 group)
    Returns:
        A Metadata instance
    """
    from py4DSTEM.io.legacy.legacy13.v13_emd_classes.metadata import Metadata
    from os.path import basename

    er = f"Group {group} is not a valid EMD Metadata group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == EMD_group_types["Metadata"], er

    # Get data
    data = {}
    for k, v in group.items():
        # get type
        try:
            t = group[k].attrs["type"].decode("utf-8")
        except KeyError:
            raise Exception(f"unrecognized Metadata value type {type(v)}")

        # None
        if t == "None":
            v = None

        # strings
        elif t == "string":
            v = v[...].item()
            v = v.decode("utf-8")
            v = v if v != "_None" else None

        # numbers
        elif t == "number":
            v = v[...].item()

        # bools
        elif t == "bool":
            v = v[...].item()

        # array
        elif t == "array":
            v = np.array(v)

        # tuples of numbers
        elif t == "tuple":
            v = tuple(v[...])

        # tuples of arrays
        elif t == "tuple_of_arrays":
            L = group[k].attrs["length"]
            tup = []
            for l in range(L):
                tup.append(np.array(v[str(l)]))
            v = tuple(tup)

        # tuples of tuples
        elif t == "tuple_of_tuples":
            L = group[k].attrs["length"]
            tup = []
            for l in range(L):
                x = v[str(l)][...]
                if x.ndim == 0:
                    x = x.item()
                else:
                    x = tuple(x)
                tup.append(x)
            v = tuple(tup)

        # tuples of strings
        elif t == "tuple_of_strings":
            L = group[k].attrs["length"]
            tup = []
            for l in range(L):
                s = v[str(l)][...].item().decode("utf-8")
                tup.append(s)
            v = tuple(tup)

        # lists of numbers
        elif t == "list":
            v = list(v[...])

        # lists of arrays
        elif t == "list_of_arrays":
            L = group[k].attrs["length"]
            _list = []
            for l in range(L):
                _list.append(np.array(v[str(l)]))
            v = _list

        # list of strings
        elif t == "list_of_strings":
            L = group[k].attrs["length"]
            _list = []
            for l in range(L):
                s = v[str(l)][...].item().decode("utf-8")
                _list.append(s)
            v = _list

        else:
            raise Exception(f"unrecognized Metadata value type {t}")

        # add data
        data[k] = v

    # make Metadata instance, add data, and return
    md = Metadata(basename(group.name))
    md._params.update(data)
    return md


## ARRAY

# write


def Array_to_h5(array, group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    Array's .name field nested inside the passed group, and saves the
    data there.
    Accepts:
        group (HDF5 group)
    """

    ## Write

    grp = group.create_group(array.name)
    grp.attrs.create("emd_group_type", 1)  # this tag indicates an Array
    grp.attrs.create("py4dstem_class", array.__class__.__name__)

    # add the data
    data = grp.create_dataset(
        "data",
        shape=array.data.shape,
        data=array.data,
        # dtype = type(array.data)
    )
    data.attrs.create(
        "units", array.units
    )  # save 'units' but not 'name' - 'name' is the group name

    # Add the normal dim vectors
    for n in range(array.rank):
        # unpack info
        dim = array.dims[n]
        name = array.dim_names[n]
        units = array.dim_units[n]
        is_linear = array._dim_is_linear(dim, array.shape[n])

        # compress the dim vector if it's linear
        if is_linear:
            dim = dim[:2]

        # write
        dset = grp.create_dataset(f"dim{n}", data=dim)
        dset.attrs.create("name", name)
        dset.attrs.create("units", units)

    # Add stack dim vector, if present
    if array.is_stack:
        n = array.rank
        name = "_labels_"
        dim = [s.encode("utf-8") for s in array.slicelabels]

        # write
        dset = grp.create_dataset(f"dim{n}", data=dim)
        dset.attrs.create("name", name)

    # Add metadata
    _write_metadata(array, grp)


## read


def Array_from_h5(group: h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode.
    Determines if this group represents an Array object and if it does, loads
    returns it. If it doesn't, raises an exception.
    Accepts:
        group (HDF5 group)
    Returns:
        An Array instance
    """
    from py4DSTEM.io.legacy.legacy13.v13_emd_classes.array import Array
    from os.path import basename

    er = f"Group {group} is not a valid EMD Array group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == EMD_group_types["Array"], er

    # get data
    dset = group["data"]
    data = dset[:]
    units = dset.attrs["units"]
    rank = len(data.shape)

    # determine if this is a stack array
    last_dim = group[f"dim{rank-1}"]
    if last_dim.attrs["name"] == "_labels_":
        is_stack = True
        normal_dims = rank - 1
    else:
        is_stack = False
        normal_dims = rank

    # get dim vectors
    dims = []
    dim_units = []
    dim_names = []
    for n in range(normal_dims):
        dim_dset = group[f"dim{n}"]
        dims.append(dim_dset[:])
        dim_units.append(dim_dset.attrs["units"])
        dim_names.append(dim_dset.attrs["name"])

    # if it's a stack array, get the labels
    if is_stack:
        slicelabels = last_dim[:]
        slicelabels = [s.decode("utf-8") for s in slicelabels]
    else:
        slicelabels = None

    # make Array
    ar = Array(
        data=data,
        name=basename(group.name),
        units=units,
        dims=dims,
        dim_names=dim_names,
        dim_units=dim_units,
        slicelabels=slicelabels,
    )

    # add metadata
    _read_metadata(ar, group)

    return ar


## POINTLIST


# write
def PointList_to_h5(pointlist, group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    PointList's .name field nested inside the passed group, and saves
    the data there.
    Accepts:
        group (HDF5 group)
    """

    ## Write
    grp = group.create_group(pointlist.name)
    grp.attrs.create("emd_group_type", 2)  # this tag indicates a PointList
    grp.attrs.create("py4dstem_class", pointlist.__class__.__name__)

    # Add data
    for f, t in zip(pointlist.fields, pointlist.types):
        group_current_field = grp.create_dataset(f, data=pointlist.data[f])
        group_current_field.attrs.create("dtype", np.string_(t))
        # group_current_field.create_dataset(
        #    "data",
        #    data = pointlist.data[f]
        # )

    # Add metadata
    _write_metadata(pointlist, grp)


# read
def PointList_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_emd_classes.pointlist import PointList
    from os.path import basename

    er = f"Group {group} is not a valid EMD PointList group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == EMD_group_types["PointList"], er

    # Get metadata
    fields = list(group.keys())
    if "_metadata" in fields:
        fields.remove("_metadata")
    dtype = []
    for field in fields:
        curr_dtype = group[field].attrs["dtype"].decode("utf-8")
        dtype.append((field, curr_dtype))
    length = len(group[fields[0]])

    # Get data
    data = np.zeros(length, dtype=dtype)
    if length > 0:
        for field in fields:
            data[field] = np.array(group[field])

    # Make the PointList
    pl = PointList(data=data, name=basename(group.name))

    # Add additional metadata
    _read_metadata(pl, group)

    return pl


## POINTLISTARRAY


# write
def PointListArray_to_h5(pointlistarray, group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    PointListArray's .name field nested inside the passed group, and
    saves the data there.
    Accepts:
        group (HDF5 group)
    """

    ## Write
    grp = group.create_group(pointlistarray.name)
    grp.attrs.create("emd_group_type", 3)  # this tag indicates a PointListArray
    grp.attrs.create("py4dstem_class", pointlistarray.__class__.__name__)

    # Add metadata
    dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    dset = grp.create_dataset("data", pointlistarray.shape, dtype)

    # Add data
    for i, j in tqdmnd(dset.shape[0], dset.shape[1]):
        dset[i, j] = pointlistarray[i, j].data

    # Add additional metadata
    _write_metadata(pointlistarray, grp)


# read
def PointListArray_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_emd_classes.pointlistarray import (
        PointListArray,
    )
    from os.path import basename

    er = f"Group {group} is not a valid EMD PointListArray group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == EMD_group_types["PointListArray"], er

    # Get the DataSet
    dset = group["data"]
    dtype = h5py.check_vlen_dtype(dset.dtype)
    shape = dset.shape

    # Initialize a PointListArray
    pla = PointListArray(dtype=dtype, shape=shape, name=basename(group.name))

    # Add data
    for i, j in tqdmnd(
        shape[0], shape[1], desc="Reading PointListArray", unit="PointList"
    ):
        try:
            pla[i, j].add(dset[i, j])
        except ValueError:
            pass

    # Add metadata
    _read_metadata(pla, group)

    return pla


# Metadata helper functions


def _write_metadata(obj, grp):
    items = obj._metadata.items()
    if len(items) > 0:
        grp_metadata = grp.create_group("_metadata")
        for name, md in items:
            obj._metadata[name].name = name
            obj._metadata[name].to_h5(grp_metadata)


def _read_metadata(obj, grp):
    try:
        grp_metadata = grp["_metadata"]
        for key in grp_metadata.keys():
            obj.metadata = Metadata_from_h5(grp_metadata[key])
    except KeyError:
        pass
