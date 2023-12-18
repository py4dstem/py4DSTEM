# File reader for py4DSTEM v13 files

import h5py
import numpy as np
import warnings
from os.path import exists, basename, dirname, join
from typing import Optional, Union

from py4DSTEM.io.legacy.read_utils import is_py4DSTEM_version13
from py4DSTEM.io.legacy.legacy13 import (
    Calibration,
    DataCube,
    DiffractionSlice,
    VirtualDiffraction,
    RealSlice,
    VirtualImage,
    Probe,
    QPoints,
    BraggVectors,
)
from py4DSTEM.io.legacy.legacy13 import Root, Metadata, Array, PointList, PointListArray
from py4DSTEM.io.legacy.legacy13 import v13_to_14


def read_legacy13(
    filepath,
    root: Optional[str] = None,
    tree: Optional[Union[bool, str]] = True,
):
    """
    File reader for legacy py4DSTEM (v=0.13.x) formated HDF5 files.

    Args:
        filepath (str or Path): the file path
        root (str): the path to the data group in the HDF5 file
            to read from. To examine an HDF5 file written by py4DSTEM
            in order to determine this path, call
            `py4DSTEM.print_h5_tree(filepath)`. If left unspecified,
            looks in the file and if it finds a single top-level
            object, loads it. If it finds multiple top-level objects,
            prints a warning and returns a list of root paths to the
            top-level object found.
        tree (bool or str): indicates what data should be loaded,
            relative to the root group specified above.  Must be in
            (`True` or `False` or `noroot`).  If set to `False`, the
            only the data in the root group is loaded, plus any
            associated calibrations.  If set to `True`, loads the root
            group, and all other data groups nested underneath it
            in the file tree.  If set to `'noroot'`, loads all other
            data groups nested under the root group in the file tree,
            but does *not* load the data inside the root group (allowing,
            e.g., loading all the data nested under a DataCube13 without
            loading the whole datacube).
    Returns:
        (the data)
    """
    # Check that filepath is valid
    assert exists(filepath), "Error: specified filepath does not exist"
    assert is_py4DSTEM_version13(
        filepath
    ), f"Error: {filepath} isn't recognized as a v13 py4DSTEM file."

    if root is None:
        # check if there is a single object in the file
        # if so, set root to that file; otherwise raise an Exception or Warning

        with h5py.File(filepath, "r") as f:
            l1keys = list(f.keys())
            if len(l1keys) == 0:
                raise Exception("No top level groups found in this HDF5 file!")
            elif len(l1keys) > 1:
                warnings.warn(
                    "Multiple top level groups found; please specify. Returning group names."
                )
                return l1keys
            else:
                l2keys = list(f[l1keys[0]].keys())
                if len(l2keys) == 0:
                    raise Exception("No top level data blocks found in this HDF5 file!")
                elif len(l2keys) > 1:
                    warnings.warn(
                        "Multiple top level data blocks found; please specify. Returning h5 paths to top level data blocks."
                    )
                    return [join(l1keys[0], k) for k in l2keys]
                else:
                    root = join(l1keys[0], l2keys[0])
                    # this is a windows fix
                    root = root.replace("\\", "/")

    # Open file
    with h5py.File(filepath, "r") as f:
        # open the selected group
        try:
            group_data = f[root]
        except KeyError:
            raise Exception(
                f"the provided root {root} is not a valid path to a recognized data group"
            )

        # Read data
        if tree is True:
            data = _read_with_tree(group_data)

        elif tree is False:
            data = _read_without_tree(group_data)

        elif tree == "noroot":
            data = _read_without_root(group_data)

        else:
            raise Exception(f"Unexpected value {tree} for `tree`")

        # Read calibration
        cal = _read_calibration(group_data)

    # convert version 13 -> 14
    data = v13_to_14(data, cal)
    return data


# utilities


def _read_without_tree(grp):
    # handle empty datasets
    if grp.attrs["emd_group_type"] == "root":
        data = Root(
            name=basename(grp.name),
        )
        return data

    # read data as v13 objects
    __class__ = _get_v13_class(grp)
    data = __class__.from_h5(grp)

    return data


def _read_with_tree(grp):
    data = _read_without_tree(grp)
    _populate_tree(data.tree, grp)
    return data


def _read_without_root(grp):
    root = Root()
    _populate_tree(root.tree, grp)
    return root


def _read_calibration(grp):
    keys = [k for k in grp.keys() if isinstance(grp[k], h5py.Group)]
    keys = [k for k in keys if (_get_v13_class(grp[k]) == Calibration)]
    if len(keys) > 0:
        k = keys[0]
        cal = Calibration.from_h5(grp[k])
        return cal
    else:
        name = dirname(grp.name)
        if name != "/":
            grp_upstream = grp.file[dirname(grp.name)]
            return _read_calibration(grp_upstream)
        else:
            return None


def _populate_tree(tree, grp):
    keys = [k for k in grp.keys() if isinstance(grp[k], h5py.Group)]
    keys = [
        k for k in keys if (k[0] != "_" and not _get_v13_class(grp[k]) == Calibration)
    ]
    for key in keys:
        tree[key] = _read_without_tree(grp[key])
        _populate_tree(tree[key].tree, grp[key])
    pass


def print_v13h5_tree(filepath, show_metadata=False):
    """
    Prints the contents of an h5 file from a filepath.
    """

    with h5py.File(filepath, "r") as f:
        print("/")
        print_v13h5pyFile_tree(f, show_metadata=show_metadata)
        print("\n")


def print_v13h5pyFile_tree(f, tablevel=0, linelevels=[], show_metadata=False):
    """
    Prints the contents of an h5 file from an open h5py File instance.
    """
    if tablevel not in linelevels:
        linelevels.append(tablevel)
    keys = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    if not show_metadata:
        keys = [k for k in keys if k != "_metadata"]
    N = len(keys)
    for i, k in enumerate(keys):
        string = ""
        string += "|" if 0 in linelevels else ""
        for idx in range(tablevel):
            l = "|" if idx + 1 in linelevels else ""
            string += "\t" + l
        print(string + "--" + k)
        if i == N - 1:
            linelevels.remove(tablevel)
        print_v13h5pyFile_tree(
            f[k],
            tablevel=tablevel + 1,
            linelevels=linelevels,
            show_metadata=show_metadata,
        )

    pass


def _get_v13_class(grp):
    lookup = {
        "Metadata": Metadata,
        "Array": Array,
        "PointList": PointList,
        "PointListArray": PointListArray,
        "Calibration": Calibration,
        "DataCube": DataCube,
        "DiffractionSlice": DiffractionSlice,
        "VirtualDiffraction": VirtualDiffraction,
        "DiffractionImage": VirtualDiffraction,
        "RealSlice": RealSlice,
        "VirtualImage": VirtualImage,
        "Probe": Probe,
        "QPoints": QPoints,
        "BraggVectors": BraggVectors,
    }

    if "py4dstem_class" in grp.attrs:
        classname = grp.attrs["py4dstem_class"]
    elif "emd_group_type" in grp.attrs:
        emd_group_type = grp.attrs["emd_group_type"]
        classname = {
            "root": "root",
            0: Metadata,
            1: Array,
            2: PointList,
            3: PointListArray,
        }[emd_group_type]
    else:
        warnings.warn(f"Can't determine class type of H5 group {grp}; skipping...")
        return None
    try:
        __class__ = lookup[classname]
        return __class__
    except KeyError:
        warnings.warn(f"Can't determine class type of H5 group {grp}; skipping...")
        return None
