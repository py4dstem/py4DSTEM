# File reader for py4DSTEM files

import h5py
import numpy as np
import warnings
from os.path import splitext, exists,basename,dirname,join
from typing import Optional, Union

from py4DSTEM.io.native.read_utils import is_py4DSTEM_file
from py4DSTEM.io.datastructure import (
    Root,
    Tree,
    ParentTree,
    Metadata,
    Array,
    PointList,
    PointListArray,
    Calibration,
    DataCube,
    DiffractionSlice,
    DiffractionImage,
    RealSlice,
    VirtualImage,
    Probe,
    QPoints,
    BraggVectors
)

from py4DSTEM.io.native.read_utils import get_py4DSTEM_version, version_is_geq

def read_py4DSTEM(
    filepath,
    root: Optional[str] = None,
    tree: Optional[Union[bool,str]] = True,
    **legacy_options,
    ):
    """
    File reader for files written by py4DSTEM.

    For files written by py4DSTEM v0.13+, the arguments this function
    accepts and their behaviors are below. For older verions, see
    the docstring for `py4DSTEM.io.native.legacy.read_py4DSTEM_legacy`
    for keyword arguments and their behaviors.


    Args:
        filepath (str or Path): the file path
        root (str): the path to the root data group in the HDF5 file
            to read from. To examine an HDF5 file written by py4DSTEM
            in order to determine this path, call
            `py4DSTEM.print_h5_tree(filepath)`. If left unspecified,
            looks in the file and if it finds a single top-level
            object, loads it. If it finds multiple top-level objects,
            prints a warning and returns a list of root paths to the
            top-level object found
        tree (bool or str): indicates what data should be loaded,
            relative to the root group specified above.  must be in
            (`True` or `False` or `noroot`).  If set to `False`, the
            only the data in the root group is loaded, plus any
            associated calibrations.  If set to `True`, loads the root
            group, and all other data groups nested underneath it
            in the file tree.  If set to `'noroot'`, loads all other
            data groups nested under the root group in the file tree,
            but does *not* load the data inside the root group (allowing,
            e.g., loading all the data nested under a DataCube without
            loading the whole datacube).

    Returns:
        (the data)
    """
    # Check that filepath is valid
    assert(exists(filepath)), "Error: specified filepath does not exist"
    assert(is_py4DSTEM_file(filepath)), "Error: {} isn't recognized as a py4DSTEM file.".format(filepath)

    # if root is None, determine if there is a single object in the file
    # if so, set root to that file; otherwise raise an Exception or Warning
    if root is None:
        with h5py.File(filepath,'r') as f:
            l1keys = list(f.keys())
            if len(l1keys)==0:
                raise Exception('No top level groups found in this HDF5 file!')
            elif len(l1keys)>1:
                warnings.warn('Multiple top level groups found; please specify. Returning group names.')
                return l1keys
            else:
                l2keys = list(f[l1keys[0]].keys())
                if len(l2keys)==0:
                    raise Exception('No top level data blocks found in this HDF5 file!')
                elif len(l2keys)>1:
                    warnings.warn('Multiple top level data blocks found; please specify. Returning h5 paths to top level data blocks.')
                    return [join(l1keys[0],k) for k in l2keys]
                else:
                    root = join(l1keys[0],l2keys[0])

    # Check the EMD version
    v = get_py4DSTEM_version(filepath, root.split("/")[0])
    # print(f"Reading EMD version {v[0]}.{v[1]}.{v[2]}")

    # Use legacy readers for older EMD files
    if v[1] <= 12:
        from py4DSTEM.io.native.legacy import read_py4DSTEM_legacy
        return read_py4DSTEM_legacy(filepath,**legacy_options)

    # Open h5 file
    with h5py.File(filepath,'r') as f:

        # Open the selected group
        try:
            group_data = f[root]

            # Read
            if tree is True:
                return _read_with_tree(group_data)

                return _read_without_tree(group_data)

            elif tree == 'noroot':
                return _read_without_root(group_data)

        except KeyError:
            raise Exception(f"the provided root {root} is not a valid path to a recognized data group")




def _read_without_tree(grp):

    # handle empty datasets
    if grp.attrs['emd_group_type'] == 'root':
        data = Root(
            name = basename(grp.name),
        )
        cal = _add_calibration(
            data.tree,
            grp
        )
        if cal is not None:
            data.tree = ParentTree(data, cal)
            data.calibration = cal
        return data

    # read all other data
    __class__ = _get_class(grp)
    data = __class__.from_h5(grp)
    if not isinstance(data, Calibration):
        cal = _add_calibration(
            data.tree,
            grp
        )
        if cal is not None:
            data.tree = ParentTree(data, cal)
            data.calibration = cal
    return data


def _read_with_tree(grp):
    data = _read_without_tree(grp)
    _populate_tree(
        data.tree,
        grp
    )
    return data


def _read_without_root(grp):
    root = Root()
    cal = _add_calibration(
        root,
        grp
    )
    if cal is not None:
        root.tree = ParentTree(root,cal)
        root.calibration = cal
    _populate_tree(
        root.tree,
        grp
    )
    return root


# TODO: case of multiple Calibration instances
def _add_calibration(tree,grp):
    keys = [k for k in grp.keys() if isinstance(grp[k],h5py.Group)]
    keys = [k for k in keys if (_get_class(grp[k]) == Calibration)]
    if len(keys)>0:
        k = keys[0]
        tree[k] = _read_without_tree(grp[k])
        return tree[k]
    else:
        name = dirname(grp.name)
        if name != '/':
            grp_upstream = grp.file[dirname(grp.name)]
            return _add_calibration(tree, grp_upstream)
        else:
            return None


def _populate_tree(tree,grp):
    keys = [k for k in grp.keys() if isinstance(grp[k],h5py.Group)]
    keys = [k for k in keys if (k[0] != '_' and not _get_class(
                grp[k]) == Calibration)]
    for key in keys:
        tree[key] = _read_without_tree(
            grp[key]
        )
        _populate_tree(
            tree[key].tree,
            grp[key]
        )
    pass








def print_h5_tree(filepath, show_metadata=False):
    """
    Prints the contents of an h5 file from a filepath.
    """

    with h5py.File(filepath,'r') as f:
        print('/')
        print_h5pyFile_tree(f, show_metadata=show_metadata)
        print('\n')

def print_h5pyFile_tree(f, tablevel=0, linelevels=[], show_metadata=False):
    """
    Prints the contents of an h5 file from an open h5py File instance.
    """
    if tablevel not in linelevels:
        linelevels.append(tablevel)
    keys = [k for k in f.keys() if isinstance(f[k],h5py.Group)]
    if not show_metadata:
        keys = [k for k in keys if k != '_metadata']
    N = len(keys)
    for i,k in enumerate(keys):
        string = ''
        string += '|' if 0 in linelevels else ''
        for idx in range(tablevel):
            l = '|' if idx+1 in linelevels else ''
            string += '\t'+l
        #print(string)
        print(string+'--'+k)
        if i == N-1:
            linelevels.remove(tablevel)
        print_h5pyFile_tree(
            f[k],
            tablevel=tablevel+1,
            linelevels=linelevels,
            show_metadata=show_metadata)

    pass








def _get_class(grp):

    lookup = {
        'Metadata' : Metadata,
        'Array' : Array,
        'PointList' : PointList,
        'PointListArray' : PointListArray,
        'Calibration' : Calibration,
        'DataCube' : DataCube,
        'DiffractionSlice' : DiffractionSlice,
        'DiffractionImage' : DiffractionImage,
        'RealSlice' : RealSlice,
        'VirtualImage' : VirtualImage,
        'Probe' : Probe,
        'QPoints' : QPoints,
        'BraggVectors' : BraggVectors
    }
    try:
        classname = grp.attrs['py4dstem_class']
        __class__ = lookup[classname]
        return __class__
    except KeyError:
        return None
        #raise Exception(f"Unknown classname {classname}")










