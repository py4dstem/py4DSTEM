# File reader for py4DSTEM files

import h5py
import numpy as np
from os.path import splitext, exists
from typing import Optional, Union
from os.path import basename,dirname

from .read_utils import is_py4DSTEM_file
from ..datastructure import (
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

from .read_utils import get_py4DSTEM_version, version_is_geq

def read_py4DSTEM(
    filepath,
    root: Optional[str] = '4DSTEM',
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
            `py4DSTEM.print_h5_tree(filepath)`.
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

    # Check the EMD version
    v = get_py4DSTEM_version(filepath, root.split("/")[0])
    # print(f"Reading EMD version {v[0]}.{v[1]}.{v[2]}")

    # Use legacy readers for older EMD files
    if v[1] <= 12:
        from .legacy import read_py4DSTEM_legacy
        return read_py4DSTEM_legacy(filepath,**legacy_options)

    # Open h5 file
    with h5py.File(filepath,'r') as f:

        # Open the selected group
        try:
            group_data = f[root]

            # Read
            if tree is True:
                return _read_with_tree(group_data)

            elif tree is False:
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











    # For HDF5 files containing multiple valid EMD type 2 files (i.e. py4DSTEM files),
    # disambiguate desired data
#    tgs = get_py4DSTEM_topgroups(filepath)
#    if 'topgroup' in kwargs.keys():
#        tg = kwargs['topgroup']
#        #assert(tg in tgs), "Error: specified topgroup, {}, not found.".format(tg)
#    else:
#        if len(tgs)==1:
#            tg = tgs[0]
#        else:
#            print("Multiple topgroups were found -- please specify one:")
#            print("")
#            for tg in tgs:
#                print(tg)
#            return
#
#    # Get py4DSTEM version and call the appropriate read function
#    version = get_py4DSTEM_version(filepath, tg)
#    if version_is_geq(version,(0,12,0)): return read_v0_12(filepath, **kwargs)
#    elif version_is_geq(version,(0,9,0)): return read_v0_9(filepath, **kwargs)
#    elif version_is_geq(version,(0,7,0)): return read_v0_7(filepath, **kwargs)
#    elif version_is_geq(version,(0,6,0)): return read_v0_6(filepath, **kwargs)
#    elif version_is_geq(version,(0,5,0)): return read_v0_5(filepath, **kwargs)
#    else:
#        raise Exception('Support for legacy v{}.{}.{} files has not been added yet.'.format(version[0],version[1],version[2]))







#    Args:
#        filepath (str or pathlib.Path): When passed a filepath only,
#            this function checks if the path points to a valid py4DSTEM
#            file, then prints its contents to screen.
#        data_id (int/str/list, optional): Specifies which data to load.
#            Use integers to specify the data index, or strings to specify
#            data names. A list or tuple returns a list of DataObjects.
#            Returns the specified data.
#        topgroup (str, optional:) Stricty, a py4DSTEM file is considered
#            to be everything inside a toplevel subdirectory within the HDF5
#            file, so that if desired one can place many py4DSTEM files inside
#            a single H5.  In this case, when loading data, the topgroup
#            argument is passed to indicate which py4DSTEM file to load.
#            If an H5 containing multiple py4DSTEM files is passed without a
#            topgroup specified, the topgroup names are printed to screen.
#        metadata (bool, optional) If True, returns the metadata as a
#            Metadata instance.
#        mem (str, optional): Only used if a single DataCube is loaded.
#            In this case, mem specifies how the data should be stored;
#            must be "RAM" or "MEMMAP". See docstring for py4DSTEM.file.io.read.
#            Default is "RAM".
#        binfactor (int, optional): Only used if a single DataCube is loaded.
#            In this case, a binfactor of > 1 causes the data to be binned by
#            this amount as it's loaded.
#        dtype (dtype, optional): Used when binning data, ignored otherwise.
#            Defaults to whatever the type of the raw data is, to avoid
#            enlarging data size. May be useful to avoid wraparound errors
#            with uint16 data.
#
#    Returns:
#        (variable): The output depends on usage:
#
#            * If no input arguments with return values (i.e. data_id or
#                metadata) are passed, nothing is returned.
#            * If metadata==True, returns a Metadata instance with the
#                file metadata.
#            * Otherwise, a single DataObject or list of DataObjects are
#                returned, based on the value of the argument data_id.



