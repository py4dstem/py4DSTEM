# Functions for reading native and non-native file types

import pathlib
from os.path import exists, splitext, basename, dirname, join
from typing import Union, Optional
import warnings
import h5py
import numpy as np

# Readers for non-native filetypes
from py4DSTEM.io.nonnative import (
    read_empad,
    read_dm,
    read_gatan_K2_bin,
    load_mib
)

# Classes for native reader
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
    VirtualDiffraction,
    RealSlice,
    VirtualImage,
    Probe,
    QPoints,
    BraggVectors
)







###### GENERAL UTILITY FUNCTIONS ######


def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    _, fext = splitext(fp)
    fext = fext.lower()
    if fext in [
        ".h5",
        ".H5",
        ".hdf5",
        ".HDF5",
        ".py4dstem",
        ".py4DSTEM",
        ".PY4DSTEM",
        ".emd",
        ".EMD",
    ]:
        return "py4DSTEM"
    elif fext in [
        ".dm",
        ".dm3",
        ".dm4",
        ".DM",
        ".DM3",
        ".DM4"
    ]:
        return "dm"
    elif fext in [".raw"]:
       return "empad"
    elif fext in [".mrc"]:
       return "mrc_relativity"
    elif fext in [".gtg", ".bin"]:
       return "gatan_K2_bin"
    elif fext in [".kitware_counted"]:
       return "kitware_counted"
    elif fext in [".mib", ".MIB"]:
        return "mib"
    else:
        raise Exception(f"Unrecognized file extension {fext}.")






###### NON-NATIVE FILE READER ######


def import_file(
    filepath: Union[str, pathlib.Path],
    mem: Optional[str] = "RAM",
    binfactor: Optional[int] = 1,
    filetype: Optional[str] = None,
    **kwargs,
):
    """
    Reader for non-native file formats.
    Parses the filetype, and calls the appropriate reader.
    Supports Gatan DM3/4, some EMPAD file versions, Gatan K2 bin/gtg, and mib
    formats.

    Args:
        filepath (str or Path): Path to the file.
        mem (str):  Must be "RAM" or "MEMMAP". Specifies how the data is
            loaded; "RAM" transfer the data from storage to RAM, while "MEMMAP"
            leaves the data in storage and creates a memory map which points to
            the diffraction patterns, allowing them to be retrieved individually
            from storage.
        binfactor (int): Diffraction space binning factor for bin-on-load.
        filetype (str): Used to override automatic filetype detection.
        **kwargs: any additional kwargs are passed to the downstream reader -
            refer to the individual filetype reader function call signatures
            and docstrings for more details.

    Returns:
        (DataCube or Array) returns a DataCube if 4D data is found, otherwise
        returns an Array

    """

    assert isinstance(
        filepath, (str, pathlib.Path)
    ), f"filepath must be a string or Path, not {type(filepath)}"
    assert exists(filepath), f"The given filepath: '{filepath}' \ndoes not exist"
    assert mem in [
        "RAM",
        "MEMMAP",
    ], 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert isinstance(
        binfactor, int
    ), "Error: argument binfactor must be an integer"
    assert binfactor >= 1, "Error: binfactor must be >= 1"
    if binfactor > 1:
        assert (
            mem != "MEMMAP"
        ), "Error: binning is not supported for memory mapping.  Either set binfactor=1 or set mem='RAM'"

    filetype = parse_filetype(filepath) if filetype is None else filetype

    assert filetype in [
        "dm",
        "empad",
        "gatan_K2_bin",
        "mib"
        # "kitware_counted",
    ], "Error: filetype not recognized"

    if filetype == "dm":
        data = read_dm(filepath, mem=mem, binfactor=binfactor, **kwargs)
    elif filetype == "empad":
        data = read_empad(filepath, mem=mem, binfactor=binfactor, **kwargs)
    elif filetype == "gatan_K2_bin":
        data = read_gatan_K2_bin(filepath, mem=mem, binfactor=binfactor, **kwargs)
    # elif filetype == "kitware_counted":
    #    data = read_kitware_counted(filepath, mem, binfactor, metadata=metadata, **kwargs)
    elif filetype == "mib":
        data = load_mib(filepath, mem=mem, binfactor=binfactor,**kwargs)
    else:
        raise Exception("Bad filetype!")

    return data











###### NATIVE FILE READER #######



# utility functions

def get_py4DSTEM_topgroups(filepath):
    """
    Returns a list of toplevel groups in an HDF5 file which are valid
    py4DSTEM file trees.
    """
    valid_emd_keys = (0,1,2,3,'root')

    topgroups = []
    with h5py.File(filepath,'r') as f:
        for key in f.keys():
            if 'emd_group_type' in f[key].attrs:
                if f[key].attrs['emd_group_type'] in valid_emd_keys:
                    topgroups.append(key)

    return topgroups

def is_py4DSTEM_file(filepath):
    """
    Returns True iff filepath points to a py4DSTEM formatted (EMD type 2) file.
    """
    try:
        topgroups = get_py4DSTEM_topgroups(filepath)
        if len(topgroups)>0:
            return True
        else:
            return False
    except OSError:
        return False

def get_py4DSTEM_version(filepath, topgroup=None):
    """
    Returns the version (major,minor,release) of a py4DSTEM file.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        if topgroup is None or topgroup not in f.keys():
            if "4DSTEM" in f.keys():
                topgroup = "4DSTEM"
            elif "4DSTEM_experiment" in f.keys():
                topgroup = "4DSTEM_experiment"
            else:
                raise ValueError("no root group exists with default names. please specify topgroup.")
        version_major = int(f[topgroup].attrs['version_major'])
        version_minor = int(f[topgroup].attrs['version_minor'])
        if 'version_release' in f[topgroup].attrs.keys():
            version_release = int(f[topgroup].attrs['version_release'])
        else:
            version_release = 0
        return version_major, version_minor, version_release

def get_UUID(filepath, topgroup='4DSTEM'):
    """
    Returns the UUID of a py4DSTEM file, or if unavailable returns -1.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        if topgroup in f.keys():
            if 'UUID' in f[topgroup].attrs:
                return f[topgroup].attrs['UUID']
    return -1

def version_is_geq(current,minimum):
    """
    Returns True iff current version (major,minor,release) is greater than or equal to minimum."
    """
    if current[0]>minimum[0]:
        return True
    elif current[0]==minimum[0]:
        if current[1]>minimum[1]:
            return True
        elif current[1]==minimum[1]:
            if current[2]>=minimum[2]:
                return True
        else:
            return False
    else:
        return False

def _get_class(grp):

    lookup = {
        'Metadata' : Metadata,
        'Array' : Array,
        'PointList' : PointList,
        'PointListArray' : PointListArray,
        'Calibration' : Calibration,
        'DataCube' : DataCube,
        'DiffractionSlice' : DiffractionSlice,
        'VirtualDiffraction' : VirtualDiffraction,
        'DiffractionImage' : VirtualDiffraction,
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


# Print the HDF5 filetree to screen

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



# native file reader

def read(
    filepath,
    root: Optional[str] = None,
    tree: Optional[Union[bool,str]] = True,
    **legacy_options,
    ):
    """
    File reader for files *written by* py4DSTEM.
    To read non-native file formats, please use py4DSTEM.import_file.

    For files written by py4DSTEM v0.13+, the arguments this function
    accepts and their behaviors are below. For older verions, pass
    kwargs (**legacy_options) according to the legacy reader args
    found in py4DSTEM.io.legacy.read_py4DSTEM_legacy`.

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
        (data, variable type) the form of the returned data depends on
            the `tree` argument - see it's docs, above.
    """
    # parse filepath
    # (must be valid, exist, and point to a py4DSTEM file)
    er1 = f"filepath must be a string or Path, not {type(filepath)}"
    assert(isinstance(filepath, (str,pathlib.Path) )), er1
    er2 = f"specified filepath '{filepath}' does not exist"
    assert(exists(filepath)), er2
    er3 = f"Error: {filepath} isn't recognized as a py4DSTEM file."
    assert(is_py4DSTEM_file(filepath)), er3

    # parse filetype
    filetype = parse_filetype(filepath)
    assert filetype == "py4DSTEM", "Non-native file type detected. To import data from non-native formats, use p4DSTEM.import_file"


    # Check the EMD version
    if root is not None:
        v = get_py4DSTEM_version(filepath, root.split("/")[0])
    else:
        try:
            with h5py.File(filepath,'r') as f:
                k = list(f.keys())[0]
                v = get_py4DSTEM_version(filepath, k)
        except:
            raise Exception('error parsing file version...')


    # Use legacy readers for EMD files with v<13
    if v[1] <= 12:
        from py4DSTEM.io.native.legacy import read_py4DSTEM_legacy
        return read_py4DSTEM_legacy(filepath,**legacy_options)


    # Read EMD files with v>=13

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
                    #this is a windows fix
                    root = root.replace("\\","/")

    # Open h5 file
    with h5py.File(filepath,'r') as f:

        # Open the selected group
        try:
            group_data = f[root]

            # Read for...

            # ...the whole file tree
            if tree is True:
                return _read_with_tree(group_data)

            # ...the specified node only
            elif tree is False:
                return _read_without_tree(group_data)

            # ...the tree under the specified node,
            # excluding the node itself
            elif tree == 'noroot':
                return _read_without_root(group_data)

        except KeyError:
            raise Exception(f"the provided root {root} is not a valid path to a recognized data group")



# utilities for performing the read in each use-case,
# walking or not walking the tree as specified,
# and attaching calibrations where appropriate

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






