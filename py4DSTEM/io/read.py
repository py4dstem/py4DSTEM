# Reader functions
# for native and non-native file types

import h5py
import pathlib
from os.path import exists, splitext, basename, dirname, join
from typing import Union, Optional
import warnings

# Classes for native reader
from py4DSTEM.io.classes import (
    Root,
    Node,
    Tree,
    Calibration
)
from py4DSTEM.io.classes.class_io_utils import _get_class, EMD_data_group_types

# Readers for non-native filetypes
from py4DSTEM.io.nonnative import (
    read_empad,
    read_dm,
    read_gatan_K2_bin,
    load_mib
)



# File parser utility

def _parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    _, fext = splitext(fp)
    fext = fext.lower()
    if fext in [
        ".h5",
        ".hdf5",
        ".py4dstem",
        ".emd",
    ]:
        return "EMD"
    elif fext in [
        ".dm",
        ".dm3",
        ".dm4",
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




# EMD utilities

def _get_EMD_rootgroups(filepath):
    """
    Returns a list of root groups in an EMD 1.0 file.
    """
    rootgroups = []
    with h5py.File(filepath,'r') as f:
        for key in f.keys():
            if 'emd_group_type' in f[key].attrs:
                if f[key].attrs['emd_group_type'] == 'root':
                    rootgroups.append(key)
    return rootgroups

def _is_EMD_file(filepath):
    """
    Returns True iff filepath points to a valid EMD 1.0 file.
    """
    # check for the 'emd_group_type'='file' attribute
    with h5py.File(filepath,'r') as f:
        try:
            assert('emd_group_type' in f.attrs.keys())
            assert(f.attrs['emd_group_type'] == 'file')
            assert('version_major' in f.attrs.keys())
            assert('version_minor' in f.attrs.keys())
        except AssertionError:
            return False
    try:
        rootgroups = _get_EMD_rootgroups(filepath)
        if len(rootgroups)>0:
            return True
        else:
            return False
    except OSError: #TODO: do we want this here?
        return False

def _get_EMD_version(filepath, rootgroup=None):
    """
    Returns the version (major,minor,release) of an EMD file.
    """
    assert(_is_EMD_file(filepath)), "Error: not recognized as an EMD file"
    with h5py.File(filepath,'r') as f:
        v_major = int(f.attrs['version_major'])
        v_minor = int(f.attrs['version_minor'])
        if 'version_release' in f.attrs.keys():
            v_release = int(f.attrs['version_release'])
        else:
            v_release = 0
        return v_major, v_minor, v_release

def _get_UUID(filepath):
    """
    Returns the UUID of an EMD file, or if unavailable returns -1.
    """
    assert(_is_EMD_file(filepath)), "Error: not recognized as an EMD file"
    with h5py.File(filepath,'r') as f:
        if 'UUID' in f.attrs:
            return f.attrs['UUID']
    return -1

def _version_is_geq(current,minimum):
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




########## EMD 1.0+ reader ##########

# TODO: tree = 'noroot' is confusing.  Better name?
def read(
    filepath,
    emdpath: Optional[str] = None,
    tree: Optional[Union[bool,str]] = True,
    **legacy_options,
    ):
    """
    File reader for EMD 1.0+ files. To read other formats, please
    use import_file().

    Args:
        filepath (str or Path): the file path
        emdpath (str): path to the node in an EMD object tree to read
            from. May be a root node or some downstream node using '/'
            delimiters between node names. emdpath is None...(TODO)
        tree (True or False or 'noroot'): indicates what data should be loaded,
            relative to the node specified by `emdpath`. If set to `False`,
            only data/metadata in the specified node is loaded, plus any
            root metadata. If set to `True`, loads that node plus the
            subtree of data objects it contains (and their metadata, and
            the root metadata). If set to `'noroot'`, loads the branch
            under this node as above, but does not load the node itself.
            If `emdpath` points to a root node, setting `tree` to `'noroot`'
            or `True` are equivalent - both return the whole data tree.

    Returns:
        (Root) returns a Root instance containing (1) any root metadata from
            the EMD tree loaded from, and (2) a tree of one or more pieces
            of data/metadata
    """
    # parse filepath
    er1 = f"filepath must be a string or Path, not {type(filepath)}"
    er2 = f"specified filepath '{filepath}' does not exist"
    er3 = f"Error: {filepath} isn't a valid EMD 1.0+ file."
    assert(isinstance(filepath, (str,pathlib.Path) )), er1
    assert(exists(filepath)), er2
    assert(_is_EMD_file(filepath)), er3

    # parse filetype
    filetype = _parse_filetype(filepath)
    assert(filetype == "EMD"), f"Expected EMD file; file parser found type {filetype}"

    # get version
    v = _get_EMD_version(filepath)

    # Support for legacy readers...
    if v[0] < 1:
        #from <somewhere> import legacy_reader
        #return legacy_reader(filepath,**legacy_options)
        raise Exception('Reader does not currently support EMD v<1.0')

    # determine `emdpath` if it was left as None
    if emdpath is None:
        rootgroups = _get_EMD_rootgroups(filepath)
        if len(rootgroups) == 0:
            raise Exception("No root groups found! This error should never occur! (You're amazing! You've broken the basic laws of logic, reason, and thermodynamics itself!)")
        elif len(rootgroups) == 1:
            emdpath = rootgroups[0]
        else:
            # TODO: what do we want here?
            raise Exception("i have to build this code which means deciding on the logic here :'''''''[[[[[[[")

    # parse the root and tree paths
    p = emdpath.split('/')
    if p[0]=='': p=p[1:]
    if p[-1]=='': p=p[:-1]
    rootpath = p[0]
    treepath = '/'.join(p[1:])

    # Open h5 file
    with h5py.File(filepath,'r') as f:

        # Find the root group
        assert(rootpath in f.keys()), f"Error: root group {rootpath} not found"
        rootgroup = f[rootpath]

        # Find the node of interest
        group_names = treepath.split('/')
        nodegroup = rootgroup
        for name in group_names:
            assert(name in nodegroup.keys()), f"Error: group {name} not ground in group {nodegroup.name}"
            nodegroup = nodegroup[name]

        # Read the root
        root = Root.from_h5(rootgroup)

        # if this is all that was requested, return
        if nodegroup is rootgroup and tree is False:
                return root


        # Read...

        # ...if the whole tree was requested
        if nodegroup is rootgroup and tree is in (True,'noroot'):
            # build the tree
            _populate_tree(root,rootgroup)

        # ...if a single node was requested
        if tree is False:
            # read the node
            node = _read_single_node(nodegroup)
            # build the tree and return
            root.add_to_tree(node)
            return root

        # ...if a branch was requested
        if tree is True:
            # read source node and add to tree
            node = _read_single_node(nodegroup)
            root.add_to_tree(node)
            # build the tree
            _populate_tree(node,nodegroup)

        # ...if a rootless branch was requsted
        if tree == 'noroot':
            # build the tree
            _populate_tree(root,nodegroup)




# group / tree reading utilities

def _read_single_node(grp):
    """
    Determines the class type of the h5py Group `grp`, then
    instantiates and returns an instance of the class with
    this group's data and metadata
    """
    __class__ = _get_class(grp)
    data = __class__.from_h5(grp)
    return data

def _populate_tree(node,group):
    """
    `node` is a Node and `group` is its parallel h5py Group.
    Reads the tree underneath this nodegroup in the h5 file and adds it
    to the runtime tree underneath this node
    """
    keys = [k for k in group.keys() if isinstance(group[k],h5py.Group)]
    keys = [k for k in keys if 'emd_group_type' is in group[k].attrs.keys()]
    keys = [k for k in keys if group[k].attrs['emd_group_type'] is in \
        EMD_data_group_types]

    for key in keys:
        new_node = _read_single_node(group[key])
        node.add_to_tree(new_node)
        _populate_tree(
            new_node,
            group[key]
        )
    pass







# Print the HDF5 filetree to screen

def print_h5_tree(filepath, show_metadata=False):
    """
    Prints the contents of an h5 file from a filepath.
    """
    with h5py.File(filepath,'r') as f:
        print('/')
        _print_h5pyFile_tree(f, show_metadata=show_metadata)
        print('\n')

def _print_h5pyFile_tree(f, tablevel=0, linelevels=[], show_metadata=False):
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

    filetype = _parse_filetype(filepath) if filetype is None else filetype

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











#def _read_without_tree(grp):
#
#    # handle empty datasets
#    if grp.attrs['emd_group_type'] == 'root':
#        data = Node(
#            name = basename(grp.name),
#        )
#        cal = _add_calibration(
#            data.tree,
#            grp
#        )
#        if cal is not None:
#            data.tree = ParentTree(data, cal)
#            data.calibration = cal
#        return data
#
#    # read all other data
#    __class__ = _get_class(grp)
#    data = __class__.from_h5(grp)
#    if not isinstance(data, Calibration):
#        cal = _add_calibration(
#            data.tree,
#            grp
#        )
#        if cal is not None:
#            data.tree = ParentTree(data, cal)
#            data.calibration = cal
#    return data
#
#
#
#def _read_without_root(grp):
#    root = Node()
#    cal = _add_calibration(
#        root,
#        grp
#    )
#    if cal is not None:
#        root.tree = ParentTree(root,cal)
#        root.calibration = cal
#    _populate_tree(
#        root.tree,
#        grp
#    )
#    return root
#
#
## TODO: case of multiple Calibration instances
#def _add_calibration(tree,grp):
#    keys = [k for k in grp.keys() if isinstance(grp[k],h5py.Group)]
#    keys = [k for k in keys if (_get_class(grp[k]) == Calibration)]
#    if len(keys)>0:
#        k = keys[0]
#        tree[k] = _read_without_tree(grp[k])
#        return tree[k]
#    else:
#        name = dirname(grp.name)
#        if name != '/':
#            grp_upstream = grp.file[dirname(grp.name)]
#            return _add_calibration(tree, grp_upstream)
#        else:
#            return None






