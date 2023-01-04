# Functions for reading native and non-native file types

import h5py
import pathlib
from os.path import exists, splitext
from typing import Union, Optional

from py4DSTEM.io.native.read import read_py4DSTEM
from py4DSTEM.io.nonnative import (
    read_empad,
    read_dm,
    read_gatan_K2_bin,
    load_mib
)



# Parse filetypes

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




# Read non-native files by
# parsing the filetype and calling the appropriate reader

def import_file(
    filepath: Union[str, pathlib.Path],
    mem: Optional[str] = "RAM",
    binfactor: Optional[int] = 1,
    filetype: Optional[str] = None,
    **kwargs,
):
    """
    Reader for non-native file formats.
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
        **kwargs:

    For documentation of kwargs, refer to the individual readers, in
    py4DSTEM.io.

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





# Read native files by
# error checking the inputs then passing to another function ... :'(

def read(
    filepath: Union[str,pathlib.Path],
    root: Optional[str] = None,
    tree: Optional[Union[bool,str]] = True,
    **kwargs,
    ):
    """
    File reader for files written by py4DSTEM. To load non-native
    file types, use py4DSTEM.import_file.

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

    # parse filetype
    er1 = f"filepath must be a string or Path, not {type(filepath)}"
    er2 = f"specified filepath '{filepath}' does not exist"
    assert(isinstance(filepath, (str,pathlib.Path) )), er1
    assert(exists(filepath)), er2

    filetype = parse_filetype(filepath)
    assert filetype == "py4DSTEM", "Incompatible file type for py4DSTEM.io.read. To non-native files must be read with py4DSTEM.io.import_file"

    # prepare kwargs
    kwargs['root'] = root
    kwargs['tree'] = tree

    # load data
    data = read_py4DSTEM(
        filepath,
        **kwargs
    )

    return data
