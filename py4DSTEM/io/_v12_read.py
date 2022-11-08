# General reader for 4D-STEM datasets.

import pathlib
from os.path import exists,splitext
from py4DSTEM.io.native import read_py4DSTEM, is_py4DSTEM_file
from py4DSTEM.io.nonnative import *

def read(
    filepath,
    mem="RAM",
    binfactor=1,
    ft=None,
    metadata=False,
    **kwargs
    ):
    """
    General read function for 4D-STEM datasets.

    `read` parses the file extension, the calls the appropriate reader.

    For non-native filetypes, returns a DataCube instance if a 4D data
    array is found, otherwise returns an Array.

    For native filetypes (HDF5 files written by py4DSTEM), behavior is
    contingent on the data_id argument. Valid entries are:
        - a string corresponding to the name of a
            block of data in this file. Loads it.
        - a 



what are all the things i want this fn to do?

- load non-native files
    - in terms of their Python class representation:
        - as datacubes
        - as arrays
    - in terms of their data storage/access:
        - load into RAM
        - load as a memory map
    - in terms of preprocessing
        - bin


- load native files
    - print the contents of the file (the tree)
    - load the file tree as a dictionary/structure of keys, which
        can be passed back to this function
    - return a single object
    - return a tree, starting from a given root






    For native .h5 files, behavior is contingent on the ``data_id`` argument, as
    follows:
        - if ``data_id`` is not passed, prints the contents of the .h5 file
          and returns nothing
        - if ``data_id`` is passed, returns one or more DataObject instances

    Args:
        fp (str or pathlib.Path): path to the file
        mem (str, optional): Specifies how the data should be stored; must be "RAM" or
            "MEMMAP", or "DASK". "RAM" loads the entire dataset into memory. "MEMMAP" and "DASK" are useful for
            large datasets; it does not load the data into memory, but instead creates a
            map describing where each diffraction pattern lives in storage, and only
            loads data into memory as needed. "MEMMAP" is numpy implementation, "DASK" is a Dask implementation.
        binfactor (int, optional): Bin the data, in diffraction space, as it's loaded.
            On-load binning enables datasets which, in storage, are larger than the
            system RAM to still be loaded into RAM, provided the amount of binning is
            sufficient. Binning by N reduces the filesize by N^2, so for instance, on a
            system with only 16 GB of RAM, its possible to load datasets of up to 64,
            144, or 256 GB using binfactors of 2, 3, or 4. Default is 1.
              * Note 1: binning is only supported with mem='RAM'.
              * Note 2: binning may cause 'wraparound' errors (e.g. if the datatype is
                uint16 and the summed pixels in a bin exceed 65536, the count 'wraps back
                around' to 0). This can be avoided by explicitly casting the datatype by
                passing the keyword argument 'dtype', however, casting will also affect
                the size of the data.
        ft (str, optional): Force py4DSTEM to attempt to read the file as a specified
            filetype, rather than trying to determine this automatically. Must be None or
            a str from 'py4DSTEM', 'dm', 'empad', 'mrc_relativity', 'gatan_K2_bin',
            'kitware_counted'. Default is None.
        dtype (dtype, optional): Used when binning data, ignored otherwise. Default to
            whatever the type of the raw data is, to avoid enlarging data size. May be
            useful to avoid 'wraparound' errors.
        data_id (int/str/list, optional): For py4DSTEM files only.  Specifies which data
            to load. Use integers to specify the data index, or strings to specify data
            names. A list or tuple returns a list of DataObjects. Returns the specified data.
        topgroup (str, optional): For py4DSTEM files only.  Stricty, a py4DSTEM file is
            considered to be everything inside a toplevel subdirectory within the HDF5
            file, so that if desired one can place many py4DSTEM files inside a single
            H5.  In this case, when loading data, the topgroup argument is passed to
            indicate which py4DSTEM file to load. If an H5 containing multiple py4DSTEM
            files is passed without a topgroup specified, the topgroup names are printed
            to screen.
        metadata (bool, optional): If True, returns the file metadata as a Metadata
            instance.
        log (bool, optional): For py4DSTEM files only.  If True, writes the processing
            log to a plaintext file called splitext(fp)[0]+'.log'.

    Returns:
        (variable): The return value depends on usage:

            * When loading non-native filetypes, the output type is a DataCube.
            * When loading from a native .h5 file, if no ``data_id`` value is specified,
              returns ``None``, and prints the file contents to screen
            * When loading from a native file, if one data block is requested with the
              ``data_id`` parameter, returns a single DataObject instance.
            * When loading from a native .h5 file, if multiple data blocks are being
              loaded (i.e. 'load' is passed a list) returns a list of DataObject instances
            * when ``metadata==True``, returns a MetaData instance
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(exists(fp)), "Error: specified filepath does not exist."
    assert(mem in ['RAM','MEMMAP', 'DASK']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"
    if binfactor > 1:
        assert (
            mem != "MEMMAP"
        ), "Error: binning is not supported for memory mapping.  Either set binfactor=1 or set mem='RAM'"
    assert ft in [
        None,
        "dm",
        "empad",
        "mrc_relativity",
        "gatan_K2_bin",
        "kitware_counted",
    ], "Error: ft argument not recognized"

    if ft is None:
        ft = parse_filetype(fp)

    if ft == "py4DSTEM":
        data = read_py4DSTEM(
            fp, mem=mem, binfactor=binfactor, metadata=metadata, **kwargs
        )
    elif ft == "dm":
        data = read_dm(fp, mem, binfactor, metadata=metadata, **kwargs)
    elif ft == "empad":
        data = read_empad(fp, mem, binfactor, metadata=metadata, **kwargs)
    elif ft == "mrc_relativity":
        data = read_mrc_relativity(fp, mem, binfactor, metadata=metadata, **kwargs)
    elif ft == "gatan_K2_bin":
        data = read_gatan_K2_bin(fp, mem, binfactor, metadata=metadata, **kwargs)
    elif ft == "kitware_counted":
        data = read_kitware_counted(fp, mem, binfactor, metadata=metadata, **kwargs)
    else:
        raise Exception(
            "Unrecognized file extension {}.  To force reading as a particular filetype, pass the 'ft' keyword argument.".format(
                fext
            )
        )

    return data


def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    assert isinstance(
        fp, (str, pathlib.Path)
    ), "Error: filepath fp must be a string or pathlib.Path"

    _, fext = splitext(fp)
    if fext in [
        ".h5",
        ".H5",
        "hdf5",
        "HDF5",
        ".py4dstem",
        ".py4DSTEM",
        ".PY4DSTEM",
        ".emd",
        ".EMD",
    ]:
        if is_py4DSTEM_file(fp):
            return "py4DSTEM"
        else:
            raise Exception(
                "Non-py4DSTEM formatted .h5 files are not presently supported."
            )
    elif fext in [".dm", ".dm3", ".dm4", ".DM", ".DM3", ".DM4"]:
        return "dm"
    elif fext in [".raw"]:
        return "empad"
    elif fext in [".mrc"]:
        # TK TODO
        return "mrc_relativity"
    elif fext in [".gtg", ".bin"]:
        return "gatan_K2_bin"
    elif fext in [".kitware_counted"]:
        # TK TODO
        return "kitware_counted"
    else:
        raise Exception(
            "Unrecognized file extension {}.  To force reading as a particular filetype, pass the 'ft' keyword argument.".format(
                fext
            )
        )
