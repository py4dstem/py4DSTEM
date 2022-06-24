# General reader for 4D-STEM datasets.

import pathlib
from os.path import exists, splitext
from typing import Union, Optional
import h5py

from .native.read import read_py4DSTEM
from .nonnative import read_dm


def read(
    filepath: Union[str,pathlib.Path],
    #mem Optional[str] = "RAM",
    #binfactor Optional[int] = 1,
    filetype: Optional[str] = None,
    #metadata=False,
    **kwargs
    ):
    """
    General read function for 4D-STEM datasets.

    Args:
        filepath: the filepath

    what are all the things i want this fn to do?


    - load native files
        - print the contents of the file (the tree)
        - load the file tree as a dictionary/structure of keys, which
            can be passed back to this function
        - return a single object
        - return a tree, starting from a given root
        - for datacubes, in terms of their data storage/access:
            - load a numpy array stored in RAM
            - load a numpy array pointing to a memory map
            - load a Dask representation

    - load non-native files
        - in terms of their Python class representation:
            - as datacubes
            - as arrays
        - in terms of their data storage/access:
            - load a numpy array stored in RAM
            - load a numpy array pointing to a memory map
            - load a Dask representation
        - in terms of preprocessing
            - bin


    """
    # parse filetype

    er1 = "filepath must be a string or Path"
    er2 = "specified filepath does not exist"
    assert(isinstance(filepath, (str,pathlib.Path) )), er1
    assert(exists(filepath)), er2

    filetype = parse_filetype(filepath) if filetype is None else filetype
    assert filetype in [
        "py4DSTEM",
        "dm",
        #"empad",
        #"mrc_relativity",
        #"gatan_K2_bin",
        #"kitware_counted",
    ], "Error: ft argument not recognized"


    # Call appropriate reader

    if filetype == 'py4DSTEM':
        data = read_py4DSTEM(
            filepath,
            #mem=mem,
            #binfactor=binfactor,
            **kwargs
        )

    elif filetype == 'dm':
        data = read_dm(
            filepath,
            #mem=mem,
            #binfactor=binfactor,
            **kwargs
        )

    return data




def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
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
    #elif fext in [".raw"]:
    #    return "empad"
    #elif fext in [".mrc"]:
    #    return "mrc_relativity"
    #elif fext in [".gtg", ".bin"]:
    #    return "gatan_K2_bin"
    #elif fext in [".kitware_counted"]:
    #    return "kitware_counted"
    else:
        raise Exception(f"Unrecognized file extension {fext}.  To force reading as a particular filetype, pass the 'filetype' keyword argument.")
























    #if ft == "py4DSTEM":
    #    data = read_py4DSTEM(
    #        fp, mem=mem, binfactor=binfactor, metadata=metadata, **kwargs
    #    )
    #elif ft == "empad":
    #    data = read_empad(fp, mem, binfactor, metadata=metadata, **kwargs)
    #elif ft == "mrc_relativity":
    #    data = read_mrc_relativity(fp, mem, binfactor, metadata=metadata, **kwargs)
    #elif ft == "gatan_K2_bin":
    #    data = read_gatan_K2_bin(fp, mem, binfactor, metadata=metadata, **kwargs)
    #elif ft == "kitware_counted":
    #    data = read_kitware_counted(fp, mem, binfactor, metadata=metadata, **kwargs)
    #else:
    #    raise Exception(
    #        "Unrecognized file extension {}.  To force reading as a particular filetype, pass the 'ft' keyword argument.".format(
    #            fext
    #        )
    #    )


    #assert(mem in ['RAM','MEMMAP', 'DASK']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    #assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    #assert(binfactor>=1), "Error: binfactor must be >= 1"
    #if binfactor > 1:
    #    assert (
    #        mem != "MEMMAP"
    #    ), "Error: binning is not supported for memory mapping.  Either set binfactor=1 or set mem='RAM'"




