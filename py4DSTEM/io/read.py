# General reader for 4D-STEM datasets.

import pathlib
from os.path import exists, splitext
from typing import Union, Optional
import h5py

from .native.read import read_py4DSTEM
from .utils import parse_filetype


def read(
    filepath: Union[str,pathlib.Path],
    **kwargs
    ):
    """
    Reader for py4DSTEM-formatted EMD files

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
    """

    # parse filetype
    er1 = f"filepath must be a string or Path, not {type(filepath)}"
    er2 = f"specified filepath '{filepath}' does not exist"
    assert(isinstance(filepath, (str,pathlib.Path) )), er1
    assert(exists(filepath)), er2

    filetype = parse_filetype(filepath) if filetype is None else filetype
    assert filetype == "py4DSTEM", "Incompatible file type for py4DSTEM.io.read. To import data from a non-py4DSTEM EMD file, use py4DSTEM.io.import_"

    data = read_py4DSTEM(
        filepath,
        **kwargs
    )

    return data
