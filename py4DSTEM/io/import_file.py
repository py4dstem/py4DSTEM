import pathlib
from os.path import exists, splitext
from typing import Union, Optional

from py4DSTEM.io.utils import parse_filetype
from py4DSTEM.io.nonnative import read_empad, read_dm, read_gatan_K2_bin, load_mib

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

    Args:
        filepath (str or Path): Path to the file.
        mem (str):  Must be "RAM" or "MEMMAP". Specifies where the data is
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


