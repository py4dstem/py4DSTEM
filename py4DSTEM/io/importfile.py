# Reader functions for non-native file types

import pathlib
from os.path import exists
from typing import Optional, Union

from py4DSTEM.io.filereaders import (
    load_mib,
    read_abTEM,
    read_arina,
    read_dm,
    read_empad,
    read_gatan_K2_bin,
)
from py4DSTEM.io.parsefiletype import _parse_filetype


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
            options include "dm", "empad", "gatan_K2_bin", "mib", "arina", "abTEM"
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
    assert isinstance(binfactor, int), "Error: argument binfactor must be an integer"
    assert binfactor >= 1, "Error: binfactor must be >= 1"
    if binfactor > 1:
        assert (
            mem != "MEMMAP"
        ), "Error: binning is not supported for memory mapping.  Either set binfactor=1 or set mem='RAM'"

    filetype = _parse_filetype(filepath) if filetype is None else filetype

    if filetype in ("emd", "legacy"):
        raise Exception(
            "EMD file or py4DSTEM detected - use py4DSTEM.read, not py4DSTEM.import_file!"
        )
    assert filetype in [
        "dm",
        "empad",
        "gatan_K2_bin",
        "mib",
        "arina",
        "abTEM",
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
        data = load_mib(filepath, mem=mem, binfactor=binfactor, **kwargs)
    elif filetype == "arina":
        data = read_arina(filepath, mem=mem, binfactor=binfactor, **kwargs)
    elif filetype == "abTEM":
        data = read_abTEM(filepath, mem=mem, binfactor=binfactor, **kwargs)
    else:
        raise Exception("Bad filetype!")

    return data
