import pathlib
from os.path import exists, splitext
from typing import Union, Optional

from .utils import parse_filetype
from .nonnative import read_empad, read_dm, read_gatan_K2_bin


def import_file(
    filepath: Union[str, pathlib.Path],
    mem: Optional[str] = "RAM",
    binfactor: Optional[int] = 1,
    filetype: Optional[str] = None,
    **kwargs,
):
    """
    Reader for non-native file formats.
    Supports Gatan DM3/4, some EMPAD file versions, and Gatan K2 bin/gtg

    Args:
        filepath:   Path to the file. For K2 raw datasets, pass the path to the gtg file
        mem:        "RAM" to load the dataset into ram, "MEMMAP" to produce a memory map
                        (For K2 raw data, only MEMMAP is supported)
        binfactor   Diffraction space binning factor for bin-on-load.
        filetype    Used to override automatic filetype detection.

    For documentation of kwargs, refer to the individual readers (currently
        only the K2 reader uses kwargs.)

    Returns:
        data    DataCube if 4D data is found, else an Array containing 2D or 3D data

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

    filetype = parse_filetype(filepath) if filetype is None else filetype

    assert filetype in [
        "dm",
        "empad",
        "gatan_K2_bin",
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
    else:
        raise Exception("Bad filetype!")

    return data


