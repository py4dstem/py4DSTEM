# Reads a Gatan K2 binary 4D-STEM file.

from pathlib import Path
from ...datastructure import DataCube

def read_gatan_K2_bin(fp, mem="RAM", binfactor=1, **kwargs):
    """
    Read a Gatan K2 4D-STEM file.

    Accepts:
        fp          str or Path Path to the file
        mem         str         (opt) Specifies how the data should be stored; must be "RAM" or "MEMMAP". See
                                docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor   int         (opt) Bin the data, in diffraction space, as it's loaded. See docstring for
                                py4DSTEM.file.io.read.  Default is 1.
        **kwargs

    Returns:
        dc          DataCube    The 4D-STEM data.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"

    if (mem,binfactor)==("RAM",1):
        # TODO
        pass
    elif (mem,binfactor)==("MEMMAP",1):
        # TODO
        pass
    elif (mem)==("RAM"):
        # TODO
        pass
    else:
        # TODO
        pass

    # TK TODO load the data
    # TK TODO load the metadata

    return dc, md


