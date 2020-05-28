# General reader for 4D-STEM datasets.

from .native import read_py4DSTEM_4D
#from .nonnative import read_boink
from ..datastructure import DataCube

def read_4D(fp, mem="RAM", bin_Q=1, order="RQ")
    """
    General read function for 4D-STEM datasets.

    Takes a filename as input, parses the filetype, and calls the appropriate read function
    for that filetype.

    Accepts:
        fp          str         Path to the file
        mem         str         (opt) Specifies how the data should be stored in memory; TK TODO TK
        bin_Q       int         (opt) Bin the data, in diffraction space, as it is loaded
        order       str         (opt) "RQ" or "QR", specifying the order of the real/diffraction space
                                        coordinates in the dataset.  Default is "RQ".

    Returns:
        dc          DataCube    the 4D datacube
    """
    ft = parse_filetype(fp)

    if ft == "py4DSTEM":
        dc = read_py4DSTEM(fp, mem, bin_Q, order)
    elif ft == "boink":
        dc = read_boink(fp, mem, bin_Q, order)
    # TK TODO TK
    else:
        raise Exception("Unknown filetype, {}".format(ft))

    return dc



def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    # TK TODO TK
    return

