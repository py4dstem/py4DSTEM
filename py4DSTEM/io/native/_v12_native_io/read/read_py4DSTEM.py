# File reader for py4DSTEM files

import h5py
import numpy as np
from os.path import splitext, exists
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups
from .read_utils import get_py4DSTEM_version, version_is_geq
from .read_v0_12 import read_v0_12
from .read_v0_9 import read_v0_9
from .read_v0_7 import read_v0_7
from .read_v0_6 import read_v0_6
from .read_v0_5 import read_v0_5

def read_py4DSTEM(filepath, metadata=False, **kwargs):
    """
    File reader for py4DSTEM formated .h5 files.  Precise behavior is
    detemined by which arguments are passed -- see below.

    Args:
        filepath (str or pathlib.Path): When passed a filepath only, this function checks
            if the path points to a valid py4DSTEM file, then prints its contents to
            screen.
        data_id (int/str/list, optional): Specifies which data to load. Use integers to
            specify the data index, or strings to specify data names. A list or tuple
            returns a list of DataObjects. Returns the specified data.
        topgroup (str, optional:) Stricty, a py4DSTEM file is considered to be everything
            inside a toplevel subdirectory within the HDF5 file, so that if desired one
            can place many py4DSTEM files inside a single H5.  In this case, when loading
            data, the topgroup argument is passed to indicate which py4DSTEM file to
            load. If an H5 containing multiple py4DSTEM files is passed without a
            topgroup specified, the topgroup names are printed to screen.
        metadata (bool, optional) If True, returns the metadata as a Metadata instance.
        mem (str, optional): Only used if a single DataCube is loaded. In this case,
            mem specifies how the data should be stored; must be "RAM" or "MEMMAP". See
            docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor (int, optional): Only used if a single DataCube is loaded. In this
            case, a binfactor of > 1 causes the data to be binned by this amount as it's
            loaded.
        dtype (dtype, optional): Used when binning data, ignored otherwise. Defaults to
            whatever the type of the raw data is, to avoid enlarging data size. May be
            useful to avoid 'wraparound' errors.

    Returns:
        (variable): The output depends on usage:

            * If no input arguments with return values (i.e. data_id or metadata) are
              passed, nothing is returned.
            * If metadata==True, returns a Metadata instance with the file metadata.
            * Otherwise, a single DataObject or list of DataObjects are returned, based
              on the value of the argument data_id.
    """
    assert(exists(filepath)), "Error: specified filepath does not exist"
    assert(is_py4DSTEM_file(filepath)), "Error: {} isn't recognized as a py4DSTEM file.".format(filepath)

    # For HDF5 files containing multiple valid EMD type 2 files (i.e. py4DSTEM files),
    # disambiguate desired data
    tgs = get_py4DSTEM_topgroups(filepath)
    if 'topgroup' in kwargs.keys():
        tg = kwargs['topgroup']
        #assert(tg in tgs), "Error: specified topgroup, {}, not found.".format(tg)
    else:
        if len(tgs)==1:
            tg = tgs[0]
        else:
            print("Multiple topgroups were found -- please specify one:")
            print("")
            for tg in tgs:
                print(tg)
            return

    # Get py4DSTEM version and call the appropriate read function
    version = get_py4DSTEM_version(filepath, tg)
    if version_is_geq(version,(0,12,0)): return read_v0_12(filepath, **kwargs)
    elif version_is_geq(version,(0,9,0)): return read_v0_9(filepath, **kwargs)
    elif version_is_geq(version,(0,7,0)): return read_v0_7(filepath, **kwargs)
    elif version_is_geq(version,(0,6,0)): return read_v0_6(filepath, **kwargs)
    elif version_is_geq(version,(0,5,0)): return read_v0_5(filepath, **kwargs)
    else:
        raise Exception('Support for legacy v{}.{}.{} files has not been added yet.'.format(version[0],version[1],version[2]))




