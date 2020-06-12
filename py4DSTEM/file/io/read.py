# General reader for 4D-STEM datasets.

import pathlib
from os.path import splitext
from .native import read_py4DSTEM, is_py4DSTEM_file
from .nonnative import *

def read(fp, mem="RAM", binfactor=1, ft=None, **kwargs):
    """
    General read function for 4D-STEM datasets.

    Takes a filename as input, parses the filetype, and calls the appropriate read function
    for that filetype.

    Accepts:
        fp          str or Path Path to the file
        mem         str         (opt) Specifies how the data should be stored; must be "RAM" or "MEMMAP".
                                "RAM" loads the entire dataset into memory. "MEMMAP" is useful for large datasets;
                                it does not load the data into memory, but instead creates a map describing where
                                each diffraction pattern lives in storage, and only loads data into memory as
                                needed.
        binfactor   int         (opt) Bin the data, in diffraction space, as it's loaded. On-load binning enables
                                datasets which, in storage, are larger than the system RAM to still be loaded into
                                RAM, provided the amount of binning is sufficient.  Binning by N reduces the
                                filesize by N^2, so for instance, on a system with only 16 GB of RAM, its possible
                                to load datasets of up to 64, 144, or 256 GB using binfactors of 2, 3, or 4.
                                Default is 1.
                                *Note 1: binning is only supported with mem='RAM'.
                                **Note 2: binning may cause 'wraparound' errors (e.g. if the datatype is uint16
                                and the summed pixels in a bin exceed 65536, the count 'wraps back around' to 0).
                                This can be avoided by explicitly casting the datatype by passing the keyword
                                argument 'dtype', however, casting will also affect the size of the data.
        ft          str         (opt) Force py4DSTEM to attempt to read the file as a specified filetype, rather
                                than trying to determine this automatically. Must be None or a str from
                                'py4DSTEM', 'dm', 'empad', 'mrc_relativity', 'gatan_K2_bin', 'kitware_counted'.
                                Default is None.
        dtype       dtype       Used when binning data, ignored otherwise. Defaults to whatever the type of the
                                raw data is, to avoid enlarging data size. May be useful to avoid 'wraparound'
                                errors.
        data_id    int/str/list For py4DSTEM files only.  Specifies which data to load. Use integers to specify
                                the data index, or strings to specify data names. A list or tuple returns a list
                                of DataObjects. Returns the specified data.
        topgroup     str        For py4DSTEM files only.  Stricty, a py4DSTEM file is considered to be
                                everything inside a toplevel subdirectory within the HDF5 file, so that if
                                desired one can place many py4DSTEM files inside a single H5.  In this case,
                                when loading data, the topgroup argument is passed to indicate which
                                py4DSTEM file to load. If an H5 containing multiple py4DSTEM files is passed
                                without a topgroup specified, the topgroup names are printed to screen.
        metadata    bool        For py4DTEM files only.  If True, returns a dictionary with the file metadata.
        log         bool        For py4DSTEM files only.  If True, writes the processing log to a plaintext file
                                called splitext(fp)[0]+'.log'.

    Returns:
        data        variable    The data. When loading non-native filetypes, the output type is a DataCube.
                                When loading from a native .h5 file, loading one data block (i.e. 'load' is
                                passed a string or integer) returns a single DataObject instance, while loading
                                several data blocks returns a list of instances.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"
    if binfactor > 1:
        assert(mem!='MEMMAP'), "Error: binning is not supported for memory mapping.  Either set binfactor=1 or set mem='RAM'"
    assert(ft in [None,'dm','empad','mrc_relativity','gatan_K2_bin','kitware_counted']), "Error: ft argument not recognized"

    if ft is None:
        ft = parse_filetype(fp)

    if ft == "py4DSTEM":
        data,md = read_py4DSTEM(fp, mem=mem, binfactor=binfactor, **kwargs)
    elif ft == "dm":
        data,md = read_dm(fp, mem, binfactor, **kwargs)
    elif ft == "empad":
        data,md = read_empad(fp, mem, binfactor, **kwargs)
    elif ft == 'mrc_relativity':
        data,md = read_mrc_relativity(fp, mem, binfactor, **kwargs)
    elif ft == "gatan_K2_bin":
        data,md = read_gatan_K2_bin(fp, mem, binfactor, **kwargs)
    elif ft == "kitware_counted":
        data,md = read_kitware_counted(fp, mem, binfactor, **kwargs)
    else:
        raise Exception("Unrecognized file extension {}.  To force reading as a particular filetype, pass the 'ft' keyword argument.".format(fext))

    return data, md


def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"

    _,fext = splitext(fp)
    if fext in ['.h5','.H5','hdf5','HDF5','.py4dstem','.py4DSTEM','.PY4DSTEM','.emd','.EMD']:
        if is_py4DSTEM_file(fp):
            return 'py4DSTEM'
        else:
            raise Exception("Non-py4DSTEM formatted .h5 files are not presently supported.")
    elif fext in ['.dm','.dm3','.dm4','.DM','.DM3','.DM4']:
        return 'dm'
    elif fext in ['.empad']:
        # TK TODO
        return 'empad'
    elif fext in ['.mrc']:
        # TK TODO
        return 'mrc_relativity'
    elif fext in ['.gatan_K2_bin']:
        # TK TODO
        return 'gatan_K2_bin'
    elif fext in ['.kitware_counted']:
        # TK TODO
        return 'kitware_counted'
    else:
        raise Exception("Unrecognized file extension {}.  To force reading as a particular filetype, pass the 'ft' keyword argument.".format(fext))



