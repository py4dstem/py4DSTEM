# General reader for 4D-STEM datasets.

import pathlib
from os.path import splitext
from .native import read_py4DSTEM
from .nonnative import read_dm, read_empad, read_mrc_relativity, read_gatan_K2_bin, read_kitware_counted

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
        ft          str         (opt) Force py4DSTEM to attempt to read the file as a specified filetype, rather
                                than trying to determine this automatically. Must be None or a str from 'dm',
                                'empad', 'mrc_relativity', 'gatan_K2_bin', 'kitware_counted'.  Default is None.
        **kwargs                (opt) When reading the native h5 file format, additional keyword arguments are
                                used to indicate loading behavior in the case where the source file contains
                                multiple data objects.

                                Recognized keywords are:

                                    TKTKkwarg1       int         descrption TKTKTK

    Returns:
        data        *           The data. The output type is contingent.
                                When loading non-native filetypes, the output type is a DataCube.
                                When loading from a native .h5 file, if a single DataObject is loaded, the output
                                type is that of the DataObject.  If multiple objects are loaded, the output is
                                a list of DataObjects.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(ft in [None,'dm','empad','mrc_relativity','gatan_K2_bin','kitware_counted']), "Error: ft argument not recognized"

    if ft is None:
        ft = parse_filetype(fp)

    if ft == "py4DSTEM":
        data,md = read_py4DSTEM(fp, mem, binfactor, **kwargs)
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
        raise Exception("Unknown filetype, {}".format(ft))

    return data, md


def parse_filetype(fp):
    """ Accepts a path to a 4D-STEM dataset, and returns the file type.
    """
    assert(isinstance(fp,(str,pathlib.Path))), "Error: filepath fp must be a string or pathlib.Path"

    _,fext = splitext(fp)
    if fext in ['dm','dm3','dm4','DM','DM3','DM4']:
        return 'dm'
    elif fext in ['empad']:
        # TK TODO
        return 'empad'
    elif fext in ['mrc']:
        # TK TODO
        return 'mrc_relativity'
    elif fext in ['gatan_K2_bin']:
        # TK TODO
        return 'gatan_K2_bin'
    elif fext in ['kitware_counted']:
        # TK TODO
        return 'kitware_counted'
    return ft

