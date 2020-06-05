# Reads a py4DSTEM formatted (EMD type 2) 4D-STEM dataset

import h5py
from pathlib import Path
from .filebrowser import FileBrowser
from ...datastructure import DataCube

def read_py4DSTEM(fp, mem="RAM", binfactor=1, **kwargs):
    """
    Read a py4DSTEM formatted (EMD type 2) 4D-STEM file.

    Because native py4DSTEM files may contain one or many data objects, load behavior depends the kwarg 'load'.
    If this argument is not passed, the function simply lists the contents of the file, prints a message
    describing how to load particular items, and returns nothing.  Otherwise, the behavior depends on the
    dtype of the argument passed with 'load':

    dtype           behavior
    -----           --------
    str             load the DataObject(s) with a name matching this string
    int             load the DataObject found at this index in a FileBrowser for this filepath
    list of ints    load the set of objects at these indices in the FileBrowser for this filepath
    list of str     load the set of objects with name matching these strings

    To determine the indices of the various objects in a py4DSTEM file, call this function without the 'load'
    kwarg and they will be printed.  Alternatively a FileBrowser can be loaded manually - see
    py4DSTEM.file.io.FileBrowser.

    The 'mem' and 'binfactor' arguments are only used if the object loaded is a single DataCube.
    Otherwise, these arguments are ignored.

    Accepts:
        fp          str or Path Path to the file
        mem         str         (opt) Specifies how the data should be stored; must be "RAM" or "MEMMAP". See
                                docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor   int         (opt) Bin the data, in diffraction space, as it's loaded. See docstring for
                                py4DSTEM.file.io.read.  Default is 1.
        **kwargs                Recognized keywords:
                                    load (int, str, or list)See above for behaviors.
                                    dtype (dtype)           Used when binning data, ignored otherwise.
                                                            By defaults to whatever the type of the raw data
                                                            is, to avoid enlarging data size. May be useful
                                                            to avoid'wraparound' errors.

    Returns:
        dc          DataCube    The 4D-STEM data.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"
    assert(is_py4DSTEM_file(fp)), "Error: {} is not recognized as a native py4DSTEM file.".format(fp)

    browser = FileBrowser(fp)

    if 'load' not in kwargs.keys():
        print("Native py4DSTEM (EMD type 2) file detected.  This file contains the following data objects:")
        print("")
        print("") # TODO
        print("")
        print("To load one or more objects, call this function again, this time passing the keyword 'load'.")
        print("For one object, use 'load = x' where x is either the object's index (integer) or name (string).")
        print("For several objects, use 'load = [x1,x2,x3,...]' where xi are all indices, or all names.")
        return

    else:
        # Get data
        load = kwargs['load']
        if type(load) == int:
            # Check if its a DataCube - if so, pass below to memmap stuff
            data = browser.get_dataobject(load)
        elif type(load) == str:
            # Check if its a DataCube - if so, get index and pass below to memmap stuff
            data = browser.get_dataobject_by_name(load)
        elif type(load) == list:
            if type(load[0]) == int:
                assert(all([isinstance(item,int) for item in load])), "Error: if load is a list, items must all be ints or all be strings. Mixed lists are not supported."
                browser.get_dataobjects(load)
            elif type(load[0]) == str:
                assert(all([isinstance(item,int) for item in load])), "Error: if load is a list, items must all be ints or all be strings. Mixed lists are not supported."
                # TODO
                # browser.get_dataobjects_by_name(load)
                data = None

        # Get metadata
        md = None # TODO



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
        raise Exception("Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'.")
        return

    return data, md





def is_py4DSTEM_file(fp):
    """ Returns True iff fp points to a py4DSTEM formatted (EMD type 2) file.
    """
    py4DSTEM_attrs = ['emd_group_type','version_major','version_minor','version_release','UUID']
    with h5py.File(fp,'r') as f:
        if '4DSTEM_experiment' in f.keys():
            if np.all([attr in f['py4DSTEM_experiment'] for attr in py4DSTEM_attrs]):
                return True
    return False

def get_py4DSTEM_version(fp):
    """ Returns the version (major,minor,release) of a py4DSTEM file.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not a py4DSTEM file of version >= 0.9.0"
    with h5py.File(fp,'r') as f:
        version_major = f['py4DSTEM_experiment'].attrs['version_major']
        version_minor = f['py4DSTEM_experiment'].attrs['version_minor']
        version_release = f['py4DSTEM_experiment'].attrs['version_release']
    return version_major, version_minor, version_release

def version_is_greater_or_equal(current,minimum):
    """ Returns True iff current version (major,minor,release) is greater than or equal to minimum."
    """
    if current[0]>minimum[0]:
        return True
    elif current[0]==minimum[0]:
        if current[1]>minimum[1]:
            return True
        elif current[1]==minimum[1]:
            if current[2]>=minimum[2]:
                return True
        else:
            return False
    else:
        return False





