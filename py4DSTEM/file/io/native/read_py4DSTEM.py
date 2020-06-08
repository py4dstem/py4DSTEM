# Reads a py4DSTEM formatted (EMD type 2) 4D-STEM dataset

import h5py
from pathlib import Path
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_py4DSTEM_version, version_is_geq
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
                                    topgroup (str)          Used with HDF5 files containing multiple py4DSTEM
                                                            formatted file trees, to specify which tree/subfile
                                                            to open. In most instances there will be only one,
                                                            and this argument need not be passed.

    Returns:
        dc          DataCube    The 4D-STEM data.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"
    assert(is_py4DSTEM_file(fp)), "Error: {} is not recognized as a native py4DSTEM file.".format(fp)

    # Identify toplevel H5 group containing the py4DSTEM tree of interest
    topgroups = get_py4DSTEM_topgroups(fp)
    if 'topgroup' in kwargs.keys():
        tg = kwargs.keys['topgroup']
        assert(tg in topgroups), "Error: specified topgroup, {}, not found.".format(tg)
    else:
        if len(topgroups)==1:
            tg = topgroups[0]
        else:
            print("Multiple topgroups detected.  Please specify one by passing the 'topgroup' keyword argument.")
            print("")
            print("Topgroups found:")
            for tg in topgroups:
                print(tg)
            return None,None

    # Get version info, and open the appropriate version FileBrowser
    version = get_py4DSTEM_version(fp,tg)
    print("py4DSTEM (EMD type 2) v{}.{}.{} file detected".format(version[0],version[1],version[2]))
    if version_is_geq(version,(0,9,0)):
        browser = FileBrowser(fp,tg)
    else:
        browser = FileBrowser_v0(fp,tg)

    # If 
    if 'load' not in kwargs.keys():
        print("This file contains the following data objects:")
        print("")
        browser.show_dataobjects()
        browser.close()
        print("")
        print("To load data, call this function again, this time passing the keyword 'load'.")
        print("To load one object, use 'load = x' where x is either the object's index (integer) or name (string).")
        print("To load several objects, use 'load = [x1,x2,x3,...]' where xi are all indices, or all names.")
        return None,None

    # Get data
    else:
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








