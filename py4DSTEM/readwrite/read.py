# Read 4DSTEM data
#
# File Formats
# .h5 files conforming to the py4DSTEM format are, of course, supported. The complete py4DSTEM file
# format is found in filestructure.txt.
# For the vast wilderness of additional electron microscopy file formats, hyperspy
# (www.hyperspy.org) is used to load data and scrape metadata. Most formats that hyperspy can read,
# py4DSTEM will likely be able to handle, however, not all formats have been tested at this stage.
#
# Tested formats which py4DSTEM should read correctly include:
#   .dm3
#   .dm4
#
# An aside: the authors gratefully thank the developers of hyperspy for all their efforts - their
# work has been a shining beacon amidst the dark and antiscientific morass of closed and proprietary
# formats that plague the world of electron scattering. Friends: a thousand times, thank you! <3, b

import hyperspy.api as hs
from .filebrowser import FileBrowser, is_py4DSTEM_file
from ..process.datastructure import DataCube
from ..process.datastructure import Metadata
from ..process.log import log

###################### BEGIN read FUNCTIONS ########################

@log
def read(filename, load=None):
    """
    General read function.

    Takes a filename as input, and outputs some py4DSTEM dataobjects.

    For non-py4DSTEM files, the output is a DataCube.

    For py4DSTEM files, the behavior depends on the kwarg load, as follows:
    load = None
        load the first DataObject found; useful for files containing only a single DataObject
    load = 'all':
        load all DataObjects found in the file
    load = 'name':
        load the DataObject(s) named 'name'. There is no catch for objects named 'all' - don't name
        DataObjects 'all'! ;)
    load_behavior = 5:
        If load behavoir is an int, loads the object found at that index in a FileBrowser
        instantiated from filename.
    load_behavior = [0,1,5,8,...]:
        If load behavoir is a list of ints, loads the set of objects found at those indices in
        a FileBrowser instantiated from filename.
    """
    if not is_py4DSTEM_file(filename):
        print("{} is not a py4DSTEM file.  Reading with hyperspy...".format(filename))
        output = read_non_py4DSTEM_file(filename)

    else:
        browser = FileBrowser(filename)
        print("{} is a py4DSTEM file, v{}.{}. Reading...".format(filename, browser.version[0], browser.version[1]))
        if load is None:
            output = browser.get_dataobject(0)
        elif load == 'all':
            output = browser.get_dataobjects('all')
        elif type(load) == str:
            output = browser.get_dataobject_by_name(name=load)
        elif type(load) == int:
            output = browser.get_dataobject(load)
        elif type(load) == list:
            assert all([isinstance(item,int) for item in load]), "If load is a list, it must be a list of ints specifying DataObject indices in the files associated FileBrowser."
            output = browser.get_dataobjects(load)
        else:
            print("Error: unknown value for parameter 'load' = {}. Returning None. See the read docstring for more info.".format(load))
            output = None

        browser.close()
    return output


def read_non_py4DSTEM_file(filename):
    """
    Read a non-py4DSTEM file using hyperspy.
    """
    # Load with hyperspy
    try:
        hyperspy_file = hs.load(filename)
    except Exception as err:
        print("Failed to load", err)
        print("Returning None")
        return None

    # Get metadata
    metadata = Metadata(is_py4DSTEM_file = False,
                        original_metadata_shortlist = hyperspy_file.metadata,
                        original_metadata_all = hyperspy_file.original_metadata)

    # Get datacube
    datacube = DataCube(data = hyperspy_file.data)

    # Set scan shape, if in metadata
    try:
        R_Nx = int(metadata.get_metadata_item('scan_size_Nx'))
        R_Ny = int(metadata.get_metadata_item('scan_size_Ny'))
        datacube.set_scan_shape(R_Nx, R_Ny)
    except ValueError:
        print("Warning: scan shape not detected in metadata; please set manually.")

    # Point to metadata from datacube
    datacube.metadata = metadata

    return datacube


