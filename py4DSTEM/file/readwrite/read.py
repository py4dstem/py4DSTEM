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
# work has been a shining beacon of light amidst the dark and antiscientific morass of closed and
# proprietary formats that plague the world of electron scattering. Friends: a thousand times,
# thank you! <3, b

import hyperspy.api as hs
from .dm import dmReader
from .empad import read_empad
from .filebrowser import FileBrowser, is_py4DSTEM_file
from ..datastructure import DataCube
from ..datastructure import Metadata
from ..log import log
from ...process.utils import bin2D

###################### BEGIN read FUNCTIONS ########################

@log
def read(filename, load=None):
    """
    General read function.  Takes a filename as input, and outputs some py4DSTEM dataobjects.

    First checks to see if filename is a .h5 file conforming to the py4DSTEM format.
    In either case, the precise behavior then depends on the kwarg load.

    For .h5 file conforming to the py4DSTEM format, behavior is as follows:
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

    For non-py4DSTEM files, the output is a DataCube, and load behavior is as follows:
    load = None
        attempt to load a datacube using hyperspy
    load = 'dmmmap'
        load a dm file (.d3 or .dm4), memory mapping the datacube, using dm.py
    load = 'empad'
        load an EMPAD formatted file, using empad.py
    load = 'relativity'
        load an MRC file written from the IDES Relativity subframing system, which generates
        multiple small, tiled diffraction patterns on each detector frame; each subframe corresponds
        to a distinct scan position, enabling faster effective frame rates than the camera readout
        time, at the expense of subframe sampling size.
        the output is a memory map to the 4D datacube, which must be sliced into subframes using the
        relativity module in py4DTEM.process.preprocess.relativity; see there for more info.
        This functionality requires the mrcfile package, which can be installed with
            pip install mrcfile
    """
    if not is_py4DSTEM_file(filename):
        print("{} is not a py4DSTEM file.".format(filename))
        if load is None:
            print("Reading with hyperspy...")
            output = read_with_hyperspy(filename)
        elif load == 'dmmmap':
            print("Memory mapping a dm file...")
            output = read_dm_mmap(filename)
        elif load == 'empad':
            print("Reading an EMPAD file...")
            output = read_empad_file(filename)
        elif load == 'relativity':
            import mrcfile
            print("Reading an IDES Relativity MRC file...")
            output = mrcfile.mmap(filename,mode='r')
        else:
            print("Error: unknown value for parameter 'load' = {}. Returning None. See the read docstring for more info.".format(load))
            output = None


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


def read_with_hyperspy(filename):
    """
    Read a non-py4DSTEM file using hyperspy.
    """
    # Get data
    try:
        hyperspy_file = hs.load(filename)
        data = hyperspy_file.data
    except Exception as err:
        print("Failed to load", err)
        print("Returning None")
        return None

    # Get metadata
    metadata = Metadata(is_py4DSTEM_file = False,
                        original_metadata_shortlist = hyperspy_file.metadata,
                        original_metadata_all = hyperspy_file.original_metadata)

    # Get datacube
    datacube = DataCube(data = data)

    # Set scan shape, if in metadata
    try:
        R_Nx = int(metadata.get_metadata_item('scan_size_Nx'))
        R_Ny = int(metadata.get_metadata_item('scan_size_Ny'))
        datacube.set_scan_shape(R_Nx, R_Ny)
    except ValueError:
        print("Warning: scan shape not detected in metadata; please check / set manually.")

    # Point to metadata from datacube
    datacube.metadata = metadata

    return datacube

def read_dm_mmap(filename):
    """
    Read a .dm3/.dm4 file, using dm.py to read data to a memory mapped np.memmap object, which
    is stored in the outpute DataCube.data4D.

    Read the metadata with hyperspy.
    """
    assert (filename.endswith('.dm3') or filename.endswith('.dm4')), 'File must be a .dm3 or .dm4'

    # Load .dm3/.dm4 files with dm.py
    data = dmReader(filename,dSetNum=0,verbose=False)['data']

    # Get metadata
    hyperspy_file = hs.load(filename, lazy=True)
    metadata = Metadata(is_py4DSTEM_file = False,
                        original_metadata_shortlist = hyperspy_file.metadata,
                        original_metadata_all = hyperspy_file.original_metadata)

    # Get datacube
    datacube = DataCube(data = data)

    # Set scan shape, if in metadata
    try:
        R_Nx = int(metadata.get_metadata_item('scan_size_Nx'))
        R_Ny = int(metadata.get_metadata_item('scan_size_Ny'))
        datacube.set_scan_shape(R_Nx, R_Ny)
    except ValueError:
        print("Warning: scan shape not detected in metadata; please check / set manually.")

    # Point to metadata from datacube
    datacube.metadata = metadata

    return datacube

def read_empad_file(filename):
    """
    Read an empad file, using empad.py to read the data.

    Additionally reads and attaches metadata. # TODO
    """
    # Get data
    data = read_empad(filename)
    data = data[:,:,:128,:]

    # Get metadata -- TODO
    metadata = None

    datacube = DataCube(data = data)
    # datacube.metadata = metadata

    return datacube













