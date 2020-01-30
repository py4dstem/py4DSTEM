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

import hyperspy.api_nogui as hs
from ncempy.io.dm import fileDM
import h5py
import numpy as np

from .empad import read_empad
from .filebrowser import FileBrowser, is_py4DSTEM_file
from ..datastructure import DataCube, PointListArray
from ..datastructure import Metadata, CountedDataCube
from ..log import log
from ...process.utils import bin2D, tqdmnd

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
    load = 5:
        If load behavoir is an int, loads the object found at that index in a FileBrowser
        instantiated from filename.
    load = [0,1,5,8,...]:
        If load behavoir is a list of ints, loads the set of objects found at those indices in
        a FileBrowser instantiated from filename.

    For non-py4DSTEM files, the output is a DataCube, and load behavior is as follows:
    load = None
        attempt to load a datacube using hyperspy
    load = 'dmmmap'
        load a dm file (.dm3 or .dm4), memory mapping the datacube, using dm.py
    load = 'empad'
        load an EMPAD formatted file, using empad.py
    load = 'gatan_bin'
        load a sequence of *.bin files output by a Gatan K2 camera. Any file in the folder can be
        passed as the argument. The reader searches for the *.gtg file that contains the metadata,
        then maps the chunked binary files.
    load = 'kitware_counted'
        load a *.h5 file in the Kitware format. (Returns a CountedDataCube)
    load = 'relativity'
        Load an MRC file written from the IDES Relativity subframing system, which generates
        multiple small, tiled diffraction patterns on each detector frame; each subframe
        corresponds to a distinct scan position, enabling faster effective frame rates than
        the camera readout time, at the expense of subframe sampling size.
        The output is a memory map to the 4D datacube, which must be sliced into subframes using
        the relativity module in py4DTEM.process.preprocess.relativity; see there for more info.
        This functionality requires the mrcfile package, which can be installed with
            pip install mrcfile
    """
    if not is_py4DSTEM_file(filename):
        print("{} is not a py4DSTEM file.".format(filename))
        if load is None:
            print("Couldn't identify input, attempting to read with hyperspy...")
            try:
                output = read_with_hyperspy(filename)
            except:
                print("Hyperspy read failed")
                return "Hyperspy read failed"
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
        elif load == 'gatan_bin':
            print('Reading Gatan binary files...')
            output = read_gatan_binary(filename)
        elif load == 'kitware_counted':
            print('Reading a Kitware electron counted dataset.')
            output = read_kitware_counted(filename)
        elif load == 'kitware_counted_mmap':
            print('Memory-mapping a Kitware electron counted dataset.')
            output = read_kitware_counted_mmap(filename)
        else:
            raise ValueError("Unknown value for parameter 'load' = {}. See the read docstring for more info.".format(load))

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
            raise ValueError("Unknown value for parameter 'load' = {}. See the read docstring for more info.".format(load))

    return output

def read_with_hyperspy(filename):
    """
    Read a non-py4DSTEM file using hyperspy.
    """
    # Get metadata
    metadata = Metadata(init='hs',filepath=filename)

    # Get data
    hyperspy_file = hs.load(filename)
    data = hyperspy_file.data

    # Get datacube
    datacube = DataCube(data = data)

    # Link metadata and data
    datacube.metadata = metadata

    # Set scan shape, if in metadata
    try:
        R_Nx = int(metadata.get_metadata_item('scan_size_Nx'))
        R_Ny = int(metadata.get_metadata_item('scan_size_Ny'))
        datacube.set_scan_shape(R_Nx, R_Ny)
    except ValueError:
        print("Warning: scan shape not detected in metadata; please check / set manually.")

    return datacube

def read_dm_mmap(filename):
    """
    Read a .dm3/.dm4 file, using dm.py to read data to a memory mapped np.memmap object, which
    is stored in the outpute DataCube.data.

    Read the metadata with hyperspy.
    """
    assert (filename.endswith('.dm3') or filename.endswith('.dm4')), 'File must be a .dm3 or .dm4'

    # Get metadata
    metadata = Metadata(init='hs',filepath=filename)

    # Load .dm3/.dm4 files with dm.py
    with fileDM(filename,verbose=False) as dmfile:
        data = dmfile.getMemmap(0)

    # Get datacube
    datacube = DataCube(data = data)

    # Link metadata and data
    datacube.metadata = metadata

    # Set scan shape, if in metadata
    try:
        R_Nx = int(metadata.get_metadata_item('scan_size_Nx'))
        R_Ny = int(metadata.get_metadata_item('scan_size_Ny'))
        datacube.set_scan_shape(R_Nx, R_Ny)
    except ValueError:
        print("Warning: scan shape not detected in metadata; please check / set manually.")

    return datacube

def read_empad_file(filename):
    """
    Read an empad file, using empad.py to read the data.

    Additionally reads and attaches metadata. # TODO
    """
    # Get data
    data = read_empad(filename)
    data = data[:,:,:128,:]

    # Get metadata
    metadata = None
    #metadata = Metadata(init='empad',filepath=filename)  # TODO: add setup_metadata_empad method
                                                          # to Metadata object

    datacube = DataCube(data = data)
    # datacube.metadata = metadata

    return datacube

def read_gatan_binary(filename):
    """
    Read a folder with Gatan binary files. The folder must contain a *.gtg file (this is where
    the metadata for the whole dataset lives) as well as a sequence of 8 *.bin files. DO NOT
    change the folder structure, as this relies on having only one scan per folder (if you
    have two scans with different names, this will fail.)

    filename can refer to any of the *.bin files, the *.gtg file, or
    the directory containing them.

    Requires ncempy: `pip install ncempy` and numba: `conda install numba`
    """

    #this import is delayed to here so that numba is not a base dependency
    from . import gatanK2

    data_map = gatanK2.K2DataArray(filename)
    datacube = DataCube(data = data_map)

    #metadata = Metadata(init='hs',filepath=datacube.data4D._gtg_file)

    #datacube.metadata = metadata

    return datacube


def read_kitware_counted(filename):
    """
    Read a Kitware counted dataset (i.e. from the NCEM 4D camera)
    (Not for py4DSTEM formatted files, which may be suported by 
    Kitware in the future.)
    """
    hfile = h5py.File(filename,'r')

    R_Nx = hfile['electron_events']['scan_positions'].attrs['Ny']
    R_Ny = hfile['electron_events']['scan_positions'].attrs['Nx']

    Q_Nx = hfile['electron_events']['frames'].attrs['Ny']
    Q_Ny = hfile['electron_events']['frames'].attrs['Nx']

    pla = PointListArray([('ind','u4')],(R_Nx,R_Ny))

    print('Importing Electron Events:',flush=True)

    for (i,j) in tqdmnd(int(R_Nx),int(R_Ny)):
        ind = np.ravel_multi_index((i,j),(R_Nx,R_Ny))
        pla.get_pointlist(i,j).add_dataarray(hfile['electron_events']['frames'][ind].astype([('ind','u4')]))

    return CountedDataCube(pla,[Q_Nx,Q_Ny],'ind',use_dask=False)

def read_kitware_counted_mmap(filename):
    """
    Read a Kitware counted dataset (i.e. from the NCEM 4D camera)
    (Not for py4DSTEM formatted files, which may be suported by 
    Kitware in the future.)
    """
    hfile = h5py.File(filename,'r')

    R_Nx = hfile['electron_events']['scan_positions'].attrs['Ny']
    R_Ny = hfile['electron_events']['scan_positions'].attrs['Nx']

    Q_Nx = hfile['electron_events']['frames'].attrs['Nx']
    Q_Ny = hfile['electron_events']['frames'].attrs['Ny']

    return CountedDataCube(hfile['electron_events']['frames'],[Q_Nx,Q_Ny],[None],
        use_dask=False,R_Nx=R_Nx,R_Ny=R_Ny)
