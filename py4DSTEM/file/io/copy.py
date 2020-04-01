# Copy an existing py4DSTEM formatted .h5 file to a new file.
#
# The new file may be a complete copy of the original, or it may contain any subset of the original
# files DataObjects.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from .filebrowser import is_py4DSTEM_file, FileBrowser
from .write import save
from .write import save_datacube_group, save_diffraction_group, save_real_group
from .write import save_pointlist_group, save_pointlistarray_group
from ..datastructure import DataCube, DiffractionSlice, RealSlice
from ..datastructure import PointList, PointListArray
from ..datastructure import DataObject, Metadata

from ..log import log, Logger
logger = Logger()

@log
def copy_from_indices(original_filepath, new_filepath, indices):
    """
    Copies DataObjects specified by indices from the py4DSTEM .h5 file at original_filepath to a
    new file at new_filepath.

    Accepts:
        indices             either an int or a list of ints. Copy the DataObjects
                            corresponding to these indices in the original file. The indexing of
                            a file can be seen with, e.g., a FileBrowser.show_dataobjects() call.
        original_filepath   path to an existing py4DSTEM .h5 file to copy
        new_filepath        path to the new file to write
    """
    if isinstance(indices, int):
        indices = [indices]
    elif isinstance(indices, list):
        assert all([isinstance(item,int) for item in indices]), "Error: indices must be an int or a list of ints"
    else:
        raise Exception("indices must be either an int or a list of ints.")

    #### Open a FileBrowser ####
    print("Opening file {}...".format(original_filepath))
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3
    assert max(indices) <= browser.N_dataobjects, "DataObject at index {} was requested, but the original file contains only {} DataObjects.".format(max(indices), browser.N_dataobjects)

    #### Get data from existing .h5 file ####
    dataobjects = browser.get_dataobjects(indices)

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_datacubes(original_filepath, new_filepath):
    """
    Copies only the DataCubes from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_datacubes()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_diffractionslices(original_filepath, new_filepath):
    """
    Copies only the DiffractionSlices from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_diffractionslices()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_realslices(original_filepath, new_filepath):
    """
    Copies only the RealSlices from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_realslices()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_pointlists(original_filepath, new_filepath):
    """
    Copies only the PointLists from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_pointlists()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_pointlistarrays(original_filepath, new_filepath):
    """
    Copies only the pointlistarrays from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_pointlistarrays()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy_all(original_filepath, new_filepath):
    """
    Copies all DataObjects from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = browser.get_dataobjects('all')

    #### Write new file ####
    save(dataobjects, new_filepath, save_metadata=False)

    ##### Finish and close #####
    browser.close()

@log
def copy_all_except_datacubes(original_filepath, new_filepath):
    """
    Copies all DataObjects from original filepath to new_filepath.
    """
    #### Open a FileBrowser ####
    assert is_py4DSTEM_file(original_filepath), "filepath paramter must point to an existing py4DSTEM file."
    browser = FileBrowser(original_filepath)
    assert browser.version == (0,3), "Copying py4DSTEM files only supported in v0.3 and higher." ##TODO: this forces v0.3, not >v0..3

    #### Get data ####
    dataobjects = []
    dataobjects += browser.get_diffractionslices()
    dataobjects += browser.get_realslices()
    dataobjects += browser.get_pointlists()
    dataobjects += browser.get_pointlistarrays()

    #### Write new file ####
    save(dataobjects, new_filepath)

    ##### Finish and close #####
    browser.close()

@log
def copy(original_filepath, new_filepath, save='all'):
    """
    Copies DataObjects specified by indices from the py4DSTEM .h5 file at original_filepath to a
    new file at new_filepath.

    Exact behavior depends on the kwarg save:
        save = 'all'        copy the complete file
        save = 'DataCube'   if save is 'DataCube', 'DiffractionSlice', 'RealSlice', 'PointList', or
                            'PointListArray', then save only the DataObjects of this type
        save = 'nocube'     save everything except any DataCube objects
        save = 5            if save is an int, copy only the DataObject corresponding to this index
                            in the original file. The indexing of a file can be seen with, e.g.,
                            a FileBrowser.show_dataobjects() call.
        save = [1,2,5]      if save is a list of ints, copy the DataObjects corresponding to these
                            indices in the original file. The indexing of

    Accepts:
        original_filepath   path to an existing py4DSTEM .h5 file to copy
        new_filepath        path to the new file to write
        save                see above
    """
    if isinstance(save, str):
        if save=='all':
            copy_all(original_filepath, new_filepath)
        elif save=='DataCube':
            copy_datacubes(original_filepath, new_filepath)
        elif save=='DiffractionSlice':
            copy_diffractionslices(original_filepath, new_filepath)
        elif save=='RealSlice':
            copy_realslices(original_filepath, new_filepath)
        elif save=='PointList':
            copy_pointlists(original_filepath, new_filepath)
        elif save=='PointListArrays':
            copy_pointlistarrays(original_filepath, new_filepath)
        elif save=='nocube':
            copy_all_except_datacubes(original_filepath, new_filepath)
        else:
            raise Exception("If save is a str, it must be 'all', 'DataCube', 'DiffractionSlice', 'RealSlice', 'PointList', 'PointListArray', or 'nocube'. Recieved argument {}.".format(save))

    elif isinstance(save, int):
        copy_from_indices(original_filepath, new_filepath, save)

    elif isinstance(save, list):
        assert all([isinstance(item, int) for item in save]), "If save is a list, all elements must be ints."
        copy_from_indices(original_filepath, new_filepath, save)

    else:
        raise Exception("save must be an int, a list of ints, or a str. See copy docstring for more info.")


################### END OF COPY FUNCTIONS #####################



