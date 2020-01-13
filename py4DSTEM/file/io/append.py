# Append additional DataObjects to an existing py4DSTEM formatted .h5 file.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from .filebrowser import is_py4DSTEM_file, FileBrowser
from .write import save_datacube_group, save_diffraction_group, save_real_group
from .write import save_pointlist_group, save_pointlistarray_group, save_counted_datacube_group
from ..datastructure import DataCube, DiffractionSlice, RealSlice
from ..datastructure import PointList, PointListArray, CountedDataCube
from ..datastructure import DataObject, Metadata

from ..log import log, Logger
logger = Logger()

@log
def append_from_dataobject_list(dataobject_list, filepath):
    """
    Appends new dataobjects to an existing py4DSTEM h5 file.

    Accepts:
        dataobject_list     a list of DataObjects to save
        filepath            path to an existing py4DSTEM .h5 file to save
    """

    assert all([isinstance(item,DataObject) for item in dataobject_list]), "Error: all elements of dataobject_list must be DataObject instances."

    #### Get info about existing .h5 file ####
    print("Opening file {}...".format(filepath))
    assert is_py4DSTEM_file(filepath), "filepath parameter must point to an existing py4DSTEM file."
    browser = FileBrowser(filepath)
    if browser.version[0] == 0:
        assert browser.version[1] >= 3, "appending to py4DSTEM files only supported in v0.3 and higher."
    N_dataobjects = browser.N_dataobjects
    N_datacubes = browser.N_datacubes
    N_counted = browser.N_counted
    N_diffractionslices = browser.N_diffractionslices
    N_realslices = browser.N_realslices
    N_pointlists = browser.N_pointlists
    N_pointlistarrays = browser.N_pointlistarrays
    browser.close()

    #### Open file for read/write ####
    try:
        f = h5py.File(filepath,"r+")
    except OSError as e:
        print(e)
        print('The file appears to be open elsewhere...')
        print('This can occur if your datacube is memory-mapped from a py4DSTEM h5 file.')
        print(f'To force close the file, closing any dataobjects open from it, run: py4DSTEM.file.io.close_h5_at_path(\'{filepath}\')')
        print('To force close all h5 files run: py4DSTEM.file.io.close_all_h5()')
        return -1

    topgroup = get_py4DSTEM_topgroup(f)
    # Find data groups
    group_data = f[topgroup]['data']
    group_datacubes = f[topgroup]['data']['datacubes']
    group_counted = f[topgroup]['data']['counted_datacubes']
    group_diffractionslices = f[topgroup]['data']['diffractionslices']
    group_realslices = f[topgroup]['data']['realslices']
    group_pointlists = f[topgroup]['data']['pointlists']
    group_pointlistarrays = f[topgroup]['data']['pointlistarrays']

    # Loop through and save all objects in the dataobjectlist
    for dataobject in dataobject_list:
        name = dataobject.name
        if isinstance(dataobject, DataCube):
            if name == '':
                name = 'datacube_'+str(N_datacubes)
                N_datacubes += 1
            try:
                group_new_datacube = group_datacubes.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_datacubes.keys())])
                name = name+"_"+str(N)
                group_new_datacube = group_datacubes.create_group(name)
            save_datacube_group(group_new_datacube, dataobject)
        elif isinstance(dataobject,CountedDataCube):
            if name == '':
                name = 'counted_datacube_'+str(N_counted)
                ind_cdcs += 1
            try:
                group_new_counted = group_counted.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_counted.keys())])
                name = name+"_"+str(N)
                group_new_counted = group_counted.create_group(name)
            save_counted_datacube_group(group_new_counted, dataobject)
        elif isinstance(dataobject, DiffractionSlice):
            if name == '':
                name = 'diffractionslice_'+str(N_diffractionslices)
                N_diffractionslices += 1
            try:
                group_new_diffractionslice = group_diffractionslices.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_diffractionslices.keys())])
                name = name+"_"+str(N)
                group_new_diffractionslice = group_diffractionslices.create_group(name)
            save_diffraction_group(group_new_diffractionslice, dataobject)
        elif isinstance(dataobject, RealSlice):
            if name == '':
                name = 'realslice_'+str(N_realslices)
                N_realslices += 1
            try:
                group_new_realslice = group_realslices.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_realslices.keys())])
                name = name+"_"+str(N)
                group_new_realslice = group_realslices.create_group(name)
            save_real_group(group_new_realslice, dataobject)
        elif isinstance(dataobject, PointList):
            if name == '':
                name = 'pointlist_'+str(N_pointlists)
                N_pointlists += 1
            try:
                group_new_pointlist = group_pointlists.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_pointlists.keys())])
                name = name+"_"+str(N)
                group_new_pointlist = group_pointlists.create_group(name)
            save_pointlist_group(group_new_pointlist, dataobject)
        elif isinstance(dataobject, PointListArray):
            if name == '':
                name = 'pointlistarray_'+str(N_pointlistarrays)
                N_pointlistarrays += 1
            try:
                group_new_pointlistarray = group_pointlistarrays.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_pointlistarrays.keys())])
                name = name+"_"+str(N)
                group_new_pointlistarray = group_pointlistarrays.create_group(name)
            save_pointlistarray_group(group_new_pointlistarray, dataobject)
        elif isinstance(dataobject, Metadata):
            pass
        else:
            print("Error: object {} has type {}, and is not a DataCube, DiffractionSlice, RealSlice, PointList, or PointListArray instance.".format(dataobject,type(dataobject)))

    ##### Finish and close #####
    print("Done.")
    f.close()

@log
def append_dataobject(dataobject, filepath, **kwargs):
    """
    Appends dataobject to existing .h5 file at filepath.
    """
    assert isinstance(dataobject, DataObject)

    # append
    append_from_dataobject_list([dataobject], filepath, **kwargs)

@log
def append_dataobjects_by_indices(index_list, filepath, **kwargs):
    """
    Appends the DataObjects found at the indices in index_list in DataObject.get_dataobjects to
    the existing py4DSTEM .h5 file at filepath.
    """
    full_dataobject_list = DataObject.get_dataobjects()
    dataobject_list = [full_dataobject_list[i] for i in index_list]

    append_from_dataobject_list(dataobject_list, filepath, **kwargs)

@log
def append(data, filepath, **kwargs):
    """
    Appends data to an existing py4DSTEM .h5 file at filepath. What is saved depends on the
    arguement data.

    If data is a DataObject, appends just this dataobject.
    If data is a list of DataObjects, appends all these objects.
    If data is an int, appends the dataobject corresponding to this index in
    DataObject.get_dataobjects().
    If data is a list of indices, appends the objects corresponding to these
    indices in DataObject.get_dataobjects().
    """
    if isinstance(data, DataObject):
        append_dataobject(data, filepath, **kwargs)
    elif isinstance(data, int):
        append_dataobjects_by_indices([data], filepath, **kwargs)
    elif isinstance(data, list):
        if all([isinstance(item,DataObject) for item in data]):
            append_from_dataobject_list(data, filepath, **kwargs)
        elif all([isinstance(item,int) for item in data]):
            append_dataobjects_by_indices(data, filepath, **kwargs)
        else:
            print("Error: if data is a list, it must contain all ints or all DataObjects.")
    else:
        print("Error: unrecognized value for argument data. Must be either a DataObject, a list of DataObjects, a list of ints, or the string 'all'.")


################### END OF APPEND FUNCTIONS #####################
def get_py4DSTEM_topgroup(h5_file):
    """
    Accepts an open h5py File boject. Returns string of the top group name. 
    """
    if ('4DSTEM_experiment' in h5_file.keys()): # or ('4D-STEM_data' in h5_file.keys()) or ('4DSTEM_simulation' in h5_file.keys())):
        return '4DSTEM_experiment/'
    elif ('4DSTEM_simulation' in h5_file.keys()):
        return '4DSTEM_simulation/'
    else:
        return '4D-STEM_data/'


