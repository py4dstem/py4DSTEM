# Write py4DSTEM formatted .h5 files.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from collections import OrderedDict
from os.path import exists
from hyperspy.misc.utils import DictionaryTreeBrowser
from ..datastructure import DataCube, DiffractionSlice, RealSlice, CountedDataCube
from ..datastructure import MetadataCollection, Metadata, DataObject, PointList
from ..datastructure import PointListArray
from ...process.utils import tqdmnd

from ..log import log, Logger
logger = Logger()

@log
def save_from_dataobject_list(dataobject_list, outputfile, topgroup=None, overwrite=False, **kwargs):
    """
    Saves an h5 file from a list of DataObjects and an output filepath.

    Accepts:
        dataobject_list     a list of DataObjects to save
        outputfile          path to an .h5 file to save
        topgroup            (str) name for the toplevel group; if None, use "4DSTEM_experiment"
    """

    assert all([isinstance(item,DataObject) for item in dataobject_list]), "Error: all elements of dataobject_list must be DataObject instances."
    if exists(outputfile):
        if overwrite is False:
            raise Exception('{} already exists.  To overwrite, use overwrite=True. To append new objects to an existing file, use append() rather than save().'.format(outputfile))

    # Handle keyword arguments
    use_compression = kwargs.get('compression',False)

    ##### Make .h5 file #####
    print("Creating file {}...".format(outputfile))
    try:
        f = h5py.File(outputfile,"w")
    except OSError as e:
        print(e)
        print('The file appears to be open elsewhere...')
        print('This can occur if your datacube was read from a py4DSTEM h5 file.')
        print(f'To forse close the file, losing any dataobjects open from it, run: py4DSTEM.file.io.close_h5_at_path(\'{outputfile}\')')
        print('To force close all h5 files run: py4DSTEM.file.io.close_all_h5()')
        return -1

    if topgroup is None:
        group_toplevel = f.create_group("4DSTEM_experiment")
    else:
        assert isinstance(topgroup, str)
        group_toplevel = f.create_group(topgroup)
    group_toplevel.attrs.create("emd_group_type",2)
    group_toplevel.attrs.create("version_major",0)
    group_toplevel.attrs.create("version_minor",7)

    ##### Metadata #####

    # Find and label all metadata objects
    metadata_list,i = [],0
    for dataobject in dataobject_list:
        if dataobject.metadata is not None:
            assert isinstance(dataobject.metadata,Metadata), "DataObject.metadata must be a Metadata object or None for all DataObjects being saved."
            if dataobject.metadata not in metadata_list:
                metadata_list.append(dataobject.metadata)
                dataobject.metadata._ind = i
                i += 1

    # Save metadata
    group_metadata = group_toplevel.create_group("metadata")
    for metadata in metadata_list:
        group_metadata_current = group_metadata.create_group("metadata_{}".format(metadata._ind))
        save_metadata(metadata,group_metadata_current)

    ##### Data #####

    # Write data groups
    group_data = group_toplevel.create_group("data")
    group_datacubes = group_data.create_group("datacubes")
    group_counted = group_data.create_group("counted_datacubes")
    group_diffractionslices = group_data.create_group("diffractionslices")
    group_realslices = group_data.create_group("realslices")
    group_pointlists = group_data.create_group("pointlists")
    group_pointlistarrays = group_data.create_group("pointlistarrays")
    ind_dcs, ind_cdcs, ind_dfs, ind_rls, ind_ptl, ind_ptla = 0,0,0,0,0,0

    # Loop through and save all objects in the dataobjectlist
    for dataobject in dataobject_list:
        name = dataobject.name
        if isinstance(dataobject, DataCube):
            if name == '':
                name = 'datacube_'+str(ind_dcs)
                ind_dcs += 1
            try:
                group_new_datacube = group_datacubes.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_datacubes.keys())])
                name = name+"_"+str(N)
                group_new_datacube = group_datacubes.create_group(name)
            save_datacube_group(group_new_datacube, dataobject, use_compression)
        elif isinstance(dataobject,CountedDataCube):
            if name == '':
                name = 'counted_datacube_'+str(ind_dcs)
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
                name = 'diffractionslice_'+str(ind_dfs)
                ind_dfs += 1
            try:
                group_new_diffractionslice = group_diffractionslices.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_diffractionslices.keys())])
                name = name+"_"+str(N)
                group_new_diffractionslice = group_diffractionslices.create_group(name)
            save_diffraction_group(group_new_diffractionslice, dataobject)
        elif isinstance(dataobject, RealSlice):
            if name == '':
                name = 'realslice_'+str(ind_rls)
                ind_rls += 1
            try:
                group_new_realslice = group_realslices.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_realslices.keys())])
                name = name+"_"+str(N)
                group_new_realslice = group_realslices.create_group(name)
            save_real_group(group_new_realslice, dataobject)
        elif isinstance(dataobject, PointList):
            if name == '':
                name = 'pointlist_'+str(ind_ptl)
                ind_ptl += 1
            try:
                group_new_pointlist = group_pointlists.create_group(name)
            except ValueError:
                N = sum([name in string for string in list(group_pointlists.keys())])
                name = name+"_"+str(N)
                group_new_pointlist = group_pointlists.create_group(name)
            save_pointlist_group(group_new_pointlist, dataobject)
        elif isinstance(dataobject, PointListArray):
            if name == '':
                name = 'pointlistarray_'+str(ind_ptla)
                ind_ptla += 1
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

    ##### Log #####
    group_log = group_toplevel.create_group("log")
    for index in range(logger.log_index):
        write_log_item(group_log, index, logger.logged_items[index])

    ##### Finish and close #####
    print("Done.",flush=True)
    f.close()

#@log
def save_dataobject(dataobject, outputfile, **kwargs):
    """
    Saves a .h5 file containing only a single DataObject instance to outputfile.
    """
    assert isinstance(dataobject, DataObject)

    # Save
    save_from_dataobject_list([dataobject], outputfile, **kwargs)

#@log
def save_dataobjects_by_indices(index_list, outputfile, **kwargs):
    """
    Saves a .h5 file containing DataObjects corresponding to the indices in index_list, a list of
    ints, in the list generated by DataObject.get_dataobjects().
    """
    full_dataobject_list = DataObject.get_dataobjects()
    dataobject_list = [full_dataobject_list[i] for i in index_list]

    save_from_dataobject_list(dataobject_list, outputfile, **kwargs)

#@log
def save(data, outputfile, **kwargs):
    """
    Saves a .h5 file to outputpath. What is saved depends on the arguement data.

    If data is a DataObject, saves a .h5 file containing just this object.
    If data is a list of DataObjects, saves a .h5 file containing all these objects.
    If data is an int, saves a .h5 file containing the dataobject corresponding to this index in
    DataObject.get_dataobjects().
    If data is a list of indices, saves a .h5 file containing the objects corresponding to these
    indices in DataObject.get_dataobjects().
    If data is 'all', saves all DataObjects in memory to a .h5 file.
    """
    if isinstance(data, DataObject):
        save_dataobject(data, outputfile, **kwargs)
    elif isinstance(data, int):
        save_dataobjects_by_indices([data], outputfile, **kwargs)
    elif isinstance(data, list):
        if all([isinstance(item,DataObject) for item in data]):
            save_from_dataobject_list(data, outputfile, **kwargs)
        elif all([isinstance(item,int) for item in data]):
            save_dataobjects_by_indices(data, outputfile, **kwargs)
        else:
            print("Error: if data is a list, it must contain all ints or all DataObjects.")
    elif data=='all':
        save_from_dataobject_list(DataObject.get_dataobjects(), outputfile, **kwargs)
    else:
        print("Error: unrecognized value for argument data. Must be either a DataObject, a list of DataObjects, a list of ints, or the string 'all'.")


################### END OF PRIMARY SAVE FUNCTIONS #####################



#### Functions for writing dataobjects to .h5 ####

def save_datacube_group(group, datacube, use_compression=False):
    group.attrs.create("emd_group_type",1)
    if datacube.metadata is not None:
        group.attrs.create("metadata",datacube.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    # TODO: consider defining data chunking here, keeping k-space slices together
    if (isinstance(datacube.data,np.ndarray) or isinstance(datacube.data,h5py.Dataset)):
        if use_compression:
            data_datacube = group.create_dataset("data", data=datacube.data,
                chunks=(1,1,datacube.Q_Nx,datacube.Q_Ny),compression='gzip')
        else:
            data_datacube = group.create_dataset("data", data=datacube.data)
    else:
        # handle K2DataArray datacubes
        data_datacube = datacube.data._write_to_hdf5(group)

    # Dimensions
    assert len(data_datacube.shape)==4, "Shape of datacube is {}".format(len(data_datacube))
    R_Nx,R_Ny,Q_Nx,Q_Ny = data_datacube.shape
    data_R_Nx = group.create_dataset("dim1",(R_Nx,))
    data_R_Ny = group.create_dataset("dim2",(R_Ny,))
    data_Q_Nx = group.create_dataset("dim3",(Q_Nx,))
    data_Q_Ny = group.create_dataset("dim4",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))
    data_Q_Nx[...] = np.arange(0,Q_Nx)
    data_Q_Nx.attrs.create("name",np.string_("Q_x"))
    data_Q_Nx.attrs.create("units",np.string_("[pix]"))
    data_Q_Ny[...] = np.arange(0,Q_Ny)
    data_Q_Ny.attrs.create("name",np.string_("Q_y"))
    data_Q_Ny.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present


def save_counted_datacube_group(group,datacube):
    if datacube.data._mmap:
        # memory mapped CDC's aren't supported yet
        print('Data not written. Memory mapped CountedDataCube not yet supported.')
        return

    group.attrs.create("emd_group_type",1)
    if datacube.metadata is not None:
        group.attrs.create("metadata",datacube.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    pointlistarray = datacube.electrons
    try:
        n_coords = len(pointlistarray.dtype.names)
    except:
        n_coords = 1
    #coords = np.string_(str([coord for coord in pointlistarray.dtype.names]))
    group.attrs.create("coordinates", np.string_(str(pointlistarray.dtype)))
    group.attrs.create("dimensions", n_coords)

    pointlist_dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    name = "data"
    dset = group.create_dataset(name,pointlistarray.shape,pointlist_dtype)

    print('Writing CountedDataCube:',flush=True)

    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data

    # indexing coordinates:
    dt = h5py.special_dtype(vlen=str)
    data_coords = group.create_dataset('index_coords',shape=(datacube.data._mode,),dtype=dt)
    if datacube.data._mode == 1:
        data_coords[0] = datacube.data.index_key
    else:
        data_coords[0] = datacube.data.index_key.ravel()[0]
        data_coords[1] = datacube.data.index_key.ravel()[1]

    # Dimensions
    R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.data.shape
    data_R_Nx = group.create_dataset("dim1",(R_Nx,))
    data_R_Ny = group.create_dataset("dim2",(R_Ny,))
    data_Q_Nx = group.create_dataset("dim3",(Q_Nx,))
    data_Q_Ny = group.create_dataset("dim4",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))
    data_Q_Nx[...] = np.arange(0,Q_Nx)
    data_Q_Nx.attrs.create("name",np.string_("Q_x"))
    data_Q_Nx.attrs.create("units",np.string_("[pix]"))
    data_Q_Ny[...] = np.arange(0,Q_Ny)
    data_Q_Ny.attrs.create("name",np.string_("Q_y"))
    data_Q_Ny.attrs.create("units",np.string_("[pix]"))

def save_diffraction_group(group, diffractionslice):
    if diffractionslice.metadata is not None:
        group.attrs.create("metadata",diffractionslice.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    group.attrs.create("depth", diffractionslice.depth)
    data_diffractionslice = group.create_dataset("data", data=diffractionslice.data)

    shape = diffractionslice.data.shape
    assert len(shape)==2 or len(shape)==3

    # Dimensions 1 and 2
    Q_Nx,Q_Ny = shape[:2]
    dim1 = group.create_dataset("dim1",(Q_Nx,))
    dim2 = group.create_dataset("dim2",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    dim1[...] = np.arange(0,Q_Nx)
    dim1.attrs.create("name",np.string_("Q_x"))
    dim1.attrs.create("units",np.string_("[pix]"))
    dim2[...] = np.arange(0,Q_Ny)
    dim2.attrs.create("name",np.string_("Q_y"))
    dim2.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present

    # Dimension 3
    if len(shape)==3:
        dim3 = group.create_dataset("dim3", data=np.array(diffractionslice.slicelabels).astype("S64"))

def save_real_group(group, realslice):
    if realslice.metadata is not None:
        group.attrs.create("metadata",realslice.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    group.attrs.create("depth", realslice.depth)
    data_realslice = group.create_dataset("data", data=realslice.data)

    shape = realslice.data.shape
    assert len(shape)==2 or len(shape)==3

    # Dimensions 1 and 2
    R_Nx,R_Ny = shape[:2]
    dim1 = group.create_dataset("dim1",(R_Nx,))
    dim2 = group.create_dataset("dim2",(R_Ny,))

    # Populate uncalibrated dimensional axes
    dim1[...] = np.arange(0,R_Nx)
    dim1.attrs.create("name",np.string_("R_x"))
    dim1.attrs.create("units",np.string_("[pix]"))
    dim2[...] = np.arange(0,R_Ny)
    dim2.attrs.create("name",np.string_("R_y"))
    dim2.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present

    # Dimension 3
    if len(shape)==3:
        dim3 = group.create_dataset("dim3", data=np.array(realslice.slicelabels).astype("S64"))

def save_pointlist_group(group, pointlist):
    if pointlist.metadata is not None:
        group.attrs.create("metadata",pointlist.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    n_coords = len(pointlist.dtype.names)
    coords = np.string_(str([coord for coord in pointlist.dtype.names]))
    group.attrs.create("coordinates", coords)
    group.attrs.create("dimensions", n_coords)
    group.attrs.create("length", pointlist.length)

    for name in pointlist.dtype.names:
        group_current_coord = group.create_group(name)
        group_current_coord.attrs.create("dtype", np.string_(pointlist.dtype[name]))
        group_current_coord.create_dataset("data", data=pointlist.data[name])

def save_pointlistarray_group(group, pointlistarray):
    if pointlistarray.metadata is not None:
        group.attrs.create("metadata",pointlistarray.metadata._ind)
    else:
        group.attrs.create("metadata",-1)

    try:
        n_coords = len(pointlistarray.dtype.names)
    except:
        n_coords = 1
    #coords = np.string_(str([coord for coord in pointlistarray.dtype.names]))
    group.attrs.create("coordinates", np.string_(str(pointlistarray.dtype)))
    group.attrs.create("dimensions", n_coords)

    pointlist_dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    name = "data"
    dset = group.create_dataset(name,pointlistarray.shape,pointlist_dtype)

    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data



#### Metadata functions ####

def save_metadata(metadata,group):
    """
    Save metadata (Metadata object) into group (HDF5 group).
    """
    # Create subgroups
    group_microscope_metadata = group.create_group("microscope")
    group_sample_metadata = group.create_group("sample")
    group_user_metadata = group.create_group("user")
    group_calibration_metadata = group.create_group("calibration")
    group_comments_metadata = group.create_group("comments")

    group_original_metadata = group.create_group("original")
    group_original_metadata_all = group_original_metadata.create_group("all")
    group_original_metadata_shortlist = group_original_metadata.create_group("shortlist")

    # Transfer metadata dictionaries
    transfer_metadata_dict(metadata.microscope,group_microscope_metadata)
    transfer_metadata_dict(metadata.sample,group_sample_metadata)
    transfer_metadata_dict(metadata.user,group_user_metadata)
    transfer_metadata_dict(metadata.calibration,group_calibration_metadata)
    transfer_metadata_dict(metadata.comments,group_comments_metadata)

    # Transfer original metadata trees
    if type(metadata.original_metadata.shortlist)==DictionaryTreeBrowser:
        transfer_metadata_tree_hs(metadata.original_metadata.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_hs(metadata.original_metadata.all,group_original_metadata_all)
    else:
        transfer_metadata_tree_py4DSTEM(metadata.original_metadata.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_py4DSTEM(metadata.original_metadata.all,group_original_metadata_all)

def transfer_metadata_dict(dictionary,group):
    """
    Transfers metadata from datacube metadata dictionaries (standard python dictionary objects)
    to attrs in a .h5 group.

    Accepts two arguments:
        dictionary - a dictionary of metadata
        group - an hdf5 file group, which will become the root node of a copy of tree
    """
    for key,val in dictionary.items():
        if type(val)==str:
            group.attrs.create(key,np.string_(val))
        else:
            group.attrs.create(key,val)

def transfer_metadata_tree_hs(tree,group):
    """
    Transfers metadata from hyperspy.misc.utils.DictionaryTreeBrowser objects to a tree of .h5
    groups (non-terminal nodes) and attrs (terminal nodes).

    Accepts two arguments:
        tree - a hyperspy.misc.utils.DictionaryTreeBrowser object, containing metadata
        group - an hdf5 file group, which will become the root node of a copy of tree
    """
    for key in tree.keys():
        if istree_hs(tree[key]):
            subgroup = group.create_group(key)
            transfer_metadata_tree_hs(tree[key],subgroup)
        else:
            if type(tree[key])==str:
                group.attrs.create(key,np.string_(tree[key]))
            else:
                group.attrs.create(key,tree[key])

def istree_hs(node):
    """
    Determines if a node in a hyperspy metadata structure is a parent or terminal leaf.
    """
    if type(node)==DictionaryTreeBrowser:
        return True
    else:
        return False

def transfer_metadata_tree_py4DSTEM(tree,group):
    """
    Transfers metadata from MetadataCollection objects to a tree of .h5
    groups (non-terminal nodes) and attrs (terminal nodes).

    Accepts two arguments:
        tree - a MetadataCollection object, containing metadata
        group - an hdf5 file group, which will become the root node of a copy of tree
    """
    for key in tree.__dict__.keys():
        if istree_py4DSTEM(tree.__dict__[key]):
            subgroup = group.create_group(key)
            transfer_metadata_tree_py4DSTEM(tree.__dict__[key],subgroup)
        elif is_metadata_dict(key):
            metadata_dict = tree.__dict__[key]
            for md_key in metadata_dict.keys():
                if type(metadata_dict[md_key])==str:
                    group.attrs.create(md_key,np.string_(metadata_dict[md_key]))
                else:
                    group.attrs.create(md_key,metadata_dict[md_key])

def istree_py4DSTEM(node):
    """
    Determines if a node in a py4DSTEM metadata structure is a parent or terminal leaf.
    """
    if type(node)==MetadataCollection:
        return True
    else:
        return False

def is_metadata_dict(key):
    """
    Determines if a node in a py4DSTEM metadata structure is a metadata dictionary.
    """
    if key=='metadata_items':
        return True
    else:
        return False

def close_all_h5():
    n = 0
    import gc
    for obj in gc.get_objects():
        try:
            t = type(obj)
            if t is h5py.Dataset:
                try:
                    obj.file.close()
                    n += 1
                except:
                    pass
        except:
            pass
    print(f'Closed {n} files.')

def close_h5_at_path(fpath):
    import gc, os
    n=0
    for obj in gc.get_objects():
        try:
            t = type(obj)
            if t is h5py.File:
                try:
                    pth = obj.filename
                    if os.path.normpath(pth) == os.path.normpath(fpath):
                        obj.close()
                        n += 1
                        print(pth)
                except:
                    pass
        except:
            pass

#### Logging functions ####

def write_log_item(group_log, index, logged_item):
    group_logitem = group_log.create_group('log_item_'+str(index))
    group_logitem.attrs.create('function', np.string_(logged_item.function))
    group_inputs = group_logitem.create_group('inputs')
    for key,value in logged_item.inputs.items():
        if type(value)==str:
            group_inputs.attrs.create(key, np.string_(value))
        elif isinstance(value,DataObject):
            if value.name == '':
                if isinstance(value,DataCube):
                    name = np.string_("DataCube_id"+str(id(value)))
                elif isinstance(value,DiffractionSlice):
                    name = np.string_("DiffractionSlice_id"+str(id(value)))
                elif isinstance(value,RealSlice):
                    name = np.string_("RealSlice_id"+str(id(value)))
                elif isinstance(value,PointList):
                    name = np.string_("PointList_id"+str(id(value)))
                elif isinstance(value,PointListArray):
                    name = np.string_("PointListArray_id"+str(id(value)))
                else:
                    name = np.string_("DataObject_id"+str(id(value)))
            else:
                name = np.string_(value.name)
            group_inputs.attrs.create(key, name)
        else:
            try:
                group_inputs.attrs.create(key, value)
            except TypeError:
                group_inputs.attrs.create(key, np.string_(str(value)))
    group_logitem.attrs.create('version', logged_item.version)
    write_time_to_log_item(group_logitem, logged_item.datetime)

def write_time_to_log_item(group_logitem, datetime):
    date = str(datetime.tm_year)+str(datetime.tm_mon)+str(datetime.tm_mday)
    time = str(datetime.tm_hour)+':'+str(datetime.tm_min)+':'+str(datetime.tm_sec)
    group_logitem.attrs.create('time', np.string_(date+'__'+time))





