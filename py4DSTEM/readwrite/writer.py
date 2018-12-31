# Reading / writing 4D-STEM data into an HDF5 file format
# 
# Files are readable as .emd files, with the datacube group conforming to the EMD data group 
# specifications.
#
# See end of file for complete file structure.

import h5py
import numpy as np
from collections import OrderedDict
from hyperspy.misc.utils import DictionaryTreeBrowser
from ..process.datastructure import RawDataCube, DataCube, DiffractionSlice, RealSlice
from ..process.datastructure import PointList, PointListArray
from ..process.datastructure import MetadataCollection, DataObjectTracker, DataObject
from ..process.log import log, Logger

logger = Logger()

@log
def save_from_dataobjecttracker(dataobjecttracker, outputfile):
    """
    Saves an h5 file from a DataObjectTracker object and an output filepath.
    """

    assert isinstance(dataobjecttracker, DataObjectTracker)

    ##### Make .h5 file #####
    print("Creating file {}...".format(outputfile))
    f = h5py.File(outputfile,"w")
    f.attrs.create("version_major",0)
    f.attrs.create("version_minor",2)
    group_data = f.create_group("4DSTEM_experiment")

    ##### Metadata #####
    print("Writing metadata...")

    # Create metadata groups
    group_metadata = group_data.create_group("metadata")
    group_original_metadata = group_metadata.create_group("original")
    group_microscope_metadata = group_metadata.create_group("microscope")
    group_sample_metadata = group_metadata.create_group("sample")
    group_user_metadata = group_metadata.create_group("user")
    group_calibration_metadata = group_metadata.create_group("calibration")
    group_comments_metadata = group_metadata.create_group("comments")
    group_original_metadata_all = group_original_metadata.create_group("all")
    group_original_metadata_shortlist = group_original_metadata.create_group("shortlist")

    # Transfer original metadata trees
    if type(dataobjecttracker.rawdatacube.metadata.original.shortlist)==DictionaryTreeBrowser:
        transfer_metadata_tree_hs(dataobjecttracker.rawdatacube.metadata.original.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_hs(dataobjecttracker.rawdatacube.metadata.original.all,group_original_metadata_all)
    else:
        transfer_metadata_tree_py4DSTEM(dataobjecttracker.rawdatacube.metadata.original.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_py4DSTEM(dataobjecttracker.rawdatacube.metadata.original.all,group_original_metadata_all)

    # Transfer dataobjecttracker.rawdatacube.metadata dictionaries
    transfer_metadata_dict(dataobjecttracker.rawdatacube.metadata.microscope,group_microscope_metadata)
    transfer_metadata_dict(dataobjecttracker.rawdatacube.metadata.sample,group_sample_metadata)
    transfer_metadata_dict(dataobjecttracker.rawdatacube.metadata.user,group_user_metadata)
    transfer_metadata_dict(dataobjecttracker.rawdatacube.metadata.calibration,group_calibration_metadata)
    transfer_metadata_dict(dataobjecttracker.rawdatacube.metadata.comments,group_comments_metadata)

    ##### Log #####
    group_log = group_data.create_group("log")
    for index in range(logger.log_index):
        write_log_item(group_log, index, logger.logged_items[index])

    ##### Data #####

    # Write data groups
    group_rawdatacube = group_data.create_group("rawdatacube")
    group_processing = group_data.create_group("processing")
    group_processed_datacubes = group_processing.create_group("datacubes")
    group_diffraction_slices = group_processing.create_group("diffractionslices")
    group_real_slices = group_processing.create_group("realslices")
    group_point_lists = group_processing.create_group("pointlists")
    group_point_list_arrays = group_processing.create_group("pointlistarrays")
    ind_rdc, ind_dcs, ind_dfs, ind_rls, ind_ptl, ind_ptla = 0,0,0,0,0,0

    # Loop through all objects in the DataObjectTracker and, if save_behavior==True, save
    for item in dataobjecttracker.dataobject_list:
        save_behavior = item[3]
        if save_behavior:
            name = item[1]
            dataobject = item[4]
            if isinstance(dataobject, RawDataCube):
                if ind_rdc == 0:
                    save_datacube_group(group_rawdatacube, dataobject)
                    ind_rdc += 1
                else:
                    print("Warning: more than one RawDataCube found. Saving additional RawDataCubes to processing>datacubes.")
                    if name == '':
                        name = 'datacube_'+str(ind_dcs)
                        ind_dcs += 1
                    group_new_datacube = group_processed_datacubes.create_group(name)
                    save_datacube_group(group_new_datacube, dataobject)
            elif isinstance(dataobject, DataCube):
                if name == '':
                    name = 'datacube_'+str(ind_dcs)
                    ind_dcs += 1
                group_new_datacube = group_processed_datacubes.create_group(name)
                save_datacube_group(group_new_datacube, dataobject)
            elif isinstance(dataobject, DiffractionSlice):
                if name == '':
                    name = 'diffractionslice_'+str(ind_dfs)
                    ind_dfs += 1
                group_new_diffraction_slice = group_diffraction_slices.create_group(name)
                save_diffraction_group(group_new_diffraction_slice, dataobject)
            elif isinstance(dataobject, RealSlice):
                if name == '':
                    name = 'realslice_'+str(ind_rls)
                    ind_rls += 1
                group_new_real_slice = group_real_slices.create_group(name)
                save_real_group(group_new_real_slice, dataobject)
            elif isinstance(dataobject, PointList):
                if name == '':
                    name = 'pointlist_'+str(ind_ptl)
                    ind_ptl += 1
                group_new_pointlist = group_point_lists.create_group(name)
                save_pointlist_group(group_new_pointlist, dataobject)
            elif isinstance(dataobject, PointListArray):
                if name == '':
                    name = 'pointlistarray_'+str(ind_ptla)
                    ind_ptla += 1
                group_new_point_list_array = group_point_list_arrays.create_group(name)
                save_pointlistarray_group(group_new_point_list_array, dataobject)
            else:
                print("Error: object {} has type {}, and is not a RawDataCube, DataCube, DiffractionSlice, RealSlice, PointList, or PointListArray instance.".format(dataobject,type(dataobject)))

    ##### Finish and close #####
    print("Done.")
    f.close()


@log
def save_dataobject(dataobject,outputfile):
    """
    Saves a .h5 file containing only a single DataObject instance to outputfile.
    """
    assert isinstance(dataobject, DataObject)

    # Get current state of tracker
    tracker = dataobject.get_dataobjecttrackers()[0]
    tracker_save_behavior_list = tracker.get_save_behavior_list()

    # Edit tracker so only dataobject has save_behavior==True
    tracker.change_all_save_behaviors(False)
    tracker.change_save_behavior(dataobject, True)

    # Save
    save_from_dataobjecttracker(tracker, outputfile)

    # Revert tracker state
    for i in range(len(tracker_save_behavior_list)):
        tracker.change_save_behavior_by_index(i, tracker_save_behavior_list[i])

@log
def save_datacube(datacube,outputfile):
    """
    Saves a .h5 file containing only a single DataCube instance to outputfile.
    """
    assert isinstance(datacube, DataCube)

    save_dataobject(datacube, outputfile)

@log
def save(dataobject,outputfile):
    """
    Saves a .h5 file to outputpath.
    If dataobject is a RawDataCube, save() identifies its DataObjectTracker and saves all objects
    currently flagged for saving.
    If dataobject is any other DataObject, save() identifies all associated trackers. If there is
    only one tracker, all objects associated with this tracker are saved. If there are multiple
    trackers, save() prints a warning message asking the user to specify a tracker.
    """
    assert isinstance(dataobject, DataObject)

    if isinstance(dataobject, RawDataCube):
        save_from_dataobjecttracker(dataobject.dataobjecttracker, outputfile)
    else:
        trackers = dataobject.get_dataobjecttrackers()
        if len(trackers)==0:
            print("Error: DataObject {} has no associated DataObjectTracker instances.".format(dataobject))
        elif len(trackers)==1:
            save_from_dataobjecttracker(trackers[0], outputfile)
        else:
            print("Error: DataObject {} has {} associated DataObjectTracker instances.".format(dataobject, len(trackers)))

################### END OF PRIMARY SAVE FUNCTIONS #####################


#### Functions for writing dataobjects to .h5 ####

def save_datacube_group(group, datacube):
    group.attrs.create("emd_group_type",1)

    # TODO: consider defining data chunking here, keeping k-space slices together
    data_datacube = group.create_dataset("datacube", data=datacube.data4D)

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

    # Calibrate axes, if calibrations are present

    # Calibrate R axes
    #try:
    #    R_pix_size = datacube.metadata.calibration["R_pix_size"]
    #    data_R_Nx[...] = np.arange(0,R_Nx*R_pix_size,R_pix_size)
    #    data_R_Ny[...] = np.arange(0,R_Ny*R_pix_size,R_pix_size)
    #    # Set R axis units
    #    try:
    #        R_units = datacube.metadata.calibration["R_units"]
    #        data_R_Nx.attrs["units"] = R_units
    #        data_R_Ny.attrs["units"] = R_units
    #    except KeyError:
    #        print("WARNING: Real space calibration found and applied, however, units were",
    #               "not identified and have been left in pixels.")
    #except KeyError:
    #    print("No real space calibration found.")
    #except TypeError:
    #    # If R_pix_size is a str, i.e. has not been entered, pass
    #    pass

    # Calibrate Q axes
    #try:
    #    Q_pix_size = datacube.metadata.calibration["Q_pix_size"]
    #    data_Q_Nx[...] = np.arange(0,Q_Nx*Q_pix_size,Q_pix_size)
    #    data_Q_Ny[...] = np.arange(0,Q_Ny*Q_pix_size,Q_pix_size)
    #    # Set Q axis units
    #    try:
    #        Q_units = datacube.metadata.calibration["Q_units"]
    #        data_Q_Nx.attrs["units"] = Q_units
    #        data_Q_Ny.attrs["units"] = Q_units
    #    except KeyError:
    #        print("WARNING: Diffraction space calibration found and applied, however, units",
    #               "were not identified and have been left in pixels.")
    #except KeyError:
    #    print("No diffraction space calibration found.")
    #except TypeError:
    #    # If Q_pix_size is a str, i.e. has not been entered, pass
    #    pass

def save_diffraction_group(group, diffractionslice):

    group.attrs.create("depth", diffractionslice.depth)
    if diffractionslice.depth==1:
        shape = diffractionslice.data2D.shape
        data_diffractionslice = group.create_dataset("diffractionslice", data=diffractionslice.data2D)
    else:
        if type(diffractionslice.data2D)==OrderedDict:
            shape = diffractionslice.data2D[list(diffractionslice.data2D.keys())[0]].shape
            for key in diffractionslice.data2D.keys():
                data_diffractionslice = group.create_dataset(str(key), data=diffractionslice.data2D[key])
        else:
            shape = diffractionslice.data2D[0].shape
            for i in range(diffractionslice.depth):
                data_diffractionslice = group.create_dataset("slice_"+str(i), data=diffractionslice.data2D[i])

    # Dimensions
    assert len(shape)==2, "Shape of diffractionslice is {}".format(len(shape))
    Q_Nx,Q_Ny = shape
    data_Q_Nx = group.create_dataset("dim1",(Q_Nx,))
    data_Q_Ny = group.create_dataset("dim2",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    data_Q_Nx[...] = np.arange(0,Q_Nx)
    data_Q_Nx.attrs.create("name",np.string_("Q_x"))
    data_Q_Nx.attrs.create("units",np.string_("[pix]"))
    data_Q_Ny[...] = np.arange(0,Q_Ny)
    data_Q_Ny.attrs.create("name",np.string_("Q_y"))
    data_Q_Ny.attrs.create("units",np.string_("[pix]"))

    # ToDo: axis calibration
    # Requires pulling metadata from associated RawDataCube

    # Calibrate axes, if calibrations are present
    #try:
    #    Q_pix_size = datacube.metadata.calibration["Q_pix_size"]
    #    data_Q_Nx[...] = np.arange(0,Q_Nx*Q_pix_size,Q_pix_size)
    #    data_Q_Ny[...] = np.arange(0,Q_Ny*Q_pix_size,Q_pix_size)
    #    # Set Q axis units
    #    try:
    #        Q_units = datacube.metadata.calibration["Q_units"]
    #        data_Q_Nx.attrs["units"] = Q_units
    #        data_Q_Ny.attrs["units"] = Q_units
    #    except KeyError:
    #        print("WARNING: Diffraction space calibration found and applied, however, units",
    #               "were not identified and have been left in pixels.")
    #except KeyError:
    #    print("No diffraction space calibration found.")
    #except TypeError:
    #    # If Q_pix_size is a str, i.e. has not been entered, pass
    #    pass

def save_real_group(group, realslice):

    group.attrs.create("depth", realslice.depth)
    if realslice.depth==1:
        shape = realslice.data2D.shape
        data_realslice = group.create_dataset("realslice", data=realslice.data2D)
    else:
        if type(realslice.data2D)==OrderedDict:
            shape = realslice.data2D[list(realslice.data2D.keys())[0]].shape
            for key in realslice.data2D.keys():
                data_realslice = group.create_dataset(str(key), data=realslice.data2D[key])
        else:
            shape = realslice.data2D[0].shape
            for i in range(realslice.depth):
                data_realslice = group.create_dataset("slice_"+str(i), data=realslice.data2D[i])

    # Dimensions
    assert len(shape)==2, "Shape of realslice is {}".format(len(shape))
    R_Nx,R_Ny = shape
    data_R_Nx = group.create_dataset("dim1",(R_Nx,))
    data_R_Ny = group.create_dataset("dim2",(R_Ny,))

    # Populate uncalibrated dimensional axes
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))

    # ToDo: axis calibration
    # Requires pulling metadata from associated RawDataCube

    # Calibrate axes, if calibrations are present
    #try:
    #    R_pix_size = datacube.metadata.calibration["R_pix_size"]
    #    data_R_Nx[...] = np.arange(0,R_Nx*R_pix_size,R_pix_size)
    #    data_R_Ny[...] = np.arange(0,R_Ny*R_pix_size,R_pix_size)
    #    # Set R axis units
    #    try:
    #        R_units = datacube.metadata.calibration["R_units"]
    #        data_R_Nx.attrs["units"] = R_units
    #        data_R_Ny.attrs["units"] = R_units
    #    except KeyError:
    #        print("WARNING: Real space calibration found and applied, however, units",
    #               "were not identified and have been left in pixels.")
    #except KeyError:
    #    print("No real space calibration found.")
    #except TypeError:
    #    # If R_pix_size is a str, i.e. has not been entered, pass
    #    pass

def save_pointlist_group(group, pointlist):

    for name in pointlist.dtype.names:
        group_current_coord = group.create_group(name)
        group_current_coord.attrs.create("dtype", np.string_(pointlist.dtype[name]))
        group_current_coord.create_dataset("data", data=pointlist.data[name])

def save_pointlistarray_group(group, pointlistarray):

    for i in range(pointlistarray.shape[0]):
        for j in range(pointlistarray.shape[1]):
            group_current_arrayposition = group.create_group("{}_{}".format(i,j))
            save_pointlist_group(group_current_arrayposition, pointlistarray.get_pointlist(i,j))



#### Functions for original metadata transfer ####

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


#### Functions for writing logs ####

def write_log_item(group_log, index, logged_item):
    group_logitem = group_log.create_group('log_item_'+str(index))
    group_logitem.attrs.create('function', np.string_(logged_item.function))
    group_inputs = group_logitem.create_group('inputs')
    for key,value in logged_item.inputs.items():
        if type(value)==str:
            group_inputs.attrs.create(key, np.string_(value))
        elif isinstance(value,DataObject):
            if value.name == '':
                if isinstance(value,RawDataCube):
                    name = np.string_("RawDataCube_id"+str(id(value)))
                elif isinstance(value,DataCube):
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
        elif isinstance(value,DataObjectTracker):
            name = np.string_("DataObjectTracker_id"+str(id(value)))
            group_inputs.attrs.create(key, name)
        else:
            group_inputs.attrs.create(key, value)
    group_logitem.attrs.create('version', logged_item.version)
    write_time_to_log_item(group_logitem, logged_item.datetime)

def write_time_to_log_item(group_logitem, datetime):
    date = str(datetime.tm_year)+str(datetime.tm_mon)+str(datetime.tm_mday)
    time = str(datetime.tm_hour)+':'+str(datetime.tm_min)+':'+str(datetime.tm_sec)
    group_logitem.attrs.create('time', np.string_(date+'__'+time))


############################### File structure ###############################
#
# /
# |--attr: version_major=0
# |--attr: version_minor=2
# |--grp: 4DSTEM_experiment
#             |
#             |--grp: datacube
#             |         |--attr: emd_group_type=1
#             |         |--data: datacube
#             |         |--data: dim1
#             |         |    |--attr: name="R_x"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim2
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim3
#             |         |    |--attr: name="Q_x"
#             |         |    |--attr: units="[n_m^-1]"
#             |         |--data: dim4
#             |               |--attr: name="Q_y"
#             |               |--attr: units="[n_m^-1]"
#             |
#             |--grp: processing
#             |         |
#             |         |--grp: datacubes
#             |         |   |
#             |         |   |--grp: processed_datacube_1
#             |         |   |    |--attr: emd_group_type=1
#             |         |   |    |--data: datacube
#             |         |   |    |--data: dim1,dim2,dim3,dim4
#             |         |   |    |--data: modification_log_indices
#             |         |   |
#             |         |   |--grp: processed_datacube_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: diffraction
#             |         |   |
#             |         |   |--grp: diffraction_slice_1
#             |         |   |    |--attr: emd_group_type=1
#             |         |   |    |--data: diffractionslice
#             |         |   |    |--data: dim1,dim2
#             |         |   |    |--data: modification_log_indices
#             |         |   |
#             |         |   |--grp: diffraction_slice_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: real
#             |         |   |
#             |         |   |--grp: real_slice_1
#             |         |   |    |--attr: emd_group_type=1
#             |         |   |    |--data: realslice
#             |         |   |    |--data: dim1,dim2
#             |         |   |    |--data: modification_log_indices
#             |         |   |
#             |         |   |--grp: real_slice_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: pointlist
#             |             |
#             |             |--grp: point_list_1
#             |             |    |--attr: coordinates='Qx, Qy, Rx, Ry, Int, ...'
#             |             |    |--attr: dimensions=val
#             |             |    |--data: point_list
#             |             |    |--data: modification_log_indices
#             |             |
#             |             |--grp: point_list_2
#             |             |    |
#             |             :    :
#             |
#             |--grp: log
#             |         |-grp: log_item_1
#             |         |   |--attr: function="function"
#             |         |   |--grp: inputs
#             |         |   |    |--attr: input1=val1
#             |         |   |    |--attr: input2=val2
#             |         |   |    |--...
#             |         |   |
#             |         |   |--attr: version=0.1
#             |         |   |--attr: time="20181015_16:09:42"
#             |         |
#             |         |-grp: log_item_2
#             |         |-...
#             |
#             |--grp: metadata
#                       |--grp: original
#                       |   |--# Raw metadata from original files
#                       |
#                       |--grp: microscope
#                       |   |--# Acquisition parameters
#                       |   |--# Accelerating voltage, camera length, convergence angle, 
#                       |   |--# C2 aperture, spot size, exposure time, scan rotation angle,
#                       |   |--# scan shape, probe FWHM
#                       |
#                       |--grp: sample
#                       |   |--# Material, preparation
#                       |
#                       |--grp: user
#                       |   |--# Name, instituion, dept, contact email
#                       |
#                       |--grp: calibration
#                       |   |--# R pixel size, Q pixel size, R/Q rotation offset
#                       |   |--# In case of duplicates here and in grp: microscope (e.g. pixel
#                       |   |--# sizes), quantities here are calculated from data rather than
#                       |   |--# being read from the instrument
#                       |
#                       |--grp: comments
#
#

############################### File structure ###############################
############################## Abridged version ##############################
#
# /
# |--grp: 4DSTEM_experiment
#             |
#             |--grp: datacube
#             |         |--data: datacube
#             |         |--data: dim1,dim2,dim3,dim4
#             |
#             |--grp: processing
#             |         |
#             |         |--grp: datacubes
#             |         |   |
#             |         |   |--grp: processed_datacube_1
#             |         |   |    |--data: datacube
#             |         |   |    |--data: dim1,dim2,dim3,dim4
#             |         |   |
#             |         |   |--grp: processed_datacube_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: diffraction
#             |         |   |
#             |         |   |--grp: diffraction_slice_1
#             |         |   |    |--data: diffractionslice
#             |         |   |    |--data: dim1,dim2
#             |         |   |
#             |         |   |--grp: diffraction_slice_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: real
#             |         |   |
#             |         |   |--grp: real_slice_1
#             |         |   |    |--data: realslice
#             |         |   |    |--data: dim1,dim2
#             |         |   |
#             |         |   |--grp: real_slice_2
#             |         |   |    |
#             |         |   :    :
#             |         |
#             |         |--grp: pointlist
#             |             |
#             |             |--grp: point_list_1
#             |             |    |--attr: coordinates='Qx, Qy, Rx, Ry, Int, ...'
#             |             |    |--attr: dimensions=val
#             |             |    |--data: point_list
#             |             |
#             |             |--grp: point_list_2
#             |             |    |
#             |             :    :
#             |
#             |--grp: log
#             |         |-grp: log_item_1
#             |         |-grp: log_item_2
#             |         |-...
#             |
#             |--grp: metadata
#                       |--grp: original
#                       |--grp: microscope
#                       |--grp: sample
#                       |--grp: user
#                       |--grp: calibration
#                       |--grp: comments



