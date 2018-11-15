# Reading / writing 4D-STEM data into an HDF5 file format
# 
# Files are readable as .emd files, with the datacube group conforming to the EMD data group 
# specifications.
#
# See end of file for complete file structure.

import h5py
import numpy as np
from hyperspy.misc.utils import DictionaryTreeBrowser
from ..process.datastructure.datacube import MetadataCollection, DataCube
from ..process.log import log, Logger

logger = Logger()

@log
def save_from_datacube(datacube,outputfile):
    """
    Saves an h5 file from a datacube object and an output filepath.
    """

    ##### Make .h5 file #####
    print("Creating file {}...".format(outputfile))
    f = h5py.File(outputfile,"w")
    f.attrs.create("version_major",0)
    f.attrs.create("version_minor",1)
    group_data = f.create_group("4D-STEM_data")


    ##### Write metadata #####
    print("Writing metadata...")

    # Create metadata groups
    group_metadata = group_data.create_group("metadata")
    group_original_metadata = group_metadata.create_group("original")
    group_microscope_metadata = group_metadata.create_group("microscope")
    group_sample_metadata = group_metadata.create_group("sample")
    group_user_metadata = group_metadata.create_group("user")
    group_processing_metadata = group_metadata.create_group("processing")
    group_calibration_metadata = group_metadata.create_group("calibration")
    group_comments_metadata = group_metadata.create_group("comments")
    group_original_metadata_all = group_original_metadata.create_group("all")
    group_original_metadata_shortlist = group_original_metadata.create_group("shortlist")

    # Transfer original metadata trees
    if type(datacube.metadata.original.shortlist)==DictionaryTreeBrowser:
        transfer_metadata_tree_hs(datacube.metadata.original.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_hs(datacube.metadata.original.all,group_original_metadata_all)
    else:
        transfer_metadata_tree_py4DSTEM(datacube.metadata.original.shortlist,group_original_metadata_shortlist)
        transfer_metadata_tree_py4DSTEM(datacube.metadata.original.all,group_original_metadata_all)

    # Transfer datacube metadata dictionaries
    transfer_metadata_dict(datacube.metadata.microscope,group_microscope_metadata)
    transfer_metadata_dict(datacube.metadata.sample,group_sample_metadata)
    transfer_metadata_dict(datacube.metadata.user,group_user_metadata)
    transfer_metadata_dict(datacube.metadata.processing,group_processing_metadata)
    transfer_metadata_dict(datacube.metadata.calibration,group_calibration_metadata)
    transfer_metadata_dict(datacube.metadata.comments,group_comments_metadata)


    ##### Write datacube group #####
    group_datacube = group_data.create_group("datacube")
    group_datacube.attrs.create("emd_group_type",1)

    # TODO: consider defining data chunking here, keeping k-space slices together
    data_datacube = group_datacube.create_dataset("datacube", data=datacube.data4D)

    # Dimensions
    assert len(data_datacube.shape)==4, "Shape of datacube is {}".format(len(data_datacube))
    R_Ny,R_Nx,K_Ny,K_Nx = data_datacube.shape
    data_R_Ny = group_datacube.create_dataset("dim1",(R_Ny,))
    data_R_Nx = group_datacube.create_dataset("dim2",(R_Nx,))
    data_K_Ny = group_datacube.create_dataset("dim3",(K_Ny,))
    data_K_Nx = group_datacube.create_dataset("dim4",(K_Nx,))

    # Populate uncalibrated dimensional axes
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_K_Ny[...] = np.arange(0,K_Ny)
    data_K_Ny.attrs.create("name",np.string_("K_y"))
    data_K_Ny.attrs.create("units",np.string_("[pix]"))
    data_K_Nx[...] = np.arange(0,K_Nx)
    data_K_Nx.attrs.create("name",np.string_("K_x"))
    data_K_Nx.attrs.create("units",np.string_("[pix]"))

    # Calibrate axes, if calibrations are present

    # Calibrate R axes
    try:
        R_pix_size = datacube.metadata.calibration["R_pix_size"]
        data_R_Ny[...] = np.arange(0,R_Ny*R_pix_size,R_pix_size)
        data_R_Nx[...] = np.arange(0,R_Nx*R_pix_size,R_pix_size)
        # Set R axis units
        try:
            R_units = datacube.metadata.calibration["R_units"]
            data_R_Nx.attrs["units"] = R_units
            data_R_Ny.attrs["units"] = R_units
        except KeyError:
            print("WARNING: Real space calibration found and applied, however, units were",
                   "not identified and have been left in pixels.")
    except KeyError:
        print("No real space calibration found.")
    except TypeError:
        # If R_pix_size is a str, i.e. has not been entered, pass
        pass

    # Calibrate K axes
    try:
        K_pix_size = datacube.metadata.calibration["K_pix_size"]
        data_K_Ny[...] = np.arange(0,K_Ny*K_pix_size,K_pix_size)
        data_K_Nx[...] = np.arange(0,K_Nx*K_pix_size,K_pix_size)
        # Set K axis units
        try:
            K_units = datacube.metadata.calibration["K_units"]
            data_K_Nx.attrs["units"] = K_units
            data_K_Ny.attrs["units"] = K_units
        except KeyError:
            print("WARNING: Diffraction space calibration found and applied, however, units",
                   "were not identified and have been left in pixels.")
    except KeyError:
        print("No diffraction space calibration found.")
    except TypeError:
        # If K_pix_size is a str, i.e. has not been entered, pass
        pass


    ##### Write log #####
    group_log = f.create_group("log")
    for index in range(logger.log_index):
        write_log_item(group_log, index, logger.logged_items[index])


    ##### Finish and close #####
    print("Done.")
    f.close()


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
        elif type(value)==DataCube:
            group_inputs.attrs.create(key, np.string_("DataCube_id"+str(id(value))))
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
# |--grp: 4D-STEM_data
#             |
#             |--grp: datacube
#             |         |--attr: emd_group_type=1
#             |         |--data: datacube
#             |         |--data: dim1
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim2
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim3
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim4
#             |               |--attr: name="R_y"
#             |               |--attr: units="[n_m]"
#             |
#             |--grp: processing
#             |         |--# This will contain objects created and used during processing
#             |         |--# e.g. vacuum probe, shifts from scan coils, binary masks,
#             |         |--#      convolution kernels, etc.
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
#                       |--grp: processing
#                       |   |--# original file name, processing perfomed - binning, cropping
#                       |   |--# Consider attaching logs
#                       |
#                       |--grp: calibration
#                       |   |--# R pixel size, K pixel size, R/K rotation offset
#                       |   |--# In case of duplicates here and in grp: microscope (e.g. pixel
#                       |   |--# sizes), quantities here are calculated from data rather than
#                       |   |--# being read from the instrument
#                       |
#                       |--grp: comments


