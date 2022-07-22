# Write py4DSTEM formatted .h5 files.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from collections import OrderedDict
from os.path import exists, dirname
from os import remove
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups
from ..datastructure import (
    Root,
    Tree,
    Metadata,
    Array,
    PointList,
    PointListArray,
    Calibration,
    DataCube,
    DiffractionSlice,
    RealSlice
)
from ...version import __version__


def save(
    filepath,
    data,
    mode = 'w',
    root = '4DSTEM',
    tree = True,
    ):
    """
    Saves data to a .h5 file at filepath.

    Specific behavior depends on the `mode`, `root`, and `tree` arguments -
    see below.

    Args:
        filepath: path where the file will be saved
        data: a block of data, in the form of a py4DSTEM data class
            instance. Numpy arrays may also be saved by passing a
            tuple (array, 'name'), which will convert the array into
            a py4DSTEM Array with name 'name', and save it.
        mode (str): must be write mode ('w','write'), overwrite mode ('o',
            'overwrite'), append mode ('a','+','append'), or appendover
            ('ao','oa','o+','+o').  Write mode writes a new file, and if a
            file of this name already exists, raises an exception.
            Overwrite mode writes a new file, and if a file of this name
            already exists, deletes it and writes a new file. Append mode
            adds data to an existing file, and if a block of data with
            the same name as the block of data being saved already exists
            at the same level, raises an exception.  Appendover mode adds
            data to an existing file, and if a block of data with the same
            name as the data block being saved already exists, overwrites
            it. Default is write mode.
        root (str): indicates where in the .h5 file to store the data.
            When writing a new file, should be a string indicating the
            top level group in which to store the data.  When appending
            to an existing file, should be a string representing an
            existing group in the .h5 file where the data will be placed.
            If an invalid group is passed, raises an exception.
        tree: indicates how the object tree nested inside `data` should
            be treated.  If `True` (default), the entire tree is saved.
            If `False`, only this object is saved, without its tree. If
            `"noroot"`, saves the entire tree structure, but the root
            object is saved with metadata only, and none of its data is
            saved.
    """

    # parse mode
    writemode = [
        'w',
        'write'
    ]
    overwritemode = [
        'o',
        'overwrite'
    ]
    appendmode = [
        'a',
        '+',
        'append'
    ]
    appendovermode = [
        'oa',
        'ao',
        'o+',
        '+o'
    ]
    allmodes = writemode + overwritemode + appendmode + appendovermode

    er = f"unrecognized mode {mode}; mode must be in {allmodes}"
    assert mode in allmodes, er

    # If a numpy array was passed, convert it into an Array
    if isinstance(data,tuple):
        assert(isinstance(data[0],np.ndarray)), "if `data` is a tuple, types must be (np.ndarray, string)"
        data = Array(
            data = data[0],
            name = data[1]
        )

    # overwrite: delete existing file
    if mode in overwritemode:
        if exists(filepath):
            remove(filepath)
        mode = 'w'

    # write a new file
    if (mode in writemode) and (exists(filepath)):
        raise Exception("A file already exists at this destination; use append or overwrite mode, or choose a new file path.")
    elif (mode in writemode) or (
        mode in appendmode+appendovermode and not exists(filepath)):
        _write_new_file(
            filepath = filepath,
            data = data,
            root = root,
            tree = tree
        )

    # append to an existing file
    elif mode in appendmode:
        _append_to_file(
            filepath = filepath,
            data = data,
            root = root,
            tree = tree,
            appendover = False
        )

    # append to an existing file, overwriting objects with the same name
    elif mode in appendovermode:
        _append_to_file(
            filepath = filepath,
            data = data,
            root = root,
            tree = tree,
            appendover = True
        )

    else:
        raise Exception(f"unknown mode {mode}")



def _write_new_file(
    filepath,
    data,
    root,
    tree,
    ):
    """
    Saves data to a new py4DSTEM .h5 file at filepath.
    """

    # open a new .h5 file
    with h5py.File(filepath, 'w') as f:

        # Make top level group
        grp_top = f.create_group(root)
        grp_top.attrs.create("emd_group_type",'root')
        grp_top.attrs.create("version_major",__version__.split('.')[0])
        grp_top.attrs.create("version_minor",__version__.split('.')[1])
        grp_top.attrs.create("version_release",__version__.split('.')[2])

        # Save
        if tree is True:
            _save_w_tree(grp_top,data)
            _add_calibration(grp_top,data)

        elif tree is False:
            _save_wout_tree(grp_top,data)
            _add_calibration(grp_top,data)

        elif tree == "noroot":
            _save_wout_root(grp_top,data)
            _add_calibration(grp_top,data)

        else:
            raise Exception(f"invalid argument {tree} passed for `tree`; must be True or False or 'noroot'")



def _append_to_file(
    filepath,
    data,
    root,
    tree,
    appendover = False
    ):
    """
    Append data to an existing py4DSTEM .h5 file at filepath.
    If an object is passed that already exists at this location,
    raises an exception or removed that object, depeding on the
    value of `appendover`.
    """
    # open a new .h5 file
    with h5py.File(filepath, 'a') as f:

        # find the root, check if it's an EMD group
        try:
            grp_top = f[root]
            er = "specified root exists, but is not an EMD group"
            assert("emd_group_type" in grp_top.attrs.keys()), er
        # if this root doesn't exist, create it
        except KeyError:
            grp_top = f.create_group(root)
            grp_top.attrs.create("emd_group_type",'root')
            grp_top.attrs.create("version_major",__version__.split('.')[0])
            grp_top.attrs.create("version_minor",__version__.split('.')[1])
            grp_top.attrs.create("version_release",__version__.split('.')[2])

        # Check if a data block with this data's name already exists
        # either raise an exception or overwrite, depending on mode
        if data.name in grp_top.keys():
            if appendover is False:
                raise Exception("Data with this name already exists in this location. Either change the data's name, or overwrite using appendover mode.")
            else:
                del grp_top[data.name]

        # Save
        if tree is True:
            _save_w_tree(grp_top,data)
            _add_calibration(grp_top,data)

        elif tree is False:
            _save_wout_tree(grp_top,data)
            _add_calibration(grp_top,data)

        elif tree == "noroot":
            _save_wout_root(grp_top,data)
            _add_calibration(grp_top,data)

        else:
            raise Exception(f"invalid argument {tree} passed for `tree`; must be True or False or 'noroot'")










def _save_wout_tree(
    group,
    data
    ):

    # Save the data
    data.to_h5(group)

    # Save only the metadata in its tree
    subgroup = group[data.name]
    for key in data.tree.keys():
        d = data.tree[key]
        if isinstance(d,Metadata):
            _save_wout_tree(
                subgroup,
                d
            )



def _save_w_tree(
    group,
    data
    ):

    # Save the data
    data.to_h5(group)

    # Save its tree
    subgroup = group[data.name]
    for key in data.tree.keys():
        # Convert np.ndarrays to Arrays
        d = data.tree[key]
        if isinstance(d,np.ndarray):
            d = Array(
                data = d,
                name = key)
        # Save
        _save_w_tree(
            subgroup,
            d
        )



def _save_wout_root(
    group,
    data
    ):

    grp_root = group.create_group(data.name)
    grp_root.attrs.create("emd_group_type",'root')
    grp_root.attrs.create("version_major",__version__.split('.')[0])
    grp_root.attrs.create("version_minor",__version__.split('.')[1])
    grp_root.attrs.create("version_release",__version__.split('.')[2])



    # Save the root group without its data
    #data.to_h5(group, include_data=False)

    # Save its tree
    #subgroup = group[data.name]
    for key in data.tree.keys():
        # Convert np.ndarrays to Arrays
        d = data.tree[key]
        if isinstance(d,np.ndarray):
            d = Array(
                data = d,
                name = key)
        # Save
        _save_w_tree(
            grp_root,
            d
        )



def _add_calibration(
    group,
    data
    ):
    """
    Checks if a Calibration instance exists at the top level;
    if not, looks for a .calibration parameter, and adds this.
    """
    # checks if there is already a Calibration instance at top level
    for key in data.tree.keys():
        d = data.tree[key]
        if isinstance(d,Calibration):
            return
    try:
        # If .calibration attr exists, save to tree
        cal = data.calibration
        subgroup = group[data.name]
        _save_wout_tree(
            subgroup,
            cal
        )
    except AttributeError:
        return







    # Open the file
#    if exists(filepath):
#        if not overwrite:
#            if is_py4DSTEM_file(filepath):
#                # If the file exists and is a py4DSTEM .h5, determine
#                # if we are writing a new topgroup to an existing .h5
#                tgs = get_py4DSTEM_topgroups(filepath)
#                if topgroup in tgs:
#                    raise Exception("This py4DSTEM .h5 file already contains a topgroup named '{}'. Overwrite the whole file using overwrite=True, or add another topgroup.".format(topgroup))
#                else:
#                    f = h5py.File(filepath,'r+')
#            else:
#                raise Exception('A file already exists at path {}.  To overwrite the file, use overwrite=True. To append new objects to an existing file, use append() rather than save().'.format(filepath))
#        else:
#            rm(filepath)
#            f = h5py.File(filepath,'w')
#    else:
#        f = h5py.File(filepath,'w')
#
#    # Construct dataobject list
#    if isinstance(data, DataObject):
#        dataobject_list = [data]
#    elif isinstance(data, list):
#        assert all([isinstance(item,DataObject) for item in data]), "If 'data' is a list, all items must be DataObjects."
#        dataobject_list = data
#    else:
#        raise TypeError("Error: unrecognized value for argument data. Must be a DataObject or list of DataObjects")
#    assert np.sum([isinstance(dataobject_list[i],Metadata) for i in range(len(dataobject_list))])<2, "Multiple Metadata instances were passed"
#
#    # Handle keyword arguments
#    use_compression = kwargs.get('compression',False)
#
#    ##### Make .h5 file #####
#    # Make topgroup
#    grp_top = f.create_group(topgroup)
#    grp_top.attrs.create("emd_group_type",2)
#    grp_top.attrs.create("version_major",__version__.split('.')[0])
#    grp_top.attrs.create("version_minor",__version__.split('.')[1])
#    grp_top.attrs.create("version_release",__version__.split('.')[2])
#
#    # Make data groups
#    group_data = grp_top.create_group("data")
#    grp_dc = group_data.create_group("datacubes")
#    grp_cdc = group_data.create_group("counted_datacubes")
#    grp_ds = group_data.create_group("diffractionslices")
#    grp_rs = group_data.create_group("realslices")
#    grp_pl = group_data.create_group("pointlists")
#    grp_pla = group_data.create_group("pointlistarrays")
#    grp_coords = group_data.create_group("coordinates")
#    ind_dcs, ind_cdcs, ind_dfs, ind_rls, ind_ptl, ind_ptla, ind_coords = 0,0,0,0,0,0,0
#
#    # Make metadata group and identify any metadata, either passed as arguments or attached to DataCubes
#    grp_md = grp_top.create_group("metadata")
#    inds = np.nonzero([isinstance(dataobject_list[i],Metadata) for i in range(len(dataobject_list))])[0]
#    metadata_list = []
#    for i in inds[::-1]:
#        metadata_list.append(dataobject_list.pop(i))
#    for dataobject in dataobject_list:
#        if isinstance(dataobject,DataCube):
#            if hasattr(dataobject,'metadata'):
#                metadata_list.append(dataobject.metadata)
#    if len(metadata_list)>1:
#        assert(all([id(metadata_list[0])==id(metadata_list[i]) for i in range(1,len(metadata_list))])), 'Error: multiple distinct Metadata objects found'
#        md = metadata_list[0]
#    elif len(metadata_list)==1:
#        md = metadata_list[0]
#    else:
#        md = None
#
#    # Function to patch HDF5 datacube compression back in
#    def save_datacube_compression_patch(grp,do):
#        save_datacube_group(grp,do,use_compression=use_compression)
#
#    # Loop through and save all objects in the dataobjectlist
#    names,grps,save_fns = [],[],[]
#    lookupTable = {
#            'DataCube':['datacube_',ind_dcs,grp_dc,
#                               save_datacube_compression_patch],
#                               #save_datacube_group],
#            'CountedDataCube':['counted_data_cube_',ind_cdcs,grp_cdc,
#                                         save_counted_datacube_group],
#            'DiffractionSlice':['diffractionslice_',ind_dfs,grp_ds,
#                                            save_diffraction_group],
#            'RealSlice':['realslice_',ind_rls,grp_rs,
#                                     save_real_group],
#            'PointList':['pointlist_',ind_ptl,grp_pl,
#                                save_pointlist_group],
#            'PointListArray':['pointlistarray_',ind_ptla,grp_pla,
#                                       save_pointlistarray_group],
#            'Coordinates':['coordinates_',ind_coords,grp_coords,
#                                       save_coordinates_group]
#             }
#    for dataobject in dataobject_list:
#        name = dataobject.name
#        dtype = type(dataobject).__name__
#        basename,inds,grp,save_fn = lookupTable[dtype]
#        if name == '':
#            name = basename+str(inds)
#            inds += 1
#        names.append(name)
#        grps.append(grp)
#        save_fns.append(save_fn)
#
#    # Save metadata
#    if md is not None:
#        metadata_to_h5(filepath,md,overwrite=overwrite,topgroup=topgroup)
#    else:
#        metadata_to_h5(filepath,Metadata(),overwrite=overwrite,topgroup=topgroup)
#    # Save data
#    for name,grp,save_fn,do in zip(names,grps,save_fns,dataobject_list):
#        new_grp = grp.create_group(name)
#        print("Saving {} '{}'...".format(type(do).__name__,name))
#        save_fn(new_grp,do)








