# Write py4DSTEM formatted .h5 files.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from collections import OrderedDict
from os.path import exists
from os import remove as rm
from ..read import is_py4DSTEM_file, get_py4DSTEM_topgroups
from ..metadata import metadata_to_h5
from ...datastructure import DataCube, save_datacube_group
from ...datastructure import DiffractionSlice, save_diffraction_group
from ...datastructure import RealSlice, save_real_group
from ...datastructure import PointList, save_pointlist_group
from ...datastructure import PointListArray, save_pointlistarray_group
from ...datastructure import CountedDataCube, save_counted_datacube_group
from ...datastructure import Coordinates, save_coordinates_group
from ...datastructure import DataObject, Metadata, Coordinates
from ....process.utils import tqdmnd
from ....version import __version__

def save(filepath, data, overwrite=False, topgroup='4DSTEM_experiment', **kwargs):
    """
    Saves data to a new py4DSTEM .h5 file at filepath.

    Accepts:
        filepath            path where the file will be saved
        data                a single DataObject or a list of DataObjects
        overwrite           boolean controlling behavior when an existing file
                            is found at filepath.  If overwrite is True, deletes the
                            existing file and writes a new one. Otherwise,
                            raises an error.
        topgroup            name of the h5 toplevel group containing the py4DSTEM
                            file of interest
    """
    # Open the file
    if exists(filepath):
        if not overwrite:
            if is_py4DSTEM_file(filepath):
                # If the file exists and is a py4DSTEM .h5, determine
                # if we are writing a new topgroup to an existing .h5
                tgs = get_py4DSTEM_topgroups(filepath)
                if topgroup in tgs:
                    raise Exception("This py4DSTEM .h5 file already contains a topgroup named '{}'. Overwrite the whole file using overwrite=True, or add another topgroup.".format(topgroup))
                else:
                    f = h5py.File(filepath,'r+')
            else:
                raise Exception('A file already exists at path {}.  To overwrite the file, use overwrite=True. To append new objects to an existing file, use append() rather than save().'.format(filepath))
        else:
            rm(filepath)
            f = h5py.File(filepath,'w')
    else:
        f = h5py.File(filepath,'w')

    # Construct dataobject list
    if isinstance(data, DataObject):
        dataobject_list = [data]
    elif isinstance(data, list):
        assert all([isinstance(item,DataObject) for item in data]), "If 'data' is a list, all items must be DataObjects."
        dataobject_list = data
    else:
        raise TypeError("Error: unrecognized value for argument data. Must be a DataObject or list of DataObjects")
    assert np.sum([isinstance(dataobject_list[i],Metadata) for i in range(len(dataobject_list))])<2, "Multiple Metadata instances were passed"

    # Handle keyword arguments
    use_compression = kwargs.get('compression',False)

    ##### Make .h5 file #####
    # Make topgroup
    grp_top = f.create_group(topgroup)
    grp_top.attrs.create("emd_group_type",2)
    grp_top.attrs.create("version_major",__version__.split('.')[0])
    grp_top.attrs.create("version_minor",__version__.split('.')[1])
    grp_top.attrs.create("version_release",__version__.split('.')[2])

    # Make data groups
    group_data = grp_top.create_group("data")
    grp_dc = group_data.create_group("datacubes")
    grp_cdc = group_data.create_group("counted_datacubes")
    grp_ds = group_data.create_group("diffractionslices")
    grp_rs = group_data.create_group("realslices")
    grp_pl = group_data.create_group("pointlists")
    grp_pla = group_data.create_group("pointlistarrays")
    grp_coords = group_data.create_group("coordinates")
    ind_dcs, ind_cdcs, ind_dfs, ind_rls, ind_ptl, ind_ptla, ind_coords = 0,0,0,0,0,0,0

    # Make metadata group and identify any metadata, either passed as arguments or attached to DataCubes
    grp_md = grp_top.create_group("metadata")
    inds = np.nonzero([isinstance(dataobject_list[i],Metadata) for i in range(len(dataobject_list))])[0]
    metadata_list = []
    for i in inds[::-1]:
        metadata_list.append(dataobject_list.pop(i))
    for dataobject in dataobject_list:
        if isinstance(dataobject,DataCube):
            if hasattr(dataobject,'metadata'):
                metadata_list.append(dataobject.metadata)
    if len(metadata_list)>1:
        assert(all([id(metadata_list[0])==id(metadata_list[i]) for i in range(1,len(metadata_list))])), 'Error: multiple distinct Metadata objects found'
        md = metadata_list[0]
    elif len(metadata_list)==1:
        md = metadata_list[0]
    else:
        md = None

    # Loop through and save all objects in the dataobjectlist
    names,grps,save_fns = [],[],[]
    lookupTable = {
            'DataCube':['datacube_',ind_dcs,grp_dc,
                               save_datacube_group],
            'CountedDataCube':['counted_data_cube_',ind_cdcs,grp_cdc,
                                         save_counted_datacube_group],
            'DiffractionSlice':['diffractionslice_',ind_dfs,grp_ds,
                                            save_diffraction_group],
            'RealSlice':['realslice_',ind_rls,grp_rs,
                                     save_real_group],
            'PointList':['pointlist_',ind_ptl,grp_pl,
                                save_pointlist_group],
            'PointListArray':['pointlistarray_',ind_ptla,grp_pla,
                                       save_pointlistarray_group],
            'Coordinates':['coordinates_',ind_coords,grp_coords,
                                       save_coordinates_group]
             }
    for dataobject in dataobject_list:
        name = dataobject.name
        dtype = type(dataobject).__name__
        basename,inds,grp,save_fn = lookupTable[dtype]
        if name == '':
            name = basename+str(inds)
            inds += 1
        names.append(name)
        grps.append(grp)
        save_fns.append(save_fn)

    # Save metadata
    if md is not None:
        metadata_to_h5(filepath,md,overwrite=overwrite,topgroup=topgroup)
    else:
        metadata_to_h5(filepath,Metadata(),overwrite=overwrite,topgroup=topgroup)
    # Save data
    for name,grp,save_fn,do in zip(names,grps,save_fns,dataobject_list):
        new_grp = grp.create_group(name)
        print("Saving {} '{}'...".format(type(do).__name__,name))
        save_fn(new_grp,do)








