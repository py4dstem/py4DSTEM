# Write py4DSTEM formatted .h5 files.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from collections import OrderedDict
from os.path import exists
from os import remove as rm
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups
from .metadata import metadata_to_h5
from ..datastructure import DataCube, DiffractionSlice, RealSlice, CountedDataCube
from ..datastructure import DataObject, PointList, PointListArray, Metadata
from ...process.utils import tqdmnd
from ...version import __version__

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
    ind_dcs, ind_cdcs, ind_dfs, ind_rls, ind_ptl, ind_ptla = 0,0,0,0,0,0

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
                                       save_pointlistarray_group]
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




#### Functions for writing dataobjects to .h5 ####

def save_datacube_group(group, datacube, use_compression=False):
    group.attrs.create("emd_group_type",1)
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
    # if datacube.metadata is not None:
    #     group.attrs.create("metadata",datacube.metadata._ind)
    # else:
    #     group.attrs.create("metadata",-1)

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
    # if diffractionslice.metadata is not None:
    #     group.attrs.create("metadata",diffractionslice.metadata._ind)
    # else:
    #     group.attrs.create("metadata",-1)

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
    # if realslice.metadata is not None:
    #     group.attrs.create("metadata",realslice.metadata._ind)
    # else:
    #     group.attrs.create("metadata",-1)

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
    #if pointlist.metadata is not None:
    #    group.attrs.create("metadata",pointlist.metadata._ind)
    #else:
    #    group.attrs.create("metadata",-1)

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
    #if pointlistarray.metadata is not None:
    #    group.attrs.create("metadata",pointlistarray.metadata._ind)
    #else:
    #    group.attrs.create("metadata",-1)

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







