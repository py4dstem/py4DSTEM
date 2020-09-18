# Write py4DSTEM formatted .h5 files.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from collections import OrderedDict
from os.path import exists
from ...datastructure import DataCube, DiffractionSlice, RealSlice, CountedDataCube
from ...datastructure import DataObject, PointList
from ...datastructure import PointListArray
from ....process.utils import tqdmnd
from ....version import __version__

def save_from_dataobject_list(fp, dataobject_list, topgroup="4DSTEM_experiment",
                                                      overwrite=False, **kwargs):
    """
    Saves an h5 file from a list of DataObjects and an output filepath.

    Accepts:
        fp                  path to save the .h5 file to
        dataobject_list     a list of DataObjects to save
        topgroup            (str) name for the toplevel group; if None, use
                            "4DSTEM_experiment"
    """

    assert(all([isinstance(item,DataObject) for item in dataobject_list])), "Error: all elements of dataobject_list must be DataObject instances."
    assert(isinstance(topgroup,str)), "Error: topgroup must be a string"
    if exists(fp):
        if overwrite is False:
            raise Exception('{} already exists.  To overwrite, use overwrite=True. To append new objects to an existing file, use append() rather than save().'.format(fp))

    # Handle keyword arguments
    use_compression = kwargs.get('compression',False)

    ##### Make .h5 file #####
    with h5py.File(fp,'w') as f:
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

        # Loop through and save all objects in the dataobjectlist
        names,grps,save_fns = [],[],[]
        for dataobject in dataobject_list:
            name = dataobject.name
            if isinstance(dataobject, DataCube):
                if name == '':
                    name = 'datacube_'+str(ind_dcs)
                    ind_dcs += 1
                grp_curr = grp_dc
                save_func_curr = save_datacube_group
            elif isinstance(dataobject,CountedDataCube):
                if name == '':
                    name = 'counted_datacube_'+str(ind_dcs)
                    ind_cdcs += 1
                grp_curr = grp_cdc
                save_func_curr = save_counted_datacube_group
            elif isinstance(dataobject, DiffractionSlice):
                if name == '':
                    name = 'diffractionslice_'+str(ind_dfs)
                    ind_dfs += 1
                grp_curr = grp_ds
                save_func_curr = save_diffraction_group
            elif isinstance(dataobject, RealSlice):
                if name == '':
                    name = 'realslice_'+str(ind_rls)
                    ind_rls += 1
                grp_curr = grp_rs
                save_func_curr = save_real_group
            elif isinstance(dataobject, PointList):
                if name == '':
                    name = 'pointlist_'+str(ind_ptl)
                    ind_ptl += 1
                grp_curr = grp_pl
                save_func_curr = save_pointlist_group
            elif isinstance(dataobject, PointListArray):
                if name == '':
                    name = 'pointlistarray_'+str(ind_ptla)
                    ind_ptla += 1
                grp_curr = grp_pla
                save_func_curr = save_pointlistarray_group
            else:
                raise Exception('Unrecognized dataobject type {}'.format(type(dataobejct)))
            names.append(name)
            grps.append(grp_curr)
            save_fns.append(save_func_curr)

        # Save objects
        for name,grp,save_fn,do in zip(names,grps,save_fns,dataobject_list):
            new_grp = grp.create_group(name)
            print("Saving {} '{}'...".format(type(do).__name__,name))
            save_fn(new_grp,do)

    print("Done.",flush=True)

def save_dataobject(fp, dataobject, **kwargs):
    """
    Saves a .h5 file containing only a single DataObject instance to fp.
    """
    assert isinstance(dataobject, DataObject)

    # Save
    save_from_dataobject_list(fp, [dataobject], **kwargs)

def save_dataobjects_by_indices(fp, index_list, **kwargs):
    """
    Saves a .h5 file containing DataObjects corresponding to the indices in index_list, a list of
    ints, in the list generated by DataObject.get_dataobjects().
    """
    full_dataobject_list = DataObject.get_dataobjects()
    dataobject_list = [full_dataobject_list[i] for i in index_list]

    save_from_dataobject_list(fp, dataobject_list, **kwargs)

def save(fp, data, **kwargs):
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
        save_dataobject(fp, data, **kwargs)
    elif isinstance(data, int):
        save_dataobjects_by_indices(fp, [data], **kwargs)
    elif isinstance(data, list):
        if all([isinstance(item,DataObject) for item in data]):
            save_from_dataobject_list(fp, data, **kwargs)
        elif all([isinstance(item,int) for item in data]):
            save_dataobjects_by_indices(fp, data, **kwargs)
        else:
            print("Error: if data is a list, it must contain all ints or all DataObjects.")
    elif data=='all':
        save_from_dataobject_list(fp, DataObject.get_dataobjects(), **kwargs)
    else:
        print("Error: unrecognized value for argument data. Must be either a DataObject, a list of DataObjects, a list of ints, or the string 'all'.")


################### END OF PRIMARY SAVE FUNCTIONS #####################



#### Functions for writing dataobjects to .h5 ####

def save_datacube_group(group, datacube, use_compression=False):
    group.attrs.create("emd_group_type",1)
    # if datacube.metadata is not None:
    #     group.attrs.create("metadata",datacube.metadata._ind)
    # else:
    #     group.attrs.create("metadata",-1)

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




