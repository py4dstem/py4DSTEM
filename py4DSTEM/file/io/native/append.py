# Append additional DataObjects to an existing py4DSTEM formatted .h5 file.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_version
from .read_utils import get_N_dataobjects, get_py4DSTEM_topgroups
from .write import save_datacube_group, save_diffraction_group, save_real_group
from .write import save_pointlist_group, save_pointlistarray_group
from .write import save_counted_datacube_group
from ...datastructure import DataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray, CountedDataCube
from ...datastructure import DataObject

def append_from_dataobject_list(fp, dataobject_list, overwrite=False,
                                topgroup='4DSTEM_experiment'):
    """
    Appends new dataobjects to an existing py4DSTEM h5 file.

    Accepts:
        fp                  path to an existing py4DSTEM .h5 file to append to
        dataobject_list     a list of DataObjects to save
        overwrite           boolean controlling behavior when a dataobject with
                            the same name as one already in the .h5 file is found.
                            If True, overwrite the object. If false, raise an error.
        topgroup            name of the h5 toplevel group containing the py4DSTEM
                            file of interest
    """

    assert all([isinstance(item,DataObject) for item in dataobject_list]), "Error: all elements of dataobject_list must be DataObject instances."
    assert(is_py4DSTEM_file(fp)), "Error: file is not recognized as a py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(fp)
    assert(topgroup in tgs), "Error: specified topgroup, {}, not found.".format(self.topgroup)

    N_dc,N_cdc,N_ds,N_rs,N_pl,N_pla,N_do = get_N_dataobjects(fp)
    with h5py.File(fp,"r+") as f:
        # Get data groups
        group_data = f[topgroup]['data']
        grp_dc = f[topgroup]['data/datacubes']
        grp_cdc = f[topgroup]['data/counted_datacubes']
        grp_ds = f[topgroup]['data/diffractionslices']
        grp_rs = f[topgroup]['data/realslices']
        grp_pl = f[topgroup]['data/pointlists']
        grp_plas = f[topgroup]['data/pointlistarrays']

        # Loop through and save all objects in the dataobjectlist
        names,grps,save_fns = [],[],[]
        for dataobject in dataobject_list:
            name = dataobject.name
            if isinstance(dataobject, DataCube):
                if name == '':
                    name = 'datacube_'+str(N_dc)
                    N_dc += 1
                grp_curr = grp_dc
                save_func_curr = save_datacube_group
            elif isinstance(dataobject, CountedDataCube):
                if name == '':
                    name = 'counted_datacube_'+str(N_cdc)
                    N_cdc += 1
                grp_curr = grp_cdc
                save_func_curr = save_counted_datacube_group
            elif isinstance(dataobject, DiffractionSlice):
                if name == '':
                    name = 'diffractionslice_'+str(N_ds)
                    N_ds += 1
                grp_curr = grp_ds
                save_func_curr = save_diffraction_group
            elif isinstance(dataobject, RealSlice):
                if name == '':
                    name = 'realslice_'+str(N_rs)
                    N_rs += 1
                grp_curr = grp_rs
                save_func_curr = save_real_group
            elif isinstance(dataobject, PointList):
                if name == '':
                    name = 'pointlist_'+str(N_pl)
                    N_pl += 1
                grp_curr = grp_pl
                save_func_curr = save_pointlist_group
            elif isinstance(dataobject, PointListArray):
                if name == '':
                    name = 'pointlistarray_'+str(N_pla)
                    N_pla += 1
                grp_curr = grp_plas
                save_func_curr = save_pointlistarray_group
            else:
                raise Exception('Unrecognized dataobject type {}'.format(type(dataobject)))
            names.append(name)
            grps.append(grp_curr)
            # Check if a group of this name already exists
            if name in grp_curr.keys():
                if overwrite==True:
                    # TODO TKTKTK deal with releasing the storage space
                    # because h5 files apparently don't do this...
                    del grp_curr[name]
                else:
                    save_func_curr = False
            save_fns.append(save_func_curr)

        # Error message if there are overwrite conflicts
        if not all(save_fns):
            inds = np.nonzero([i==False for i in save_fns])[0]
            print('WARNING: attempting to append one or more DataObject type/names which already exist in this h5 file. Conflicts were found with the following objects:')
            print('')
            for i in inds:
                print("{} '{}'".format(type(dataobject_list[i]).__name__,dataobject_list[i].name))
            print('')
            print("Either rename these objects before saving, or pass the 'overwrite=True' keyword to save over the corresponding objects in the .h5.  Note that this will *not* release the storage space on the .h5 file, so if these are large objects consider re-writing the file.")
            print("No objects saved.")
            return

        # Save objects
        else:
            for name,grp,save_fn,do in zip(names,grps,save_fns,dataobject_list):
                new_grp = grp.create_group(name)
                print("Saving {} '{}'...".format(type(do).__name__,name))
                save_fn(new_grp,do)


#            if isinstance(dataobject, DataCube):
#                if name == '':
#                    name = 'datacube_'+str(N_dc)
#                    N_dc += 1
#                try:
#                    group_new_datacube = grp_dc.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_dc.keys())])
#                    name = name+"_"+str(N)
#                    group_new_datacube = grp_dc.create_group(name)
#                save_datacube_group(group_new_datacube, dataobject)
#            elif isinstance(dataobject,CountedDataCube):
#                if name == '':
#                    name = 'counted_datacube_'+str(N_cdc)
#                    ind_cdcs += 1
#                try:
#                    group_new_counted = grp_cdc.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_cdc.keys())])
#                    name = name+"_"+str(N)
#                    group_new_counted = grp_cdc.create_group(name)
#                save_counted_datacube_group(group_new_counted, dataobject)
#            elif isinstance(dataobject, DiffractionSlice):
#                if name == '':
#                    name = 'diffractionslice_'+str(N_ds)
#                    N_ds += 1
#                try:
#                    group_new_diffractionslice = grp_ds.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_ds.keys())])
#                    name = name+"_"+str(N)
#                    group_new_diffractionslice = grp_ds.create_group(name)
#                save_diffraction_group(group_new_diffractionslice, dataobject)
#            elif isinstance(dataobject, RealSlice):
#                if name == '':
#                    name = 'realslice_'+str(N_rs)
#                    N_rs += 1
#                try:
#                    group_new_realslice = grp_rs.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_rs.keys())])
#                    name = name+"_"+str(N)
#                    group_new_realslice = grp_rs.create_group(name)
#                save_real_group(group_new_realslice, dataobject)
#            elif isinstance(dataobject, PointList):
#                if name == '':
#                    name = 'pointlist_'+str(N_pl)
#                    N_pl += 1
#                try:
#                    group_new_pointlist = grp_pl.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_pl.keys())])
#                    name = name+"_"+str(N)
#                    group_new_pointlist = grp_pl.create_group(name)
#                save_pointlist_group(group_new_pointlist, dataobject)
#            elif isinstance(dataobject, PointListArray):
#                if name == '':
#                    name = 'pointlistarray_'+str(N_pla)
#                    N_pla += 1
#                try:
#                    group_new_pointlistarray = grp_plas.create_group(name)
#                except ValueError:
#                    N = sum([name in string for string in list(grp_plas.keys())])
#                    name = name+"_"+str(N)
#                    group_new_pointlistarray = grp_plas.create_group(name)
#                save_pointlistarray_group(group_new_pointlistarray, dataobject)
#            else:
#                print("Error: object {} has type {}, and is not a DataCube, DiffractionSlice, RealSlice, PointList, or PointListArray instance.".format(dataobject,type(dataobject)))

        ##### Finish and close #####
        print("Done.")
        f.close()

def append_dataobject(fp, dataobject, **kwargs):
    """
    Appends dataobject to existing .h5 file at fp.
    """
    assert isinstance(dataobject, DataObject)

    # append
    append_from_dataobject_list(fp, [dataobject], **kwargs)

def append_dataobjects_by_indices(fp, index_list, **kwargs):
    """
    Appends the DataObjects found at the indices in index_list in DataObject.get_dataobjects to
    the existing py4DSTEM .h5 file at fp.
    """
    full_dataobject_list = DataObject.get_dataobjects()
    dataobject_list = [full_dataobject_list[i] for i in index_list]

    append_from_dataobject_list(fp, dataobject_list, **kwargs)

def append(fp, data, **kwargs):
    """
    Appends data to an existing py4DSTEM .h5 file at fp. What is saved depends on the
    arguement data.

    If data is a DataObject, appends just this dataobject.
    If data is a list of DataObjects, appends all these objects.
    If data is an int, appends the dataobject corresponding to this index in
    DataObject.get_dataobjects().
    If data is a list of indices, appends the objects corresponding to these
    indices in DataObject.get_dataobjects().
    """
    if isinstance(data, DataObject):
        append_dataobject(fp, data, **kwargs)
    elif isinstance(data, int):
        append_dataobjects_by_indices(fp, [data], **kwargs)
    elif isinstance(data, list):
        if all([isinstance(item,DataObject) for item in data]):
            append_from_dataobject_list(fp, data, **kwargs)
        elif all([isinstance(item,int) for item in data]):
            append_dataobjects_by_indices(fp, data, **kwargs)
        else:
            print("Error: if data is a list, it must contain all ints or all DataObjects.")
    else:
        print("Error: unrecognized value for argument data. Must be either a DataObject, a list of DataObjects, a list of ints, or the string 'all'.")



