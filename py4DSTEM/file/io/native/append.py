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
        grp_pla = f[topgroup]['data/pointlistarrays']

        # Loop through and save all objects in the dataobjectlist
        names,grps,save_fns = [],[],[]
        lookupTable = {
                'DataCube':['datacube_',N_dc,grp_dc,
                                   save_datacube_group],
                'CountedDataCube':['counted_data_cube_',N_cdc,grp_cdc,
                                             save_counted_datacube_group],
                'DiffractionSlice':['diffractionslice_',N_ds,grp_ds,
                                                save_diffraction_group],
                'RealSlice':['realslice_',N_rs,grp_rs,
                                         save_real_group],
                'PointList':['pointlist_',N_pl,grp_pl,
                                    save_pointlist_group],
                'PointListArray':['pointlistarray_',N_pla,grp_pla,
                                           save_pointlistarray_group]
                 }
        for dataobject in dataobject_list:
            name = dataobject.name
            dtype = type(dataobject).__name__
            basename,N,grp,save_fn = lookupTable[dtype]
            if name == '':
                name = basename+str(N)
                N += 1
            names.append(name)
            grps.append(grp)
            # Check if a group of this name already exists
            if name in grp.keys():
                if overwrite==True:
                    # TODO TKTKTK deal with releasing the storage space
                    # because h5 files apparently don't do this...
                    del grp[name]
                else:
                    save_fn = False
            save_fns.append(save_fn)

        # Error message if there are overwrite conflicts
        if not all(save_fns):
            inds = np.nonzero([i==False for i in save_fns])[0]
            dtypes,names = type(dataobject_list[inds]).__name__,dataobject_list[inds].name
            print('WARNING: attempting to append one or more DataObject type/names which already exist in this h5 file. Conflicts were found with the following objects:')
            print('')
            for dtype,name in zip(dtypes,names):
                print("{} '{}'".format(dtype,name))
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

        # Finish and close
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



