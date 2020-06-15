# Remove existing DataObjects from a py4DSTEM formatted .h5 file.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_py4DSTEM_dataobject_info

def remove_from_index_list(fp, indices, topgroup='4DSTEM_experiment'):
    """
    Remove existing dataobjects from a py4DSTEM h5 file.

    Accepts:
        fp            path to the py4DSTEM .h5 file
        indices             (list of ints) the indices of the DataObjects to remove
    """
    assert(all([isinstance(item,(int,np.integer)) for item in indices])), "Error: indices must be ints."
    assert is_py4DSTEM_file(fp), "fp parameter must point to an existing py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(fp)
    assert(topgroup in tgs), "Error: topgroup '{}' not found.".format(topgroup)

    info = get_py4DSTEM_dataobject_info(fp,topgroup)
    with h5py.File(fp,'a') as f:
        for i in indices:
            name = info[i]['name']
            objtype = info[i]['type']
            if objtype == "DataCube":
                group = f[topgroup + '/data/datacubes']
            elif objtype == "CountedDataCube":
                group = f[topgroup + '/data/counted_datacubes']
            elif objtype == "DiffractionSlice":
                group = f[topgroup + '/data/diffractionslices']
            elif objtype == "RealSlice":
                group = f[topgroup + '/data/realslices']
            elif objtype == "PointList":
                group = f[topgroup + '/data/pointlists']
            elif objtype == "PointListArray":
                group = f[topgroup + '/data/pointlistarrays']
            else:
                raise ValueError("Unknown DataObject type {}".format(objtype))
            print("Removing {} object '{}'".format(objtype,name))
            del group[name]

    ##### Finish and close #####
    print("Done.")
    f.close()

def remove(fp, dataobjects):
    """
    Remove existing dataobjects from a py4DSTEM h5 file.

    Accepts:
        fp              path to the py4DSTEM .h5 file
        dataobjects     (int or list of ints) the index or indices or name of the DataObjects to
                        remove.
    """
    assert is_py4DSTEM_file(fp), "fp parameter must point to an existing py4DSTEM file."

    if isinstance(dataobjects, (int,np.integer)):
        remove_from_index_list(fp, [dataobjects])
    else:
        remove_from_index_list(fp, dataobjects)


