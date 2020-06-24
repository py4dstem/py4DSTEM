# Copy an existing py4DSTEM formatted .h5 file to a new file.
#
# The new file may be a complete copy of the original, or it may contain any subset of the original
# files DataObjects.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from os.path import exists
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_N_dataobjects, get_py4DSTEM_dataobject_info
from .read_py4DSTEM import read_py4DSTEM
from .write import save
from .append import append
from ...datastructure import DataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray
from ...datastructure import DataObject, Metadata

def copy(fp_orig, fp_new, topgroup='4DSTEM_experiment', **kwargs):
    """
    Copies DataObjects specified by indices from the py4DSTEM .h5 file at fp_orig to a
    new file at fp_new.

    Accepts:
        fp_orig             path to an existing py4DSTEM .h5 file to copy
        fp_new              path to the new file to write
        indices             if unspecified, copy the entire file.  If speficied, must be
                            either an int or a list of ints.  Copies the DataObjects
                            corresponding to these indices in the original file.
    """
    assert(is_py4DSTEM_file(fp_orig)), "Error: not recognized as a py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(fp_orig)
    assert(topgroup in tgs), "Error: topgroup '{}' not found.".format(topgroup)
    if exists(fp_new):
        raise Exception('{} already exists.'.format(fp_new))

    # Parse kwargs
    indices = kwargs.get('indices')
    if indices is None:
        _,_,_,_,_,_,N = get_N_dataobjects(fp_orig,topgroup)
        indices = list(np.arange(N))
    else:
        if isinstance(indices, int):
            indices = [indices]
        assert(all([isinstance(item,(int,np.integer)) for item in indices])), "Error: indices must be ints."

    info = get_py4DSTEM_dataobject_info(fp_orig,topgroup)
    for i in indices:
        data,_ = read_py4DSTEM(fp_orig,ft='py4DSTEM',topgroup=topgroup,data_id=i)
        if not exists(fp_new):
            print("Creating new file...")
            print("Copying {} object '{}'".format(info[i]['type'],info[i]['name']))
            save(fp_new,data=data,topgroup=topgroup)
        else:
            print("Copying {} object '{}'".format(info[i]['type'],info[i]['name']))
            append(fp_new,data=data,topgroup=topgroup)

    return



