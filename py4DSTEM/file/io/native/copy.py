# Copy a py4DSTEM file, or some subset of a py4DSTEM file

import h5py
import numpy as np
from os import remove, rename
from os.path import exists, dirname, basename
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups
from .read_utils import get_N_dataobjects, get_py4DSTEM_dataobject_info
from .read_py4DSTEM import read_py4DSTEM
from .write import save
from ._append import _append
from ...datastructure import DataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray
from ...datastructure import DataObject

def copy(fp_orig, fp_new, indices=None, delete=False,
         topgroup_orig='4DSTEM_experiment',topgroup_new='4DSTEM_experiment'):
    """
    Copies DataObjects specified by indices from the py4DSTEM .h5 file at fp_orig
    to avnew file at fp_new.

    Accepts:
        fp_orig             path to an existing py4DSTEM .h5 file to copy
        fp_new              path to the new file to write
        indices             if None, copy the entire file.
                            Otherwise must be either an int or a list of ints.
                            Copies the DataObjects corresponding to these indices
                            in the original file.
        topgroup_orig       The toplevel group for the original file
        topgroup_new        and for the new file
    """
    assert(is_py4DSTEM_file(fp_orig)), "Error: not recognized as a py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(fp_orig)
    assert(topgroup_orig in tgs), "Error: topgroup '{}' not found.".format(topgroup)
    if exists(fp_new):
        assert(is_py4DSTEM_file(fp_new)), "Error: a file with the target filename already exists, and is not recognized as a py4DSTEM file."
        tgs = get_py4DSTEM_topgroups(fp_new)
        if topgroup_new in tgs:
            raise Exception('A file with the target filename exists and already contains a toplevel group with the specified topgroup name.')

    # Determine what needs to be copied
    if indices is None:
        _,_,_,_,_,_,N = get_N_dataobjects(fp_orig,topgroup_orig)
        indices = list(np.arange(N))
    else:
        if isinstance(indices, int):
            indices = [indices]
        assert(all([isinstance(item,(int,np.integer)) for item in indices])), "Error: indices must be ints."

    # Make infrastructure for the new file
    save(fp_new,[],topgroup=topgroup_new)

    # Write the new file
    info = get_py4DSTEM_dataobject_info(fp_orig,topgroup_orig)
    for i in indices:
        data = read_py4DSTEM(fp_orig,ft='py4DSTEM',topgroup=topgroup_orig,data_id=i)
        _append(fp_new,data=data,topgroup=topgroup_new)

    # Delete the old file
    if delete:
        print("Deleting the old file...")
        remove(fp_orig)

    return



