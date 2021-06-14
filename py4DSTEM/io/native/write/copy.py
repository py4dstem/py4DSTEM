# Copy a py4DSTEM file, or some subset of a py4DSTEM file

import h5py
import numpy as np
from os import remove, rename
from os.path import exists, dirname, basename
from .write import save
from ._append import _append
from ..read import is_py4DSTEM_file, get_py4DSTEM_topgroups
from ..read import get_N_dataobjects, read_py4DSTEM
from ..read.read_v0_12 import get_py4DSTEM_dataobject_info
from ...datastructure import DataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray
from ...datastructure import DataObject

def copy(filepath_orig, filepath_new, indices=None, delete=False,
         topgroup_orig='4DSTEM_experiment',topgroup_new='4DSTEM_experiment'):
    """
    Copies DataObjects specified by indices from the py4DSTEM .h5 file at filepath_orig
    to avnew file at filepath_new.

    Accepts:
        filepath_orig       path to an existing py4DSTEM .h5 file to copy
        filepath_new        path to the new file to write
        indices             if None, copy the entire file.
                            Otherwise must be either an int or a list of ints.
                            Copies the DataObjects corresponding to these indices
                            in the original file.
        topgroup_orig       The toplevel group for the original file
        topgroup_new        and for the new file
    """
    assert(is_py4DSTEM_file(filepath_orig)), "Error: not recognized as a py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(filepath_orig)
    assert(topgroup_orig in tgs), "Error: topgroup '{}' not found.".format(topgroup)
    if exists(filepath_new):
        assert(is_py4DSTEM_file(filepath_new)), "Error: a file with the target filename already exists, and is not recognized as a py4DSTEM file."
        tgs = get_py4DSTEM_topgroups(filepath_new)
        if topgroup_new in tgs:
            raise Exception('A file with the target filename exists and already contains a toplevel group with the specified topgroup name.')

    # Determine what needs to be copied
    if indices is None:
        _,_,_,_,_,_,_,N = get_N_dataobjects(filepath_orig,topgroup_orig)
        indices = list(np.arange(N))
    else:
        if isinstance(indices, int):
            indices = [indices]
        assert(all([isinstance(item,(int,np.integer)) for item in indices])), "Error: indices must be ints."

    # Make infrastructure for the new file
    save(filepath_new,[],topgroup=topgroup_new)

    # Write the new file
    info = get_py4DSTEM_dataobject_info(filepath_orig,topgroup_orig)
    for i in indices:
        data = read_py4DSTEM(filepath_orig,ft='py4DSTEM',topgroup=topgroup_orig,data_id=i)
        _append(filepath_new,data=data,topgroup=topgroup_new)

    # Delete the old file
    if delete:
        print("Deleting the old file...")
        remove(filepath_orig)

    return



