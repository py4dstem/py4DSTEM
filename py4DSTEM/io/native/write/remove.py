import h5py
import numpy as np
from os import rename
from os.path import exists, dirname, basename
from .copy import copy
from ..read import is_py4DSTEM_file, get_py4DSTEM_topgroups
from ..read.read_v0_12 import get_py4DSTEM_dataobject_info

def remove(filepath, data, topgroup='4DSTEM_experiment', delete=True):
    """
    Remove some subset of dataobjects from a py4DSTEM h5 file.

    Accepts:
        filepath        path to the py4DSTEM .h5 file
        data            (int or list of ints) the index or indices or name of
                        the DataObjects to remove.
        topgroup        the toplevel group
        delete          (bool) if True, fully remove objects from the file.
                        Otherwise, just removes the links and names of these
                        objects, without releasing the storage space. If you've
                        already used delete=False and want to release the space,
                        run io.native.repack(filepath). For more info, see the docstring
                        for io.native.append.
    """
    assert is_py4DSTEM_file(filepath), "filepath parameter must point to an existing py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(filepath)
    assert(topgroup in tgs), "Error: topgroup '{}' not found.".format(topgroup)
    if isinstance(data, (int,np.integer)):
        dataobjects = [data]
    else:
        dataobjects = data
    assert(all([isinstance(item,(int,np.integer)) for item in dataobjects])), "Error: data must be ints."

    info = get_py4DSTEM_dataobject_info(filepath,topgroup)
    with h5py.File(filepath,'a') as f:
        for i in dataobjects:
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
    f.close()

    if delete:
        _filepath = dirname(filepath)+'_'+basename(filepath)
        while exists(_filepath):
            _filepath = dirname(_filepath)+'_'+basename(_filepath)
        copy(filepath,_filepath,topgroup_orig=topgroup,topgroup_new=topgroup)
        rename(_filepath,filepath)



