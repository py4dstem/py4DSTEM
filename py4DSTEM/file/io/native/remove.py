import h5py
import numpy as np
from os import remove, rename
from os.path import exists, dirname, basename
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups
from .read_utils import get_py4DSTEM_dataobject_info
from .copy import copy

def remove(fp, data, topgroup='4DSTEM_experiment', delete=True):
    """
    Remove some subset of dataobjects from a py4DSTEM h5 file.

    Accepts:
        fp              path to the py4DSTEM .h5 file
        data            (int or list of ints) the index or indices or name of
                        the DataObjects to remove.
        topgroup        the toplevel group
        delete          (bool) if True, fully remove objects from the file.
                        Otherwise, just removes the links and names of these
                        objects, without releasing the storage space. If you've
                        already used delete=False and want to release the space,
                        run io.repack(fp). For more info, see the docstring for
                        io.append.
    """
    assert is_py4DSTEM_file(fp), "fp parameter must point to an existing py4DSTEM file."
    tgs = get_py4DSTEM_topgroups(fp)
    assert(topgroup in tgs), "Error: topgroup '{}' not found.".format(topgroup)
    if isinstance(data, (int,np.integer)):
        dataobjects = [data]
    else:
        dataobjects = data
    assert(all([isinstance(item,(int,np.integer)) for item in dataobjects])), "Error: data must be ints."

    info = get_py4DSTEM_dataobject_info(fp,topgroup)
    with h5py.File(fp,'a') as f:
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
        _fp = dirname(fp)+'_'+basename(fp)
        while exists(_fp):
            _fp = dirname(_fp)+'_'+basename(_fp)
        copy(fp,_fp,topgroup_orig=topgroup,topgroup_new=topgroup)
        rename(_fp,fp)



