# Remove existing DataObjects from a py4DSTEM formatted .h5 file.
# 
# See filestructure.txt for a description of the file structure.

import h5py
import numpy as np
from .filebrowser import is_py4DSTEM_file, FileBrowser

from ..log import log, Logger
logger = Logger()

@log
def remove_from_index_list(indices, filepath):
    """
    Remove existing dataobjects from a py4DSTEM h5 file.

    Accepts:
        indices             (list of ints) the indices of the DataObjects to remove, in a FileBrowser
                            associated with filepath
        filepath            path to the py4DSTEM .h5 file
    """
    assert all([isinstance(item,(int,np.integer)) for item in indices]), "Error: indices must be ints."

    #### Get info about .h5 file and objects to delete ####
    print("Opening file {}...".format(filepath))
    assert is_py4DSTEM_file(filepath), "filepath parameter must point to an existing py4DSTEM file."
    browser = FileBrowser(filepath)
    if browser.version[0] == 0:
        assert browser.version[1] >= 3, "removing DataObjects from py4DSTEM files is supported in v0.3 and higher."
    names,types = [],[]
    for i in range(len(indices)):
        info = browser.get_dataobject_info(indices[i])
        names.append(info['name'])
        types.append(info['type'])
    browser.close()

    #### Open file for read/write ####
    f = h5py.File(filepath,"a")
    topgroup = get_py4DSTEM_topgroup(f)

    # Delete objects
    for i in range(len(indices)):
        name,objtype = names[i],types[i]
        if objtype == "DataCube":
            group = f[topgroup + 'data/datacubes']
        elif objtype == "RealSlice":
            group = f[topgroup + 'data/realslices']
        elif objtype == "DiffractionSlice":
            group = f[topgroup + 'data/diffractionslices']
        elif objtype == "PointList":
            group = f[topgroup + 'data/pointlists']
        elif objtype == "PointListArray":
            group = f[topgroup + 'data/pointlistarrays']
        else:
            raise ValueError("Unknown DataObject type {}".format(objtype))
        del group[name]

    ##### Finish and close #####
    print("Done.")
    f.close()

@log
def remove(dataobjects, filepath):
    """
    Remove existing dataobjects from a py4DSTEM h5 file.

    Accepts:
        dataobjects     (int or list of ints) the index or indices or name of the DataObjects to
                        remove. If an int or list of ints, indices are those of a FileBrowser
                        associated with filepath
        filepath        path to the py4DSTEM .h5 file
    """
    assert is_py4DSTEM_file(filepath), "filepath parameter must point to an existing py4DSTEM file."

    if isinstance(dataobjects, (int,np.integer)):
        remove_from_index_list([dataobjects], filepath)
    else:
        remove_from_index_list(dataobjects, filepath)


################### END OF REMOVE FUNCTIONS #####################
def get_py4DSTEM_topgroup(h5_file):
    """
    Accepts an open h5py File boject. Returns string of the top group name. 
    """
    if ('4DSTEM_experiment' in h5_file.keys()): # or ('4D-STEM_data' in h5_file.keys()) or ('4DSTEM_simulation' in h5_file.keys())):
        return '4DSTEM_experiment/'
    elif ('4DSTEM_simulation' in h5_file.keys()):
        return '4DSTEM_simulation/'
    else:
        return '4D-STEM_data/'



