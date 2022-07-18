# Utility functions

import h5py
import numpy as np

def get_py4DSTEM_topgroups(filepath):
    """ Returns a list of toplevel groups in an HDF5 file which are valid py4DSTEM file trees.
    """
    topgroups = []
    with h5py.File(filepath,'r') as f:
        for key in f.keys():
            if 'emd_group_type' in f[key].attrs:
                if f[key].attrs['emd_group_type']==2:
                    topgroups.append(key)
    return topgroups

def is_py4DSTEM_file(filepath):
    """ Returns True iff filepath points to a py4DSTEM formatted (EMD type 2) file.
    """
    try:
        topgroups = get_py4DSTEM_topgroups(filepath)
        if len(topgroups)>0:
            return True
        else:
            return False
    except OSError:
        return False

def get_py4DSTEM_version(filepath, topgroup='4DSTEM_experiment'):
    """ Returns the version (major,minor,release) of a py4DSTEM file.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        version_major = int(f[topgroup].attrs['version_major'])
        version_minor = int(f[topgroup].attrs['version_minor'])
        if 'version_release' in f[topgroup].attrs.keys():
            version_release = int(f[topgroup].attrs['version_release'])
        else:
            version_release = 0
        return version_major, version_minor, version_release

def get_UUID(filepath, topgroup='4DSTEM_experiment'):
    """ Returns the UUID of a py4DSTEM file, or if unavailable returns -1.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        if topgroup in f.keys():
            if 'UUID' in f[topgroup].attrs:
                return f[topgroup].attrs['UUID']
    return -1

def version_is_geq(current,minimum):
    """ Returns True iff current version (major,minor,release) is greater than or equal to minimum."
    """
    if current[0]>minimum[0]:
        return True
    elif current[0]==minimum[0]:
        if current[1]>minimum[1]:
            return True
        elif current[1]==minimum[1]:
            if current[2]>=minimum[2]:
                return True
        else:
            return False
    else:
        return False

def get_N_dataobjects(filepath, topgroup='4DSTEM_experiment'):
    """ Returns a 7-tuple of ints with the numbers of: DataCubes, CountedDataCubes,
        DiffractionSlices, RealSlices, PointLists, PointListArrays, total DataObjects.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        assert(topgroup in f.keys()), "Error: unrecognized topgroup"
        N_dc = len(f[topgroup]['data/datacubes'].keys())
        N_cdc = len(f[topgroup]['data/counted_datacubes'].keys())
        N_ds = len(f[topgroup]['data/diffractionslices'].keys())
        N_rs = len(f[topgroup]['data/realslices'].keys())
        N_pl = len(f[topgroup]['data/pointlists'].keys())
        N_pla = len(f[topgroup]['data/pointlistarrays'].keys())
        try:
            N_coords = len(f[topgroup]['data/coordinates'].keys())
        except:
            N_coords = 0
        N_do = N_dc+N_cdc+N_ds+N_rs+N_pl+N_pla+N_coords
        return N_dc,N_cdc,N_ds,N_rs,N_pl,N_pla,N_coords,N_do


