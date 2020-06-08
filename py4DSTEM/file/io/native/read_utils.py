# Utility functions

import h5py

def get_py4DSTEM_topgroups(fp):
    """ Returns a list of toplevel groups in an HDF5 file which are valid py4DSTEM file trees.
    """
    topgroups = []
    with h5py.File(fp,'r') as f:
        for key in f.keys():
            if 'emd_group_type' in f[key].attrs:
                if f[key].attrs['emd_group_type']==2:
                    topgroups.append(key)
    return topgroups


def is_py4DSTEM_file(fp):
    """ Returns True iff fp points to a py4DSTEM formatted (EMD type 2) file.
    """
    topgroups = get_py4DSTEM_topgroups(fp)
    if len(topgroups)>0:
        return True
    else:
        return False

def get_py4DSTEM_version(fp, topgroup='4DSTEM_experiment'):
    """ Returns the version (major,minor,release) of a py4DSTEM file.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(fp,'r') as f:
        version_major = int(f[topgroup].attrs['version_major'])
        version_minor = int(f[topgroup].attrs['version_minor'])
        if 'version_release' in f[topgroup].attrs.keys():
            version_release = int(f[topgroup].attrs['version_release'])
            return version_major, version_minor, version_release
        else:
            return version_major, version_minor, 0

def get_UUID(fp, topgroup='4DSTEM_experiment'):
    """ Returns the UUID of a py4DSTEM file, or if unavailable returns -1.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(fp,'r') as f:
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





