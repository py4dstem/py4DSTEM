# Utility functions

def is_py4DSTEM_file(fp, topgroup='4DSTEM_experiment'):
    """ Returns True iff fp points to a py4DSTEM formatted (EMD type 2) file.
    """
    with h5py.File(fp,'r') as f:
        if topgroup in f.keys():
            if 'emd_group_type' in f[topgroup].attrs:
                if f[topgroup].attrs['emd_group_type']==2:
                    return True
    return False

def get_py4DSTEM_version(fp):
    """ Returns the version (major,minor,release) of a py4DSTEM file.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(fp,'r') as f:
        version_major = int(f['4DSTEM_experiment'].attrs['version_major'])
        version_minor = int(f['4DSTEM_experiment'].attrs['version_minor'])

        version_release = int(f['4DSTEM_experiment'].attrs['version_release'])
    return version_major, version_minor, version_release

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

def version_is_geq_majorminor(current,minimum):
    """ Returns True iff current version (major,minor) is greater than or equal to minimum."
    """
    if current[0]>minimum[0]:
        return True
    elif current[0]==minimum[0]:
        if current[1]>=minimum[1]:
            return True
        else:
            return False
    else:
        return False

def get_UUID(fp, topgroup='4DSTEM_experiment'):
    """ Returns the UUID of a py4DSTEM file, or if unavailable returns -1.
    """
    with h5py.File(fp,'r') as f:
        if topgroup in f.keys():
            if 'UUID' in f[topgroup].attrs:
                return f[topgroup].attrs['UUID']
    return -1



