# Utility functions

import h5py
import numpy as np

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
    try:
        topgroups = get_py4DSTEM_topgroups(fp)
        if len(topgroups)>0:
            return True
        else:
            return False
    except OSError:
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
        else:
            version_release = 0
        return version_major, version_minor, version_release

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

def get_N_dataobjects(fp, topgroup='4DSTEM_experiment'):
    """ Returns a 7-tuple of ints with the numbers of: DataCubes, CountedDataCubes,
        DiffractionSlices, RealSlices, PointLists, PointListArrays, total DataObjects.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(fp,'r') as f:
        assert(topgroup in f.keys()), "Error: unrecognized topgroup"
        N_dc = len(f[topgroup]['data/datacubes'].keys())
        N_cdc = len(f[topgroup]['data/counted_datacubes'].keys())
        N_ds = len(f[topgroup]['data/diffractionslices'].keys())
        N_rs = len(f[topgroup]['data/realslices'].keys())
        N_pl = len(f[topgroup]['data/pointlists'].keys())
        N_pla = len(f[topgroup]['data/pointlistarrays'].keys())
        N_do = N_dc+N_cdc+N_ds+N_rs+N_pl+N_pla
        return N_dc,N_cdc,N_ds,N_rs,N_pl,N_pla,N_do

def get_py4DSTEM_dataobject_info(fp, topgroup='4DSTEM_experiment'):
    """ Returns a numpy structured array with basic metadata for all contained dataobjects.
        Keys for the info array are: 'index','type','shape','name'.
    """
    assert(is_py4DSTEM_file(fp)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(fp,'r') as f:
        assert(topgroup in f.keys()), "Error: unrecognized topgroup"
    i = 0
    l_md = []
    with h5py.File(fp,'r') as f:
        grp_dc = f[topgroup+'/data/datacubes/']
        grp_cdc = f[topgroup+'/data/counted_datacubes/']
        grp_ds = f[topgroup+'/data/diffractionslices/']
        grp_rs = f[topgroup+'/data/realslices/']
        grp_pl = f[topgroup+'/data/pointlists/']
        grp_pla = f[topgroup+'/data/pointlistarrays/']
        N = len(grp_dc)+len(grp_cdc)+len(grp_ds)+len(grp_rs)+len(grp_pl)+len(grp_pla)
        info = np.zeros(N,dtype=[('index',int),('type','U16'),('shape',tuple),('name','U64')])
        for name in sorted(grp_dc.keys()):
            shape = grp_dc[name+'/data/'].shape
            dtype = 'DataCube'
            info[i] = i,dtype,shape,name
            i += 1
        for name in sorted(grp_cdc.keys()):
            # TODO
            shape = grp_cdc[name+'/data/'].shape
            dtype = 'CountedDataCube'
            info[i] = i,dtype,shape,name
            i += 1
        for name in sorted(grp_ds.keys()):
            shape = grp_ds[name+'/data/'].shape
            dtype = 'DiffractionSlice'
            info[i] = i,dtype,shape,name
            i += 1
        for name in sorted(grp_rs.keys()):
            shape = grp_rs[name+'/data/'].shape
            dtype = 'RealSlice'
            info[i] = i,dtype,shape,name
            i += 1
        for name in sorted(grp_pl.keys()):
            coordinates = list(grp_pl[name].keys())
            length = grp_pl[name+'/'+coordinates[0]+'/data'].shape[0]
            shape = (len(coordinates),length)
            dtype = 'PointList'
            info[i] = i,dtype,shape,name
            i += 1
        for name in sorted(grp_pla.keys()):
            ar_shape = grp_pla[name+'/data'].shape
            N_coords = len(grp_pla[name+'/data'][0,0].dtype)
            shape = (ar_shape[0],ar_shape[1],N_coords,-1)
            dtype = 'PointListArray'
            info[i] = i,dtype,shape,name
            i += 1

    return info



