# Utility functions

import h5py
import numpy as np
from .read_utils import is_py4DSTEM_file

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
        grp_ds = f[topgroup+'/data/diffractionslices/']
        grp_rs = f[topgroup+'/data/realslices/']
        grp_pl = f[topgroup+'/data/pointlists/']
        grp_pla = f[topgroup+'/data/pointlistarrays/']
        N = len(grp_dc)+len(grp_ds)+len(grp_rs)+len(grp_pl)+len(grp_pla)
        info = np.zeros(N,dtype=[('index',int),('type','U16'),('shape',tuple),('name','U64')])
        for name in sorted(grp_dc.keys()):
            shape = grp_dc[name+'/data/'].shape
            dtype = 'DataCube'
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
            l = list(grp_pla[name])
            ar = np.array([l[j].split('_') for j in range(len(l))]).astype(int)
            ar_shape = (np.max(ar[:,0])+1,np.max(ar[:,1])+1)
            N_coords = len(list(grp_pla[name+'/0_0']))
            shape = (ar_shape[0],ar_shape[1],N_coords,-1)
            dtype = 'PointListArray'
            info[i] = i,dtype,shape,name
            i += 1

    return info


