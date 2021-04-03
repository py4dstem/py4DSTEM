# Utility functions

import h5py
import numpy as np
from .read_utils import is_py4DSTEM_file

def get_py4DSTEM_dataobject_info(filepath, topgroup='4DSTEM_experiment'):
    """ Returns a numpy structured array with basic metadata for all contained dataobjects.
        Keys for the info array are: 'index','type','shape','name'.
    """
    assert(is_py4DSTEM_file(filepath)), "Error: not recognized as a py4DSTEM file"
    with h5py.File(filepath,'r') as f:
        assert(topgroup in f.keys()), "Error: unrecognized topgroup"
    i = 0
    l_md = []
    with h5py.File(filepath,'r') as f:
        grp_dc = f[topgroup+'/data/datacubes/']
        grp_cdc = f[topgroup+'/data/counted_datacubes/']
        grp_ds = f[topgroup+'/data/diffractionslices/']
        grp_rs = f[topgroup+'/data/realslices/']
        grp_pl = f[topgroup+'/data/pointlists/']
        grp_pla = f[topgroup+'/data/pointlistarrays/']
        grp_coords = f[topgroup+'/data/coordinates/']
        N = len(grp_dc)+len(grp_cdc)+len(grp_ds)+len(grp_rs)+len(grp_pl)+len(grp_pla)+len(grp_coords)
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
        for name in sorted(grp_coords.keys()):
            shape=0 #TODO?
            dtype = 'Coordinates'
            info[i] = i,dtype,shape,name
            i += 1

    return info



