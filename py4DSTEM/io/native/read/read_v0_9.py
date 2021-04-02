# Reader for py4DSTEM v0.9 - v0.11 files

import h5py
import numpy as np
from os.path import splitext
from ..read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_py4DSTEM_version, version_is_geq
from ...datastructure import DataCube, CountedDataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray
from ....process.utils import tqdmnd
from .read_utils_v0_9 import get_py4DSTEM_dataobject_info



def function(**kwargs):

    # If metadata is requested
    if metadata:
        return metadata_from_h5(filepath, tg)

    # If data is requested
    elif 'data_id' in kwargs.keys():
        data_id = kwargs['data_id']
        assert(isinstance(data_id,(int,np.int_,str,list,tuple))), "Error: data must be specified with strings or integers only."
        if not isinstance(data_id,(int,np.int_,str)):
            assert(all([isinstance(d,(int,np.int_,str)) for d in data_id])), "Error: data must be specified with strings or integers only."

        # Parse optional arguments
        if 'mem' in kwargs.keys():
            mem = kwargs['mem']
            assert(mem in ('RAM','MEMMAP'))
        else:
            mem='RAM'
        if 'binfactor' in kwargs.keys():
            binfactor = kwargs['binfactor']
            assert(isinstance(binfactor,(int,np.int_)))
        else:
            binfactor=1
        if 'dtype' in kwargs.keys():
            bindtype = kwargs['dtype']
            assert(isinstance(bindtype,type))
        else:
            bindtype = None

        return get_data(filepath,tg,data_id,mem,binfactor,bindtype)

    # If no data is requested
    else:
        print_py4DSTEM_file(filepath,tg)
        return


############ Helper functions ############

def print_py4DSTEM_file(filepath,tg):
    """ Accepts a filepath to a valid py4DSTEM file and prints to screen the file contents.
    """
    info = get_py4DSTEM_dataobject_info(filepath,tg)
    print("{:10}{:18}{:24}{:54}".format('Index', 'Type', 'Shape', 'Name'))
    print("{:10}{:18}{:24}{:54}".format('-----', '----', '-----', '----'))
    for el in info:
        print("  {:8}{:18}{:24}{:54}".format(str(el['index']),str(el['type']),str(el['shape']),str(el['name'])))
    return

def get_data(filepath,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a filepath to a valid py4DSTEM file and an int/str/list specifying data, and returns the data.
    """
    if isinstance(data_id,(int,np.int_)):
        return get_data_from_int(filepath,tg,data_id,mem=mem,binfactor=binfactor,bindtype=bindtype)
    elif isinstance(data_id,str):
        return get_data_from_str(filepath,tg,data_id,mem=mem,binfactor=binfactor,bindtype=bindtype)
    else:
        return get_data_from_list(filepath,tg,data_id)

def get_data_from_int(filepath,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a filepath to a valid py4DSTEM file and an integer specifying data, and returns the data.
    """
    assert(isinstance(data_id,(int,np.int_)))
    with h5py.File(filepath,'r') as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_cdc = f[tg+'/data/counted_datacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlistarrays/']
        grps = [grp_dc,grp_cdc,grp_ds,grp_rs,grp_pl,grp_pla]

        Ns = np.cumsum([len(grp.keys()) for grp in grps])
        i = np.nonzero(data_id<Ns)[0][0]
        grp = grps[i]
        N = data_id-Ns[i]
        name = sorted(grp.keys())[N]

        grp_data = f[grp.name+'/'+name]
        data = get_data_from_grp(grp_data,mem=mem,binfactor=binfactor,bindtype=bindtype)

    return data

def get_data_from_str(filepath,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a filepath to a valid py4DSTEM file and a string specifying data, and returns the data.
    """
    assert(isinstance(data_id,str))
    with h5py.File(filepath,'r') as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_cdc = f[tg+'/data/counted_datacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlistarrays/']
        grps = [grp_dc,grp_cdc,grp_ds,grp_rs,grp_pl,grp_pla]

        l_dc = list(grp_dc.keys())
        l_cdc = list(grp_cdc.keys())
        l_ds = list(grp_ds.keys())
        l_rs = list(grp_rs.keys())
        l_pl = list(grp_pl.keys())
        l_pla = list(grp_pla.keys())
        names = l_dc+l_cdc+l_ds+l_rs+l_pl+l_pla

        inds = [i for i,name in enumerate(names) if name==data_id]
        assert(len(inds)!=0), "Error: no data named {} found.".format(data_id)
        assert(len(inds)<2), "Error: multiple data blocks named {} found.".format(data_id)
        ind = inds[0]

        Ns = np.cumsum([len(grp.keys()) for grp in grps])
        i_grp = np.nonzero(ind<Ns)[0][0]
        grp = grps[i_grp]

        grp_data = f[grp.name+'/'+data_id]
        data = get_data_from_grp(grp_data,mem=mem,binfactor=binfactor,bindtype=bindtype)

    return data

def get_data_from_list(filepath,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a filepath to a valid py4DSTEM file and a list or tuple specifying data, and returns the data.
    """
    assert(isinstance(data_id,(list,tuple)))
    assert(all([isinstance(d,(int,np.int_,str)) for d in data_id]))
    data = []
    for el in data_id:
        if isinstance(el,(int,np.int_)):
            data.append(get_data_from_int(filepath,tg,data_id=el,mem=mem,binfactor=binfactor,bindtype=bindtype))
        elif isinstance(el,str):
            data.append(get_data_from_str(filepath,tg,data_id=el,mem=mem,binfactor=binfactor,bindtype=bindtype))
        else:
            raise Exception("Data must be specified with strings or integers only.")
    return data

def get_data_from_grp(g,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts an h5py Group corresponding to a single dataobject in an open, correctly formatted H5 file,
        and returns a py4DSTEM DataObject.
    """
    dtype = g.name.split('/')[-2]
    if dtype == 'datacubes':
        return get_datacube_from_grp(g,mem,binfactor,bindtype)
    elif dtype == 'counted_datacubes':
        return get_counted_datacube_from_grp(g)
    elif dtype == 'diffractionslices':
        return get_diffractionslice_from_grp(g)
    elif dtype == 'realslices':
        return get_realslice_from_grp(g)
    elif dtype == 'pointlists':
        return get_pointlist_from_grp(g)
    elif dtype == 'pointlistarrays':
        return get_pointlistarray_from_grp(g)
    else:
        raise Exception('Unrecognized data object type {}'.format(dtype))

def get_datacube_from_grp(g,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts an h5py Group corresponding to a single datacube in an open, correctly formatted H5 file,
        and returns a DataCube.
    """
    # TODO: add memmapping, binning
    data = np.array(g['data'])
    name = g.name.split('/')[-1]
    return DataCube(data=data,name=name)

def get_counted_datacube_from_grp(g):
    """ Accepts an h5py Group corresponding to a counted datacube in an open, correctly formatted H5 file,
        and returns a CountedDataCube.
    """
    return #TODO

def get_diffractionslice_from_grp(g):
    """ Accepts an h5py Group corresponding to a diffractionslice in an open, correctly formatted H5 file,
        and returns a DiffractionSlice.
    """
    data = np.array(g['data'])
    name = g.name.split('/')[-1]
    Q_Nx,Q_Ny = data.shape[:2]
    if len(data.shape)==2:
        return DiffractionSlice(data=data,Q_Nx=Q_Nx,Q_Ny=Q_Ny,name=name)
    else:
        lbls = g['dim3']
        if('S' in lbls.dtype.str): # Checks if dim3 is composed of fixed width C strings
            with lbls.astype('S64'):
                lbls = lbls[:]
            lbls = [lbl.decode('UTF-8') for lbl in lbls]
        return DiffractionSlice(data=data,Q_Nx=Q_Nx,Q_Ny=Q_Ny,name=name,slicelabels=lbls)

def get_realslice_from_grp(g):
    """ Accepts an h5py Group corresponding to a realslice in an open, correctly formatted H5 file,
        and returns a RealSlice.
    """
    data = np.array(g['data'])
    name = g.name.split('/')[-1]
    R_Nx,R_Ny = data.shape[:2]
    if len(data.shape)==2:
        return RealSlice(data=data,R_Nx=R_Nx,R_Ny=R_Ny,name=name)
    else:
        lbls = g['dim3']
        if('S' in lbls.dtype.str): # Checks if dim3 is composed of fixed width C strings
            with lbls.astype('S64'):
                lbls = lbls[:]
            lbls = [lbl.decode('UTF-8') for lbl in lbls]
        return RealSlice(data=data,R_Nx=R_Nx,R_Ny=R_Ny,name=name,slicelabels=lbls)

def get_pointlist_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlist in an open, correctly formatted H5 file,
        and returns a PointList.
    """
    name = g.name.split('/')[-1]
    coordinates = []
    coord_names = list(g.keys())
    length = len(g[coord_names[0]+'/data'])
    if length==0:
        for coord in coord_names:
            coordinates.append((coord,None))
    else:
        for coord in coord_names:
            dtype = type(g[coord+'/data'][0])
            coordinates.append((coord, dtype))
    data = np.zeros(length,dtype=coordinates)
    for coord in coord_names:
        data[coord] = np.array(g[coord+'/data'])
    return PointList(data=data,coordinates=coordinates,name=name)

def get_pointlistarray_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlistarray in an open, correctly formatted H5 file,
        and returns a PointListArray.
    """
    name = g.name.split('/')[-1]
    dset = g['data']
    shape = g['data'].shape
    coordinates = g['data'][0,0].dtype
    pla = PointListArray(coordinates=coordinates,shape=shape,name=name)
    for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
        pla.get_pointlist(i,j).add_dataarray(dset[i,j])
    return pla



