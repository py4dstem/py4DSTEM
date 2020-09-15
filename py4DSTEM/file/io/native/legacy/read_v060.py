# File reader for files written by py4DSTEM v0.8.0 or older.

import h5py
import numpy as np
from os.path import splitext
from ..read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_py4DSTEM_version, version_is_geq
from ....datastructure import DataCube, CountedDataCube, DiffractionSlice, RealSlice
from ....datastructure import PointList, PointListArray
from .....process.utils import tqdmnd
from .read_utils_v060 import get_py4DSTEM_dataobject_info


def read_v060(fp, **kwargs):
    """
    File reader for files written by py4DSTEM v0.8.0.  Precise behavior is detemined by which
    arguments are passed -- see below.

    ***NOTE: this function has not yet been tested on all legacy py4DSTEM formats. Please report
    any problems by filing an issue on our github!

    Accepts:
        filepath    str or Path     When passed a filepath only, this function checks if the path
                                    points to a valid py4DSTEM file, then prints its contents to screen.
        data_id     int/str/list    Specifies which data to load. Use integers to specify the
                                    data index, or strings to specify data names. A list or
                                    tuple returns a list of DataObjects. Returns the specified data.
        topgroup     str            Stricty, a py4DSTEM file is considered to be
                                    everything inside a toplevel subdirectory within the
                                    HDF5 file, so that if desired one can place many py4DSTEM
                                    files inside a single H5.  In this case, when loading
                                    data, the topgroup argument is passed to indicate which
                                    py4DSTEM file to load. If an H5 containing multiple
                                    py4DSTEM files is passed without a topgroup specified,
                                    the topgroup names are printed to screen.
        metadata    bool            If True, returns a dictionary with the file metadata.
        log         bool            If True, writes the processing log to a plaintext file
                                    called splitext(fp)[0]+'.log'.
        mem         str             Only used if a single DataCube is loaded. In this case, mem
                                    specifies how the data should be stored; must be "RAM"
                                    or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default
                                    is "RAM".
        binfactor   int             Only used if a single DataCube is loaded. In this case,
                                    a binfactor of > 1 causes the data to be binned by this amount
                                    as it's loaded.
        dtype       dtype           Used when binning data, ignored otherwise. Defaults to whatever
                                    the type of the raw data is, to avoid enlarging data size. May be
                                    useful to avoid 'wraparound' errors.

    Returns:
        data,md                     The function always returns a length 2 tuple corresponding
                                    to data and md.  If no input arguments with return values (i.e.
                                    data, metadata), these will return None.  Otherwise, their return
                                    values are as described above. E.f. passing data=[0,1,2],metadata=True
                                    will return a length two tuple, the first element being a list of 3
                                    DataObject instances and the second a MetaData instance.
    """
    assert(is_py4DSTEM_file(fp)), "Error: {} isn't recognized as a py4DSTEM file.".format(fp)
    version = get_py4DSTEM_version(fp)
    print("py4DSTEM v{}.{}.{} file detected.".format(version[0],version[1],version[2]))

    # For HDF5 files containing multiple valid EMD type 2 files, disambiguate desired data
    tgs = get_py4DSTEM_topgroups(fp)
    if 'topgroup' in kwargs.keys():
        tg = kwargs['topgroup']
        assert(self.topgroup in topgroups), "Error: specified topgroup, {}, not found.".format(self.topgroup)
    else:
        if len(tgs)==1:
            tg = tgs[0]
        else:
            print("Multiple topgroups detected.  Please specify one by passing the 'topgroup' keyword argument.")
            print("")
            print("Topgroups found:")
            for tg in topgroups:
                print(tg)
            return None,None

    # Triage - determine what needs doing
    _data_id = 'data_id' in kwargs.keys()
    _metadata = 'metadata' in kwargs.keys()
    _log = 'log' in kwargs.keys()

    # Validate inputs
    if _data_id:
        data_id = kwargs['data_id']
        assert(isinstance(data_id,(int,str,list,tuple))), "Error: data must be specified with strings or integers only."
        if not isinstance(data_id,(int,str)):
            assert(all([isinstance(d,(int,str)) for d in data_id])), "Error: data must be specified with strings or integers only."
    if _metadata:
        assert(isinstance(kwargs.keys['metdata'],bool))
        _metadata = kwargs.keys['metadata']
    if _metadata:
        assert(isinstance(kwargs.keys['metdata'],bool))
        _log = kwargs.keys['log']

    # Parse optional arguments
    if 'mem' in kwargs.keys():
        mem = kwargs['mem']
        assert(mem in ('RAM','MEMMAP'))
    else:
        mem='RAM'
    if 'binfactor' in kwargs.keys():
        binfactor = kwargs['binfactor']
        assert(isinstance(binfactor,int))
    else:
        binfactor=1
    if 'dtype' in kwargs.keys():
        bindtype = kwargs['dtype']
        assert(isinstance(bindtype,type))
    else:
        bindtype = None

    # Perform requested operations
    if not (_data_id or _metadata or _log):
        print_py4DSTEM_file(fp,tg)
        return None,None
    if _data_id:
        loaded_data = get_data(fp,tg,data_id,mem,binfactor,bindtype)
    if _metadata:
        md = get_metadata(fp,tg)
    if _log:
        write_log(fp,tg)

    # Return
    if (_data_id and _metadata):
        return loaded_data,md
    elif _data_id:
        return loaded_data,None
    elif _metadata:
        return None,md
    else:
        return None,None


############ Helper functions ############

def print_py4DSTEM_file(fp,tg):
    """ Accepts a fp to a valid py4DSTEM file and prints to screen the file contents.
    """
    info = get_py4DSTEM_dataobject_info(fp,tg)
    print("{:10}{:18}{:24}{:54}".format('Index', 'Type', 'Shape', 'Name'))
    print("{:10}{:18}{:24}{:54}".format('-----', '----', '-----', '----'))
    for el in info:
        print("  {:8}{:18}{:24}{:54}".format(str(el['index']),str(el['type']),str(el['shape']),str(el['name'])))

    return

def get_data(fp,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a fp to a valid py4DSTEM file and an int/str/list specifying data, and returns the data.
    """
    if isinstance(data_id,int):
        return get_data_from_int(fp,tg,data_id,mem=mem,binfactor=binfactor,bindtype=bindtype)
    elif isinstance(data_id,str):
        return get_data_from_str(fp,tg,data_id,mem=mem,binfactor=binfactor,bindtype=bindtype)
    else:
        return get_data_from_list(fp,tg,data_id)

def get_data_from_int(fp,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a fp to a valid py4DSTEM file and an integer specifying data, and returns the data.
    """
    assert(isinstance(data_id,int))
    with h5py.File(fp,'r') as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlistarrays/']
        grps = [grp_dc,grp_ds,grp_rs,grp_pl,grp_pla]

        Ns = np.cumsum([len(grp.keys()) for grp in grps])
        i = np.nonzero(data_id<Ns)[0][0]
        grp = grps[i]
        N = data_id-Ns[i]
        name = sorted(grp.keys())[N]

        grp_data = f[grp.name+'/'+name]
        data = get_data_from_grp(grp_data,mem=mem,binfactor=binfactor,bindtype=bindtype)

    return data

def get_data_from_str(fp,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a fp to a valid py4DSTEM file and a string specifying data, and returns the data.
    """
    assert(isinstance(data_id,str))
    with h5py.File(fp,'r') as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlistarrays/']
        grps = [grp_dc,grp_ds,grp_rs,grp_pl,grp_pla]

        l_dc = list(grp_dc.keys())
        l_ds = list(grp_ds.keys())
        l_rs = list(grp_rs.keys())
        l_pl = list(grp_pl.keys())
        l_pla = list(grp_pla.keys())
        names = l_dc+l_ds+l_rs+l_pl+l_pla

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

def get_data_from_list(fp,tg,data_id,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts a fp to a valid py4DSTEM file and a list or tuple specifying data, and returns the data.
    """
    assert(isinstance(data_id,(list,tuple)))
    assert(all([isinstance(d,(int,str)) for d in data_id]))
    data = []
    for el in data_id:
        if isinstance(el,int):
            data.append(get_data_from_int(fp,tg,data_id=el,mem=mem,binfactor=binfactor,bindtype=bindtype))
        elif isinstance(el,str):
            data.append(get_data_from_str(fp,tg,data_id=el,mem=mem,binfactor=binfactor,bindtype=bindtype))
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
    l = list(g)
    ar = np.array([l[i].split('_') for i in range(len(l))]).astype(int)
    shape = (np.max(ar[:,0])+1,np.max(ar[:,1])+1)
    coord_names = list(g['0_0'])
    N = len(coord_names)
    coord_types = [type(np.array(g['0_0/'+coord_names[i]+'/data'])[0]) for i in range(N)]
    coordinates = [(coord_names[i],coord_types[i]) for i in range(N)]
    pla = PointListArray(coordinates=coordinates,shape=shape,name=name)
    for (i,j) in tqdmnd(range(shape[0]),range(shape[1]),desc="Reading PointListArray",unit="PointList"):
        g_pl = g[str(i)+'_'+str(j)]
        L = len(np.array(g_pl[coordinates[0][0]+'/data']))
        data = np.zeros(L,dtype=coordinates)
        for i in range(N):
            coord = coordinates[i][0]
            data[coord] = np.array(g_pl[coord+'/data'])
        pla.get_pointlist(i,j).add_dataarray(data)
    return pla




