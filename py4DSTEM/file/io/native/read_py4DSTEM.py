# File reader for files written by py4DSTEM v0.9.0+

import h5py
import numpy as np
from os.path import splitext
from .read_utils import is_py4DSTEM_file, get_py4DSTEM_topgroups, get_py4DSTEM_version, version_is_geq
from ...datastructure import DataCube, CountedDataCube, DiffractionSlice, RealSlice
from ...datastructure import PointList, PointListArray
from ....process.utils import tqdmnd


def read_py4DSTEM(fp, **kwargs):
    """
    File reader for files written by py4DSTEM v0.9.0+.  Precise behavior is detemined by which
    arguments are passed -- see below.

    Accepts:
        filepath    str or Path     When passed a filepath only, this function checks if the path
                                    points to a valid py4DSTEM file, then prints its contents to screen.
        data        int/str/list    Specifies which data to load. Use integers to specify the
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
        Variable - see above.       If multiple arguments which have return values are specified,
                                    they're returned in the order of the arguments above - i.e.
                                    load=[0,1,2],metadata=True will return a length two tuple, the
                                    first element being a list of 3 DataObject instances and the
                                    second a MetaData instance.
    """
    assert(is_py4DSTEM_file(fp)), "Error: {} isn't recognized as a py4DSTEM file.".format(fp)
    version = get_py4DSTEM_version(fp)
    if not version_is_geq(version,(0,9,0)):
        return read_py4DSTEM_legacy(fp, **kwargs)

    # For HDF5 files containing multiple valid EMD type 2 files, disambiguate desired data
    tgs = get_py4DSTEM_topgroups(fp)
    if 'topgroup' in kwargs.keys():
        tg = kwargs.keys['topgroup']
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
            return

    # Triage - determine what needs doing
    _data = 'data' in kwargs.keys()
    _metadata = 'metadata' in kwargs.keys()
    _log = 'log' in kwargs.keys()

    # Validate inputs
    if _data:
        data = kwargs['data']
        assert(isinstance(data,(int,str,list,tuple))), "Error: data must be specified with strings or integers only."
        if not isinstance(data,(int,str)):
            assert(all([isinstance(d,(int,str)) for d in data])), "Error: data must be specified with strings or integers only."
    if _metadata:
        assert(isinstance(kwargs.keys['metdata'],bool))
        _metadata = kwargs.keys['metadata']
    if _metadata:
        assert(isinstance(kwargs.keys['metdata'],bool))
        _log = kwargs.keys['log']

    # Perform requested operations
    if not (_data or _metadata or _log):
        print_py4DSTEM_file(fp,tg)
        return
    if _data:
        loaded_data = get_data(fp,tg,data,**kwargs)
    if _metadata:
        md = get_metadata(fp,tg)
    if _log:
        write_log(fp,tg)

    # Return
    if (_data and _metadata):
        return data,metadata
    elif _data:
        return loaded_data
    elif _metadata:
        return md
    else:
        return


############ Helper functions ############

def print_py4DSTEM_file(fp,tg):
    """ Accepts a fp to a valid py4DSTEM file and prints to screen the file contents.
    """
    i = 0
    l_md = []
    with h5py.File(fp) as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_cdc = f[tg+'/data/counteddatacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlists/']
        for name in sorted(grp_dc.keys()):
            shape = grp_dc[name+'/data/'].shape
            dtype = 'DataCube'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1
        for name in sorted(grp_cdc.keys()):
            # TODO
            shape = grp_cdc[name+'/data/'].shape
            dtype = 'CountedDataCube'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1
        for name in sorted(grp_ds.keys()):
            shape = grp_ds[name+'/data/'].shape
            dtype = 'DiffractionSlice'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1
        for name in sorted(grp_rs.keys()):
            shape = grp_dc[name+'/data/'].shape
            dtype = 'RealSlice'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1
        for name in sorted(grp_pl.keys()):
            coordinates = list(grp_pl[name].keys())
            length = grp_pl[name+'/'+coordinates[0]+'data'].shape[0]
            shape = (len(coordinates),length)
            dtype = 'PointList'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1
        for name in sorted(grp_pla.keys()):
            shape = grp_pla[name+'/data'].shape
            dtype = 'PointListArray'
            d = {'index':i,'type':dtype,'shape':shape,'name':name}
            l_md.append(d)
            i += 1

    print("{:^8}{:^18}{:^18}{:^54}".format('Index', 'Type', 'Shape', 'Name'))
    print("{:^8}{:^18}{:^18}{:^54}".format('-----', '----', '-----', '----'))
    for d in l_md:
        print("{:^8}{:^18}{:^18}{:^54}".format(d['index'],d['type'],d['shape'],d['name']))

    return

def get_data(fp,tg,data_id):
    """ Accepts a fp to a valid py4DSTEM file and an int/str/list specifying data, and returns the data.
    """
    if isinstance(data_id,int):
        return get_data_from_int(fp,tg,data_id)
    elif isinstance(data_id,str):
        return get_data_from_str(fp,tg,data_id)
    else:
        return get_data_from_list(fp,tg,data_id)

def get_data_from_int(fp,tg,data_id):
    """ Accepts a fp to a valid py4DSTEM file and an integer specifying data, and returns the data.
    """
    assert(isinstance(data_id,int))
    with h5py.File(fp) as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_cdc = f[tg+'/data/counteddatacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlists/']
        grps = [grp_dc,grp_cdc,grp_ds,grp_rs,grp_pl,grp_pla]

        Ns = np.cumsum([len(grp.keys()) for grp in grps])
        i = np.nonzero(data_id<Ns)[0][0]
        grp = grps[i]
        N = data_id-Ns[i]
        name = sorted(grp.keys())[N]

        grp_data = f[grp.name+'/'+name]
        data = get_data_from_grp(grp_data)

    return data

def get_data_from_str(fp,tg,data_id):
    """ Accepts a fp to a valid py4DSTEM file and a string specifying data, and returns the data.
    """
    assert(isinstance(data_id,str))
    with h5py.File(fp) as f:
        grp_dc = f[tg+'/data/datacubes/']
        grp_cdc = f[tg+'/data/counteddatacubes/']
        grp_ds = f[tg+'/data/diffractionslices/']
        grp_rs = f[tg+'/data/realslices/']
        grp_pl = f[tg+'/data/pointlists/']
        grp_pla = f[tg+'/data/pointlists/']
        grps = [grp_dc,grp_cdc,grp_ds,grp_rs,grp_pl,grp_pla]

        l_dc = list(grp_dc.keys())
        l_cdc = list(grp_cdc.keys())
        l_ds = list(grp_ds.keys())
        l_rs = list(grp_rs.keys())
        l_pl = list(grp_pl.keys())
        l_pla = list(grp_pla.keys())
        names = l_dc+l_cdc+l_ds+l_rs+l_pl+l_pla

        inds = [i for i,name in enumerate(name) if name==data_id]
        assert(len(inds)!=0), "Error: no data named {} found.".format(data_id)
        assert(len(inds)<2), "Error: multiple data blocks named {} found.".format(data_id)
        ind = inds[0]

        Ns = np.cumsum([len(grp.keys()) for grp in grps])
        i_grp = np.nonzero(ind<Ns)[0][0]
        grp = grps[i_grp]

        grp_data = f[grp.name+'/'+data_id]
        data = get_data_from_grp(grp_data)

    return data

def get_data_from_list(fp,tg,data_id):
    """ Accepts a fp to a valid py4DSTEM file and a list or tuple specifying data, and returns the data.
    """
    assert(isinstance(data_id,(list,tuple)))
    assert(all([isinstance(d,(int,str)) for d in data]))
    data = []
    for el in data_id:
        if isinstance(el,int):
            data.append(get_data_from_int)
        elif isinstance(el,str):
            data.append(get_data_from_str)
        else:
            raise Exception("Data must be specified with strings or integers only.")
    return data

def get_data_from_grp(g):
    """ Accepts an h5py Group corresponding to a single dataobject in an open, correctly formatted H5 file,
        and returns a py4DSTEM DataObject.
    """
    dtype = g.name.split('/')[-2]
    if dtype == 'datacubes':
        return get_datacube_from_grp(g)
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

def get_datacube_from_grp(g):
    """ Accepts an h5py Group corresponding to a single datacube in an open, correctly formatted H5 file,
        and returns a DataCube.
    """
    return #TODO

def get_counted_datacube_from_grp(g):
    """ Accepts an h5py Group corresponding to a counted datacube in an open, correctly formatted H5 file,
        and returns a CountedDataCube.
    """
    return #TODO

def get_diffractionslice_from_grp(g):
    """ Accepts an h5py Group corresponding to a diffractionslice in an open, correctly formatted H5 file,
        and returns a DiffractionSlice.
    """
    return #TODO

def get_realslice_from_grp(g):
    """ Accepts an h5py Group corresponding to a realslice in an open, correctly formatted H5 file,
        and returns a RealSlice.
    """
    return #TODO

def get_pointlist_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlist in an open, correctly formatted H5 file,
        and returns a PointList.
    """
    return #TODO

def get_pointlistarray_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlistarray in an open, correctly formatted H5 file,
        and returns a PointListArray.
    """
    return #TODO


def get_metadata(fp,tg):
    """ Accepts a fp to a valid py4DSTEM file, and return a dictionary with its metadata.
    """
    return #TODO

def write_log(fp,tg):
    """ Accepts a fp to a valid py4DSTEM file, then prints its processing log to splitext(fp)[0]+'.log'.
    """
    return #TODO


