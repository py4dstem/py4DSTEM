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
                                        tuple returns a list of DataObjects.  Note that mixed
                                        int/str lists are not supported. Returns the specified data.
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

        # In cases of H5 files containing multiple valid EMD type 2 files, disambiguate desired data
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
            assert(isinstance(data,(int,str,list,tuple)))
            if not isinstance(data,(int,str)):
                assert(all([isinstance(data[i],str) for i in range(len(data))]) or
                       all([isinstance(data[i],int) for i in range(len(data))])    )
        if _metadata:
            assert(isinstance(kwargs.keys['metdata'],bool))
            _metadata = kwargs.keys['metadata']
        if _metadata:
            assert(isinstance(kwargs.keys['metdata'],bool))
            _log = kwargs.keys['log']

        # Perform requested operations
        if not (_data or _metadata or _log):
            print_py4DSTEM_file(fp)
            return
        if _data:
            loaded_data = get_data(fp,data)
        if _metadata:
            md = get_metadata(fp)
        if _log:
            write_log(fp)

        # Return
        if (_data and _metadata):
            return data,metadata
        elif _data:
            return loaded_data
        elif _metadata:
            return md
        else:
            return


###### Helper functions ######

def print_py4DSTEM_file(fp):
    """ Accepts a fp to a valid py4DSTEM file and prints to screen the file contents.
    """
    metadata_ar = []



    return

def get_data(fp, data_id):
    """ Accepts a fp to a valid py4DSTEM file and an int/str/list specifying data, and returns the data.
    """
    return

def get_metadata(fp):
    """ Accepts a fp to a valid py4DSTEM file, and return a dictionary with its metadata.
    """
    return

def write_log(fp):
    """ Accepts a fp to a valid py4DSTEM file, then prints its processing log to splitext(fp)[0]+'.log'.
    """
    return





















        if objecttype == 'DataCube':
            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}







        obj_classes = ['DataCube','CountedDataCube','DiffractionSlice','RealSlice','PointList','PointListArray']
        obj_types = ['datacubes','counted_datacubes','diffractionslices','realslices','pointlists','pointlistarrays']
        obj_type_paths = [self.topgroup+'/data/'+obj_type for obj_type in obj_types]
        self.obj_lookup_ar = []
        index = 0
        for dtype,path in zip(obj_classes,obj_type_paths):
            

            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}


            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}


        self.N_datacubes = len(self.file[self.topgroup + '/data/datacubes'])
        self.N_counted = len(self.file[self.topgroup + '/data/counted_datacubes'])
        self.N_diffractionslices = len(self.file[self.topgroup + '/data/diffractionslices'])
        self.N_realslices = len(self.file[self.topgroup + '/data/realslices'])
        self.N_pointlists = len(self.file[self.topgroup + '/data/pointlists'])
        self.N_pointlistarrays = len(self.file[self.topgroup + '/data/pointlistarrays'])
        self.N_dataobjects = np.sum([self.N_datacubes, self.N_counted, self.N_diffractionslices, self.N_realslices, self.N_pointlists, self.N_pointlistarrays])

        self.dataobject_lookup_arr = []
        self.dataobject_lookup_arr += ['DataCube' for i in range(self.N_datacubes)]
        self.dataobject_lookup_arr += ['CountedDataCube' for i in range(self.N_counted)]
        self.dataobject_lookup_arr += ['DiffractionSlice' for i in range(self.N_diffractionslices)]
        self.dataobject_lookup_arr += ['RealSlice' for i in range(self.N_realslices)]
        self.dataobject_lookup_arr += ['PointList' for i in range(self.N_pointlists)]
        self.dataobject_lookup_arr += ['PointListArray' for i in range(self.N_pointlistarrays)]
        self.dataobject_lookup_arr = np.array(self.dataobject_lookup_arr)

    def __exit__(self):
        if self.file:
            self.file.close()

    ######## Methods for looking up info about dataobjects ########






    def get_dataobject_info_v0_7(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.7.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'CountedDataCube':
            name = list(self.file[self.topgroup + 'data/counted_datacubes'].keys())[objectindex]
            R_Nx, R_Ny = self.file[self.topgroup + 'data/counted_datacubes'][name]['data'].shape
            Q_Nx = self.file[self.topgroup + 'data/counted_datacubes'][name]['dim3'].shape[0]
            Q_Ny = self.file[self.topgroup + 'data/counted_datacubes'][name]['dim4'].shape[0]
            shape = (R_Nx, R_Ny, Q_Nx, Q_Ny)
            #metadata = self.file[self.topgroup + 'data/counted_datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file[self.topgroup + 'data/diffractionslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/diffractionslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/diffractionslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/diffractionslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'RealSlice':
            name = list(self.file[self.topgroup + 'data/realslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/realslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/realslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/realslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'PointList':
            name = list(self.file[self.topgroup + 'data/pointlists'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlists'][name].keys())
            length = self.file[self.topgroup + 'data/pointlists'][name][coordinates[0]]['data'].shape[0]
            #metadata = self.file[self.topgroup + 'data/pointlists'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointListArray':
            name = list(self.file[self.topgroup + 'data/pointlistarrays'].keys())[objectindex]
            coordinates = self.file[self.topgroup + 'data/pointlistarrays'][name]["data"][0,0].dtype
            shape = self.file[self.topgroup + 'data/pointlistarrays'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/pointlistarrays'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index} #, 'metadata':'unsupported'}
        return objectinfo








    ###### Display object info ######

    def show_dataobject(self, index):
        """
        Display the info about dataobject at index.
        Two verbosity levels are supported through the 'v' boolean.
        """
        info = self.get_dataobject_info(index)

        if info['type'] == 'RawDataCube':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Shape', str(info['shape'])))
            print("{:<8}: {:<50}".format('Index', index))
        elif info['type'] == 'DataCube':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Shape', str(info['shape'])))
            print("{:<8}: {:<50}".format('Index', index))
        elif info['type'] == 'DiffractionSlice':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Depth', info['depth']))
            print("{:<8}: {:<50}".format('Slices', str(info['slices'])))
            print("{:<8}: {:<50}".format('Shape', str(info['shape'])))
            print("{:<8}: {:<50}".format('Index', index))
        elif info['type'] == 'RealSlice':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Depth', info['depth']))
            print("{:<8}: {:<50}".format('Slices', str(info['slices'])))
            print("{:<8}: {:<50}".format('Shape', str(info['shape'])))
            print("{:<8}: {:<50}".format('Index', index))
        elif info['type'] == 'PointList':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Length', str(info['length'])))
            print("{:<8}: {:<50}".format('Coords', str(info['coordinates'])))
            print("{:<8}: {:<50}".format('Index', index))
        elif info['type'] == 'PointListArray':
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Shape', str(info['shape'])))
            print("{:<8}: {:<50}".format('Coords', str(info['coordinates'])))
            print("{:<8}: {:<50}".format('Index', index))
        else:
            print("{:<8}: {:<50}".format('Type', info['type']))
            print("{:<8}: {:<50}".format('Name', info['name']))
            print("{:<8}: {:<50}".format('Index', index))

    def show_dataobjects(self, index=None, objecttype=None):
        if index is not None:
            self.show_dataobject(index)

        elif objecttype is not None:
            if objecttype == 'RawDataCube':
                self.show_rawdatacubes()
            elif objecttype == 'DataCube':
                self.show_datacubes()
            elif objecttype == 'CountedDataCube':
                self.show_counted_datacubes()
            elif objecttype == 'DiffractionSlice':
                self.show_diffractionslices()
            elif objecttype == 'RealSlice':
                self.show_realslices()
            elif objecttype == 'PointList':
                self.show_pointlists()
            elif objecttype == 'PointListArray':
                self.show_pointlistarrays()
            else:
                print("Error: unknown objecttype {}.".format(objecttype))

        else:
            print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Type'))
            for index in range(self.N_dataobjects):
                info = self.get_dataobject_info(index)
                name = info['name']
                objecttype = info['type']
                print("{:^8}{:<36}{:<20}".format(index, name, objecttype))

    def show_rawdatacubes(self):
        if self.version==(0,2):
            if self.N_rawdatacubes == 0:
                print("No RawDataCubes present.")
            else:
                print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Shape'))
                for index in (self.dataobject_lookup_arr=='RawDataCube').nonzero()[0]:
                    info = self.get_dataobject_info(index)
                    name = info['name']
                    shape = info['shape']
                    print("{:^8}{:<36}{:^20}".format(index, name, str(shape)))
        else:
            print("RawDataCubes are not supported past v0.2.")

    def show_datacubes(self):
        if self.N_datacubes == 0:
            print("No DataCubes present.")
        else:
            print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Shape'))
            for index in (self.dataobject_lookup_arr=='DataCube').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                shape = info['shape']
                print("{:^8}{:<36}{:^20}".format(index, name, str(shape)))

    def show_counted_datacubes(self):
        if self.N_counted == 0:
            print("No CountedDataCubes present.")
        else:
            print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Shape'))
            for index in (self.dataobject_lookup_arr=='CountedDataCube').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                shape = info['shape']
                print("{:^8}{:<36}{:^20}".format(index, name, str(shape)))

    def show_diffractionslices(self):
        if self.N_diffractionslices == 0:
            print("No DiffractionSlices present.")
        else:
            print("{:^8}{:^36}{:^10}{:^20}".format('Index', 'Name', 'Depth', 'Shape'))
            for index in (self.dataobject_lookup_arr=='DiffractionSlice').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                depth = info['depth']
                shape = info['shape']
                print("{:^8}{:<36}{:^10}{:^20}".format(index, name, depth, str(shape)))

    def show_realslices(self):
        if self.N_realslices == 0:
            print("No RealSlices present.")
        else:
            print("{:^8}{:^36}{:^10}{:^20}".format('Index', 'Name', 'Depth', 'Shape'))
            for index in (self.dataobject_lookup_arr=='RealSlice').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                depth = info['depth']
                shape = info['shape']
                print("{:^8}{:<36}{:^10}{:^20}".format(index, name, depth, str(shape)))

    def show_pointlists(self):
        if self.N_pointlists == 0:
            print("No PointLists present.")
        else:
            print("{:^8}{:^36}{:^8}{:^24}".format('Index', 'Name', 'Length', 'Coordinates'))
            for index in (self.dataobject_lookup_arr=='PointList').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                length = info['length']
                coordinates = info['coordinates']
                print("{:^8}{:<36}{:^8}{:^24}".format(index, name, length, str(coordinates)))

    def show_pointlistarrays(self):
        if self.N_pointlistarrays == 0:
            print("No PointListArrays present.")
        else:
            print("{:^8}{:^36}{:^12}{:^24}".format('Index', 'Name', 'Shape', 'Coordinates'))
            for index in (self.dataobject_lookup_arr=='PointListArray').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                shape = info['shape']
                coordinates = info['coordinates']
                print("{:^8}{:<36}{:^8}{:^24}".format(index, name, shape, str(coordinates)))

    ###### Get dataobject info #####

    def get_dataobject_info(self, index):
        """
        Returns a dictionary containing information about the object at index.
        Dict keys includes 'name', 'type', and 'index'.
        The following additional keys are object type dependent:
            DataCube: 'shape'
            DiffractionSlice, RealSlice: 'depth', 'shape'
            PointList: 'coordinates', 'length'
            PointListArray: 'coordinates', 'shape'
            RawDataCube (v0.2 only): 'shape'
        """
        if self.version == (0,7):
            return self.get_dataobject_info_v0_7(index)
        elif self.version == (0,6):
            return self.get_dataobject_info_v0_6(index)
        elif self.version == (0,5):
            return self.get_dataobject_info_v0_5(index)
        elif self.version == (0,4):
            return self.get_dataobject_info_v0_4(index)
        elif self.version == (0,3):
            return self.get_dataobject_info_v0_3(index)
        elif self.version == (0,2):
            return self.get_dataobject_info_v0_2(index)
        else:
            print("Error: unknown py4DSTEM version {}.{}.".format(self.version[0],self.version[1]))
            return None

    def get_dataobject_info_v0_7(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.7.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'CountedDataCube':
            name = list(self.file[self.topgroup + 'data/counted_datacubes'].keys())[objectindex]
            R_Nx, R_Ny = self.file[self.topgroup + 'data/counted_datacubes'][name]['data'].shape
            Q_Nx = self.file[self.topgroup + 'data/counted_datacubes'][name]['dim3'].shape[0]
            Q_Ny = self.file[self.topgroup + 'data/counted_datacubes'][name]['dim4'].shape[0]
            shape = (R_Nx, R_Ny, Q_Nx, Q_Ny)
            #metadata = self.file[self.topgroup + 'data/counted_datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file[self.topgroup + 'data/diffractionslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/diffractionslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/diffractionslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/diffractionslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'RealSlice':
            name = list(self.file[self.topgroup + 'data/realslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/realslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/realslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/realslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'PointList':
            name = list(self.file[self.topgroup + 'data/pointlists'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlists'][name].keys())
            length = self.file[self.topgroup + 'data/pointlists'][name][coordinates[0]]['data'].shape[0]
            #metadata = self.file[self.topgroup + 'data/pointlists'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointListArray':
            name = list(self.file[self.topgroup + 'data/pointlistarrays'].keys())[objectindex]
            coordinates = self.file[self.topgroup + 'data/pointlistarrays'][name]["data"][0,0].dtype
            shape = self.file[self.topgroup + 'data/pointlistarrays'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/pointlistarrays'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index} #, 'metadata':'unsupported'}
        return objectinfo


    def get_dataobject_info_v0_6(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.6.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file[self.topgroup + 'data/diffractionslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/diffractionslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/diffractionslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/diffractionslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'RealSlice':
            name = list(self.file[self.topgroup + 'data/realslices'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/realslices'][name]['data'].shape
            #metadata = self.file[self.topgroup + 'data/realslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/realslices'][name]['dim3']
                if('S' in dim3.dtype.str): # Checks if dim3 is composed of fixed width C strings
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = np.array(bstrings)

        elif objecttype == 'PointList':
            name = list(self.file[self.topgroup + 'data/pointlists'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlists'][name].keys())
            length = self.file[self.topgroup + 'data/pointlists'][name][coordinates[0]]['data'].shape[0]
            #metadata = self.file[self.topgroup + 'data/pointlists'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointListArray':
            name = list(self.file[self.topgroup + 'data/pointlistarrays'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlistarrays'][name]['0_0'].keys())
            i,j=0,0
            for key in list(self.file[self.topgroup + 'data/pointlistarrays'][name].keys()):
                i0,j0 = int(key.split('_')[0]),int(key.split('_')[1])
                i,j = max(i0,i),max(j0,j)
            shape = (i+1,j+1)
            #metadata = self.file[self.topgroup + 'data/pointlistarrays'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index} #, 'metadata':'unsupported'}
        return objectinfo

    def get_dataobject_info_v0_5(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.5.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file[self.topgroup + 'data/datacubes'].keys())[objectindex]
            shape = self.file[self.topgroup + 'data/datacubes'][name]['datacube'].shape
            #metadata = self.file[self.topgroup + 'data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file[self.topgroup + 'data/diffractionslices'].keys())[objectindex]
            slices = list(self.file[self.topgroup + 'data/diffractionslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            if('dim3' in slices):
                slices.remove('dim3')
            shape = self.file[self.topgroup + 'data/diffractionslices'][name][slices[0]].shape
            #metadata = self.file[self.topgroup + 'data/diffractionslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'slices':slices, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/diffractionslices'][name]['dim3']
                if('S' in dim3.dtype.str): #Checks if fixed width C string
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = bstrings

        elif objecttype == 'RealSlice':
            name = list(self.file[self.topgroup + 'data/realslices'].keys())[objectindex]
            slices = list(self.file[self.topgroup + 'data/realslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            if('dim3' in slices):
                slices.remove('dim3')
            shape = self.file[self.topgroup + 'data/realslices'][name][slices[0]].shape
            #metadata = self.file[self.topgroup + 'data/realslices'][name].attrs['metadata']
            depth = 1
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
            if(len(shape) == 3):
                objectinfo['depth'] = shape[2]
                dim3 = self.file[self.topgroup+'data/realslices'][name]['dim3']
                if('S' in dim3.dtype.str): #Checks if fixed width C string
                    with dim3.astype('S64'):
                        bstrings = dim3[:]
                    bstrings = [bstring.decode('UTF-8') for bstring in bstrings]
                    #write key if list is string based IE inhomogenous 3D array
                    objectinfo['dim3'] = bstrings

        elif objecttype == 'PointList':
            name = list(self.file[self.topgroup + 'data/pointlists'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlists'][name].keys())
            length = self.file[self.topgroup + 'data/pointlists'][name][coordinates[0]]['data'].shape[0]
            #metadata = self.file[self.topgroup + 'data/pointlists'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointListArray':
            name = list(self.file[self.topgroup + 'data/pointlistarrays'].keys())[objectindex]
            coordinates = list(self.file[self.topgroup + 'data/pointlistarrays'][name]['0_0'].keys())
            i,j=0,0
            for key in list(self.file[self.topgroup + 'data/pointlistarrays'][name].keys()):
                i0,j0 = int(key.split('_')[0]),int(key.split('_')[1])
                i,j = max(i0,i),max(j0,j)
            shape = (i+1,j+1)
            #metadata = self.file[self.topgroup + 'data/pointlistarrays'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index} #, 'metadata':'unsupported'}
        return objectinfo

    def get_dataobject_info_v0_4(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.4.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file['4DSTEM_experiment/data/datacubes'].keys())[objectindex]
            shape = self.file['4DSTEM_experiment/data/datacubes'][name]['datacube'].shape
            #metadata = self.file['4DSTEM_experiment/data/datacubes'][name].attrs['metadata']
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype,
                          'index':index} #, 'metadata':metadata}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file['4DSTEM_experiment/data/diffractionslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/data/diffractionslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/data/diffractionslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[0]].shape
            #metadata = self.file['4DSTEM_experiment/data/diffractionslices'][name].attrs['metadata']
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'RealSlice':
            name = list(self.file['4DSTEM_experiment/data/realslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/data/realslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/data/realslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/data/realslices'][name][slices[0]].shape
            #metadata = self.file['4DSTEM_experiment/data/realslices'][name].attrs['metadata']
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointList':
            name = list(self.file['4DSTEM_experiment/data/pointlists'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/data/pointlists'][name].keys())
            length = self.file['4DSTEM_experiment/data/pointlists'][name][coordinates[0]]['data'].shape[0]
            #metadata = self.file['4DSTEM_experiment/data/pointlists'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        elif objecttype == 'PointListArray':
            name = list(self.file['4DSTEM_experiment/data/pointlistarrays'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['0_0'].keys())
            i,j=0,0
            for key in list(self.file['4DSTEM_experiment/data/pointlistarrays'][name].keys()):
                i0,j0 = int(key.split('_')[0]),int(key.split('_')[1])
                i,j = max(i0,i),max(j0,j)
            shape = (i+1,j+1)
            #metadata = self.file['4DSTEM_experiment/data/pointlistarrays'][name].attrs['metadata']
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape,
                          'type':objecttype, 'index':index} #, 'metadata':metadata}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index} #, 'metadata':'unsupported'}
        return objectinfo

    def get_dataobject_info_v0_3(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0.3.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'DataCube':
            name = list(self.file['4DSTEM_experiment/data/datacubes'].keys())[objectindex]
            shape = self.file['4DSTEM_experiment/data/datacubes'][name]['datacube'].shape
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file['4DSTEM_experiment/data/diffractionslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/data/diffractionslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/data/diffractionslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'RealSlice':
            name = list(self.file['4DSTEM_experiment/data/realslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/data/realslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/data/realslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/data/realslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'PointList':
            name = list(self.file['4DSTEM_experiment/data/pointlists'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/data/pointlists'][name].keys())
            length = self.file['4DSTEM_experiment/data/pointlists'][name][coordinates[0]]['data'].shape[0]
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length, 'type':objecttype, 'index':index}
        elif objecttype == 'PointListArray':
            name = list(self.file['4DSTEM_experiment/data/pointlistarrays'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['0_0'].keys())
            i,j=0,0
            for key in list(self.file['4DSTEM_experiment/data/pointlistarrays'][name].keys()):
                i0,j0 = int(key.split('_')[0]),int(key.split('_')[1])
                i,j = max(i0,i),max(j0,j)
            shape = (i+1,j+1)
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape, 'type':objecttype, 'index':index}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index}
        return objectinfo

    def get_dataobject_info_v0_2(self, index):
        """
        Returns a dictionary containing information about the object at index for files written
        by py4DSTEM v0..
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'RawDataCube':
            name = 'rawdatacube'
            shape = self.file['4DSTEM_experiment/rawdatacube/datacube'].shape
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'DataCube':
            name = list(self.file['4DSTEM_experiment/processing/datacubes'].keys())[objectindex]
            shape = self.file['4DSTEM_experiment/processing/datacubes'][name]['datacube'].shape
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file['4DSTEM_experiment/processing/diffractionslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/processing/diffractionslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/processing/diffractionslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/processing/diffractionslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'RealSlice':
            name = list(self.file['4DSTEM_experiment/processing/realslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment/processing/realslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment/processing/realslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment/processing/realslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'PointList':
            name = list(self.file['4DSTEM_experiment/processing/pointlists'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/processing/pointlists'][name].keys())
            length = self.file['4DSTEM_experiment/processing/pointlists'][name][coordinates[0]]['data'].shape[0]
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length, 'type':objecttype, 'index':index}
        elif objecttype == 'PointListArray':
            name = list(self.file['4DSTEM_experiment/processing/pointlistarrays'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment/processing/pointlistarrays'][name]['0_0'].keys())
            i,j=0,0
            for key in list(self.file['4DSTEM_experiment/processing/pointlistarrays'][name].keys()):
                i0,j0 = int(key.split('_')[0]),int(key.split('_')[1])
                i,j = max(i0,i),max(j0,j)
            shape = (i+1,j+1)
            objectinfo = {'name':name, 'coordinates':coordinates, 'shape':shape, 'type':objecttype, 'index':index}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index}
        return objectinfo

    ###### Get dataobjects ######

    def get_object_lookup_info(self, index):
        objecttype = self.dataobject_lookup_arr[index]
        mask = (self.dataobject_lookup_arr == objecttype)
        objectindex = index - np.nonzero(mask)[0][0]

        return objecttype, objectindex

    def get_dataobject(self, dataobject, **kwargs):
        """
        Finds a single DataObject in the file, and returns it.

        If dataobject is an int, finds the DataObject corresponding to the .h5 data pointed to
        by this index.
        If dataobject is a string, finds the DataObject with a matching name in the .h5 file.
        If the file is EMD v0.4 or greater, you can specify memory_map=True to
        return a memory map for DataCube and CountedDataCube objects.
        """
        assert(isinstance(dataobject,(int,np.integer,str))), "dataobject must be int or str"
        if isinstance(dataobject, str):
            dataobject = self.get_dataobject_index_from_name(dataobject)
        if self.version==(0,7):
            return self.get_dataobject_v0_7(dataobject, **kwargs)
        if self.version==(0,6):
            return self.get_dataobject_v0_6(dataobject, **kwargs)
        elif self.version==(0,5):
            return self.get_dataobject_v0_5(dataobject, **kwargs)
        elif self.version==(0,4):
            return self.get_dataobject_v0_4(dataobject, **kwargs)
        elif self.version==(0,3):
            return self.get_dataobject_v0_3(dataobject)
        elif self.version==(0,2):
            return self.get_dataobject_v0_2(dataobject)
        else:
            raise Exception("Error: unrecognized py4DSTEM version {}.{}.".format(self.version[0],self.version[1]))

    def get_dataobject_index_from_name(self, name):
        objects = []
        for i in range(self.N_dataobjects):
            objectname = self.get_dataobject_info(i)['name']
            if name == objectname:
                objects.append(i)
        if len(objects) > 1:
            raise Exception("Multiple dataobjects found with name {}, at indices {}.".format(name, objects))
        elif len(objects) == 1:
            return objects[0]
        else:
            raise Exception("No dataobject found with name {}.".format(name))

    def get_dataobject_v0_7(self, index, **kwargs):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']
        #metadata_index = info['metadata']

        mmap = kwargs.get('memory_map', False)

        #if metadata_index == -1:
        #    metadata = None
        #else:
        #    if self.metadataobjects_in_memory[metadata_index] is None:
        #        self.metadataobjects_in_memory[metadata_index] = self.get_metadataobject(metadata_index)
        #    metadata = self.metadataobjects_in_memory[metadata_index]

        if objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            if mmap:
                data = self.file[self.topgroup + 'data/datacubes'][name]['data']
            else:
                # TODO: preallocate array and copy into it for better RAM usage
                data = np.array(self.file[self.topgroup + 'data/datacubes'][name]['data'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'CountedDataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape

            dset = self.file[self.topgroup + 'data/counted_datacubes'][name]["data"]
            coordinates = dset[0,0].dtype

            if coordinates.fields is None:
                index_coords = [None]
            else:
                index_coords = self.file[self.topgroup + 'data/counted_datacubes'][name]["index_coords"]

            if mmap:
                # run in HDF5 memory mapping mode
                dataobject = CountedDataCube(dset,[Q_Nx,Q_Ny],index_coords[:])
            else:
                # run in PointListArray mode
                pla = PointListArray(coordinates=coordinates, shape=shape[:2], name=name)
            
                for (i,j) in tqdmnd(shape[0],shape[1],desc='Reading electrons',unit='DP'):
                    pla.get_pointlist(i,j).add_dataarray(dset[i,j])

                dataobject = CountedDataCube(pla,[Q_Nx,Q_Ny],index_coords[:])

        elif objecttype == 'DiffractionSlice':
            shape = info['shape']
            Q_Nx = shape[0]
            Q_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/diffractionslices'][name]['data'])
            if(len(shape)==2):
                dataobject = DiffractionSlice(data=data, Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)
            else:
                dim3 = info['dim3']
                dataobject = DiffractionSlice(data=data, slicelabels=dim3,
                                              Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)

        elif objecttype == 'RealSlice':
            shape = info['shape']
            R_Nx = shape[0]
            R_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/realslices'][name]['data'])
            if(len(shape)==2):
                dataobject = RealSlice(data=data, R_Nx=R_Nx, R_Ny=R_Ny, name=name)
            else:
                dim3 = info['dim3']
                dataobject = RealSlice(data=data, slicelabels=dim3,
                                       R_Nx=R_Nx, R_Ny=R_Ny, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                try:
                    dtype = type(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'][0])
                except ValueError:
                    print("dtype can't be retrieved. PointList may be empty.")
                    dtype = None
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']

            dset = self.file[self.topgroup + 'data/pointlistarrays'][name]["data"]
            coordinates = dset[0,0].dtype
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            
            for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit='PointList'):
                dataobject.get_pointlist(i,j).add_dataarray(dset[i,j])

        else:
            raise Exception("Unknown object type {}. Returning None.".format(objecttype))

        #dataobject.metadata = metadata
        return dataobject

    def get_dataobject_v0_6(self, index, **kwargs):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']
        #metadata_index = info['metadata']

        mmap = kwargs.get('memory_map', False)

        #if metadata_index == -1:
        #    metadata = None
        #else:
        #    if self.metadataobjects_in_memory[metadata_index] is None:
        #        self.metadataobjects_in_memory[metadata_index] = self.get_metadataobject(metadata_index)
        #    metadata = self.metadataobjects_in_memory[metadata_index]

        if objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            if mmap:
                data = self.file[self.topgroup + 'data/datacubes'][name]['data']
            else:
                # TODO: preallocate array and copy into it for better RAM usage
                data = np.array(self.file[self.topgroup + 'data/datacubes'][name]['data'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DiffractionSlice':
            shape = info['shape']
            Q_Nx = shape[0]
            Q_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/diffractionslices'][name]['data'])
            if(len(shape)==2):
                dataobject = DiffractionSlice(data=data, Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)
            else:
                dim3 = info['dim3']
                dataobject = DiffractionSlice(data=data, slicelabels=dim3,
                                              Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)

        elif objecttype == 'RealSlice':
            shape = info['shape']
            R_Nx = shape[0]
            R_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/realslices'][name]['data'])
            if(len(shape)==2):
                dataobject = RealSlice(data=data, R_Nx=R_Nx, R_Ny=R_Ny, name=name)
            else:
                dim3 = info['dim3']
                dataobject = RealSlice(data=data, slicelabels=dim3,
                                       R_Nx=R_Nx, R_Ny=R_Ny, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                try:
                    dtype = type(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'][0])
                except ValueError:
                    print("dtype can't be retrieved. PointList may be empty.")
                    dtype = None
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']
            coordinates = []
            for coord in coords:
                dtype = type(self.file[self.topgroup + 'data/pointlistarrays'][name]['0_0'][coord]['data'][0])
                coordinates.append((coord, dtype))
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pointlist = dataobject.get_pointlist(i,j)
                    data_dict = {}
                    for coord in coords:
                        data_dict[coord] = np.array(self.file[self.topgroup + 'data/pointlistarrays'][name]['{}_{}'.format(i,j)][coord]['data'])
                    for k in range(len(data_dict[coords[0]])):
                        new_point = tuple([data_dict[coord][k] for coord in coords])
                        pointlist.add_point(new_point)

        else:
            raise Exception("Unknown object type {}. Returning None.".format(objecttype))

        #dataobject.metadata = metadata
        return dataobject

    def get_dataobject_v0_5(self, index, **kwargs):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']
        #metadata_index = info['metadata']

        mmap = kwargs.get('memory_map', False)

        #if metadata_index == -1:
        #    metadata = None
        #else:
        #    if self.metadataobjects_in_memory[metadata_index] is None:
        #        self.metadataobjects_in_memory[metadata_index] = self.get_metadataobject(metadata_index)
        #    metadata = self.metadataobjects_in_memory[metadata_index]

        if objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            if mmap:
                data = self.file[self.topgroup + 'data/datacubes'][name]['datacube']
            else:
                # TODO: preallocate array and copy into it for better RAM usage
                data = np.array(self.file[self.topgroup + 'data/datacubes'][name]['datacube'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DiffractionSlice':
            slices = info['slices']
            shape = info['shape']
            Q_Nx = shape[0]
            Q_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/diffractionslices'][name][slices[0]])
            if(len(shape)==2):
                dataobject = DiffractionSlice(data=data, slicelabels=slices, Q_Nx=Q_Nx, Q_Ny=Q_Ny ,Q_Nz=None, name=name)
            else:
                if('dim3' in info.keys()):
                    ##TODO: slicelabels requires tuples of strings-> need to read in tuples of strings when dataobject is gotten
                    dataobject = DiffractionSlice(data=data, slicelabels=info['dim3'], Q_Nx=Q_Nx, Q_Ny=Q_Ny, Q_Nz=None, name=name)
                else:
                    dataobject = DiffractionSlice(data=data, slicelabels=None, Q_Nx=Q_Nx, Q_Ny=Q_Ny, Q_Nz=shape[2], name=name)

        elif objecttype == 'RealSlice':
            slices = info['slices']
            shape = info['shape']
            print(shape)
            R_Nx = shape[0]
            R_Ny = shape[1]
            data = np.array(self.file[self.topgroup + 'data/realslices'][name][slices[0]])
            if(len(shape)==2):
                dataobject = RealSlice(data=data, slicelabels=slices, R_Nx=R_Nx, R_Ny=R_Ny, R_Nz=None, name=name)
            else:
                if('dim3' in info.keys()):
                    ##TODO: slicelabels requires tuples of strings-> need to read in tuples of strings when dataobject is gotten
                    dataobject = RealSlice(data=data, slicelabels=info['dim3'], R_Nx=R_Nx, R_Ny=R_Ny, R_Nz=None, name=name)
                else:
                    dataobject = RealSlice(data=data, slicelabels=None, R_Nx=R_Nx, R_Ny=R_Ny, R_Nz=shape[2], name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                try:
                    dtype = type(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'][0])
                except ValueError:
                    print("dtype can't be retrieved. PointList may be empty.")
                    dtype = None
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file[self.topgroup + 'data/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']
            coordinates = []
            for coord in coords:
                dtype = type(self.file[self.topgroup + 'data/pointlistarrays'][name]['0_0'][coord]['data'][0])
                coordinates.append((coord, dtype))
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pointlist = dataobject.get_pointlist(i,j)
                    data_dict = {}
                    for coord in coords:
                        data_dict[coord] = np.array(self.file[self.topgroup + 'data/pointlistarrays'][name]['{}_{}'.format(i,j)][coord]['data'])
                    for k in range(len(data_dict[coords[0]])):
                        new_point = tuple([data_dict[coord][k] for coord in coords])
                        pointlist.add_point(new_point)

        else:
            raise Exception("Unknown object type {}. Returning None.".format(objecttype))

        #dataobject.metadata = metadata
        return dataobject

    def get_dataobject_v0_4(self, index, **kwargs):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']
        #metadata_index = info['metadata']

        mmap = kwargs.get('memory_map', False)

        #if metadata_index == -1:
        #    metadata = None
        #else:
        #    if self.metadataobjects_in_memory[metadata_index] is None:
        #        self.metadataobjects_in_memory[metadata_index] = self.get_metadataobject(metadata_index)
        #    metadata = self.metadataobjects_in_memory[metadata_index]

        if objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            if mmap:
                data = self.file['4DSTEM_experiment/data/datacubes'][name]['datacube']
            else:
                # TODO: preallocate array and copy into it for better RAM usage
                data = np.array(self.file['4DSTEM_experiment/data/datacubes'][name]['datacube'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DiffractionSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            Q_Nx, Q_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[0]])
            else:
                data = np.empty((shape[0], shape[1], depth))
                for i in range(depth):
                    data[:,:,i] = np.array(self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[i]])
            dataobject = DiffractionSlice(data=data, slicelabels=slices, Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)

        elif objecttype == 'RealSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            R_Nx, R_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/data/realslices'][name][slices[0]])
            else:
                data = np.empty((shape[0], shape[1], depth))
                for i in range(depth):
                    data[:,:,i] = np.array(self.file['4DSTEM_experiment/data/realslices'][name][slices[i]])
            dataobject = RealSlice(data=data, slicelabels=slices, R_Nx=R_Nx, R_Ny=R_Ny, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                try:
                    dtype = type(self.file['4DSTEM_experiment/data/pointlists'][name][coord]['data'][0])
                except ValueError:
                    print("dtype can't be retrieved. PointList may be empty.")
                    dtype = None
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file['4DSTEM_experiment/data/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']
            coordinates = []
            for coord in coords:
                dtype = type(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['0_0'][coord]['data'][0])
                coordinates.append((coord, dtype))
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pointlist = dataobject.get_pointlist(i,j)
                    data_dict = {}
                    for coord in coords:
                        data_dict[coord] = np.array(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['{}_{}'.format(i,j)][coord]['data'])
                    for k in range(len(data_dict[coords[0]])):
                        new_point = tuple([data_dict[coord][k] for coord in coords])
                        pointlist.add_point(new_point)

        else:
            raise Exception("Unknown object type {}. Returning None.".format(objecttype))

        #dataobject.metadata = metadata
        return dataobject

    def get_dataobject_v0_3(self, index):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']

        if objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            data = np.array(self.file['4DSTEM_experiment/data/datacubes'][name]['datacube'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DiffractionSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            Q_Nx, Q_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[0]])
            else:
                data = np.empty((shape[0], shape[1], depth))
                for i in range(depth):
                    data[:,:,i] = np.array(self.file['4DSTEM_experiment/data/diffractionslices'][name][slices[i]])
            dataobject = DiffractionSlice(data=data, slicelabels=slices, Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)

        elif objecttype == 'RealSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            R_Nx, R_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/data/realslices'][name][slices[0]])
            else:
                data = np.empty((shape[0], shape[1], depth))
                for i in range(depth):
                    data[:,:,i] = np.array(self.file['4DSTEM_experiment/data/realslices'][name][slices[i]])
            dataobject = RealSlice(data=data, slicelabels=slices, R_Nx=R_Nx, R_Ny=R_Ny, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                try:
                    dtype = type(self.file['4DSTEM_experiment/data/pointlists'][name][coord]['data'][0])
                except ValueError:
                    print("dtype can't be retrieved. PointList may be empty.")
                    dtype = None
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file['4DSTEM_experiment/data/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']
            coordinates = []
            for coord in coords:
                dtype = type(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['0_0'][coord]['data'][0])
                coordinates.append((coord, dtype))
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pointlist = dataobject.get_pointlist(i,j)
                    data_dict = {}
                    for coord in coords:
                        data_dict[coord] = np.array(self.file['4DSTEM_experiment/data/pointlistarrays'][name]['{}_{}'.format(i,j)][coord]['data'])
                    for k in range(len(data_dict[coords[0]])):
                        new_point = tuple([data_dict[coord][k] for coord in coords])
                        pointlist.add_point(new_point)

        else:
            print("Unknown object type {}. Returning None.".format(objecttype))
            dataobject = None

        return dataobject

    def get_dataobject_v0_2(self, index):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']

        if objecttype == 'RawDataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            data = np.array(self.file['4DSTEM_experiment/rawdatacube/datacube'])

            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DataCube':
            shape = info['shape']
            R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            data = np.array(self.file['4DSTEM_experiment/processing/datacubes'][name]['datacube'])
            dataobject = DataCube(data=data, name=name)

        elif objecttype == 'DiffractionSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            Q_Nx, Q_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/processing/diffractionslices'][name][slices[0]])
            else:
                data = np.empty((depth, shape[0], shape[1]))
                for i in range(depth):
                    data[i,:,:] = np.array(self.file['4DSTEM_experiment/processing/diffractionslices'][name][slices[i]])
            dataobject = DiffractionSlice(data=data, slicelabels=slices, Q_Nx=Q_Nx, Q_Ny=Q_Ny, name=name)

        elif objecttype == 'RealSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            R_Nx, R_Ny = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment/processing/realslices'][name][slices[0]])
            else:
                data = np.empty((depth, shape[0], shape[1]))
                for i in range(depth):
                    data[i,:,:] = np.array(self.file['4DSTEM_experiment/processing/realslices'][name][slices[i]])
            dataobject = RealSlice(data=data, slicelabels=slices, R_Nx=R_Nx, R_Ny=R_Ny, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                dtype = type(self.file['4DSTEM_experiment/processing/pointlists'][name][coord]['data'][0])
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file['4DSTEM_experiment/processing/pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        elif objecttype == 'PointListArray':
            shape = info['shape']
            coords = info['coordinates']
            coordinates = []
            for coord in coords:
                dtype = type(self.file['4DSTEM_experiment/processing/pointlistarrays'][name]['0_0'][coord]['data'][0])
                coordinates.append((coord, dtype))
            dataobject = PointListArray(coordinates=coordinates, shape=shape, name=name)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pointlist = dataobject.get_pointlist(i,j)
                    data_dict = {}
                    for coord in coords:
                        data_dict[coord] = np.array(self.file['4DSTEM_experiment/processing/pointlistarrays'][name]['{}_{}'.format(i,j)][coord]['data'])
                    for k in range(len(data_dict[coords[0]])):
                        new_point = tuple([data_dict[coord][k] for coord in coords])
                        pointlist.add_point(new_point)

        else:
            print("Unknown object type {}. Returning None.".format(objecttype))
            dataobject = None

        return dataobject

    def get_dataobjects(self, indices):
        if indices=='all':
            indices = range(self.N_dataobjects)
        else:
            assert all([isinstance(index,(int,np.integer)) for index in indices]), "Arguments to get_dataobjects must be integers; to retrieve a dataobject by name, use get_dataobject"
        objects = []
        for index in indices:
            objects.append(self.get_dataobject(index))
        return objects

    def get_all_dataobjects(self):
        return self.get_dataobjects('all')

    def get_rawdatacubes(self):
        indices = (self.dataobject_lookup_arr=='RawDataCube').nonzero()[0]
        indices = [index.item() for index in indices]  # Converts np.int64 -> int
        return self.get_dataobjects(indices)

    def get_datacubes(self):
        indices = (self.dataobject_lookup_arr=='DataCube').nonzero()[0]
        indices = [index.item() for index in indices]
        return self.get_dataobjects(indices)

    def get_diffractionslices(self):
        indices = (self.dataobject_lookup_arr=='DiffractionSlice').nonzero()[0]
        indices = [index.item() for index in indices]
        return self.get_dataobjects(indices)

    def get_realslices(self):
        indices = (self.dataobject_lookup_arr=='RealSlice').nonzero()[0]
        indices = [index.item() for index in indices]
        return self.get_dataobjects(indices)

    def get_pointlists(self):
        indices = (self.dataobject_lookup_arr=='PointList').nonzero()[0]
        indices = [index.item() for index in indices]
        return self.get_dataobjects(indices)

    def get_pointlistarrays(self):
        indices = (self.dataobject_lookup_arr=='PointListArray').nonzero()[0]
        indices = [index.item() for index in indices]
        return self.get_dataobjects(indices)


###################### END FileBrowser CLASS #######################




