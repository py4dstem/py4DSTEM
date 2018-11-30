# Reads 4D-STEM data

import h5py
import numpy as np
import hyperspy.api as hs
from ..process.datastructure import RawDataCube, DataCube
from ..process.datastructure import DiffractionSlice, RealSlice
from ..process.datastructure import PointList
from ..process.log import log


class FileBrowser(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.is_py4DSTEM_file = is_py4DSTEM_file(self.filepath)
        if self.is_py4DSTEM_file:
            self.version = get_py4DSTEM_version(self.filepath)
            self.file = h5py.File(filepath, 'r')
            self.set_object_lookup_info()
            self.rawdatacube = RawDataCube(data=np.array([0]),R_Ny=1,R_Nx=1,Q_Ny=1,Q_Nx=1)
            #self.N_logentries = self.get_N_logentries()

    ###### Open/close methods ######

    def open(self):
        self.file = h5py.File(filepath, 'r')

    def close(self):
        if 'file' in self.__dict__.keys():
            self.file.close()

    ###### Query dataobject info #####

    def get_dataobject_info(self, index):
        """
        Returns a dictionary containing information about the object at index.
        Dict keys includes 'name', 'type', and 'index'.
        The following additional keys are object type dependent:
            RawDataCube, DataCube: 'shape'
            DiffractionSlice, RealSlice: 'depth', 'slices', 'shape'
            PointList: 'coordinates', 'length'
        """
        objecttype, objectindex = self.get_object_lookup_info(index)

        if objecttype == 'RawDataCube':
            name = 'rawdatacube'
            shape = self.file['4DSTEM_experiment']['rawdatacube']['datacube'].shape
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'DataCube':
            name = list(self.file['4DSTEM_experiment']['processing']['datacubes'].keys())[objectindex]
            shape = self.file['4DSTEM_experiment']['processing']['datacubes'][name]['datacube'].shape
            objectinfo = {'name':name, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'DiffractionSlice':
            name = list(self.file['4DSTEM_experiment']['processing']['diffractionslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment']['processing']['diffractionslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment']['processing']['diffractionslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment']['processing']['diffractionslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'RealSlice':
            name = list(self.file['4DSTEM_experiment']['processing']['realslices'].keys())[objectindex]
            depth = self.file['4DSTEM_experiment']['processing']['realslices'][name].attrs['depth']
            slices = list(self.file['4DSTEM_experiment']['processing']['realslices'][name].keys())
            slices.remove('dim1')
            slices.remove('dim2')
            shape = self.file['4DSTEM_experiment']['processing']['realslices'][name][slices[0]].shape
            objectinfo = {'name':name, 'depth':depth, 'slices':slices, 'shape':shape, 'type':objecttype, 'index':index}
        elif objecttype == 'PointList':
            name = list(self.file['4DSTEM_experiment']['processing']['pointlists'].keys())[objectindex]
            coordinates = list(self.file['4DSTEM_experiment']['processing']['pointlists'][name].keys())
            length = self.file['4DSTEM_experiment']['processing']['pointlists'][name][coordinates[0]]['data'].shape[0]
            objectinfo = {'name':name, 'coordinates':coordinates, 'length':length, 'type':objecttype, 'index':index}
        else:
            print("Error: unknown dataobject type {}.".format(objecttype))
            objectinfo = {'name':'unsupported', 'type':'unsupported', 'index':index}
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
            elif objecttype == 'DiffractionSlice':
                self.show_diffractionslices()
            elif objecttype == 'RealSlice':
                self.show_realslices()
            elif objecttype == 'PointList':
                self.show_pointlists()
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
        if self.N_rawdatacubes == 0:
            print("No RawDataCubes present.")
        else:
            print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Shape'))
            for index in (self.dataobject_lookup_arr=='RawDataCube').nonzero()[0]:
                info = self.get_dataobject_info(index)
                name = info['name']
                shape = info['shape']
                print("{:^8}{:<36}{:^20}".format(index, name, str(shape)))

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

    ###### Retrieve dataobjects ######

    def set_object_lookup_info(self):
        if len(self.file['4DSTEM_experiment']['rawdatacube'])==0:
            self.N_rawdatacubes = 0
        else:
            self.N_rawdatacubes = 1
        self.N_datacubes = len(self.file['4DSTEM_experiment']['processing']['datacubes'])
        self.N_diffractionslices = len(self.file['4DSTEM_experiment']['processing']['diffractionslices'])
        self.N_realslices = len(self.file['4DSTEM_experiment']['processing']['realslices'])
        self.N_pointlists = len(self.file['4DSTEM_experiment']['processing']['pointlists'])
        self.N_dataobjects = np.sum([self.N_rawdatacubes, self.N_datacubes, self.N_diffractionslices, self.N_realslices, self.N_pointlists])

        self.dataobject_lookup_arr = []
        self.dataobject_lookup_arr += ['RawDataCube' for i in range(self.N_rawdatacubes)]
        self.dataobject_lookup_arr += ['DataCube' for i in range(self.N_datacubes)]
        self.dataobject_lookup_arr += ['DiffractionSlice' for i in range(self.N_diffractionslices)]
        self.dataobject_lookup_arr += ['RealSlice' for i in range(self.N_realslices)]
        self.dataobject_lookup_arr += ['PointList' for i in range(self.N_pointlists)]
        self.dataobject_lookup_arr = np.array(self.dataobject_lookup_arr)

    def get_object_lookup_info(self, index):
        objecttype = self.dataobject_lookup_arr[index]
        mask = (self.dataobject_lookup_arr == objecttype)
        objectindex = index - np.nonzero(mask)[0][0]

        return objecttype, objectindex

    def get_dataobject(self, index):
        """
        Instantiates a DataObject corresponding to the .h5 data pointed to by index.
        """
        objecttype, objectindex = self.get_object_lookup_info(index)
        info = self.get_dataobject_info(index)
        name = info['name']

        if objecttype == 'RawDataCube':
            shape = info['shape']
            self.rawdatacube.data4D = np.array(self.file['4DSTEM_experiment']['rawdatacube']['datacube'])
            R_Ny, R_Nx, Q_Ny, Q_Nx = shape
            self.rawdatacube.R_Ny = R_Ny
            self.rawdatacube.R_Nx = R_Nx
            self.rawdatacube.Q_Ny = Q_Ny
            self.rawdatacube.Q_Nx = Q_Nx
            self.rawdatacube.R_N = R_Ny*R_Nx
            dataobject = self.rawdatacube

        elif objecttype == 'DataCube':
            shape = info['shape']
            R_Ny, R_Nx, Q_Ny, Q_Nx = shape
            data = np.array(self.file['4DSTEM_experiment']['processing']['datacubes'][name]['datacube'])
            dataobject = DataCube(data=data, parentDataCube=self.rawdatacube, name=name)

        elif objecttype == 'DiffractionSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            Q_Ny, Q_Nx = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment']['processing']['diffractionslices'][name][slices[0]])
            else:
                data = np.empty((depth, shape[0], shape[1]))
                for i in range(depth):
                    data[i,:,:] = np.array(self.file['4DSTEM_experiment']['processing']['diffractionslices'][name][slices[i]])
            dataobject = DiffractionSlice(data=data, parentDataCube=self.rawdatacube, slicelabels=slices, Q_Ny=Q_Ny, Q_Nx=Q_Nx, name=name)

        elif objecttype == 'RealSlice':
            depth = info['depth']
            slices = info['slices']
            shape = info['shape']
            R_Ny, R_Nx = shape
            if depth==1:
                data = np.array(self.file['4DSTEM_experiment']['processing']['realslices'][name][slices[0]])
            else:
                data = np.empty((depth, shape[0], shape[1]))
                for i in range(depth):
                    data[i,:,:] = np.array(self.file['4DSTEM_experiment']['processing']['realslices'][name][slices[i]])
            dataobject = RealSlice(data=data, parentDataCube=self.rawdatacube, slicelabels=slices, R_Ny=R_Ny, R_Nx=R_Nx, name=name)

        elif objecttype == 'PointList':
            coords = info['coordinates']
            length = info['length']
            coordinates = []
            data_dict = {}
            for coord in coords:
                dtype = type(self.file['4DSTEM_experiment']['processing']['pointlists'][name][coord]['data'][0])
                coordinates.append((coord, dtype))
                data_dict[coord] = np.array(self.file['4DSTEM_experiment']['processing']['pointlists'][name][coord]['data'])
            dataobject = PointList(coordinates=coordinates, parentDataCube=self.rawdatacube, name=name)
            for i in range(length):
                new_point = tuple([data_dict[coord][i] for coord in coords])
                dataobject.add_point(new_point)

        else:
            print("Unknown object type {}. Returning None.".format(objecttype))
            dataobject = None

        return dataobject

    def get_dataobjects(self, indices):
        objects = []
        for index in indices:
            objects.append(self.get_dataobject(index))
        return objects

    def get_rawdatacubes(self):
        objects = []
        for index in (self.dataobject_lookup_arr=='RawDataCube').nonzero()[0]:
            objects.append(self.get_dataobject(index))
        return objects

    def get_datacubes(self):
        objects = []
        for index in (self.dataobject_lookup_arr=='DataCube').nonzero()[0]:
            objects.append(self.get_dataobject(index))
        return objects

    def get_diffractionslices(self):
        objects = []
        for index in (self.dataobject_lookup_arr=='DiffractionSlice').nonzero()[0]:
            objects.append(self.get_dataobject(index))
        return objects

    def get_realslices(self):
        objects = []
        for index in (self.dataobject_lookup_arr=='RealSlice').nonzero()[0]:
            objects.append(self.get_dataobject(index))
        return objects

    def get_pointlists(self):
        objects = []
        for index in (self.dataobject_lookup_arr=='PointList').nonzero()[0]:
            objects.append(self.get_dataobject(index))
        return objects

    ###### Log display and querry ######

    def get_N_logentries(self):
        return

###################### END FileBrowser CLASS #######################



###################### File reading functions #######################

@log
def read(filename, load_behavior='all'):
    """
    Takes a filename as input.
    Outputs some dataobjects.

    For non-py4DSTEM files, the output is a rawdatacube.
    For py4DSTEM v0.1 files, the output is a rawdatacube.

    For py4DSTEM v0.2 files, which dataobjects are loaded is controlled by the optional input
    parameter load_behavior, as follows:
    load_behavior = 'all':
        load all dataobjects found in the file
    load_behavior = 'datacube':
        load a single datacube. If present, loads the rawdatacube, otherwise, loads the first
        processed datacube found.
    load_behavior = [0,1,5,8,...]:
        If load behavoir is a list of ints, loads the set of objects found at those indices in the
        a FileBrowser instantiated from filename.
    """
    browser = FileBrowser(filename)

    if not browser.is_py4DSTEM_file:
        print("{} is not a py4DSTEM file.  Reading with hyperspy...".format(filename))
        output = read_non_py4DSTEM_file(filename)

    else:
        print("{} is a py4DSTEM file, v{}.{}. Reading...".format(filename, browser.version[0], browser.version[1]))

        if browser.version == (0,1):
            output = read_version_0_1(browser)

        elif browser.version == (0,2):
            output = read_version_0_2(browser, load_behavior=load_behavior)

        else:
            print("Error. Unsupported py4DSTEM file version - v{}.{}. Returning None.".format(browser.version[0], browser.version[1]))
            output = None

    browser.close()
    browser = None

    return output

def read_non_py4DSTEM_file(filename):
    try:
        hyperspy_file = hs.load(filename)
        if len(hyperspy_file.data.shape)==3:
            R_N, Q_Ny, Q_Nx = hyperspy_file.data.shape
            R_Ny, R_Nx = R_N, 1
        elif len(hyperspy_file.data.shape)==4:
            R_Ny, R_Nx, Q_Ny, Q_Nx = hyperspy_file.data.shape
        else:
            print("Error: unexpected raw data shape of {}".format(hyperspy_file.data.shape))
            print("Returning None")
            return None
        return RawDataCube(data=hyperspy_file.data, R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            is_py4DSTEM_file=False,
                            original_metadata_shortlist=hyperspy_file.metadata,
                            original_metadata_all=hyperspy_file.original_metadata)
    except Exception as err:
        print("Failed to load", err)
        print("Returning None")
        return None

def read_version_0_1(browser):
    R_Ny,R_Nx,Q_Ny,Q_Nx = browser.file['4D-STEM_data']['datacube']['datacube'].shape
    rawdatacube = RawDataCube(data=browser.file['4D-STEM_data']['datacube']['datacube'].value,
                    R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                    is_py4DSTEM_file=True, h5_file=browser.file)
    return rawdatacube

def read_version_0_2(browser, load_behavior='all'):
    """
    Which dataobjects are returned is controlled by the load_behavior optional parameter, as follows:

    load_behavior = 'all':
        load all dataobjects found in the file
    load_behavior = 'datacube':
        load a single datacube. If present, loads the rawdatacube, otherwise, loads the first
        processed datacube found.
    load_behavior = [0,1,5,8,...]:
        If load behavoir is a list of ints, loads the set of objects found at those indices in the
        a FileBrowser instantiated from filename.
    """
    if load_behavior == 'all':
        output = []
        for i in range(browser.N_dataobjects):
            output.append(browser.get_dataobject(i))

    elif load_behavior == 'datacube':
        try:
            output = browser.get_rawdatacubes()[0]
        except IndexError:
            try:
                output = browser.get_datacubes()[0]
            except IndexError:
                print("Error: no datacubes found. Returning None.")
                output = None

    elif isinstance(load_behavior, list):
        assert all([isinstance(x,int) for x in load_behavior]), "Error: when load_behavior is a list, all elements must be of type int."
        output = []
        for i in load_behavior:
            output.append(browser.get_dataobject(i))

    else:
        print("Error: unrecognized load_behavior {}. Must be 'all', 'datacube', or a list of ints.".format(load_behavior))

    return output

def read_datacube(filename):
    return read(filename, load_behavior="datacube")

def read_objects(filename, object_list):
    """
    Loads the set of objects found at the indices in object list in a FileBrowser instantiated from
    filename. object_list should be a list of ints.
    """
    return read(filename, load_behavior=object_list)







@log
def read_data(filename):
    """
    Takes a filename as input, and outputs a RawDataCube object.

    If filename is a .h5 file, read_data() checks if the file was written by py4DSTEM.  If it
    was, the metadata are read and saved directly.  Otherwise, the file is read with hyperspy,
    and metadata is scraped and saved from the hyperspy file.
    """
    print("Reading file {}...\n".format(filename))
    # Check if file was written by py4DSTEM
    try:
        h5_file = h5py.File(filename,'r')
        if is_py4DSTEM_file(h5_file):
            print("{} is a py4DSTEM HDF5 file.  Reading...".format(filename))
            R_Ny,R_Nx,Q_Ny,Q_Nx = h5_file['4DSTEM_experiment']['rawdatacube']['datacube'].shape
            rawdatacube = RawDataCube(data=h5_file['4DSTEM_experiment']['rawdatacube']['datacube'].value,
                            R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            is_py4DSTEM_file=True, h5_file=h5_file)
            h5_file.close()
            return rawdatacube
        else:
            h5_file.close()
    except IOError:
        pass

    # Use hyperspy
    print("{} is not a py4DSTEM file.  Reading with hyperspy...".format(filename))
    try:
        hyperspy_file = hs.load(filename)
        if len(hyperspy_file.data.shape)==3:
            R_N, Q_Ny, Q_Nx = hyperspy_file.data.shape
            R_Ny, R_Nx = R_N, 1
        elif len(hyperspy_file.data.shape)==4:
            R_Ny, R_Nx, Q_Ny, Q_Nx = hyperspy_file.data.shape
        else:
            print("Error: unexpected raw data shape of {}".format(hyperspy_file.data.shape))
            print("Initializing random datacube...")
            return RawDataCube(data=np.random.rand(100,512,512),
                            R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                            is_py4DSTEM_file=False)
        return RawDataCube(data=hyperspy_file.data, R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            is_py4DSTEM_file=False,
                            original_metadata_shortlist=hyperspy_file.metadata,
                            original_metadata_all=hyperspy_file.original_metadata)
    except Exception as err:
        print("Failed to load", err)
        print("Initializing random datacube...")
        return RawDataCube(data=np.random.rand(100,512,512),R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                           is_py4DSTEM_file=False)

@log
def read_data_v0_1(filename):
    """
    Takes a filename as input, and outputs a RawDataCube object.

    If filename is a .h5 file, read_data() checks if the file was written by py4DSTEM.  If it
    was, the metadata are read and saved directly.  Otherwise, the file is read with hyperspy,
    and metadata is scraped and saved from the hyperspy file.
    """
    print("Reading file {}...\n".format(filename))
    # Check if file was written by py4DSTEM
    try:
        h5_file = h5py.File(filename,'r')
        if is_py4DSTEM_file(h5_file):
            print("{} is a py4DSTEM HDF5 file.  Reading...".format(filename))
            R_Ny,R_Nx,Q_Ny,Q_Nx = h5_file['4D-STEM_data']['datacube']['datacube'].shape
            rawdatacube = RawDataCube(data=h5_file['4D-STEM_data']['datacube']['datacube'].value,
                            R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            is_py4DSTEM_file=True, h5_file=h5_file, py4DSTEM_version=(0,1))
            h5_file.close()
            return rawdatacube
        else:
            h5_file.close()
    except IOError:
        pass

    # Use hyperspy
    print("{} is not a py4DSTEM file.  Reading with hyperspy...".format(filename))
    try:
        hyperspy_file = hs.load(filename)
        if len(hyperspy_file.data.shape)==3:
            R_N, Q_Ny, Q_Nx = hyperspy_file.data.shape
            R_Ny, R_Nx = R_N, 1
        elif len(hyperspy_file.data.shape)==4:
            R_Ny, R_Nx, Q_Ny, Q_Nx = hyperspy_file.data.shape
        else:
            print("Error: unexpected raw data shape of {}".format(hyperspy_file.data.shape))
            print("Initializing random datacube...")
            return RawDataCube(data=np.random.rand(100,512,512),
                            R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                            is_py4DSTEM_file=False)
        return RawDataCube(data=hyperspy_file.data, R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            is_py4DSTEM_file=False,
                            original_metadata_shortlist=hyperspy_file.metadata,
                            original_metadata_all=hyperspy_file.original_metadata)
    except Exception as err:
        print("Failed to load", err)
        print("Initializing random datacube...")
        return RawDataCube(data=np.random.rand(100,512,512),R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                           is_py4DSTEM_file=False)


def show_log(filename):
    """
    Takes a filename as input, determines if it was written by py4DSTEM, and if it was prints
    the file's log.
    """
    print("Reading log for file {}...\n".format(filename))
    # Check if file was written by py4DSTEM
    try:
        h5_file = h5py.File(filename,'r')
        if is_py4DSTEM_file(h5_file):
            try:
                log_group = h5_file['log']
                for i in range(len(log_group.keys())):
                    key = 'log_item_'+str(i)
                    show_log_item(i, log_group[key])
                h5_file.close()
            except KeyError:
                print("Log cannot be read - HDF5 file has no log group.")
                h5_file.close()
        else:
            h5_file.close()
    except IOError:
        print("Log cannot be read - file is not an HDF5 file.")

def show_log_item(index, log_item):
    time = log_item.attrs['time'].decode('utf-8')
    function = log_item.attrs['function'].decode('utf-8')
    version = log_item.attrs['version']

    print("*** Log index {}, at time {} ***".format(index, time))
    print("Function: \t{}".format(function))
    print("Inputs:")
    for key,value in log_item['inputs'].attrs.items():
        if type(value)==np.bytes_:
            print("\t\t{}\t{}".format(key,value.decode('utf-8')))
        else:
            print("\t\t{}\t{}".format(key,value))
    print("Version: \t{}\n".format(version))



################# Utility functions ################

def is_py4DSTEM_file(h5_file):
    """
    Accepts either a filepath or an open h5py File object. Returns true if the file was written by
    py4DSTEM.
    """
    if isinstance(h5_file, h5py._hl.files.File):
        if ('version_major' in h5_file.attrs) and ('version_minor' in h5_file.attrs) and (('4DSTEM_experiment' in h5_file.keys()) or ('4D-STEM_data' in h5_file.keys())):
            return True
        else:
            return False
    else:
        try:
            f = h5py.File(h5_file, 'r')
            result = is_py4DSTEM_file(f)
            f.close()
            return result
        except OSError:
            return False

def get_py4DSTEM_version(h5_file):
    """
    Accepts either a filepath or an open h5py File object. Returns true if the file was written by
    py4DSTEM.
    """
    if isinstance(h5_file, h5py._hl.files.File):
        version_major = h5_file.attrs['version_major']
        version_minor = h5_file.attrs['version_minor']
        return version_major, version_minor
    else:
        try:
            f = h5py.File(h5_file, 'r')
            result = get_py4DSTEM_version(f)
            f.close()
            return result
        except OSError:
            print("Error: file cannot be opened with h5py, and may not be in HDF5 format.")
            return (0,0)


####### For filestructure summary, see end of writer.py #######


