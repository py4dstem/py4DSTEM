# Defines the DataObject class.
#
# The purpose of the DataObject class is to create a single, uniform interface for all of the types 
# of data py4DSTEM creates. It enables:
#       -searching and retrieving/listing dataobjects in memory, by name, by child class type
#       -naming dataobjects
#       -linking to metadata
#
# All objects containing py4DSTEM data - e.g. DataCube, DiffractionSlice, RealSlice, and PointLists
# - inherit from DataObject.

#from ..log import Logger
#logger = Logger()

import weakref
from functools import wraps

# Decorator which enables more human-readable display of tracked dataobjects
def show_object_list(method):
    @wraps(method)
    def wrapper(*args, show=False, **kwargs):
        objectlist = method(*args, **kwargs)
        if show:
            print("{:^8}{:^36}{:^20}".format('Index', 'Name', 'Type'))
            for item in objectlist:
                print("{:^8}{:<36s}{:<20}".format(item[0],item[1],item[2].__name__))
            return
        else:
            return objectlist
    return wrapper


################## BEGIN DataObject Class ###################

class DataObject(object):
    """
    A DataObject:
        -stores a name and a pointer to metadata
        -enables searching/listing all py4DSTEM dataobject instances in a session

    If the searchable keyword is set to False, a dataobject will not be tracked by the DataObject
    class and will not be found or returned by its search methods.
    """
    _instances = []

    def __init__(self, name='', metadata=None, searchable=True, **kwargs):
        """
        Instantiate a DataObject instance.

        Inputs:
            name      a string which will be used to identify the object in .h5 files and logs
            metadata  if specified, should point to a Metadata object, or to a DataObject.
                      if metadata is a dataobject, self.metadata will point to DataObject.metadata.
        """
        self.name = name
        if isinstance(metadata, DataObject):
            self.metadata = metadata.metadata
        else:
            self.metadata = None
        if searchable==True:
            self._instances.append(weakref.ref(self))

        # TODO: add logging of instantiation

    def link_metadata(self, dataobject):
        """
        Sets self.metadata to point to dataobject.metadata.
        If dataobject is a Metadata object, sets self.metadata to point to dataobject.
        """
        assert isinstance(dataobject, DataObject)
        self.metadata = dataobject.metadata

    ############ Searching methods ############

    @classmethod
    def get_dataobjects(cls):
        """
        Return a list of all dataobjects.
        """
        dataobjects = []
        remove = []
        for i in range(len(cls._instances)):
            obj = cls._instances[i]()
            if obj is not None:
                dataobjects.append(obj)
            else:
                remove.append(i)
        for i in range(len(remove)):
            del(cls._instances[remove[::-1][i]])
        return dataobjects

    @staticmethod
    @show_object_list
    def get_dataobject_list():
        """
        Returns a list containing, for each dataobject, a list of its:
            [index     name      objecttype      dataobject]
        """
        dataobjects = DataObject.get_dataobjects()
        dataobject_list = []
        for index in range(len(dataobjects)):
            dataobject = dataobjects[index]
            assert isinstance(dataobject, DataObject), "{} is not a DataObject instance".format(dataobject)
            name = dataobject.name
            objecttype = type(dataobject)
            dataobject_list.append([index, name, objecttype, dataobject])
        return dataobject_list

    @staticmethod
    def show_dataobjects():
        DataObject.get_dataobject_list(show=True)

    @staticmethod
    @show_object_list
    def sort_dataobjects_by_name():
        dataobject_list = DataObject.get_dataobject_list()
        return [item for item in dataobject_list if item[1]!=''] + \
               [item for item in dataobject_list if item[1]=='']

    @staticmethod
    @show_object_list
    def sort_dataobjects_by_type(objecttype=None):
        dataobject_list = DataObject.get_dataobject_list()
        if objecttype is None:
            types=[]
            for item in dataobject_list:
                if item[2] not in types:
                    types.append(item[2])
            l=[]
            for objecttype in types:
                l += [item for item in dataobject_list if item[2]==objecttype]
        else:
            l = [item for item in dataobject_list if item[2]==objecttype]
        return l

    @staticmethod
    @show_object_list
    def get_dataobject_by_name(name, exactmatch=False):
        dataobject_list = DataObject.get_dataobject_list()
        if exactmatch:
            return [item[3] for item in dataobject_list if name == item[1]]
        else:
            return [item[3] for item in dataobject_list if name in item[1]]

    @staticmethod
    @show_object_list
    def get_dataobject_by_index(index):
        dataobject_list = DataObject.get_dataobject_list()
        return dataobject_list[index][3]

    @staticmethod
    @show_object_list
    def get_dataobject_by_type(objecttype):
        dataobject_list = DataObject.get_dataobject_list()
        return [item[3] for item in dataobject_list if isinstance(item[3], objecttype)]



