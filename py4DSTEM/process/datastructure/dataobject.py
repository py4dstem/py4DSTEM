"""
Once we have processed DataObjects…
	-Placement in base DataObjectTracker:
		-Placed correctly in tracker on instantiation
		-Correctly saved from tracker
			-With save_behavior flag set to True and False
	-Other DataObjectTrackers
		-Objects can be placed in multiple RawDataObject’s trackers
		-DataObject parent list and tracker object lists are both correctly updated
			-Regardless of where adding is done (i.e. from object or tracker)
		-Saved from either / both RawDataCubes
			-save_behavior is correct in each case, even when differs between two parents
	-DataObject level logging
		-Add object level logging in DataObject scope
		-In @log function:
			-check each input to see if it’s a DataObject
			-If it is, add log index to DataObject.modification_log
"""

# Defines the DataObject class.
#
# The purpose of the DataObject class is to:
#   -create a single, uniform interface for all of the types of data py4DSTEM creates
#   -facilitate tracking of which objects were created from, refer to, or interact with
#    which parent DataCubes
#   -(*eventually*) enable object level logging.
# Each instance maintains:
#   -a list of parent RawDataCube instances
#   -log indices when the object was created or modified
#   -save info, determining whether the object should be saved
#
# All objects containing py4DSTEM data - e.g. RawDataCube, DataCube, DiffractionSlice, 
# RealSlice, and PointList objects - inherit from DataObject.
# Only RawDataCube instances may have an empty parent list.
#
# The log_modification() method is called once on instantiation of a DataObject.
# For object-level logging, the @log decorator function should be called whenever it identifies 
# any of its arguments as DataObjects.

from functools import wraps
from ..log import Logger
logger = Logger()

class DataObject(object):
    """
    A DataObject:
        -maintains list of parent RawDataCubes
        -stores a name
        -stores a (default) save behavior
        -maintains a list of log indices when the object was created/modified
    """
    def __init__(self, parent, save_behavior=False, name=''):

        self.save_behavior = save_behavior
        self.name = name

        self.parents = list()
        self.new_parent(parent=parent, save_behavior=self.save_behavior)

        self.modification_log = list()
        self.log_modification()

    def new_parent(self, parent, **kwargs):
        if 'save_behavior' in kwargs.keys():
            save_behavior = kwargs['save_behavior']
        else:
            save_behavior = self.save_behavior
        if parent is not None:
            if not parent in self.parents:
                self.parents.append(parent)
        # Check if the DataObject is in the parent's DataObjectTracker(s)
        # (If parent is not a raw datacube, note that it could have multiple trackers)
        # If not, add this DataObject and its save behavior to the parent's DataObjectTracker
        if parent is None:
            dataobjecttrackers = self.get_dataobjecttrackers()
        else:
            dataobjecttrackers = parent.get_dataobjecttrackers()
        for tracker in dataobjecttrackers:
            if not tracker.contains_dataobject(self):
                tracker.new_dataobject(self, save_behavior=save_behavior)
            else:
                tracker.change_save_behavior(self, save_behavior=save_behavior)

    def get_dataobjecttrackers(self):
        # Get all DataObjectTrackers associated with dataobject
        # Does not do a recursive search - rather, looks in dataobject and its direct parents
        dataobjecttrackers = []
        try:
            tracker = self.dataobjecttracker
            dataobjecttrackers.append(tracker)
        except AttributeError:
            pass
        for parent in self.parents:
            try:
                tracker = parent.dataobjecttracker
                dataobjecttrackers.append(tracker)
            except AttributeError:
                pass
        return dataobjecttrackers

    def change_save_behavior(self, save_behavior, parent=None):
        if parent is None:
            trackers = self.get_dataobjecttrackers()
        else:
            assert parent in self.parents
            trackers = parent.get_dataobjecttrackers()
        for tracker in trackers:
            tracker.change_save_behavior(self, save_behavior)

    def get_save_behavior(self, parent=None):
        if parent is None:
            trackers = self.get_dataobjecttrackers()
        else:
            assert parent in self.parents()
            trackers = parent.get_dataobjecttrackers()
        ans = []
        for tracker in trackers:
            index = tracker.get_object_index(self)
            save_behavior = tracker.dataobject_list[index][2]
            parent = tracker.rawdatacube
            ans.append((parent, save_behavior))
        return ans

    def has_parent(self, datacube):
        return datacube in self.parents

    def log_modification(self):
        index = self.get_current_log_index()-1
        self.modification_log.append(index)

    @staticmethod
    def get_current_log_index():
        global logger
        return logger.log_index


# Defines the DataObjectTracker class.
#
# Each RawDataCube object contains a DataObjectTracker instance, which keeps track of all the
# data objects created - DataCube, DiffractionSlice, RealSlice, and PointList objects - with 
# reference to this dataset.
# The DataObjectTracker stores a list of DataObject instances, and knows how to retreive or
# modify their attributes, in particular:
#   -name
#   -object type
#   -save behavior
#   -pointer to the object
# To be implemented (eventually?)
#   -log info
#       -log index of object creation
#       -log indices of object modification
# When an object is added to a RawDataCube's DataObjectTracker, the original DataObject adds that
# RawDataCube instance to its list of parents, ensuring the relationships can be deterimined in
# either direction.
# This interface works for now, but seems awfully kludgy.  The right move may be a database, but
# it would be nice to avoid having an extra .db file floating around (and also needing to use
# SQL / some database language), at least for now.

# Decorator which enables more human-readable display of tracked dataobjects
def show_object_list(method):
    @wraps(method)
    def wrapper(self, *args, show=False, **kwargs):
        objectlist = method(self, *args, **kwargs)
        if show:
            print("{:^8}{:^36}{:^20}{:^10}".format('Index', 'Name', 'Type', 'Save'))
            for item in objectlist:
                if item[3]:
                    save='Y'
                else:
                    save='N'
                print("{:^8}{:<36s}{:<20}{:^10}".format(item[0],item[1],item[2].__name__,save))
            return
        else:
            return objectlist
    return wrapper

class DataObjectTracker(object):

    def __init__(self, rawdatacube, save_behavior=True):
        """
        Instantiate a DataObjectTracker class instance with rawdatacube as its parent.
        save_behavior refers to the save behavior of this RawDataCube instance.
        """
        self.rawdatacube = rawdatacube
        self.dataobject_list = list()
        self.new_dataobject(self.rawdatacube, save_behavior=True)

    def new_dataobject(self, dataobject, **kwargs):
        assert isinstance(dataobject, DataObject), "{} is not a DataObject instance".format(dataobject)
        if not dataobject in self.dataobject_list:
            index = len(self.dataobject_list)
            if 'name' in kwargs.keys():
                name = kwargs['name']
            else:
                name = dataobject.name
            objecttype = type(dataobject)
            if 'save_behavior' in kwargs.keys():
                save_behavior = kwargs['save_behavior']
            else:
                save_behavior = dataobject.save_behavior
            l = [index, name, objecttype, save_behavior, dataobject]
            self.dataobject_list.append(l)
        # Check if the DataObject's parent list contains this tracker's top level RawDataCube.
        # If not, add that RawDataCube to the DataObjects parent list.
        if not dataobject.has_parent(self.rawdatacube):
            dataobject.new_parent(self.rawdatacube)

    def contains_dataobject(self, dataobject):
        return dataobject in [item[4] for item in self.dataobject_list]

    def get_object_index(self, dataobject):
        return [item[4] for item in self.dataobject_list].index(dataobject)

    def change_save_behavior(self, dataobject, save_behavior):
        """
        If dataobject is a single DataObject instance, change its save behavior to save_behavior.
        If dataobject is the string 'all', apply to all objects in the tracker.
        If dataobject is a list of dataobjects, apply to all objects in the list.
        """
        if isinstance(dataobject, DataObject):
            index = self.get_object_index(dataobject)
            self.dataobject_list[index][3] = save_behavior
        elif dataobject=='all':
            self.change_all_save_behaviors(save_behavior)
        elif isinstance(dataobject, list):
            for obj in dataobject:
                self.change_save_behavior(obj, save_behavior)
        else:
            print("{} is not a valid argument".format(dataobject))

    def change_all_save_behaviors(self, save_behavior):
        for i in range(len(self.dataobject_list)):
            self.dataobject_list[i][3] = save_behavior

    def change_save_behavior_by_index(self, index, save_behavior):
        self.dataobject_list[index][3] = save_behavior

    def get_save_behavior_list(self):
        return [item[3] for item in self.dataobject_list]

    def show_dataobjects(self):
        self.get_dataobjects(show=True)

    @show_object_list
    def get_dataobjects(self):
        return self.dataobject_list

    @show_object_list
    def sort_dataobjects_by_name(self):
        return [item for item in self.dataobject_list if item[1]!=''] + \
               [item for item in self.dataobject_list if item[1]=='']

    @show_object_list
    def sort_dataobjects_by_type(self, objecttype=None):
        if objecttype is None:
            types=[]
            for item in self.dataobject_list:
                if item[2] not in types:
                    types.append(item[2])
            l=[]
            for objecttype in types:
                l += [item for item in self.dataobject_list if item[2]==objecttype]
        else:
            l = [item for item in self.dataobject_list if item[2]==objecttype]
        return l

    @show_object_list
    def get_object_by_name(self, name, exactmatch=False):
        if exactmatch:
            return [item[4] for item in self.dataobject_list if name == item[1]]
        else:
            return [item[4] for item in self.dataobject_list if name in item[1]]

    @show_object_list
    def get_object_by_index(self, index):
        return self.dataobject_list[index][4]


