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
# The primary purpose of the DataObject class is to facilitate object level logging.
# Each instance maintains:
#   -a list of parent RawDataCube instances
#   -log indices when the object was created or modified
#   -save info, determining whether the complete data associated with the object is saved
# With respect to save info, note that if the complete data is not saved, the object name and
# log info still is, allowing it to be recreated. Save info must contain separate Boolean values
# for each parent RawDataCube.
#
# All objects containing py4DSTEM data - e.g. RawDataCube, DataCube, DiffractionSlice, 
# RealSlice, and PointList objects - inherit from DataObject.
# Only RawDataCube instances may have an empty parent list.
#
# The log_modification() method is called once on instantiation of a DataObject, and again by
# the @log decorator function whenever it identifies any of its arguments as DataObjects.

from ..log import Logger
logger = Logger()

class DataObject(object):
    """
    A DataObject:
        -maintains list of parent RawDataCubes
        -maintins a list of save information for each parent RawDataCube
        -maintains a list of log indices when the object was created/modified
    """
    def __init__(self, parent, default_save_behavior=True):

        self.default_save_behavior = default_save_behavior

        self.parents_and_save_behavior = list()
        self.new_parent(parent=parent, save_behavior=self.default_save_behavior)

        self.modification_log = list()
        self.log_modification()

    def new_parent(self, parent, **kwargs):
        if 'save_behavior' in kwargs.keys():
            save_behavior = kwargs['save_behavior']
        else:
            save_behavior = self.default_save_behavior
        if parent is not None:
            if not parent in self.get_parent_list():
                self.parents_and_save_behavior.append((parent,save_behavior))
            else:
                self.change_save_behavior(parent, save_behavior)
            # Check if the DataObject is in the parent's DataObjectTracker(s)
            # (If parent is not a raw datacube, note that it could have multiple trackers)
            # If not, add this DataObject to the parent's DataObjectTracker
            dataobjecttrackers = self.get_dataobjecttrackers(parent)
            for tracker in dataobjecttrackers:
                if not tracker.contains_dataobject(self):
                    tracker.new_dataobject(self, save_behavior=save_behavior)

    def get_dataobjecttrackers(self, dataobject):
        # Get all DataObjectTrackers associated with dataobject
        # Does not do a recursive search - rather, looks in dataobject and its direct parents
        dataobjecttrackers = []
        try:
            tracker = dataobject.dataobjecttracker
            dataobjecttrackers.append(tracker)
        except AttributeError:
            pass
        for parent in dataobject.get_parent_list():
            try:
                tracker = parent.dataobjecttracker
                dataobjecttrackers.append(tracker)
            except AttributeError:
                pass
        return dataobjecttrackers

    def get_parent_list(self):
        return [item[0] for item in self.parents_and_save_behavior]

    def change_save_behavior(self, parent, save_behavior):
        assert parent in self.get_parent_list()
        index = self.get_parent_list().index()
        self.parents_and_save_behavior[index][1] = save_behavior

    def get_save_behavior(self, parent):
        assert parent in self.get_parent_list()
        index = self.get_parent_list().index(parent)
        return self.parents_and_save_behavior[index][1]

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
#   -log info
#       -log index of object creation
#       -log indices of object modification
#   -save info. Boolean which determines behavior for this object on saving:
#       -if True, save this object in its entirity
#       -if False, save object name and log info, but not the actual data
# When an object is added to a RawDataCube's DataObjectTracker, the original DataObject adds that
# RawDataCube instance to its list of parents, ensuring the relationships can be deterimined in
# either direction.

class DataObjectTracker(object):

    def __init__(self, rawdatacube):

        self.rawdatacube = rawdatacube
        self.dataobject_list = list()

    def new_dataobject(self, dataobject, **kwargs):
        assert isinstance(dataobject, DataObject), "{} is not a DataObject instance".format(dataobject)
        if not dataobject in self.dataobject_list:
            self.dataobject_list.append(dataobject)
        # Check if the DataObject's parent list contains this tracker's top level RawDataCube.
        # If not, add that RawDataCube to the DataObjects parent list.
        if not self.contains_rawdatacube(dataobject, self.rawdatacube):
            if 'save_behavior' in kwargs.keys():
                dataobject.new_parent(self.rawdatacube, kwargs['save_behavior'])
            else:
                dataobject.new_parent(self.rawdatacube)

    def contains_rawdatacube(self, dataobject, rawdatacube):
        return rawdatacube in dataobject.get_parent_list()

    def contains_dataobject(self, dataobject):
        return dataobject in self.dataobject_list


