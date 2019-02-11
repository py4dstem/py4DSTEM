# Defines the DataObject class.
#
# The purpose of the DataObject class is to create a single, uniform interface for all of the types 
# of data py4DSTEM creates. It enables:
#       -storing a name, which will be associated with the object on saving
#       -linking to metadata
#       -listing and searching all DataObjects, and DataObjects by the various child class types
#
# All objects containing py4DSTEM data - e.g. DataCube, DiffractionSlice, RealSlice, and PointLists
# - inherit from DataObject.

#from functools import wraps
#from ..log import Logger
#logger = Logger()

import weakref

class DataObject(object):
    """
    A DataObject:
        -stores a name
        -stores pointer to metadata
        -enables listing all py4DSTEM dataobject instances in a session
    """
    _instances = set()

    def __init__(self, name='', metadata=None, **kwargs):

        self.name = name
        self.metadata = metadata # TODO: point to metadata
        self._instances.add(weakref.ref(self))

        # TODO: add logging of instantiation

    @classmethod
    def getinstances(cls):
        """
        Returns a generator of all DataObject instances.
        """
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead



# # Decorator which enables more human-readable display of tracked dataobjects
# def show_object_list(method):
#     @wraps(method)
#     def wrapper(self, *args, show=False, **kwargs):
#         objectlist = method(self, *args, **kwargs)
#         if show:
#             print("{:^8}{:^36}{:^20}{:^10}".format('Index', 'Name', 'Type', 'Save'))
#             for item in objectlist:
#                 if item[3]:
#                     save='Y'
#                 else:
#                     save='N'
#                 print("{:^8}{:<36s}{:<20}{:^10}".format(item[0],item[1],item[2].__name__,save))
#             return
#         else:
#             return objectlist
#     return wrapper


