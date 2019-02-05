# Defines the DataObject class.
#
# The purpose of the DataObject class is to create a single, uniform interface for all of the types 
# of data py4DSTEM creates. It enables:
#       -storing a name, which will be associated with the object on saving
#       -linking to metadata
#       -listing all py4DSTEM dataobject instances present in a python session
#
# All objects containing py4DSTEM data - e.g. DataCube, DiffractionSlice, 
# RealSlice, and PointList objects - inherit from DataObject.

#from functools import wraps
#from ..log import Logger
#logger = Logger()

class DataObject(object):
    """
    A DataObject:
        -stores a name
        -stores pointer to metadata
        -enables listing all py4DSTEM dataobject instances in a session
    """
    def __init__(self, name='', metadata=None, **kwargs):

        self.name = name
        self.metadata = metadata # TODO: point to metadata

        # TODO: add logging of instantiation

    # TODO
    @classmethod
    def getinstances(cls):
        pass


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


