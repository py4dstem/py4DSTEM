# Defines a class - PointList - for storing / accessing / manipulating data in the form of
# lists of vectors.
#
# Coordinates must be defined on instantiation.  Often, the first four coordinates will be
# Qy,Qx,Ry,Rx, however, the class is flexible.

from .dataobject import DataObject

class PointList(DataObject):

    def __init__(self, coordinates, data, parentDataCube):
        """
        Instantiate a PointList object.
        Defines the coordinates, data, and parentDataCube.
        """
        DataObject.__init__(self, parent=parentDataCube)

        self.parentDataCube = parentDataCube
        self.coordinates = coordinates
        self.data = data

