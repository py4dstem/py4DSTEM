# Defines a class - PointList - for storing / accessing / manipulating data in the form of
# lists of vectors.
#
# Coordinates must be defined on instantiation.  Often, the first four coordinates will be
# Qy,Qx,Ry,Rx, however, the class is flexible.

class PointList(object):

    def __init__(self, coordinates, data, parentDataCube):
        """
        Instantiate a PointList object.
        Defines the coordinates, data, and parentDataCube.
        """
        self.parentDataCube = parentDataCube
        self.coordinates = coordinates
        self.data = data

