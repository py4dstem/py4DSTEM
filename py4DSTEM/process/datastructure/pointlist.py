# Defines a class - PointList - for storing / accessing / manipulating data in the form of
# lists of vectors.
#
# Coordinates must be defined on instantiation.  Often, the first four coordinates will be
# Qy,Qx,Ry,Rx, however, the class is flexible.

import numpy as np
from .dataobject import DataObject

class PointList(DataObject):
    """
    Essentially a wrapper around a structured numpy array, with the py4DSTEM DataObject
    interface enabling saving and logging.
    """
    def __init__(self, coordinates, parentDataCube, data=None, dtype=float):
        """
		Instantiate a PointList object.
		Defines the coordinates, data, and parentDataCube.

		Inputs:
			coordinates - a list specifying the coordinates. Can be:
                          (1) a list of strings, which specify coordinate names. datatypes will
                          all default to the dtype kwarg
                          (2) a list of length-2 tuples, each a (string, dtype) pair, specifying
                          coordinate names and types
            data - an (n,m)-shape ndarray, where m=len(coordinates), and n is the number of
                   points being specified additional data can be added later with the
                   add_point() method
            parentDataCube - a DataCube object
            dtype - optional, used if coordinates don't explicitly specify dtypes.
        """
        DataObject.__init__(self, parent=parentDataCube)

        self.parentDataCube = parentDataCube
        self.default_dtype = dtype

        # Define the the data type for the PointList structured array
        if type(coordinates[0])==str:
            self.dtype = np.dtype([(name,default_type) for name in coordinates])
        elif type(coordinates[0])==tuple:
            self.dtype = np.dtype(coordinates)
        else:
            raise TypeError("coordinates must be a list of strings, or a list of 2-tuples of structure (name, dtype).")

        self.data = np.array([],dtype=self.dtype)

        if data is not None:
            self.new_pointarray(data)

    def new_point(self, point):
        point = tuple(point)
        assert(len(point)==len(self.dtype))
        self.data = np.append(self.data, np.array(point,dtype=self.dtype))

    def new_pointarray(self, pointarray):
        """
        pointarray must be an (n,m)-shaped ndarray, where m=len(coordinates), and n is the number of newpoints
        """
        assert(pointarray.shape[1]==len(self.dtype))
        for point in pointarray:
            self.new_point(point)

    def append_pointlist(self, pointlist):
        """
        Appends the data from another PointList object to this one.  Their dtypes must agree.
        """
        assert(self.dtype==pointlist.dtype)
        self.data = np.append(self.data, pointlist.data)

    def sort(self, coordinate, order='descending'):
        """
        Sorts the point list according to coordinate.
        coordinate must be a field in self.dtype.
        order should be 'descending' or 'ascending'.
        """
        assert(coordinate in self.dtype.names)
        assert((order=='descending') or (order=='ascending'))
        if order=='ascending':
            self.data = np.sort(self.data, order=coordinate)
        else:
            self.data = np.sort(self.data, order=coordinate)[::-1]



