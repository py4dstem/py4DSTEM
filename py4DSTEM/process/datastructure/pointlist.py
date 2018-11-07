# Defines a class - PointList - for storing / accessing / manipulating data in the form of
# lists of vectors.
#
# Coordinates must be defined on instantiation.  Often, the first four coordinates will be
# Qy,Qx,Ry,Rx, however, the class is flexible.

import numpy as np
from copy import copy
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
            self.add_pointarray(data)

    def add_point(self, point):
        point = tuple(point)
        assert(len(point)==len(self.dtype))
        self.data = np.append(self.data, np.array(point,dtype=self.dtype))

    def add_pointarray(self, pointarray):
        """
        pointarray must be an (n,m)-shaped ndarray, where m=len(coordinates), and n is the number of newpoints
        """
        assert(pointarray.shape[1]==len(self.dtype))
        for point in pointarray:
            self.add_point(point)

    def add_pointlist(self, pointlist):
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

    def get_subpointlist(self, coords_vals):
        """
        Returns a new PointList class instance, populated with points in self.data whose values for particular
        fields agree with those specified in coords_vals.

        Accepts:
            coords_vals - a list of 2-tuples (name, val) or 3-tuples (name, minval, maxval),
                          name should be a field in pointlist.dtype
        Returns:
            subpointlist - a new PointList class instance
        """
        deletemask = np.zeros(len(self.data),dtype=bool)
        for tup in coords_vals:
            assert((len(tup)==2) or (len(tup)==3))
            assert(tup[0] in self.dtype.names)
            if len(tup)==2:
                name,val = tup
                deletemask = deletemask | (self.data[name]!=val)
            else:
                name,minval,maxval = tup
                deletemask = deletemask | (self.data[name]<minval) | (self.data[name]>=maxval)
        new_pointlist = copy(self)
        new_pointlist.remove_points(deletemask)
        return new_pointlist

    def remove_points(self, deletemask):
        self.data = np.delete(self.data, deletemask.nonzero()[0])

