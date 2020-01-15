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
    def __init__(self, coordinates, data=None, dtype=float, **kwargs):
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
            dtype - optional, used if coordinates don't explicitly specify dtypes.
        """
        DataObject.__init__(self, **kwargs)

        self.coordinates = coordinates
        self.default_dtype = dtype
        self.length = 0

        # Define the data type for the PointList structured array
        if type(coordinates) in (type, np.dtype):
            self.dtype = coordinates
        elif type(coordinates[0])==str:
            self.dtype = np.dtype([(name,self.default_dtype) for name in coordinates])
        elif type(coordinates[0])==tuple:
            self.dtype = np.dtype(coordinates)
        else:
            raise TypeError("coordinates must be a list of strings, or a list of 2-tuples of structure (name, dtype).")

        self.data = np.array([],dtype=self.dtype)

        if data is not None:
            if isinstance(data, PointListArray):
                self.add_pointlist(data)  # If types agree, add all at once
            elif isinstance(data, np.ndarray):
                self.add_dataarray(data)  # If types agree, add all at once
            elif isinstance(data, tuple):
                self.add_tuple_of_nparrays(data)
            else:
                self.add_pointarray(data) # Otherwise, add one by one

    def add_point(self, point):
        point = tuple(point)
        assert len(point)==len(self.dtype)
        self.data = np.append(self.data, np.array(point,dtype=self.dtype))
        self.length += 1

    def add_pointarray(self, pointarray):
        """
        pointarray must be an (n,m)-shaped ndarray, where m=len(coordinates), and n is the number of newpoints
        """
        assert len(pointarray[0])==len(self.dtype)
        for point in pointarray:
            self.add_point(point)

    def add_pointlist(self, pointlist):
        """
        Appends the data from another PointList object to this one.  Their dtypes must agree.
        """
        assert self.dtype==pointlist.dtype, "Error: dtypes must agree"
        self.data = np.append(self.data, pointlist.data)
        self.length += pointlist.length

    def add_dataarray(self, data):
        """
        Appends a properly formated numpy structured array.  Their dtypes must agree.
        """
        assert self.dtype==data.dtype, "Error: dtypes must agree"
        self.data = np.append(self.data, data)
        self.length += len(data)

    def add_unstructured_dataarray(self, data):
        """
        Appends data in the form of an unstructured data array.  User warning: be careful to make
        sure the columns of the data are in the correct order, i.e. correspond to the order of the
        PointList coordinates.
        """
        assert len(self.coordinates)==(data.shape[1]), "Error: number of data columns must match the number of PointList coords."
        length = data.shape[1]

        # Make structured numpy array
        structured_data = np.empty(data.shape[0], dtype=self.dtype)
        for i in range(length):
            name = self.dtype.names[i]
            structured_data[name] = data[:,i]

        # Append to pointlist
        self.add_dataarray(structured_data)

    def add_tuple_of_nparrays(self, data):
        """
        Appends data in the form of a tuple of ndarrays.
        """
        assert isinstance(data,tuple)
        assert len(self.coordinates)==len(data), "Error: if data is a tuple, it must contain (# coords) numpy arrays."
        length = len(data[0])
        assert all([length==len(ar) for ar in data]), "Error: if data is a tuple, it must contain lists of equal length"

        # Make structured numpy array
        structured_data = np.empty(length, dtype=self.dtype)
        for i in range(len(data)):
            name = self.dtype.names[i]
            structured_data[name] = data[i]

        # Append to pointlist
        self.add_dataarray(structured_data)

    def sort(self, coordinate, order='descending'):
        """
        Sorts the point list according to coordinate.
        coordinate must be a field in self.dtype.
        order should be 'descending' or 'ascending'.
        """
        assert coordinate in self.dtype.names
        assert (order=='descending') or (order=='ascending')
        if order=='ascending':
            self.data = np.sort(self.data, order=coordinate)
        else:
            self.data = np.sort(self.data, order=coordinate)[::-1]

    def get_subpointlist(self, coords_vals):
        """
        Returns a new PointList class instance, populated with points in self.data whose values
        for particular fields agree with those specified in coords_vals.

        Accepts:
            coords_vals - a list of 2-tuples (name, val) or 3-tuples (name, minval, maxval),
                          name should be a field in pointlist.dtype
        Returns:
            subpointlist - a new PointList class instance
        """
        deletemask = np.zeros(len(self.data),dtype=bool)
        for tup in coords_vals:
            assert (len(tup)==2) or (len(tup)==3)
            assert tup[0] in self.dtype.names
            if len(tup)==2:
                name,val = tup
                deletemask = deletemask | (self.data[name]!=val)
            else:
                name,minval,maxval = tup
                deletemask = deletemask | (self.data[name]<minval) | (self.data[name]>=maxval)
        new_pointlist = self.copy()
        new_pointlist.remove_points(deletemask)
        return new_pointlist

    def remove_points(self, deletemask):
        self.data = np.delete(self.data, deletemask.nonzero()[0])
        self.length -= len(deletemask.nonzero()[0])

    def copy(self, **kwargs):
        new_pointlist = PointList(coordinates=self.coordinates,
                                  data=np.copy(self.data),
                                  dtype=self.dtype,
                                  **kwargs)
        return new_pointlist

    def add_coordinates(self, new_coords):
        """
        Creates a copy of the PointList, but with additional coordinates given by new_coords.
        new_coords must be a string of 2-tuples, ('name', dtype)
        """
        coords = []
        for key in self.dtype.fields.keys():
            coords.append((key,self.dtype.fields[key][0]))
        for coord in new_coords:
            coords.append((coord[0],coord[1]))

        data = np.zeros(self.length, np.dtype(coords))
        for key in self.dtype.fields.keys():
            data[key] = np.copy(self.data[key])

        return PointList(coordinates=coords, data=data)


class PointListArray(DataObject):
    """
    An object containing an array of PointLists.
    Facilitates more rapid access of subpointlists which have known, well structured coordinates, such
    as real space scan positions R_Nx,R_Ny.
    """
    def __init__(self, coordinates, shape, dtype=float, **kwargs):
        """
		Instantiate a PointListArray object.
		Creates a PointList with coordinates at each point of a 2D grid with a shape specified by
        the shape argument.

		Inputs:
			coordinates - see PointList documentation
            shape - a 2-tuple of ints specifying the array shape.  Often the desired shape
                    will be the real space shape (R_Nx, R_Ny).
        """
        DataObject.__init__(self, **kwargs)

        self.coordinates = coordinates
        self.default_dtype = dtype

        assert isinstance(shape,tuple), "Shape must be a tuple."
        assert len(shape) == 2, "Shape must be a length 2 tuple."
        self.shape = shape

        # Define the data type for the structured arrays in the PointLists
        if type(coordinates) in (type, np.dtype):
            self.dtype = coordinates
        elif type(coordinates[0])==str:
            self.dtype = np.dtype([(name,self.default_dtype) for name in coordinates])
        elif type(coordinates[0])==tuple:
            self.dtype = np.dtype(coordinates)
        else:
            raise TypeError("coordinates must be a list of strings, or a list of 2-tuples of structure (name, dtype).")

        kwargs['searchable']=False   # Ensure that the subpointlists don't all appear in searches
        self.pointlists = [[PointList(coordinates=self.coordinates,
                            dtype = self.default_dtype,
                            **kwargs) for j in range(self.shape[1])] for i in range(self.shape[0])]

    def get_pointlist(self, i, j):
        """
        Returns the pointlist at i,j
        """
        return self.pointlists[i][j]

    def copy(self, **kwargs):
        """
        Returns a copy of itself.
        """
        new_pointlistarray = PointListArray(coordinates=self.coordinates,
                                            shape=self.shape,
                                            dtype=self.default_dtype,
                                            **kwargs)

        for i in range(new_pointlistarray.shape[0]):
            for j in range(new_pointlistarray.shape[1]):
                curr_pointlist = new_pointlistarray.get_pointlist(i,j)
                curr_pointlist.add_pointlist(self.get_pointlist(i,j).copy())

        return new_pointlistarray

    def add_coordinates(self, new_coords, **kwargs):
        """
        Creates a copy of the PointListArray, but with additional coordinates given by new_coords.
        new_coords must be a string of 2-tuples, ('name', dtype)
        """
        coords = []
        for key in self.dtype.fields.keys():
            coords.append((key,self.dtype.fields[key][0]))
        for coord in new_coords:
            coords.append((coord[0],coord[1]))

        new_pointlistarray = PointListArray(coordinates=coords,
                                            shape=self.shape,
                                            dtype=self.default_dtype,
                                            **kwargs)

        for i in range(new_pointlistarray.shape[0]):
            for j in range(new_pointlistarray.shape[1]):
                curr_pointlist_new = new_pointlistarray.get_pointlist(i,j)
                curr_pointlist_old = self.get_pointlist(i,j)

                data = np.zeros(curr_pointlist_old.length, np.dtype(coords))
                for key in self.dtype.fields.keys():
                    data[key] = np.copy(curr_pointlist_old.data[key])

                curr_pointlist_new.add_dataarray(data)

        return new_pointlistarray


