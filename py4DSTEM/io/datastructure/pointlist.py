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
    Essentially a wrapper around a structured numpy array, which py4DSTEM can save and load.

    Args:
        coordinates (list): specifies the coordinates. Can be either (1) a list of
            strings, or (2) a list of length-2 tuples.  In the latter case each tuple
            specifies a coordinate name a dtype.  In the former case only the name is
            specified, and all datatypes will default to the ``dtype`` kwarg
        data (ndarray): the data, which shape (n,m), where m=len(coordinates), and n
            is the number of points being specified additional data can be added
            later with the add_point() method
        dtype (dtype, optional): used if coordinates don't explicitly specify dtypes.
    """
    def __init__(self, coordinates, data=None, dtype=float, **kwargs):
        """
		Instantiate a PointList object.
		Defines the coordinates, data, and parentDataCube.
        """
        DataObject.__init__(self, **kwargs)

        #: either a list of length 2 tuples, each a (string,dtype) pair, specifying
        #: each of the coordinates' name and type, or, a list of strings, specifying
        #: each of the coordinates' names. In the latter case all datatypes will
        #: use the default dtype (defaults to float)
        self.coordinates = coordinates
        self.default_dtype = dtype  #: the default datatype
        self.length = 0  #: the number of coordinates

        # Define the data type for the PointList structured array
        if isinstance(coordinates,np.dtype):
            self.dtype = coordinates  #: the custom datatype, generated from the coordinates
        elif type(coordinates[0])==str:
            self.dtype = np.dtype([(name,self.default_dtype) for name in coordinates])
        elif type(coordinates[0])==tuple:
            self.dtype = np.dtype(coordinates)
        else:
            raise TypeError("coordinates must be a list of strings, or a list of 2-tuples of structure (name, dtype).")

        if data is not None:
            if isinstance(data, PointList):
                # If types agree, add all at once
                assert self.dtype==data.data.dtype, "Error: dtypes must agree"
                self.data = data.data
                self.length = np.atleast_1d(data.data).shape[0]
            elif isinstance(data, np.ndarray):
                # If types agree, add all at once
                assert self.dtype==data.dtype, "Error: dtypes must agree"
                self.data = data
                self.length = np.atleast_1d(data).shape[0]
            elif isinstance(data, tuple):
                self.data = np.array([],dtype=self.dtype)  #: the data; a numpy structured array
                self.add_tuple_of_nparrays(data)
            else:
                self.data = np.array([],dtype=self.dtype)  #: the data; a numpy structured array
                self.add_pointarray(data) # Otherwise, add one by one
        else:
            self.data = np.array([],dtype=self.dtype)  #: the data; a numpy structured array

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
        self.length += np.atleast_1d(data).shape[0]

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
        """ Rms points wherever deletemask==True
        """
        assert np.atleast_1d(deletemask).shape[0] == self.length, "deletemask must be same length as the data"
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


### Read/Write

def save_pointlist_group(group, pointlist):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    n_coords = len(pointlist.dtype.names)
    coords = np.string_(str([coord for coord in pointlist.dtype.names]))
    group.attrs.create("coordinates", coords)
    group.attrs.create("dimensions", n_coords)
    group.attrs.create("length", pointlist.length)

    for name in pointlist.dtype.names:
        group_current_coord = group.create_group(name)
        group_current_coord.attrs.create("dtype", np.string_(pointlist.dtype[name]))
        group_current_coord.create_dataset("data", data=pointlist.data[name])

def get_pointlist_from_grp(g):
    """ Accepts an h5py Group corresponding to a pointlist in an open, correctly formatted H5 file,
        and returns a PointList.
    """
    name = g.name.split('/')[-1]
    coordinates = []
    coord_names = list(g.keys())
    length = len(g[coord_names[0]+'/data'])
    if length==0:
        for coord in coord_names:
            coordinates.append((coord,None))
    else:
        for coord in coord_names:
            dtype = type(g[coord+'/data'][0])
            coordinates.append((coord, dtype))
    data = np.zeros(length,dtype=coordinates)
    for coord in coord_names:
        data[coord] = np.array(g[coord+'/data'])
    return PointList(data=data,coordinates=coordinates,name=name)



