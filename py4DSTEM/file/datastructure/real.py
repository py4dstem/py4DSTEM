# Defines a class - RealSlice - for storing / accessing / manipulating data that lives in
# real space.  RealSlice inherits from DataSlice.

from collections import OrderedDict
from .dataobject import DataObject
from .dataslice import DataSlice

class RealSlice(DataObject):

    def __init__(self, data, R_Nx=None, R_Ny=None, slicelabels=None, **kwargs):
        """
        Instantiate a RealSlice object.  Set the data and dimensions.

        If data is two dimensional, it is stored as self.data, and has shape (R_Nx, R_Ny).
        If data is three dimensional, self.data is a list of slices of some depth,
        where self.depth is data.shape[2], i.e. the shape is (R_Nx, R_Ny, depth).
        If slicelabels is unspecified, 2D slices can be accessed as self.data[i].
        If slicelabels is specified, it should be an n-tuple of strings, where
        n==self.depth, and 2D slices can be accessed as self.data[slicelabels[i]].
        """
        # Get shape
        if R_Nx is None:
            self.R_Nx = data.shape[0]
        else:
            self.R_Nx = R_Nx
        if R_Ny is None:
            self.R_Ny = data.shape[1]
        else:
            self.R_Ny = R_Ny

        # Instantiate as DataSlice, setting up dimensions, depth, and slicelabels
        DataSlice.__init__(self, data=data,
                           Nx=self.R_Nx, Ny=self.R_Ny,
                           slicelabels=slicelabels, **kwargs)


