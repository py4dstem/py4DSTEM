# Defines a class - DiffractionSlice - for storing / accessing / manipulating data that lives in
# diffraction space.  DiffractionSlice inherits from DataSlice.

from collections import OrderedDict
from .dataobject import DataObject
from .dataslice import DataSlice

class DiffractionSlice(DataSlice):

    def __init__(self, data, Q_Nx=None, Q_Ny=None, slicelabels=None, **kwargs):
        """
        Instantiate a DiffractionSlice object.  Set the data and dimensions.

        If data is two dimensional, it is stored as self.data, and has shape (Q_Nx, Q_Ny).
        If data is three dimensional, self.data is a list of slices of some depth,
        where self.depth is data.shape[2], i.e. the shape is (Q_Nx, Q_Ny, depth).
        If slicelabels is unspecified, 2D slices can be accessed as self.data[i].
        If slicelabels is specified, it should be an n-tuple of strings, where
        n==self.depth, and 2D slices can be accessed as self.data[slicelabels[i]].
        """
        # Get shape
        if Q_Nx is None:
            self.Q_Nx = data.shape[0]
        else:
            self.Q_Nx = Q_Nx
        if Q_Ny is None:
            self.Q_Ny = data.shape[1]
        else:
            self.Q_Ny = Q_Ny

        # Instantiate as DataSlice, setting up dimensions, depth, and slicelabels
        DataSlice.__init__(self, data=data,
                           Nx=self.Q_Nx, Ny=self.Q_Ny,
                           slicelabels=slicelabels, **kwargs)



