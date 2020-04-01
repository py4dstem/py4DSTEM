# Defines a class - DataSlice - for storing / accessing / manipulating data which represents a 2D
# slice of the 4D datacube, either in real or diffraction space.
#
# Typically users will not define DataSlices, but will instead use the DiffractionSlice and RealSlice
# classes, which inherit from DataSlice.

import numpy as np
from collections import OrderedDict
from .dataobject import DataObject

class DataSlice(DataObject):

    def __init__(self, data, Nx, Ny, slicelabels=None, **kwargs):
        """
        Instantiate a DataSlice object.  Set the data and dimensions.

        The data is stored in self.data.  If it is 2D it has shape (self.Nx,self.Ny);
        if it is 3D it has shape (self.Nx,self.Ny,self.depth).  For 3D data, an array
        self.slicelabels of length self.depth specifies the data found in each slices, i.e.
        the self.data[:,:,i] is labelled by self.slicelabels[i].  self.slicelabels may be
        strings or numbers, and is instantiated by passing a list or array with the slicelabels
        keyword.  If left unspecified, these default to np.arange(self.depth).  Data can be
        accessed directly from self.data, or may be accessed using the labels in
        self.slicelabels using the syntax
            self.slices[key]
        where key is an element of slicelabels.
        """
        DataObject.__init__(self, **kwargs)

        self.Nx = Nx
        self.Ny = Ny

        self.data = data
        shape = data.shape
        assert (len(shape)==2) or (len(shape)==3)
        if len(shape)==2:
            assert shape==(self.Nx,self.Ny), "Shape of data is {}, but (DataSlice.Nx, DataSlice.Ny) = ({}, {}).".format(shape, self.Nx, self.Ny)
            self.depth=1
        else:
            assert shape[:2]==(self.Nx,self.Ny), "Shape of data is {}, but (DataSlice.Nx, DataSlice.Ny) = ({}, {}).".format(shape, self.Nx, self.Ny)
            self.depth = shape[2]
            if slicelabels is None:
                self.slicelabels = np.arange(self.depth)
            else:
                assert len(slicelabels)==self.depth
                self.slicelabels = slicelabels
            self.slices = OrderedDict()
            for i in range(self.depth):
                self.slices[self.slicelabels[i]]=self.data[:,:,i]



