# Defines a class - RealSlice - for storing / accessing / manipulating data that lives in
# real space.  RealSlice inherits from DataSlice.

from collections import OrderedDict
from .dataobject import DataObject
from .dataslice import DataSlice

class RealSlice(DataObject):

    def __init__(self, data, R_Nx=None, R_Ny=None, slicelabels=None, **kwargs):
        """
        Instantiate a RealSlice object.  Set the data and dimensions.

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


