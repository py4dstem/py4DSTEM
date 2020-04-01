# Defines a class - DiffractionSlice - for storing / accessing / manipulating data that lives in
# diffraction space.  DiffractionSlice inherits from DataSlice.

from collections import OrderedDict
from .dataobject import DataObject
from .dataslice import DataSlice

class DiffractionSlice(DataSlice):

    def __init__(self, data, Q_Nx=None, Q_Ny=None, slicelabels=None, **kwargs):
        """
        Instantiate a DiffractionSlice object.  Set the data and dimensions.

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



