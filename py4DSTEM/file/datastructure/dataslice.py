# Defines a class - DataSlice - for storing / accessing / manipulating data which represents a 2D
# slice of the 4D datacube, either in real or diffraction space.
#
# Typically users will not define DataSlices, but will instead use the DiffractionSlice and RealSlice
# classes, which inherit from DataSlice.

from collections import OrderedDict
from .dataobject import DataObject

class DataSlice(DataObject):

    def __init__(self, data, Nx, Ny, Nz=None, slicelabels=None, **kwargs):
        """
        Instantiate a DataSlice object.  Set the data and dimensions.
        ##TODO: Update this class description.
        If data is two dimensional, it is stored as self.data2D, and has shape (Nx, Ny).
        If data is three dimensional, self.data2D is a list of slices of some depth,
        where self.depth is data.shape[2], i.e. the shape is (Nx, Ny, depth).
        If slicelabels is unspecified, 2D slices can be accessed as self.data2D[i].
        If slicelabels is specified, it should be an n-tuple of strings, where
        n==self.depth, and 2D slices can be accessed as self.data2D[slicelabels[i]].
        """
        DataObject.__init__(self, **kwargs)

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        shape = data.shape
        assert (len(shape)==2) or (len(shape)==3)
        if len(shape)==2:
            assert shape==(self.Nx,self.Ny), "Shape of data is {}, but (DataSlice.Nx, DataSlice.Ny) = ({}, {}).".format(shape, self.Nx, self.Ny)
            self.depth=1
            self.data2D = data
        else:
            assert shape[:2]==(self.Nx,self.Ny), "Shape of data is {}, but (DataSlice.Nx, DataSlice.Ny) = ({}, {}).".format(shape, self.Nx, self.Ny)
            if slicelabels is None:
                """
                this implciitly checks if read version is 0.5 or above; from 0.1 to 0.4, slicelabels are passed even for depth == 1
                to maintain compability without version checking in the class definition, depth object is created for v0.5 when slicelabels
                are included despite lacking the attribute, and data are still stored into a data2D object
                On the other hand, if v0.5 or above is implicitly detected, data are instead passed to a data3D object for clarity of use
                from the user perspective.
                ##TODO: Determine if this is a stable/safe way to adapt future flexibility and compabitability 
                """
                self.data3D = data
                #self.data2D = []
                #for i in range(self.depth):
                #    self.data2D.append(data[:,:,i])
            else:
                self.depth=shape[2]
                assert len(slicelabels)==self.depth
                self.data2D = OrderedDict()
                for i in range(self.depth):
                    self.data2D[slicelabels[i]]=data[:,:,i]




