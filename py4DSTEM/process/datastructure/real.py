# Defines a class - RealSlice - for storing / accessing / manipulating data that lives in
# real space.
#
# RealSlice objects are generated from parent DataCube objects with various processing
# functions, and must contain a reference to their parent DataCube instance.

from collections import OrderedDict
from .dataobject import DataObject

class RealSlice(DataObject):

    def __init__(self, data, parentDataCube, slicelabels=None, R_Nx=None, R_Ny=None, **kwargs):
        """
        Instantiate a RealSlice object.  Set the parent datacube, dimensions, and data.
        Confirms that the data shape agrees with real space of the parent datacube.
        If data is two dimensional, it is stored as self.data2D.
        If data is three dimensional, self.data2D is a list of slices of some depth,
        where self.depth is data.shape[0].
        If slicelabels is unspecified, 2D slices can be accessed as self.data2D[i].
        If slicelabels is specified, it should be an n-tuple of strings, where
        n==self.depth, and 2D slices can be accessed as self.data2D[slicelabels[i]].
        """
        DataObject.__init__(self, parent=parentDataCube, **kwargs)

        self.parentDataCube = parentDataCube
        if R_Nx is None:
            self.R_Nx = self.parentDataCube.R_Nx
        else:
            self.R_Nx = R_Nx
        if R_Ny is None:
            self.R_Ny = self.parentDataCube.R_Ny
        else:
            self.R_Ny = R_Ny

        shape = data.shape
        assert (len(shape)==2) or (len(shape)==3)
        if len(shape)==2:
            assert shape==(self.R_Nx,self.R_Ny), "Shape of data is {}, but shape of real space in parent DataCube is ({},{}).".format(data.shape,self.R_Nx,self.R_Ny)
            self.depth=1
            self.data2D = data
        else:
            assert shape[1:]==(self.R_Nx,self.R_Ny), "Shape of data is {}, but shape of real space in parent DataCube is ({},{}).".format(data.shape,self.R_Nx,self.R_Ny)
            self.depth=shape[0]
            if slicelabels is None:
                self.data2D = []
                for i in range(self.depth):
                    self.data2D.append(data[i,:,:])
            else:
                assert len(slicelabels)==self.depth
                self.data2D = OrderedDict()
                for i in range(self.depth):
                    self.data2D[slicelabels[i]]=data[i,:,:]



