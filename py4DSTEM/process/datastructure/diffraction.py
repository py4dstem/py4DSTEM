# Defines a class - DiffractionSlice - for storing / accessing / manipulating data that lives in
# diffraction space.
#
# DiffractionSlice objects are generated from parent DataCube objects with various processing
# functions, and must contain a reference to their parent DataCube instance.

from collections import OrderedDict
from .dataobject import DataObject

class DiffractionSlice(DataObject):

    def __init__(self, data, parentDataCube, slicelabels=None, Q_Ny=None, Q_Nx=None, **kwargs):
        """
        Instantiate a DiffractionSlice object.  Set the parent datacube, dimensions, and data.
        Confirms that the data shape agrees with diffraction space of the parent datacube.
        If data is two dimensional, it is stored as self.data2D.
        If data is three dimensional, self.data2D is a list of slices of some depth,
        where self.depth is data.shape[0].
        If slicelabels is unspecified, 2D slices can be accessed as self.data2D[i].
        If slicelabels is specified, it should be an n-tuple of strings, where
        n==self.depth, and 2D slices can be accessed as self.data2D[slicelabels[i]].
        """
        DataObject.__init__(self, parent=parentDataCube, **kwargs)

        self.parentDataCube = parentDataCube
        if Q_Ny is None:
            self.Q_Ny = self.parentDataCube.Q_Ny
        else:
            self.Q_Ny = Q_Ny
        if Q_Nx is None:
            self.Q_Nx = self.parentDataCube.Q_Nx
        else:
            self.Q_Nx = Q_Nx

        shape = data.shape
        assert (len(shape)==2) or (len(shape)==3)
        if len(shape)==2:
            assert shape==(self.Q_Ny,self.Q_Nx), "Shape of data is {}, but shape of diffraction space in parent DataCube is ({},{}).".format(shape, self.Q_Ny, self.Q_Nx)
            self.depth=1
            self.data2D = data
        else:
            assert shape[1:]==(self.Q_Ny,self.Q_Nx), "Shape of data is {}, but shape of diffraction space in parent DataCube is ({},{}).".format(shape, self.Q_Ny, self.Q_Nx)
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




