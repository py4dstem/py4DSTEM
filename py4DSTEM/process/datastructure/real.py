# Defines a class - RealSlice - for storing / accessing / manipulating data that lives in
# real space.
#
# RealSlice objects are generated from parent DataCube objects with various processing
# functions, and must contain a reference to their parent DataCube instance.

from .dataobject import DataObject

class RealSlice(DataObject):

    def __init__(self, data, parentDataCube):
        """
        Instantiate a RealSlice object.  Set the parent datacube, dimensions, and data.
        Confirms that the data shape agrees with real space of the parent datacube.
        """
        DataObject.__init__(self, parent=parentDataCube)

        self.parentDataCube = parentDataCube
        self.R_Ny, self.R_Nx = self.parentDataCube.R_Ny, self.parentDataCube.R_Nx
        assert data.shape == (self.R_Ny,self.R_Nx), "Shape of data is {}, but shape of real space in parent DataCube is ({},{}).".format(data.shape,self.R_Ny,self.R_Nx)
        self.data2D = data


