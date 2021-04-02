# Defines a class - RealSlice - for storing / accessing / manipulating data that lives in
# real space.  RealSlice inherits from DataSlice.

from collections import OrderedDict
import numpy as np
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

def save_real_group(group, realslice):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    group.attrs.create("depth", realslice.depth)
    data_realslice = group.create_dataset("data", data=realslice.data)

    shape = realslice.data.shape
    assert len(shape)==2 or len(shape)==3

    # Dimensions 1 and 2
    R_Nx,R_Ny = shape[:2]
    dim1 = group.create_dataset("dim1",(R_Nx,))
    dim2 = group.create_dataset("dim2",(R_Ny,))

    # Populate uncalibrated dimensional axes
    dim1[...] = np.arange(0,R_Nx)
    dim1.attrs.create("name",np.string_("R_x"))
    dim1.attrs.create("units",np.string_("[pix]"))
    dim2[...] = np.arange(0,R_Ny)
    dim2.attrs.create("name",np.string_("R_y"))
    dim2.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present

    # Dimension 3
    if len(shape)==3:
        dim3 = group.create_dataset("dim3", data=np.array(realslice.slicelabels).astype("S64"))


