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
            self.R_Nx = data.shape[0]  #: the extent of the data along its first axis
        else:
            self.R_Nx = R_Nx
        if R_Ny is None:
            self.R_Ny = data.shape[1]  #: the extent of the data along its second axis
        else:
            self.R_Ny = R_Ny

        # Instantiate as DataSlice, setting up dimensions, depth, and slicelabels
        DataSlice.__init__(self, data=data,
                           Nx=self.R_Nx, Ny=self.R_Ny,
                           slicelabels=slicelabels, **kwargs)


### Read/Write

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

def get_realslice_from_grp(g):
    """ Accepts an h5py Group corresponding to a realslice in an open, correctly formatted H5 file,
        and returns a RealSlice.
    """
    data = np.array(g['data'])
    name = g.name.split('/')[-1]
    R_Nx,R_Ny = data.shape[:2]
    if len(data.shape)==2:
        return RealSlice(data=data,R_Nx=R_Nx,R_Ny=R_Ny,name=name)
    else:
        lbls = g['dim3']
        if('S' in lbls.dtype.str): # Checks if dim3 is composed of fixed width C strings
            with lbls.astype('S64'):
                lbls = lbls[:]
            lbls = [lbl.decode('UTF-8') for lbl in lbls]
        return RealSlice(data=data,R_Nx=R_Nx,R_Ny=R_Ny,name=name,slicelabels=lbls)


