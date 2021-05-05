# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and methods
# pointing to processing functions - generally defined in other files in the process directory.

from collections.abc import Sequence
from tempfile import TemporaryFile

import numpy as np
import numba as nb
import h5py

import numpy as np
from .dataobject import DataObject
from ...process import preprocess
from ...process import virtualimage_viewer as virtualimage
from ...process.utils import tqdmnd, bin2D

class DataCube(DataObject):

    def __init__(self, data, **kwargs):
        """
        Instantiate a DataCube object. Set the data and scan dimensions.
        """
        # Initialize DataObject
        DataObject.__init__(self, **kwargs)
        self.data = data

        # Set shape
        assert (len(data.shape)==3 or len(data.shape)==4)
        if len(data.shape)==3:
            self.R_N, self.Q_Nx, self.Q_Ny = data.shape
            self.R_Nx, self.R_Ny = self.R_N, 1
            self.set_scan_shape(self.R_Nx,self.R_Ny)
        else:
            self.R_Nx, self.R_Ny, self.Q_Nx, self.Q_Ny = data.shape
            self.R_N = self.R_Nx*self.R_Ny

        self.update_slice_parsers()
        # Set shape
        # TODO: look for shape in metadata
        # TODO: AND/OR look for R_Nx... in kwargs
        #self.R_Nx, self.R_Ny, self.Q_Nx, self.Q_Ny = self.data.shape
        #self.R_N = self.R_Nx*self.R_Ny
        #self.set_scan_shape(self.R_Nx,self.R_Ny)

    ############### Processing functions, organized by file in process directory ##############

    ############### preprocess.py ##############

    def set_scan_shape(self,R_Nx,R_Ny):
        """
        Reshape the data given the real space scan shape.
        """
        self = preprocess.set_scan_shape(self,R_Nx,R_Ny)
        self.update_slice_parsers()

    def swap_RQ(self):
        """
        Swap real and reciprocal space coordinates.
        """
        self = preprocess.swap_RQ(self)
        self.update_slice_parsers()

    def swap_Rxy(self):
        """
        Swap real space x and y coordinates.
        """
        self = preprocess.swap_Rxy(self)
        self.update_slice_parsers()

    def swap_Qxy(self):
        """
        Swap reciprocal space x and y coordinates.
        """
        self = preprocess.swap_Qxy(self)

    def crop_data_diffraction(self,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max):
        self = preprocess.crop_data_diffraction(self,crop_Qx_min,crop_Qx_max,crop_Qy_min,crop_Qy_max)

    def crop_data_real(self,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max):
        self = preprocess.crop_data_real(self,crop_Rx_min,crop_Rx_max,crop_Ry_min,crop_Ry_max)

    def bin_data_diffraction(self, bin_factor):
        self = preprocess.bin_data_diffraction(self, bin_factor)

    def bin_data_mmap(self, bin_factor):
        self = preprocess.bin_data_mmap(self, bin_factor)

    def bin_data_real(self, bin_factor):
        self = preprocess.bin_data_real(self, bin_factor)



    ################ Slice data #################

    def update_slice_parsers(self):
        # define index-sanitizing functions:
        self.normX = lambda x: np.maximum(0,np.minimum(self.R_Nx-1,x))
        self.normY = lambda x: np.maximum(0,np.minimum(self.R_Ny-1,x))

    def get_diffraction_space_view(self,Rx=0,Ry=0):
        """
        Returns the image in diffraction space, and a Bool indicating success or failure.
        """
        self.Rx,self.Ry = self.normX(Rx),self.normY(Ry)
        try:
            return self.data[self.Rx,self.Ry,:,:], 1
        except IndexError:
            return 0, 0
        except ValueError:
            return 0,0

    # Virtual images -- integrating

    def get_virtual_image_rect_integrate(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in integration
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_rect_integrate(self,slice_x,slice_y)

    def get_virtual_image_circ_integrate(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in integration
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_circ_integrate(self,slice_x,slice_y)

    def get_virtual_image_annular_integrate(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in integration
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virtualimage.get_virtual_image_annular_integrate(self,slice_x,slice_y,R)

    # Virtual images -- difference

    def get_virtual_image_rect_diffX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_rect_diffX(self,slice_x,slice_y)

    def get_virtual_image_rect_diffY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_rect_diffY(self,slice_x,slice_y)

    def get_virtual_image_circ_diffX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_circ_diffX(self,slice_x,slice_y)

    def get_virtual_image_circ_diffY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_circ_diffY(self,slice_x,slice_y)

    def get_virtual_image_annular_diffX(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virtualimage.get_virtual_image_annular_diffX(self,slice_x,slice_y,R)

    def get_virtual_image_annular_diffY(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in difference
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virtualimage.get_virtual_image_annular_diffY(self,slice_x,slice_y,R)

    # Virtual images -- CoM

    def get_virtual_image_rect_CoMX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_rect_CoMX(self,slice_x,slice_y)

    def get_virtual_image_rect_CoMY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a rectangular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_rect_CoMY(self,slice_x,slice_y)

    def get_virtual_image_circ_CoMX(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_circ_CoMX(self,slice_x,slice_y)

    def get_virtual_image_circ_CoMY(self,slice_x,slice_y):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure.
        """
        return virtualimage.get_virtual_image_circ_CoMY(self,slice_x,slice_y)

    def get_virtual_image_annular_CoMX(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virtualimage.get_virtual_image_annular_CoMX(self,slice_x,slice_y,R)

    def get_virtual_image_annular_CoMY(self,slice_x,slice_y,R):
        """
        Returns a virtual image as an ndarray, generated from a circular detector in CoM
        mode. Also returns a bool indicating success or failure. The input parameter R is the ratio
        of the inner to the outer detector radii.
        """
        return virtualimage.get_virtual_image_annular_CoMY(self,slice_x,slice_y,R)


### Read/Write

def save_datacube_group(group, datacube, use_compression=False):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    group.attrs.create("emd_group_type",1)
    if (isinstance(datacube.data,np.ndarray) or isinstance(datacube.data,h5py.Dataset)):
        if use_compression:
            data_datacube = group.create_dataset("data", data=datacube.data,
                chunks=(1,1,datacube.Q_Nx,datacube.Q_Ny),compression='gzip')
        else:
            data_datacube = group.create_dataset("data", data=datacube.data)
    else:
        # handle K2DataArray datacubes
        data_datacube = datacube.data._write_to_hdf5(group)

    # Dimensions
    assert len(data_datacube.shape)==4, "Shape of datacube is {}".format(len(data_datacube))
    R_Nx,R_Ny,Q_Nx,Q_Ny = data_datacube.shape
    data_R_Nx = group.create_dataset("dim1",(R_Nx,))
    data_R_Ny = group.create_dataset("dim2",(R_Ny,))
    data_Q_Nx = group.create_dataset("dim3",(Q_Nx,))
    data_Q_Ny = group.create_dataset("dim4",(Q_Ny,))

    # Populate uncalibrated dimensional axes
    data_R_Nx[...] = np.arange(0,R_Nx)
    data_R_Nx.attrs.create("name",np.string_("R_x"))
    data_R_Nx.attrs.create("units",np.string_("[pix]"))
    data_R_Ny[...] = np.arange(0,R_Ny)
    data_R_Ny.attrs.create("name",np.string_("R_y"))
    data_R_Ny.attrs.create("units",np.string_("[pix]"))
    data_Q_Nx[...] = np.arange(0,Q_Nx)
    data_Q_Nx.attrs.create("name",np.string_("Q_x"))
    data_Q_Nx.attrs.create("units",np.string_("[pix]"))
    data_Q_Ny[...] = np.arange(0,Q_Ny)
    data_Q_Ny.attrs.create("name",np.string_("Q_y"))
    data_Q_Ny.attrs.create("units",np.string_("[pix]"))

    # TODO: Calibrate axes, if calibrations are present


def get_datacube_from_grp(g,mem='RAM',binfactor=1,bindtype=None):
    """ Accepts an h5py Group corresponding to a single datacube in an open, correctly formatted H5 file,
        and returns a DataCube.
    """
    # TODO: add memmapping, binning
    data = np.array(g['data'])
    name = g.name.split('/')[-1]
    return DataCube(data=data,name=name)







