# Defines the DataCube class.
#
# DataCube objects contain a 4DSTEM dataset, attributes describing its shape, and methods
# pointing to processing functions - generally defined in other files in the process directory.

from collections.abc import Sequence

import numpy as np
import dask as da
import numba as nb

import numpy as np
from .dataobject import DataObject
from ...process import preprocess
from ...process import virtualimage
from ...process.utils import tqdmnd

class DataCube(DataObject):

    def __init__(self, data, **kwargs):
        """
        Instantiate a DataCube object. Set the data, scan dimensions, and metadata.
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

    def swap_RQ(self):
        """
        Swap real and reciprocal space coordinates.
        """
        self = preprocess.swap_RQ(self)

    def swap_Rxy(self):
        """
        Swap real space x and y coordinates.
        """
        self = preprocess.swap_Rxy(self)

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

    def get_diffraction_space_view(self,Rx=0,Ry=0):
        """
        Returns the image in diffraction space, and a Bool indicating success or failure.
        """
        self.Rx,self.Ry = Rx,Ry
        try:
            return self.data[Rx,Ry,:,:], 1
        except IndexError:
            return 0, 0

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


########################## END OF DATACUBE OBJECT ########################

class CountedDataCube(DataObject):

    def __init__(self,electrons,detector_shape,index_keys='ind',
                    use_dask=False, **kwargs):
        super().__init__(self,**kwargs)

        self.electrons = electrons
        self.detector_shape = detector_shape

        if use_dask:
            sa = Sparse4D(self.electrons,detector_shape,index_keys)
            self.data = da.from_array(sa,chunks=(1,1,detector_shape[0],detector_shape[1]))
        else:
            self.data = Sparse4D(self.electrons,detector_shape,index_keys)

        self.R_Nx = electrons.shape[0]
        self.R_Ny = electrons.shape[1]
        self.Q_Nx = detector_shape[0]
        self.Q_Ny = detector_shape[1]

        self.R_N = self.R_Nx * self.R_Ny

    def bin_data_diffraction(self, bin_factor):
        pass

    def bin_data_real(self, bin_factor):
        pass

    def densify(self,bin_R=1, bin_Q=1, memmap=False):
        """
        Convert to a fully dense DataCube object, with 
        optional binning in real and reciprocal space.
        If memmap=True, the dense DC will be stored on disk
        in a temporary file (using numpy).
        """
        pass


class Sparse4D(Sequence):
    """
    A wrapper for a PointListArray of electron events that returns
    a reconstructed diffraction pattern when sliced.
    NOTE: This class is meant to be constructed by the
    CountedDataCube object.
    """
    def __init__(self,electrons,detector_shape,index_key='ind'):
        super().__init__()

        self.electrons = electrons
        self.detector_shape = detector_shape
        self.index_key = np.array([index_key])

        # check if 1D or 2D
        if np.count_nonzero(self.index_key) == 1:
            # use 1D indexing mode
            self._mode = 1
            self._key1 = self.index_key.ravel()[0]
        elif np.count_nonzero(self.index_key) == 2:
            # use 2D indexing mode
            self._mode = 2
            self._key1 = self.index_key.ravel()[0]
            self._key2 = self.index_key.ravel()[1]
        else:
            assert False, "index_key specified incorrectly"

        # Needed for dask:
        self.shape = (electrons.shape[0], electrons.shape[1],
            detector_shape[0], detector_shape[1])
        self.dtype = np.uint8
        self.ndim = 4

    def __getitem__(self,i):
        pl = self.electrons.get_pointlist(i[0],i[1])

        if self._mode == 1:
            dp = points_to_DP_numba_ravel(pl.data[self._key1],
                int(self.detector_shape[0]),int(self.detector_shape[1]))
        elif self._mode == 2:
            dp = points_to_DP_numba_unravel(pl.data[self._key1],
                pl.data[self._key2],int(self.detector_shape[0]),
                int(self.detector_shape[1]))

        return dp[i[2],i[3]]

    def __len__(self):
        return np.prod(self.shape)


# Numba accelerated conversion of electron event lists
# to full diffraction patterns
@nb.njit
def points_to_DP_numba_ravel(pl,sz1,sz2):
    dp = np.zeros((sz1*sz2),dtype=np.uint8)
    for i in nb.prange(len(pl)):
        dp[pl[i]] += 1
    return dp.reshape((sz1,sz2))

@nb.njit
def points_to_DP_numba_unravel(pl1,pl2,sz1,sz2):
    dp = np.zeros((sz1,sz2),dtype=np.uint8)
    for i in nb.prange(len(pl1)):
        dp[pl1[i],pl2[i]] += 1
    return dp

