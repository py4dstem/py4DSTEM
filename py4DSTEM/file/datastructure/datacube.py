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


########################## END OF DATACUBE OBJECT ########################

class CountedDataCube(DataObject):
    """
    A 4D-STEM dataset using an electron event list as the data source. 

    Accepts:
        electrons:      (PointListArray or h5py.Dataset) array of lists of 
                        electron strike events. *DO NOT MODIFY THIS AFTER CREATION*
        detector_shape: (list of ints) size Q_Nx and Q_Ny of detector
        index_keys:     (list) if the data arrays in electrons are structured, specify
                        the keys that correspond to the electron data. If electrons
                        is unstructured, pass [None] (as a list!)
        use_dask:       (bool) by default, the CountedDataCube.data object DOES NOT
                        SUPPORT slicing along the realspace axes (i.e. you can ONLY
                        pass single scan positions). By setting use_dask = True, 
                        a Dask array will be created that enables all slicing modes
                        supported by Dask. This can add substantial overhead.
    """

    def __init__(self,electrons,detector_shape,index_keys='ind',
                    use_dask=False, **kwargs):
        DataObject.__init__(self,**kwargs)

        self.electrons = electrons
        self.detector_shape = detector_shape

        if use_dask:
            import dask.array as da
            sa = Sparse4D(self.electrons,detector_shape,index_keys,**kwargs)
            self.data = da.from_array(sa,chunks=(1,1,detector_shape[0],detector_shape[1]))
        else:
            self.data = Sparse4D(self.electrons,detector_shape,index_keys,**kwargs)

        self.R_Nx = int(self.data.shape[0])
        self.R_Ny = int(self.data.shape[1])
        self.Q_Nx = int(self.data.shape[2])
        self.Q_Ny = int(self.data.shape[3])

        self.R_N = self.R_Nx * self.R_Ny

    def bin_data_diffraction(self, bin_factor):
        # bin the underlying data (keeping in sparse storage)
        raise NotImplementedError("Binning only supported by densify().")

    def bin_data_real(self, bin_factor):
        # bin the underlying data (keeping sparse storage)
        raise NotImplementedError("Binning only supported by densify().")

    def densify(self,bin_R=1, bin_Q=1, memmap=False, dtype=np.uint16):
        """
        Convert to a fully dense DataCube object, with 
        optional binning in real and reciprocal space.
        If memmap=True, the dense DC will be stored on disk
        in a temporary file (using numpy).
        """
        newRx = int(np.ceil(self.R_Nx / bin_R))
        newRy = int(np.ceil(self.R_Ny / bin_R))
        newQx = int(np.ceil(self.Q_Nx / bin_Q))
        newQy = int(np.ceil(self.Q_Ny / bin_Q))

        if memmap:
            #make temp file
            tf = TemporaryFile()
            data4D = np.memmap(tf,dtype,'r+',shape=(newRx,newRy,newQx,newQy))
        else:
            data4D = np.zeros((newRx,newRy,newQx,newQy),dtype=dtype)

        for (Rx, Ry) in tqdmnd(self.R_Nx, self.R_Ny, desc="Creating dense DC"):
            rx = Rx // bin_R
            ry = Ry // bin_R

            DP = self.data[Rx,Ry,:,:]
            data4D[rx,ry,:,:] += DP if bin_Q == 1 else bin2D(DP,bin_Q,dtype=dtype)

        return DataCube(data4D,name=self.name)


class Sparse4D(Sequence):
    """
    A wrapper for a PointListArray or HDF5 dataset of electron events 
    that returns a reconstructed diffraction pattern when sliced.
    NOTE: This class is meant to be constructed by the
    CountedDataCube object, and should not be invoked directly.
    """
    def __init__(self,electrons,detector_shape,index_key='ind',**kwargs):
        super().__init__()

        self.electrons = electrons
        self.detector_shape = detector_shape

        # check if using the Kitware 1D scheme
        if len(electrons.shape) == 1:
            self._1Didx = True
            self.R_Nx = kwargs.get('R_Nx')
            self.R_Ny = kwargs.get('R_Ny')
        else:
            self._1Didx = False
            self.R_Nx = electrons.shape[0]
            self.R_Ny = electrons.shape[1]

        # choose PointListArray mode or HDF5 mode
        if isinstance(electrons,DataObject):
            # running in PLA mode
            self._mmap = False
        elif isinstance(electrons,h5py.Dataset):
            self._mmap = True

        # check if 1D or 2D event coordinates
        if index_key[0] is None:
            # using an unstructured 1D indexing scheme
            self._mode = 0
            self.index_key = None
        elif np.count_nonzero(index_key) == 1:
            # use 1D indexing mode
            self.index_key = np.array([index_key])
            self._mode = 1
            self._key1 = self.index_key.ravel()[0]
        elif np.count_nonzero(index_key) == 2:
            # use 2D indexing mode
            self.index_key = np.array([index_key])
            self._mode = 2
            self._key1 = self.index_key.ravel()[0]
            self._key2 = self.index_key.ravel()[1]
        else:
            assert False, "index_key specified incorrectly"

        # Needed for dask:
        self.shape = (self.R_Nx, self.R_Ny,
            detector_shape[0], detector_shape[1])
        self.dtype = np.uint8
        self.ndim = 4

    def __getitem__(self,i):

        if self._mmap == False:
            # PLA mode
            pl = self.electrons.get_pointlist(i[0],i[1])

            if self._mode == 0:
                dp = points_to_DP_numba_ravel(pl.data,
                    int(self.detector_shape[0]),int(self.detector_shape[1]))
            elif self._mode == 1:
                dp = points_to_DP_numba_ravel(pl.data[self._key1],
                    int(self.detector_shape[0]),int(self.detector_shape[1]))
            elif self._mode == 2:
                dp = points_to_DP_numba_unravel(pl.data[self._key1],
                    pl.data[self._key2],int(self.detector_shape[0]),
                    int(self.detector_shape[1]))

        else:
            # HDF5 mode
            if self._1Didx:
                idx = np.ravel_multi_index(i[:2],(self.R_Nx,self.R_Ny))
                data = self.electrons[idx]
            else:
                data = self.electrons[i[0],i[1]]

            if self._mode == 0:
                dp = points_to_DP_numba_ravel(data,
                    int(self.detector_shape[0]),int(self.detector_shape[1]))
            elif self._mode == 1:
                dp = points_to_DP_numba_ravel(data[self._key1],
                    int(self.detector_shape[0]),int(self.detector_shape[1]))
            elif self._mode == 2:
                dp = points_to_DP_numba_unravel(data[self._key1],
                    data[self._key2],int(self.detector_shape[0]),
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

