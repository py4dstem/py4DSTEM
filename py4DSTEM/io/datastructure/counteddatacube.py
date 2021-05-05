from collections.abc import Sequence
from tempfile import TemporaryFile

import numpy as np
import numba as nb
import h5py

import numpy as np
from .dataobject import DataObject
from .datacube import DataCube
from ...process import preprocess
from ...process import virtualimage_viewer as virtualimage
from ...process.utils import tqdmnd, bin2D

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



### Read/Write

def save_counted_datacube_group(group,datacube):
    """
    Expects an open .h5 group and a DataCube; saves the DataCube to the group
    """
    if datacube.data._mmap:
        # memory mapped CDC's aren't supported yet
        print('Data not written. Memory mapped CountedDataCube not yet supported.')
        return

    group.attrs.create("emd_group_type",1)

    pointlistarray = datacube.electrons
    try:
        n_coords = len(pointlistarray.dtype.names)
    except:
        n_coords = 1
    #coords = np.string_(str([coord for coord in pointlistarray.dtype.names]))
    group.attrs.create("coordinates", np.string_(str(pointlistarray.dtype)))
    group.attrs.create("dimensions", n_coords)

    pointlist_dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    name = "data"
    dset = group.create_dataset(name,pointlistarray.shape,pointlist_dtype)

    print('Writing CountedDataCube:',flush=True)

    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data

    # indexing coordinates:
    dt = h5py.special_dtype(vlen=str)
    data_coords = group.create_dataset('index_coords',shape=(datacube.data._mode,),dtype=dt)
    if datacube.data._mode == 1:
        data_coords[0] = datacube.data.index_key
    else:
        data_coords[0] = datacube.data.index_key.ravel()[0]
        data_coords[1] = datacube.data.index_key.ravel()[1]

    # Dimensions
    R_Nx,R_Ny,Q_Nx,Q_Ny = datacube.data.shape
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

def get_counted_datacube_from_grp(g):
    """ Accepts an h5py Group corresponding to a counted datacube in an open, correctly formatted H5 file,
        and returns a CountedDataCube.
    """
    return #TODO



