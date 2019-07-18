# Open an interface to a Gatan K2 binary fileset, loading frames from disk as called.
# While slicing (i.e. calling dc.data4D[__,__,__,__]) returns a numpy ndarray, the 
# object is not itself a numpy array, so most numpy functions do not operate on this.

from collections.abc import Sequence
import numpy as np
import numba as nb

class K2DataArray(Sequence):
    """
    K2DataArray provides an interface to a set of Gatan K2IS binary output files.
    This object behaves *similar* to a numpy memmap into the data, and supports 4-D indexing
    and slicing. Slices into this object return np.ndarray objects.
    
    The object is created by passing the path to any of: (i) the folder containing the
    raw data, (ii) the *.gtg metadata file, or (iii) one of the raw data *.bin files.
    In any case, there should be only one dataset (8 *.bin's and a *.gtg) in the folder.
    
    ===== Filtering and Noise Reduction =====
    This object is read-only---you cannot edit the data on disk, which means that some
    DataCube functions like swap_RQ() will not work.
    
    The K2IS has a "resolution" of 1920x1792, but actually saves hidden stripes in the raw data. 
    By setting the hidden_stripe_noise_reduction flag to True, the electronic noise in these 
    stripes is used to reduce the readout noise. (This is on by default.)
    
    If you want to take a separate background to subtract, set `dark_reference` to specify this
    background. This is then subtracted from the frames as they are called out (no matter where
    the object is referenced! So, for instance, Bragg disk detection will operate on the background-
    subtracted diffraction patterns!). However, mixing the auto-background and specified background 
    is potentially dangerous and (currently!) not allowed. To switch back from user-background to
    auto-background, just delete the user background, i.e. `del(dc.data4D.dark_reference)`
    
    ===== NOTE =====
    If you call dc.data4D[:,:,:,:] on a DataCube with a K2DataArray this will read the entire stack
    into memory. To reduce RAM pressure, only call small slices or loop over each diffraction pattern.
    """
    
    def __init__(self, filepath, hidden_stripe_noise_reduction=True):
        from ncempy.io import dm
        import os, glob
        # first parse the input and get the path to the *.gtg
        if not os.path.isdir(filepath): filepath = os.path.dirname(filepath)
        os.chdir(filepath)

        assert len(glob.glob('*.bin')) == 8, "Wrong path, or wrong number of bin files."
        assert len(glob.glob('*.gtg')) == 1, "Wrong path, or wrong number of gtg files."

        gtgpath = os.path.join(os.path.dirname(filepath), glob.glob('*.gtg')[0])
        binprefix = gtgpath[:-4]

        self._gtg_file = gtgpath
        self._bin_prefix = binprefix

        # open the *.gtg and read the metadata
        gtg = dm.fileDM(gtgpath)
        gtg.parseHeader()

        #get the important metadata
        try:
            R_Ny = gtg.allTags['.SI Image Tags.SI.Acquisition.Spatial Sampling.Height (pixels)']
            R_Nx = gtg.allTags['.SI Image Tags.SI.Acquisition.Spatial Sampling.Width (pixels)']
        except ValueError:
            print('Warning: scan shape not detected. Please check/set manually.')
            R_Nx = self._guess_number_frames()
            R_Ny = 1

        try:
            # this may be wrong for binned data... in which case the reader doesn't work anyway!
            Q_Nx = gtg.allTags['.SI Image Tags.Acquisition.Parameters.Detector.height']
            Q_Ny = gtg.allTags['.SI Image Tags.Acquisition.Parameters.Detector.width']
        except:
            print('Warning: diffraction pattern shape not detected!')
            print('Assuming 1920x1792 as the diffraction pattern size!')
            Q_Nx = 1792
            Q_Ny = 1920
        
        self.shape = (R_Nx, R_Ny, Q_Nx, Q_Ny)
        self._hidden_stripe_noise_reduction = hidden_stripe_noise_reduction

        self._attach_to_files()

        self._stripe_dtype = np.dtype([ ('sync','>u4',1), \
            ('header',np.void,24),('coords','>u2',4),('pad',np.void,4),('data','>u1',22320) ])

                
        super().__init__()


    def __del__(self):
        #detatch from the file handles
        for i in range(8):
            self._bin_files[i].close()

    #======== HANDLE SLICING AND len CALLS =========#
    def __getitem__(self,i):
        # first check that the slicing is valid:
        assert len(i) == 4, f"Incorrect number of indices given. {len(i)} given, 4 required."
        # take the input and parse it into coordinate arrays
        Rx, Ry = self._parse_slices(i[:2],'real')
        Qx,Qy = self._parse_slices(i[2:],'diffraction')

        assert Rx.max() < self.shape[0], 'index out of range'
        assert Ry.max() < self.shape[1], 'index out of range'
        assert Qx.max() < self.shape[2], 'index out of range'
        assert Qy.max() < self.shape[3], 'index out of range'

        # preallocate the output data array
        outdata = np.zeros((Rx.shape[0],Rx.shape[1],Qx.shape[0],Qx.shape[1]),dtype=np.int16)
        
        #loop over all the requested frames
        for sy in range(Rx.shape[1]):
            for sx in range(Rx.shape[0]):
                scanx = Rx[sx,sy]
                scany = Ry[sx,sy]

                frame = np.ravel_multi_index( (scanx,scany), (self.shape[0],self.shape[1]), order='F' )
                DP = self._grab_frame(frame).astype(np.int16)
                if self._hidden_stripe_noise_reduction:
                    self._subtract_readout_noise(DP)
                elif self._user_noise_reduction:
                    DP = DP - self._user_dark_reference
                    
                #set_trace()
                outdata[sx,sy,:,:] = DP[Qx,Qy].reshape([Qx.shape[0],Qx.shape[1]])
        
        return np.squeeze(outdata)
        
        
    def __len__(self):
        return np.prod(self.shape)
    
    
    #====== READING FROM BINARY AND NOISE REDUCTION ======#
    def _attach_to_files(self):
        self._bin_files = np.empty(8,dtype=object)
        for i in range(8):
            binName = self._bin_prefix + str(i+1) + '.bin'
            self._bin_files[i] = open(binName,'rb')

    def _grab_frame(self,frame):
        fullImage = np.zeros([1860,2048],dtype=np.uint16)
        for ii in range(8):
            binfile = self._bin_files[ii]

            xOffset = ii*256 #the x location of the sector for each BIN file
            syncBytes = 0 #set this to a default value to check for the proper syncBytes value
            binfile.seek(frame * 0x5758 * 32,0) # frame * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME

            # read a set of stripes:
            stripe = np.fromfile(binfile, count=32, dtype=self._stripe_dtype)

            # parse the stripes
            for jj in range(0,32):
                if stripe[jj]['sync'] != 0xffff0055:
                    print('The binary file is unsynchronized and cannot be read. You must use Digital Micrograph to extract to *.dm4.')
                    break #stop reading if the sync byte is not correct. Ideally, this would read the next byte, etc... until this exact value is found

                coords = stripe[jj]['coords'] #first x, first y, last x, last y; ref to 0;inclusive;should indicate 16x930 pixels

                #place the data in the image
                fullImage[coords[1]:coords[3]+1,coords[0]+xOffset:coords[2]+xOffset+1] = \
                    _convert_uint12(stripe[jj]['data']).reshape([930, 16])
        return fullImage
    
    @staticmethod
    def _subtract_readout_noise(DP):
        #subtract readout noise using the hidden stripes
        darkref = np.average(DP[1792:,:],axis=0).astype(np.int16)
        DP -= darkref[np.newaxis,:]
        
    # Handle the user specifying a dark reference (fix the size and make sure auto gets turned off)
    @property
    def dark_reference(self):
        return self._user_dark_reference[:1792,:1920]
    
    @dark_reference.setter
    def dark_reference(self,dr):
        assert dr.shape == (1792,1920), "Dark reference must be the size of an active frame"
        #assert dr.dtype == np.uint16, "Dark reference must be 16 bit unsigned"
        self._user_dark_reference = np.zeros((1860,2048),dtype=np.int16)
        self._user_dark_reference[:1792,:1920] = dr
        
        #disable auto noise reduction
        self._hidden_stripe_noise_reduction = False
        self._user_noise_reduction = True
        
    @dark_reference.deleter
    def dark_reference(self):
        del self._user_dark_reference
        self._user_noise_reduction = False
    
    #======== UTILITY FUNCTIONS ========#
    def _parse_slices(self,i,mode):
        assert len(i)==2, 'Wrong size input'
        
        if mode == 'real':
            xMax = self.shape[0] # R_Ny
            yMax = self.shape[1] # R_Ny
        elif mode == 'diffraction':
            xMax = self.shape[2] # Q_Nx
            yMax = self.shape[3] # Q_Ny
        else:
            raise ValueError('incorrect slice mode')
        
        if isinstance(i[0],slice):
            xInds = np.arange(i[0].start or 0,(i[0].stop or xMax),(i[0].step or 1))
        else:
            xInds = i[0]
            
        if isinstance(i[1],slice):
            yInds = np.arange(i[1].start or 0,(i[1].stop or yMax),(i[1].step or 1))
        else:
            yInds = i[1]
            
        x,y = np.meshgrid(xInds,yInds,indexing='ij')
        return x,y

    def _guess_number_frames(self):
        import os
        nbytes = os.path.getsize(self._bin_prefix + '1.bin')
        return nbytes // (0x5758 * 32)

    def _write_to_hdf5(self,group):
        """
        Write the entire dataset to an HDF5 file.
        group should be an HDF5 Group object.
        ( This function is normally called via py4DSTEM.file.io.save() )
        """
        dset = group.create_dataset("datacube",(self.shape),'i2')

        for sy in range(self.shape[1]):
            for sx in range(self.shape[0]):
                dset[sx,sy,:,:] = self[sx,sy,:,:]

        return dset

    
# ======= UTILITIES OUTSIDE THE CLASS ======#
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=False,parallel=False)
def _convert_uint12(data_chunk):
  """
  data_chunk is a contigous 1D array of uint8 data)
  eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
  """

  #ensure that the data_chunk has the right length
  assert np.mod(data_chunk.shape[0],3)==0

  out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)

  for i in nb.prange(data_chunk.shape[0]//3):
    fst_uint8=np.uint16(data_chunk[i*3])
    mid_uint8=np.uint16(data_chunk[i*3+1])
    lst_uint8=np.uint16(data_chunk[i*3+2])

    out[i*2] =   fst_uint8 | (mid_uint8 & 0x0F) << 8  # (fst_uint8 << 4) + (mid_uint8 >> 4)
    out[i*2+1] = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4 # ((mid_uint8 % 16) << 8) + lst_uint8

  return out

