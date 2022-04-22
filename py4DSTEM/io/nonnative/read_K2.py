# Open an interface to a Gatan K2 binary fileset, loading frames from disk as called.
# While slicing (i.e. calling dc.data4D[__,__,__,__]) returns a numpy ndarray, the
# object is not itself a numpy array, so most numpy functions do not operate on this.

from collections.abc import Sequence
import numpy as np
try:
    import numba as nb
except ImportError:
    pass
from ...process.utils import tqdmnd
from ..datastructure import DataCube


def read_gatan_K2_bin(fp, mem="MEMMAP", binfactor=1, metadata=False, **kwargs):
    """
    Read a K2 binary 4D-STEM file.

    Args:
        fp: str Path to the file
        mem (str, optional): Specifies how the data should be stored; must be "RAM"
            or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default is "MEMMAP".
        binfactor: (int, optional): Bin the data, in diffraction space, as it's loaded.
            See docstring for py4DSTEM.file.io.read.  Must be 1, retained only for
            compatibility.
        metadata (bool, optional): if True, returns the file metadata as a Metadata
            instance.

    Returns:
        (variable): The return value depends on usage:


            * if metadata==False, returns the 4D-STEM dataset as a DataCube
            * if metadata==True, returns the metadata as a Metadata instance

        Note that metadata is read either way - in the latter case ONLY
        metadata is read and returned, in the former case a DataCube
        is returned with the metadata attached at datacube.metadata
    """
    assert mem == "MEMMAP", "K2 files can only be memory-mapped, sorry."
    assert binfactor == 1, "K2 files can only be read at full resolution, sorry."

    if metadata is True:
        return None

    block_sync = kwargs.get("K2_sync_block_IDs", True)
    NR = kwargs.get("K2_hidden_stripe_noise_reduction",True)
    return DataCube(data=K2DataArray(fp,sync_block_IDs=block_sync, hidden_stripe_noise_reduction=NR))


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

    Note:
        If you call dc.data4D[:,:,:,:] on a DataCube with a K2DataArray this will read the entire stack
        into memory. To reduce RAM pressure, only call small slices or loop over each diffraction pattern.
    """

    def __init__(self, filepath, sync_block_IDs = True, hidden_stripe_noise_reduction=True):
        from ncempy.io import dm
        import os
        import glob

        # first parse the input and get the path to the *.gtg
        if not os.path.isdir(filepath):
            filepath = os.path.dirname(filepath)

        assert (
            len(glob.glob(os.path.join(filepath, "*.bin"))) == 8
        ), "Wrong path, or wrong number of bin files."
        assert (
            len(glob.glob(os.path.join(filepath, "*.gtg"))) == 1
        ), "Wrong path, or wrong number of gtg files."

        gtgpath = os.path.join(filepath, glob.glob(os.path.join(filepath, "*.gtg"))[0])
        binprefix = gtgpath[:-4]

        self._gtg_file = gtgpath
        self._bin_prefix = binprefix

        # open the *.gtg and read the metadata
        gtg = dm.fileDM(gtgpath)
        gtg.parseHeader()

        # get the important metadata
        try:
            R_Ny = gtg.allTags[".SI Dimensions.Size Y"]
            R_Nx = gtg.allTags[".SI Dimensions.Size X"]
        except ValueError:
            print("Warning: scan shape not detected. Please check/set manually.")
            R_Nx = self._guess_number_frames() // 32
            R_Ny = 1

        try:
            # this may be wrong for binned data... in which case the reader doesn't work anyway!
            Q_Nx = gtg.allTags[".SI Image Tags.Acquisition.Parameters.Detector.height"]
            Q_Ny = gtg.allTags[".SI Image Tags.Acquisition.Parameters.Detector.width"]
        except:
            print("Warning: diffraction pattern shape not detected!")
            print("Assuming 1920x1792 as the diffraction pattern size!")
            Q_Nx = 1792
            Q_Ny = 1920

        self.shape = (int(R_Nx), int(R_Ny), int(Q_Nx), int(Q_Ny))
        self._hidden_stripe_noise_reduction = hidden_stripe_noise_reduction
        self.sync_block_IDs = sync_block_IDs

        self._stripe_dtype = np.dtype(
            [
                ("sync", ">u4"),
                ("pad1", np.void, 5),
                ("shutter", ">u1"),
                ("pad2", np.void, 6),
                ("block", ">u4",),
                ("pad4", np.void, 4),
                ("frame", ">u4"),
                ("coords", ">u2", (4,)),
                ("pad3", np.void, 4),
                ("data", ">u1", (22320,)),
            ]
        )

        self._attach_to_files()

        self._shutter_offsets = np.zeros((8,), dtype=np.uint32)
        self._find_offsets()
        print("Shutter flags are:", self._shutter_offsets)

        self._gtg_meta = gtg.allTags

        self._user_noise_reduction = False

        self._temp = np.zeros((32,), dtype=self._stripe_dtype)
        self._Qx, self._Qy = self._parse_slices(
            (slice(None), slice(None)), "diffraction"
        )

        # needed for Dask support:
        self.ndims = 4
        self.dtype = np.int16

        super().__init__()

    # ======== HANDLE SLICING AND len CALLS =========#
    def __getitem__(self, i):
        # first check that the slicing is valid:
        assert (
            len(i) == 4
        ), f"Incorrect number of indices given. {len(i)} given, 4 required."
        # take the input and parse it into coordinate arrays
        if isinstance(i[0], slice) | isinstance(i[1], slice):
            Rx, Ry = self._parse_slices(i[:2], "real")
            R_Nx = Rx.shape[0]
            R_Ny = Rx.shape[1]
            assert Rx.max() < self.shape[0], "index out of range"
            assert Ry.max() < self.shape[1], "index out of range"
        else:  # skip _parse_slices for single input
            Rx = np.array([i[0]], ndmin=2)
            Ry = np.array([i[1]], ndmin=2)
            R_Nx = 1
            R_Ny = 1
            assert Rx < self.shape[0], "index out of range"
            assert Ry < self.shape[1], "index out of range"
        if (i[2] == slice(None)) & (i[3] == slice(None)):
            Qx, Qy = self._Qx, self._Qy
        else:
            Qx, Qy = self._parse_slices(i[2:], "diffraction")

        assert Qx.max() < self.shape[2], "index out of range"
        assert Qy.max() < self.shape[3], "index out of range"

        # preallocate the output data array
        outdata = np.zeros((R_Nx, R_Ny, Qx.shape[0], Qx.shape[1]), dtype=np.int16)

        # loop over all the requested frames
        for sy in range(R_Ny):
            for sx in range(R_Nx):
                scanx = Rx[sx, sy]
                scany = Ry[sx, sy]

                frame = np.ravel_multi_index(
                    (scanx, scany), (self.shape[0], self.shape[1]), order="F"
                )
                DP = self._grab_frame(frame)
                if self._hidden_stripe_noise_reduction:
                    self._subtract_readout_noise(DP)
                elif self._user_noise_reduction:
                    DP = DP - self._user_dark_reference

                outdata[sx, sy, :, :] = DP[Qx, Qy].reshape([Qx.shape[0], Qx.shape[1]])

        return np.squeeze(outdata)

    def __len__(self):
        return np.prod(self.shape)

    # ====== DUCK-TYPED NUMPY FUNCTIONS ======#

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        assert axis in [(0, 1), (2, 3)], "Only average DP and average image supported."

        # handle average DP
        if axis == (0, 1):
            avgDP = np.zeros((self.shape[2], self.shape[3]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                avgDP += self[Rx, Ry, :, :]

            return avgDP / (self.shape[0] * self.shape[1])

        # handle average image
        if axis == (2, 3):
            avgImg = np.zeros((self.shape[0], self.shape[1]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                avgImg[Rx, Ry] = np.mean(self[Rx, Ry, :, :])
            return avgImg

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        assert axis in [(0, 1), (2, 3)], "Only sum DP and sum image supported."

        # handle average DP
        if axis == (0, 1):
            sumDP = np.zeros((self.shape[2], self.shape[3]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                sumDP += self[Rx, Ry, :, :]

            return sumDP

        # handle average image
        if axis == (2, 3):
            sumImg = np.zeros((self.shape[0], self.shape[1]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                sumImg[Rx, Ry] = np.sum(self[Rx, Ry, :, :])
            return sumImg

    def max(self, axis=None, out=None):
        assert axis in [(0, 1), (2, 3)], "Only max DP and max image supported."

        # handle average DP
        if axis == (0, 1):
            maxDP = np.zeros((self.shape[2], self.shape[3]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                maxDP = np.maximum(maxDP, self[Rx, Ry, :, :])

            return maxDP

        # handle average image
        if axis == (2, 3):
            maxImg = np.zeros((self.shape[0], self.shape[1]))
            for (Ry, Rx) in tqdmnd(self.shape[1], self.shape[0]):
                maxImg[Rx, Ry] = np.max(self[Rx, Ry, :, :])
            return maxImg

    # ====== READING FROM BINARY AND NOISE REDUCTION ======#
    def _attach_to_files(self):
        self._bin_files = np.empty(8, dtype=object)
        for i in range(8):
            binName = self._bin_prefix + str(i + 1) + ".bin"

            # Synchronize to the magic sync word
            # First, open the file in binary mode and read ~1 MB
            with open(binName, 'rb') as f:
                s = f.read(1_000_000)

            # Scan the chunk and find everywhere the sync word appears
            sync = [s.find(b'\xff\xff\x00\x55'),]
            while sync[-1] >= 0:
                sync.append(s.find(b'\xff\xff\x00\x55',sync[-1]+1))

            # Since the sync word can conceivably occur within the data region,
            # check that there is another sync word 22360 bytes away
            sync_idx = 0
            while 0 not in [s - sync[sync_idx] - 22360 for s in sync]:
                sync_idx += 1

            if sync_idx > 0:
                print(f"Beginning file {i} at offset {sync[sync_idx]} due to incomplete data block!")

            # Start the memmap at the offset of the sync byte
            self._bin_files[i] = np.memmap(
                binName,
                dtype=self._stripe_dtype,
                mode="r",
                shape=(self._guess_number_frames(),),
                offset=sync[sync_idx],
            )

    def _find_offsets(self):
        # first, line up the block counts (LiberTEM calls this sync_sectors)
        if self.sync_block_IDs:
            print("Synchronizing block IDs.")
            first_blocks = np.zeros((8,), dtype=np.uint32)
            for i in range(8):
                binfile = self._bin_files[i]
                first_blocks[i] = binfile[0]["block"]

            # find the first frame in each with the starting block
            block_id = np.max(first_blocks)
            print("First block syncs to block #", block_id)
            for i in range(8):
                sync = False
                frame = 0
                while sync == False:
                    sync = self._bin_files[i][frame]["block"] == block_id
                    if sync == False:
                        frame += 1
                self._shutter_offsets[i] += frame
            print("Offsets are currently ", self._shutter_offsets)
        else:
            print("Skipping block ID synchronization step...")

        first_frame = self._bin_files[0][self._shutter_offsets[0]]["frame"]
        # next, check if the frames are complete (the next 32 blocks should have the same block #)
        print("Checking if first frame is complete...")
        sync = True
        for i in range(8):
            stripe = self._bin_files[i][
                self._shutter_offsets[i] : self._shutter_offsets[i] + 32
            ]
            for j in range(32):
                if stripe[j]["frame"] != first_frame:
                    sync = False
                    next_frame = stripe[j]["frame"]

        if sync == False:
            # the first frame is incomplete, so we need to seek the next one
            print(
                f"First frame ({first_frame}) incomplete, seeking frame {next_frame}..."
            )
            for i in range(8):
                sync = False
                frame = 0
                while sync == False:
                    sync = (
                        self._bin_files[i][self._shutter_offsets[i] + frame]["frame"]
                        == next_frame
                    )
                    if sync == False:
                        frame += 1
                self._shutter_offsets[i] += frame
            print("Offsets are now ", self._shutter_offsets)

            # JUST TO BE SAFE, CHECK AGAIN THAT FRAME IS COMPLETE
            print("Checking if new frame is complete...")
            first_frame = np.max(stripe["frame"])
            # check if the frames are complete (the next 32 blocks should have the same block #)
            sync = True
            for i in range(8):
                stripe = self._bin_files[i][
                    self._shutter_offsets[i] : self._shutter_offsets[i] + 32
                ]
                if np.any(stripe[:]["frame"] != first_frame):
                    sync = False
            if sync == True:
                print("New frame is complete!")
            else:
                print("Next frame also incomplete!!!! Data may be corrupt?")

        # in each file, find the first frame with open shutter (LiberTEM calls this sync_to_first_frame)
        print("Synchronizing to shutter open...")
        for i in range(8):
            shutter = False
            frame = 0
            while shutter == False:
                offset = self._shutter_offsets[i] + (frame * 32)
                stripe = self._bin_files[i][offset : offset + 32]
                shutter = stripe[0]["shutter"]
                if shutter == 0:
                    frame += 1

            self._shutter_offsets[i] += frame * 32

    def _grab_frame(self, frame):
        fullImage = np.zeros([1860, 2048], dtype=np.int16)
        for ii in range(8):
            xOffset = ii * 256  # the x location of the sector for each BIN file
            # read a set of stripes:
            start = self._shutter_offsets[ii] + (frame * 32)
            np.copyto(self._temp, self._bin_files[ii][start : start + 32])

            if np.any(self._temp[:]["sync"] != 0xFFFF0055):
                print(
                    "The binary file is unsynchronized and cannot be read. You must use Digital Micrograph to extract to *.dm4."
                )
                break  # stop reading if the sync byte is not correct. Ideally, this would read the next byte, etc... until this exact value is found
            # parse the stripes
            for jj in range(0, 32):
                coords = self._temp[jj][
                    "coords"
                ]  # first x, first y, last x, last y; ref to 0;inclusive;should indicate 16x930 pixels
                # place the data in the image
                fullImage[
                    coords[1] : coords[3] + 1,
                    coords[0] + xOffset : coords[2] + xOffset + 1,
                ] = _convert_uint12(self._temp[jj]["data"]).reshape([930, 16])
        return fullImage

    @staticmethod
    def _subtract_readout_noise(DP):
        # subtract readout noise using the hidden stripes
        darkref = np.floor_divide(
            np.sum(DP[1792:, :], axis=0, dtype=np.int16),
            np.int16(1860 - 1792),
            dtype=np.int16,
        )
        DP -= darkref[np.newaxis, :]

    # Handle the user specifying a dark reference (fix the size and make sure auto gets turned off)
    @property
    def dark_reference(self):
        return self._user_dark_reference[:1792, :1920]

    @dark_reference.setter
    def dark_reference(self, dr):
        assert dr.shape == (
            1792,
            1920,
        ), "Dark reference must be the size of an active frame"
        # assert dr.dtype == np.uint16, "Dark reference must be 16 bit unsigned"
        self._user_dark_reference = np.zeros((1860, 2048), dtype=np.int16)
        self._user_dark_reference[:1792, :1920] = dr

        # disable auto noise reduction
        self._hidden_stripe_noise_reduction = False
        self._user_noise_reduction = True

    @dark_reference.deleter
    def dark_reference(self):
        del self._user_dark_reference
        self._user_noise_reduction = False

    # ======== UTILITY FUNCTIONS ========#
    def _parse_slices(self, i, mode):
        assert len(i) == 2, "Wrong size input"

        if mode == "real":
            xMax = self.shape[0]  # R_Nx
            yMax = self.shape[1]  # R_Ny
        elif mode == "diffraction":
            xMax = self.shape[2]  # Q_Nx
            yMax = self.shape[3]  # Q_Ny
        else:
            raise ValueError("incorrect slice mode")

        if isinstance(i[0], slice):
            xInds = np.arange(i[0].start or 0, (i[0].stop or xMax), (i[0].step or 1))
        else:
            xInds = i[0]

        if isinstance(i[1], slice):
            yInds = np.arange(i[1].start or 0, (i[1].stop or yMax), (i[1].step or 1))
        else:
            yInds = i[1]

        if mode == "diffraction":
            x, y = np.meshgrid(xInds, yInds, indexing="ij")
        elif mode == "real":
            x, y = np.meshgrid(xInds, yInds, indexing="xy")
        return x, y

    def _guess_number_frames(self):
        import os

        nbytes = np.array(
            [os.path.getsize(self._bin_prefix + f"{n}.bin") for n in range(1, 9)]
        )
        return np.min(nbytes) // 0x5758

    def _write_to_hdf5(self, group):
        """
        Write the entire dataset to an HDF5 file.
        group should be an HDF5 Group object.
        ( This function is normally called via py4DSTEM.file.io.save() )
        """
        dset = group.create_dataset("data", (self.shape), "i2")

        for sy in range(self.shape[1]):
            for sx in range(self.shape[0]):
                dset[sx, sy, :, :] = self[sx, sy, :, :]

        return dset


# ======= UTILITIES OUTSIDE THE CLASS ======#
import sys
if 'numba' in sys.modules:
    @nb.njit(nb.int16[::1](nb.uint8[::1]), fastmath=False, parallel=False)
    def _convert_uint12(data_chunk):
        """
        data_chunk is a contigous 1D array of uint8 data)
        eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
        """

        # ensure that the data_chunk has the right length
        assert np.mod(data_chunk.shape[0], 3) == 0

        out = np.empty(data_chunk.shape[0] // 3 * 2, dtype=np.uint16)

        for i in nb.prange(data_chunk.shape[0] // 3):
            fst_uint8 = np.uint16(data_chunk[i * 3])
            mid_uint8 = np.uint16(data_chunk[i * 3 + 1])
            lst_uint8 = np.uint16(data_chunk[i * 3 + 2])

            out[i * 2] = (
                fst_uint8 | (mid_uint8 & 0x0F) << 8
            )  # (fst_uint8 << 4) + (mid_uint8 >> 4)
            out[i * 2 + 1] = (
                mid_uint8 & 0xF0
            ) >> 4 | lst_uint8 << 4  # ((mid_uint8 % 16) << 8) + lst_uint8

        DP = out.astype(np.int16)
        return DP
else:
    def _convert_uint12(data_chunk):
        """
        data_chunk is a contigous 1D array of uint8 data)
        eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
        """

        # ensure that the data_chunk has the right length
        assert np.mod(data_chunk.shape[0], 3) == 0

        out = np.empty(data_chunk.shape[0] // 3 * 2, dtype=np.uint16)

        for i in range(data_chunk.shape[0] // 3):
            fst_uint8 = np.uint16(data_chunk[i * 3])
            mid_uint8 = np.uint16(data_chunk[i * 3 + 1])
            lst_uint8 = np.uint16(data_chunk[i * 3 + 2])

            out[i * 2] = (
                fst_uint8 | (mid_uint8 & 0x0F) << 8
            )  # (fst_uint8 << 4) + (mid_uint8 >> 4)
            out[i * 2 + 1] = (
                mid_uint8 & 0xF0
            ) >> 4 | lst_uint8 << 4  # ((mid_uint8 % 16) << 8) + lst_uint8

        DP = out.astype(np.int16)
        return DP



