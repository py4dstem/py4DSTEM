# Read the mib file captured using the Merlin detector
# Author: Tara Mishra, tara.matsci@gmail. 
# Based on the PyXEM load_mib module https://github.com/pyxem/pyxem/blob/563a3bb5f3233f46cd3e57f3cd6f9ddf7af55ad0/pyxem/utils/io_utils.py

from collections.abc import Sequence
import numpy as np
try:
    import numba as nb
except ImportError:
    pass
from ...utils.tqdmnd import tqdmnd
from ..datastructure import DataCube
import os

def manageHeader(fname):
    """Get necessary information from the header of the .mib file.
    Parameters
    ----------
    fname : str
        Filename for header file.
    Returns
    -------
    hdr : tuple
        (DataOffset,NChips,PixelDepthInFile,sensorLayout,Timestamp,shuttertime,bitdepth)
    Examples
    --------
    #Output for 6bit 256*256 data:
    #(768, 4, 'R64', '2x2', '2019-06-14 11:46:12.607836', 0.0002, 6)
    #Output for 12bit single frame nor RAW:
    #(768, 4, 'U16', '2x2', '2019-06-06 11:12:42.001309', 0.001, 12)
    """
    Header = str()
    with open(fname, "rb") as input:
        aByte = input.read(1)
        Header += str(aByte.decode("ascii"))
        # This gets rid of the header
        while aByte and ord(aByte) != 0:

            aByte = input.read(1)
            Header += str(aByte.decode("ascii"))

    elements_in_header = Header.split(",")

    DataOffset = int(elements_in_header[2])

    NChips = int(elements_in_header[3])

    PixelDepthInFile = elements_in_header[6]
    sensorLayout = elements_in_header[7].strip()
    Timestamp = elements_in_header[9]
    shuttertime = float(elements_in_header[10])

    if PixelDepthInFile == "R64":
        bitdepth = int(elements_in_header[18])  # RAW
    elif PixelDepthInFile == "U16":
        bitdepth = 12
    elif PixelDepthInFile == "U08":
        bitdepth = 6
    elif PixelDepthInFile == "U32":
        bitdepth = 24

    hdr = (
        DataOffset,
        NChips,
        PixelDepthInFile,
        sensorLayout,
        Timestamp,
        shuttertime,
        bitdepth,
    )

    return hdr

def parse_hdr(fp):
    """Parse information from mib file header info from _manageHeader function.
    Parameters
    ----------
    fp : str
        Filepath to .mib file.
    Returns
    -------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.
        The entries of the dictionary are as follows:
        'width': int
            pixels, detector number of pixels in x direction,
        'height': int
            pixels detector number of pixels in y direction,
        'Assembly Size': str
            configuration of the detector chips, e.g. '2x2' for quad,
        'offset': int
            number of characters in the header before the first frame starts,
        'data-type': str
            always 'unsigned',
        'data-length': str
            identifying dtype,
        'Counter Depth (number)': int
            counter bit depth,
        'raw': str
            regular binary 'MIB' or raw binary 'R64',
        'byte-order': str
            always 'dont-care',
        'record-by': str
            'image' or 'vector' - only 'image' encountered,
        'title': str
            path of the mib file without extension, e.g. '/dls/e02/data/2020/cm26481-1/Merlin/testing/20200204 115306/test',
        'date': str
            date created, e.g. '20200204',
        'time': str
            time created, e.g. '11:53:32.295336',
        'data offset': int
            number of characters at the header.
    """
    hdr_info = {}

    read_hdr = manageHeader(fp)

    # Set the array size of the chip

    if read_hdr[3] == "1x1":
        hdr_info["width"] = 256
        hdr_info["height"] = 256
    elif read_hdr[3] == "2x2":
        hdr_info["width"] = 512
        hdr_info["height"] = 512

    hdr_info["Assembly Size"] = read_hdr[3]

    # Set mib offset
    hdr_info["offset"] = read_hdr[0]
    # Set data-type
    hdr_info["data-type"] = "unsigned"
    # Set data-length
    if read_hdr[6] == "1":
        # Binary data recorded as 8 bit numbers
        hdr_info["data-length"] = "8"
    else:
        # Changes 6 to 8 , 12 to 16 and 24 to 32 bit
        cd_int = int(read_hdr[6])
        hdr_info["data-length"] = str(int((cd_int + cd_int / 3)))

    hdr_info["Counter Depth (number)"] = int(read_hdr[6])
    if read_hdr[2] == "R64":
        hdr_info["raw"] = "R64"
    else:
        hdr_info["raw"] = "MIB"
    # Set byte order
    hdr_info["byte-order"] = "dont-care"
    # Set record by to stack of images
    hdr_info["record-by"] = "image"

    # Set title to file name
    hdr_info["title"] = fp.split(".")[0]
    # Set time and date
    # Adding the try argument to accommodate the new hdr formatting as of April 2018
    try:
        year, month, day_time = read_hdr[4].split("-")
        day, time = day_time.split(" ")
        hdr_info["date"] = year + month + day
        hdr_info["time"] = time
    except BaseException:
        day, month, year_time = read_hdr[4].split("/")
        year, time = year_time.split(" ")
        hdr_info["date"] = year + month + day
        hdr_info["time"] = time

    hdr_info["data offset"] = read_hdr[0]

    return hdr_info

def mib_to_daskarr(fp, mmap_mode="r"):
    """Reads the binary mib file into a numpy memmap object and returns as dask array object.
    Parameters
    ----------
    fp: str
        MIB file name / path
    mmap_mode: str
        memmpap read mode - default is 'r'
    Returns
    -------
    data_da: dask array
        data as a dask array object
    """
    hdr_info = parse_hdr(fp)
    data_length = hdr_info["data-length"]
    data_type = hdr_info["data-type"]
    endian = hdr_info["byte-order"]
    read_offset = 0

    if data_type == "signed":
        data_type = "int"
    elif data_type == "unsigned":
        data_type = "uint"
    elif data_type == "float":
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian
    endian = ">"
    data_type += str(int(data_length))
    # uint1 not a valid dtype
    if data_type == "uint1":
        data_type = "uint8"
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    data_mem = np.memmap(fp, offset=read_offset, dtype=data_type, mode=mmap_mode)
    #data_da = da.from_array(data_mem, chunks="auto")
    data_da = data_mem
    return data_da

def get_mib_depth(hdr_info, fp):
    """Determine the total number of frames based on .mib file size.
    Parameters
    ----------
    hdr_info : dict
        Dictionary containing header info extracted from .mib file.
    fp : filepath
        Path to .mib file.
    Returns
    -------
    depth : int
        Number of frames in the stack
    """
    # Define standard frame sizes for quad and single medipix chips
    if hdr_info["Assembly Size"] == "2x2":
        mib_file_size_dict = {
            "1": 33536,
            "6": 262912,
            "12": 525056,
            "24": 1049344,
        }
    if hdr_info["Assembly Size"] == "1x1":
        mib_file_size_dict = {
            "1": 8576,
            "6": 65920,
            "12": 131456,
            "24": 262528,
        }

    file_size = os.path.getsize(fp[:-3] + "mib")
    if hdr_info["raw"] == "R64":

        single_frame = mib_file_size_dict.get(str(hdr_info["Counter Depth (number)"]))
        depth = int(file_size / single_frame)
    elif hdr_info["raw"] == "MIB":
        if hdr_info["Counter Depth (number)"] == "1":
            # 1 bit and 6 bit non-raw frames have the same size
            single_frame = mib_file_size_dict.get("6")
            depth = int(file_size / single_frame)
        else:
            single_frame = mib_file_size_dict.get(
                str(hdr_info["Counter Depth (number)"])
            )
            depth = int(file_size / single_frame)

    return depth

def get_hdr_bits(hdr_info):
    """Gets the number of character bits for the header for each frame given the data type.
    Parameters
    ----------
    hdr_info: dict
        output of the parse_hdr function
    Returns
    -------
    hdr_bits: int
        number of characters in the header
    """
    data_length = hdr_info["data-length"]
    data_type = hdr_info["data-type"]

    if data_type == "signed":
        data_type = "int"
    elif data_type == "unsigned":
        data_type = "uint"
    elif data_type == "float":
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    # mib data always big-endian
    endian = ">"
    data_type += str(int(data_length))
    # uint1 not a valid dtype
    if data_type == "uint1":
        data_type = "uint8"
        data_type = np.dtype(data_type)
    else:
        data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    if data_length == "1":
        hdr_multiplier = 1
    else:
        hdr_multiplier = (int(data_length) / 8) ** -1

    hdr_bits = int(hdr_info["data offset"] * hdr_multiplier)

    return hdr_bits

def load_mib(
    file_path, 
    mem=None, 
    binfactor=None,
    reshape=True,
    flip=True,
    scan=(256,256)):
    """
    Read a MIB file and return as py4DSTEM DATACUBE"""
    hdr_stuff = parse_hdr(file_path)
    width = hdr_stuff["width"]
    height = hdr_stuff["height"]
    width_height = width * height
    
    data = mib_to_daskarr(file_path)
    depth = get_mib_depth(hdr_stuff, file_path)
    hdr_bits = get_hdr_bits(hdr_stuff)

    if hdr_stuff["Counter Depth (number)"] == 1:
        # RAW 1 bit data: the header bits are written as uint8 but the frames
        # are binary and need to be unpacked as such.
        data = data.reshape(-1, int(width_height / 8 + hdr_bits))
        data = data[:, hdr_bits:]
        # get the shape axis 1 before unpackbit
        s0 = data.shape[0]
        s1 = data.shape[1]
        data = np.unpackbits(data)
        data.reshape(s0, s1 * 8)
    else:
        data = data.reshape(-1, int(width_height + hdr_bits))
        data = data[:, hdr_bits:]

    if hdr_stuff["raw"] == "MIB":
        data = data.reshape(depth,width,height)
    else:
        print('Data type not supported as MIB reader')
    
    if reshape:
        """
        Reshapes the lazy-imported stack of dimensions: (xxxxxx|Det_X, Det_Y) to the correct scan pattern
        shape: (x, y | Det_X, Det_Y).
        """
      
        data = data.reshape(scan[0],scan[1],width,height)
        # print(data.shape)

    
    py4dstem_data = DataCube(data=data)
    return py4dstem_data


#def read_mib(fp, mem="MEMMAP", binfactor=1, metadata=False, sig_shape = (256,256),**kwargs):
#
#    """
#    READ an MIB file
#
#    Args:
#        fp: str Path to the file
#        mem (str, optional): Specifies how the data should be stored; must be "RAM"
#            or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default is "MEMMAP".
#        binfactor: (int, optional): Bin the data, in diffraction space, as it's loaded.
#            See docstring for py4DSTEM.file.io.read.  Must be 1, retained only for
#            compatibility.
#        metadata (bool, optional): if True, returns the file metadata as a Metadata
#            instance.
#        sig_shape: (tuple): Shape of the signal. By default we use a 256x256 scan shape
#
#    Returns:
#        (variable): The return value depends on usage:
#
#
#            * if metadata==False, returns the 4D-STEM dataset as a DataCube
#            * if metadata==True, returns the metadata as a Metadata instance
#
#        Note that metadata is read either way - in the latter case ONLY
#        metadata is read and returned, in the former case a DataCube
#        is returned with the metadata attached at datacube.metadata
#    """
#   
#    assert mem == "MEMMAP", "K2 files can only be memory-mapped, sorry."
#    assert binfactor == 1, "K2 files can only be read at full resolution, sorry."
#
#    if metadata is True:
#        return None
#
#    data = 0
#    return DataCube(data)
#