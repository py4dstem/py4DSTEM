# -*- coding: utf-8 -*-

# Reads an EMPAD file
# 
# Created on Tue Jan 15 13:06:03 2019
# @author: percius
# Edited on 20190409 by bsavitzky and rdhall

import numpy as np
from pathlib import Path

def read_empad(filename):
    """
    Reads the EMPAD file at filename, returning a numpy array.

    EMPAD files are shaped as 130x128 arrays, consisting of 128x128 arrays of data followed by
    two rows of metadata.  For each frame, its position in the scan is embedded in the metadata.
    By extracting the scan position of the first and last frames, the function determines the scan
    size. Then, the full dataset is loaded.

    Note that the output of this function is a datacube which includes the 2 rows of metadata.

    Accepts:
        filename    (str) path to the empad file

    Returns:
        data        (ndarray) the 4D datacube, including the metadata.
                    raw data only can be accessed data[:,:,:128,:]
    """
    row = 130
    col = 128
    fPath = Path(filename)

    # Parse the EMPAD metadata for first and last images
    empadDTYPE = np.dtype([('data','16384float32'),('metadata','256float32')])
    with open(fPath,'rb') as fid:
        imFirst = np.fromfile(fid,dtype=empadDTYPE,count=1)
        fid.seek(-128*130*4,2)
        imLast = np.fromfile(fid,dtype=empadDTYPE,count=1)

    # Get the scan shape
    shape0 = imFirst['metadata'][0][128+12:128+16]
    shape1 = imLast['metadata'][0][128+12:128+16]
    kShape = shape0[2:4]                        # detector shape
    rShape = 1 + shape1[0:2] - shape0[0:2]      # scan shape

    # Load the full data set
    with open(fPath,'rb') as fid:
        data = np.fromfile(fid,np.float32)
        data = np.reshape(data, (int(rShape[0]), int(rShape[1]), row, col))

    return data


