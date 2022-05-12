# Reads an EMPAD file
#
# Created on Tue Jan 15 13:06:03 2019
# @author: percius
# Edited on 20190409 by bsavitzky and rdhall
# Edited on 20210628 by sez

import numpy as np
from pathlib import Path
from ..datastructure import DataCube
from ...process.utils import bin2D, tqdmnd

def read_empad(filename, mem="RAM", binfactor=1, metadata=False, **kwargs):
    """
    Reads the EMPAD file at filename, returning a DataCube.

    EMPAD files are shaped as 130x128 arrays, consisting of 128x128 arrays of data followed by
    two rows of metadata.  For each frame, its position in the scan is embedded in the metadata.
    By extracting the scan position of the first and last frames, the function determines the scan
    size. Then, the full dataset is loaded and cropped to the 128x128 valid region.

    Accepts:
        filename    (str) path to the EMPAD file

    Returns:
        data        (DataCube) the 4D datacube, excluding the metadata rows.
    """
    assert(isinstance(filename, (str, Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM', 'MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor, int)), "Error: argument binfactor must be an integer"
    assert(binfactor >= 1), "Error: binfactor must be >= 1"
    assert(metadata is False), "Error: EMPAD Reader does not support metadata."

    row = 130
    col = 128
    fPath = Path(filename)

    # Parse the EMPAD metadata for first and last images
    empadDTYPE = np.dtype([("data", "16384float32"), ("metadata", "256float32")])
    with open(fPath, "rb") as fid:
        imFirst = np.fromfile(fid, dtype=empadDTYPE, count=1)
        fid.seek(-128 * 130 * 4, 2)
        imLast = np.fromfile(fid, dtype=empadDTYPE, count=1)

    # Get the scan shape
    shape0 = imFirst["metadata"][0][128 + 12 : 128 + 16]
    shape1 = imLast["metadata"][0][128 + 12 : 128 + 16]
    rShape = 1 + shape1[0:2] - shape0[0:2]  # scan shape
    data_shape = (int(rShape[0]), int(rShape[1]), row, col)

    # Load the data
    if (mem, binfactor) == ("RAM", 1):
        with open(fPath, "rb") as fid:
            data = np.fromfile(fid, np.float32).reshape(data_shape)[:, :, :128, :]
    elif (mem, binfactor) == ("MEMMAP", 1):
        data = np.memmap(fPath, dtype=np.float32, mode="r", shape=data_shape)[
            :, :, :128, :
        ]
    elif (mem) == ("RAM"):
        # binned read into RAM
        memmap = np.memmap(fPath, dtype=np.float32, mode="r", shape=data_shape)[
            :, :, :128, :
        ]
        R_Nx, R_Ny, Q_Nx, Q_Ny = memmap.shape
        Q_Nx, Q_Ny = Q_Nx // binfactor, Q_Ny // binfactor
        data = np.zeros((R_Nx, R_Ny, Q_Nx, Q_Ny), dtype=np.float32)
        for Rx, Ry in tqdmnd(
            R_Nx, R_Ny, desc="Binning data", unit="DP", unit_scale=True
        ):
            data[Rx, Ry, :, :] = bin2D(
                memmap[Rx, Ry, :, :], binfactor, dtype=np.float32
            )
    else:
        # memory mapping + bin-on-load is not supported
        raise Exception(
            "Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'."
        )
        return

    return DataCube(data=data)
