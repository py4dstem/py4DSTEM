# Reads a digital micrograph 4D-STEM dataset

import numpy as np
from pathlib import Path
from ncempy.io import dm
from ...datastructure import DataCube
from ....process.utils import bin2D

def read_dm(fp, mem="RAM", binfactor=1, **kwargs):
    """
    Read a digital micrograph 4D-STEM file.

    Accepts:
        fp          str or Path Path to the file
        mem         str         (opt) Specifies how the data should be stored; must be "RAM" or "MEMMAP". See
                                docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor   int         (opt) Bin the data, in diffraction space, as it's loaded. See docstring for
                                py4DSTEM.file.io.read.  Default is 1.

    Returns:
        dc          DataCube    The 4D-STEM data.
        md          MetaData    The metadata.
    """
    assert(isinstance(fp,(str,Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"

    if (mem,binfactor)==("RAM",1):
        with dm.fileDM(fp) as dmFile:
            dataSet = dmFile.getDataset(0)
            dc = DataCube(data=dataSet['data'])
            md = None # TODO
    elif (mem,binfactor)==("MEMMAP",1):
        with dm.fileDM(fp, on_memory=False) as dmFile:
            memmap = dmFile.getMemmap(0)
            dc = DataCube(data=memmap)
            md = None # TODO
    elif (mem)==("RAM"):
        with dm.fileDM(fp, on_memory=False) as dmFile:
            memmap = dmFile.getMemmap(0)
            md = None # TODO
        if 'dtype' in kwargs.keys():
            dtype = kwargs['dtype']
        else:
            dtype = memmap.dtype
        R_Nx,R_Ny,Q_Nx,Q_Ny = memmap.shape
        Q_Nx, Q_Ny = Q_Nx//binfactor, Q_Ny//binfactor
        data = np.empty((R_Nx,R_Ny,Q_Nx,Q_Ny),dtype=dtype)
        for Rx in range(R_Nx):
            for Ry in range(R_Ny):
                data[Rx,Ry,:,:] = bin2D(memmap[Rx,Ry,:,:,],binfactor,dtype=dtype)
        dc = DataCube(data=data)
    else:
        raise Exception("Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'.")
        return

    return dc, md


