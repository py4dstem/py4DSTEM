# Reads a digital micrograph 4D-STEM dataset

import numpy as np
from pathlib import Path
from ncempy.io import dm
from ..datastructure import DataCube, Metadata
from ...process.utils import bin2D


def read_dm(fp, mem="RAM", binfactor=1, metadata=False, **kwargs):
    """
    Read a digital micrograph 4D-STEM file.

    Args:
        fp: str or Path Path to the file
        mem (str, optional): Specifies how the data should be stored; must be "RAM",
            or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor (int, optional): Bin the data, in diffraction space, as it's loaded.
            See docstring for py4DSTEM.file.io.read.  Default is 1.
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
    assert(isinstance(fp,(str,Path))), "Error: filepath fp must be a string or pathlib.Path"
    assert(mem in ['RAM','MEMMAP']), 'Error: argument mem must be either "RAM" or "MEMMAP"'
    assert(isinstance(binfactor,int)), "Error: argument binfactor must be an integer"
    assert(binfactor>=1), "Error: binfactor must be >= 1"

    md = get_metadata_from_dmFile(fp)
    if metadata:
        return md

    if (mem, binfactor) == ("RAM", 1):
        with dm.fileDM(fp, on_memory=True) as dmFile:
            # loop through the datasets until a >2D one is found:
            i = 0
            valid_data = False
            while not valid_data:
                data = dmFile.getMemmap(i)
                if len(np.squeeze(data).shape) > 2 :
                    valid_data = True
                    dataSet = dmFile.getDataset(i)
                i += 1
            dc = DataCube(data=dataSet["data"])
            _process_NCEM_TitanX_Tags(dmFile, dc)
    elif (mem, binfactor) == ("MEMMAP", 1):
        with dm.fileDM(fp, on_memory=False) as dmFile:
            # loop through the datasets until a >2D one is found:
            i = 0
            valid_data = False
            while not valid_data:
                memmap = dmFile.getMemmap(i)
                if len(np.squeeze(memmap).shape) > 2 :
                    valid_data = True
                i += 1
            dc = DataCube(data=memmap)
            _process_NCEM_TitanX_Tags(dmFile, dc)
    elif (mem) == ("RAM"):
        with dm.fileDM(fp, on_memory=True) as dmFile:
            # loop through the datasets until a >2D one is found:
            i = 0
            valid_data = False
            while not valid_data:
                memmap = dmFile.getMemmap(i)
                if len(np.squeeze(memmap).shape) > 2 :
                    valid_data = True
                i += 1
            if "dtype" in kwargs.keys():
                dtype = kwargs["dtype"]
            else:
                dtype = memmap.dtype
            shape = memmap.shape
            rank = len(shape)
            if rank==4:
                R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            elif rank==3:
                titanTags = _process_NCEM_TitanX_Tags(dmFile)
                if titanTags is not None:
                    R_Nx, R_Ny = titanTags
                    Q_Nx, Q_Ny = shape[1:]
                else:
                    R_Nx, Q_Nx, Q_Ny = shape
                    R_Ny = 1
            else:
                raise Exception(f"Data should be 4-dimensional; found {rank} dimensions")
            Q_Nx, Q_Ny = Q_Nx // binfactor, Q_Ny // binfactor
            data = np.empty((R_Nx, R_Ny, Q_Nx, Q_Ny), dtype=dtype)
            for Rx in range(R_Nx):
                for Ry in range(R_Ny):
                    if rank==4:
                        data[Rx, Ry, :, :] = bin2D(
                            memmap[Rx, Ry, :, :,], binfactor, dtype=dtype
                        )
                    else:
                        data[Rx, Ry, :, :] = bin2D(
                            memmap[Rx, :, :,], binfactor, dtype=dtype
                        )
            dc = DataCube(data=data)
    else:
        raise Exception(
            "Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'."
        )
        return

    dc.metadata = md
    return dc

def _process_NCEM_TitanX_Tags(dmFile, dc=None):
    """
    Check the metadata in the DM File for certain tags which are added by the NCEM TitanX,
    and reshape the 3D datacube into 4D using these tags. If no datacube is passed, 
    return R_Nx and R_Ny
    """
    scanx = [v for k,v in dmFile.allTags.items() if "4D STEM Tags.Scan shape X" in k]
    scany = [v for k,v in dmFile.allTags.items() if "4D STEM Tags.Scan shape Y" in k]
    if len(scanx) == 1 and len(scany) == 1:
        # TitanX tags found!
        R_Nx = int(scanx[0])
        R_Ny = int(scany[0])

        if dc is not None:
            dc.set_scan_shape(R_Nx,R_Ny)
        else:
            return R_Nx, R_Ny


def get_metadata_from_dmFile(fp):
    """ Accepts a filepath to a dm file and returns a Metadata instance
    """
    metadata = Metadata()

    with dm.fileDM(fp, on_memory=False) as dmFile:
        pixelSizes = dmFile.scale
        pixelUnits = dmFile.scaleUnit
        assert pixelSizes[0] == pixelSizes[1], "Rx and Ry pixel sizes don't match"
        assert pixelSizes[2] == pixelSizes[3], "Qx and Qy pixel sizes don't match"
        assert pixelUnits[0] == pixelUnits[1], "Rx and Ry pixel units don't match"
        assert pixelUnits[2] == pixelUnits[3], "Qx and Qy pixel units don't match"
        for i in range(len(pixelUnits)):
            if pixelUnits[i] == "":
                pixelUnits[i] = "pixels"
        metadata.set_R_pixel_size__microscope(pixelSizes[0])
        metadata.set_R_pixel_size_units__microscope(pixelUnits[0])
        metadata.set_Q_pixel_size__microscope(pixelSizes[2])
        metadata.set_Q_pixel_size_units__microscope(pixelUnits[2])

    return metadata
