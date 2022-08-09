# Reads a digital micrograph 4D-STEM dataset

import numpy as np
from pathlib import Path
from ncempy.io import dm
from ..datastructure import DataCube, Array, Metadata
from ...process.utils import bin2D


def read_dm(
    filepath,
    name = 'dm_dataset',
    mem = "RAM",
    binfactor = 1,
    **kwargs
    ):
    """
    Read a digital micrograph 4D-STEM file.

    Args:
        filepath: str or Path Path to the file
        mem (str, optional): Specifies how the data should be stored; must be "RAM",
            or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default is "RAM".
        binfactor (int, optional): Bin the data, in diffraction space, as it's loaded.
            See docstring for py4DSTEM.file.io.read.  Default is 1.
        metadata (bool, optional): if True, returns the file metadata as a Metadata
            instance.

    Returns:
       DataCube, if a 4D dataset is found, else Array if a 2D dataset is found
    """

    with dm.fileDM(filepath, on_memory=False) as dmFile:
        # loop through datasets looking for one with more than 2D
        # This is needed because:
        #   NCEM TitanX files store 4D data in a 3D array
        #   K3 sometimes stores 2D images alongside the 4D array
        thumbanil_count = 1 if dmFile.thumbnail else 0
        dataset_index = 0
        for i in range(dmFile.numObjects - thumbanil_count):
            temp_data = dmFile.getMemmap(dataset_index)
            if len(np.squeeze(temp_data).shape) > 2 :
                dataset_index = i
                break

        # Handle 3D NCEM TitanX data
        titan_shape = _process_NCEM_TitanX_Tags(dmFile)

        if (mem, binfactor) == ("RAM", 1):
                _data = dmFile.getDataset(dataset_index)['data']
        elif (mem, binfactor) == ("MEMMAP", 1):
            _data = dmFile.getMemmap(dataset_index)
        elif (mem) == ("RAM"):
            raise NotImplementedError("Bin-on-load currently broken.")
            # with dm.fileDM(filepath, on_memory=True) as dmFile:
            #     # loop through the datasets until a >2D one is found:
            #     i = 0
            #     valid_data = False
            #     while not valid_data:
            #         memmap = dmFile.getMemmap(i)
            #         if len(np.squeeze(memmap).shape) > 2 :
            #             valid_data = True
            #         i += 1
            #     if "dtype" in kwargs.keys():
            #         dtype = kwargs["dtype"]
            #     else:
            #         dtype = memmap.dtype
            #     shape = memmap.shape
            #     rank = len(shape)
            #     if rank==4:
            #         R_Nx, R_Ny, Q_Nx, Q_Ny = shape
            #     elif rank==3:
            #         titanTags = _process_NCEM_TitanX_Tags(dmFile)
            #         if titanTags is not None:
            #             R_Nx, R_Ny = titanTags
            #             Q_Nx, Q_Ny = shape[1:]
            #         else:
            #             R_Nx, Q_Nx, Q_Ny = shape
            #             R_Ny = 1
            #     else:
            #         raise Exception(f"Data should be 4-dimensional; found {rank} dimensions")
            #     Q_Nx, Q_Ny = Q_Nx // binfactor, Q_Ny // binfactor
            #     data = np.empty((R_Nx, R_Ny, Q_Nx, Q_Ny), dtype=dtype)
            #     for Rx in range(R_Nx):
            #         for Ry in range(R_Ny):
            #             if rank==4:
            #                 data[Rx, Ry, :, :] = bin2D(
            #                     memmap[Rx, Ry, :, :,], binfactor, dtype=dtype
            #                 )
            #             else:
            #                 data[Rx, Ry, :, :] = bin2D(
            #                     memmap[Rx, :, :,], binfactor, dtype=dtype
            #                 )
            #     dc = DataCube(data=data)
        else:
            raise Exception("Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'.")
            return

        if titan_shape is not None:
            # NCEM TitanX tags were found
            _data = np.reshape(_data,titan_shape + _data.shape[-2:])

        if len(_data.shape) == 4:
            data = DataCube(_data, name=name)
        else:
            data = Array(_data, name=name)

    return data



def _process_NCEM_TitanX_Tags(dmFile, dc=None):
    """
    Check the metadata in the DM File for certain tags which are added by
    the NCEM TitanX, and reshape the 3D datacube into 4D using these tags.
    Also fixes the two-pixel roll issue present in TitanX data. If no datacube
    is passed, return R_Nx and R_Ny
    """
    scanx = [v for k,v in dmFile.allTags.items() if "4D STEM Tags.Scan shape X" in k]
    scany = [v for k,v in dmFile.allTags.items() if "4D STEM Tags.Scan shape Y" in k]
    if len(scanx) == 1 and len(scany) == 1:
        # TitanX tags found!
        R_Nx = int(scany[0]) # need to flip x/y
        R_Ny = int(scanx[0])

        if dc is not None:
            dc.set_scan_shape(R_Nx,R_Ny)
            dc.data = np.roll(dc.data,shift=-2,axis=1)
        else:
            return R_Nx, R_Ny


#def get_metadata_from_dmFile(fp):
#    """ Accepts a filepath to a dm file and returns a Metadata instance
#    """
#    metadata = Metadata()
#
#    with dm.fileDM(fp, on_memory=False) as dmFile:
#        pixelSizes = dmFile.scale
#        pixelUnits = dmFile.scaleUnit
#        assert pixelSizes[0] == pixelSizes[1], "Rx and Ry pixel sizes don't match"
#        assert pixelSizes[2] == pixelSizes[3], "Qx and Qy pixel sizes don't match"
#        assert pixelUnits[0] == pixelUnits[1], "Rx and Ry pixel units don't match"
#        assert pixelUnits[2] == pixelUnits[3], "Qx and Qy pixel units don't match"
#        for i in range(len(pixelUnits)):
#            if pixelUnits[i] == "":
#                pixelUnits[i] = "pixels"
#        metadata.set_R_pixel_size__microscope(pixelSizes[0])
#        metadata.set_R_pixel_size_units__microscope(pixelUnits[0])
#        metadata.set_Q_pixel_size__microscope(pixelSizes[2])
#        metadata.set_Q_pixel_size_units__microscope(pixelUnits[2])
#
#    return metadata




