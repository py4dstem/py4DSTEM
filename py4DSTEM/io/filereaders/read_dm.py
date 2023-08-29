# Reads a digital micrograph 4D-STEM dataset

import numpy as np
from pathlib import Path
from ncempy.io import dm
from emdfile import tqdmnd, Array

from py4DSTEM.datacube import DataCube
from py4DSTEM.preprocess.utils import bin2D


def read_dm(filepath, name="dm_dataset", mem="RAM", binfactor=1, **kwargs):
    """
    Read a digital micrograph 4D-STEM file.

    Args:
        filepath: str or Path Path to the file
        mem (str, optional): Specifies how the data is stored. Must be
            "RAM", or "MEMMAP". See docstring for py4DSTEM.file.io.read. Default
            is "RAM".
        binfactor (int, optional): Bin the data, in diffraction space, as it's
            loaded. See docstring for py4DSTEM.file.io.read.  Default is 1.
        metadata (bool, optional): if True, returns the file metadata as a
            Metadata instance.
        kwargs:
            "dtype": a numpy dtype specifier to use for data binned on load,
                defaults to the data's current dtype

    Returns:
       DataCube if a 4D dataset is found, else an ND Array
    """

    # open the file
    with dm.fileDM(filepath, on_memory=False) as dmFile:
        # loop through datasets looking for one with more than 2D
        # This is needed because:
        #   NCEM TitanX files store 4D data in a 3D array
        #   K3 sometimes stores 2D images alongside the 4D array
        thumbanil_count = 1 if dmFile.thumbnail else 0
        dataset_index = 0
        for i in range(dmFile.numObjects - thumbanil_count):
            temp_data = dmFile.getMemmap(i)
            if len(np.squeeze(temp_data).shape) > 2:
                dataset_index = i
                break

        # We will only try to read pixel sizes for 4D data for now
        pixel_size_found = False
        if dmFile.dataShape[dataset_index + thumbanil_count] > 2:
            # The pixel sizes of all datasets are chained together, so
            # we have to figure out the right offset
            try:
                scale_offset = (
                    sum(dmFile.dataShape[:dataset_index]) + 2 * thumbanil_count
                )
                pixelsize = dmFile.scale[scale_offset:]
                pixelunits = dmFile.scaleUnit[scale_offset:]

                # Get the calibration pixel sizes
                Q_pixel_size = pixelsize[0]
                Q_pixel_units = "pixels" if pixelunits[0] == "" else pixelunits[0]
                R_pixel_size = pixelsize[2]
                R_pixel_units = "pixels" if pixelunits[2] == "" else pixelunits[2]

                # Check that the units are sensible
                # On microscopes that do not have live communication with the detector
                # the calibrations can be invalid
                if Q_pixel_units in ("nm", "µm"):
                    Q_pixel_units = "pixels"
                    Q_pixel_size = 1
                    R_pixel_units = "pixels"
                    R_pixel_size = 1

                # Convert mrad to Å^-1 if possible
                if Q_pixel_units == "mrad":
                    voltage = [
                        v
                        for t, v in dmFile.allTags.items()
                        if "Microscope Info.Voltage" in t
                    ]
                    if len(voltage) >= 1:
                        from py4DSTEM.process.utils import electron_wavelength_angstrom

                        wavelength = electron_wavelength_angstrom(voltage[0])
                        Q_pixel_units = "A^-1"
                        Q_pixel_size = (
                            Q_pixel_size / wavelength / 1000.0
                        )  # convert mrad to 1/Å
                elif Q_pixel_units == "1/nm":
                    Q_pixel_units = "A^-1"
                    Q_pixel_size /= 10

                pixel_size_found = True
            except Exception as err:
                pass

        # Handle 3D NCEM TitanX data
        titan_shape = _process_NCEM_TitanX_Tags(dmFile)

        if mem == "RAM":
            if binfactor == 1:
                _data = dmFile.getDataset(dataset_index)["data"]
            else:
                # get a memory map
                _mmap = dmFile.getMemmap(dataset_index)

                # get the dtype for the binned data
                dtype = kwargs.get("dtype", _mmap[0, 0].dtype)

                if titan_shape is not None:
                    # NCEM TitanX tags were found
                    _mmap = np.reshape(_mmap, titan_shape + _mmap.shape[-2:])
                new_shape = (
                    *_mmap.shape[:2],
                    _mmap.shape[2] // binfactor,
                    _mmap.shape[3] // binfactor,
                )
                _data = np.zeros(new_shape, dtype=dtype)

                for rx, ry in tqdmnd(*_data.shape[:2]):
                    _data[rx, ry] = bin2D(_mmap[rx, ry], binfactor, dtype=dtype)

        elif (mem, binfactor) == ("MEMMAP", 1):
            _data = dmFile.getMemmap(dataset_index)
        else:
            raise Exception(
                "Memory mapping and on-load binning together is not supported.  Either set binfactor=1 or mem='RAM'."
            )
            return

        if titan_shape is not None:
            # NCEM TitanX tags were found
            _data = np.reshape(_data, titan_shape + _data.shape[-2:])

        if len(_data.shape) == 4:
            data = DataCube(_data, name=name)
            if pixel_size_found:
                try:
                    data.calibration.set_Q_pixel_size(Q_pixel_size * binfactor)
                    data.calibration.set_Q_pixel_units(Q_pixel_units)
                    data.calibration.set_R_pixel_size(R_pixel_size)
                    data.calibration.set_R_pixel_units(R_pixel_units)
                except Exception as err:
                    print(
                        f"Setting pixel sizes of the datacube failed with error {err}"
                    )
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
    scanx = [v for k, v in dmFile.allTags.items() if "4D STEM Tags.Scan shape X" in k]
    scany = [v for k, v in dmFile.allTags.items() if "4D STEM Tags.Scan shape Y" in k]
    if len(scanx) >= 1 and len(scany) >= 1:
        # TitanX tags found!
        R_Nx = int(scany[0])  # need to flip x/y
        R_Ny = int(scanx[0])

        if dc is not None:
            dc.set_scan_shape(R_Nx, R_Ny)
            dc.data = np.roll(dc.data, shift=-2, axis=1)
        else:
            return R_Nx, R_Ny


# def get_metadata_from_dmFile(fp):
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
