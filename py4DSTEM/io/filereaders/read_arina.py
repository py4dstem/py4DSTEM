import h5py
import hdf5plugin # noqa: F401
import numpy as np
from py4DSTEM.datacube import DataCube
from py4DSTEM.preprocess.utils import bin2D


def read_arina(
    filename,
    scan_width=1,
    mem="RAM",
    binfactor: int = 1,
    dtype_bin: float = None,
    flatfield: np.ndarray = None,
):
    """
    File reader for arina 4D-STEM datasets
    Args:
        filename: str with path to master file
        scan_width: x dimension of scan
        mem (str):  Must be "RAM" or "MEMMAP". Specifies how the data is
            loaded; "RAM" transfer the data from storage to RAM, while "MEMMAP"
            leaves the data in storage and creates a memory map which points to
            the diffraction patterns, allowing them to be retrieved individually
            from storage.
        binfactor (int): Diffraction space binning factor for bin-on-load.
        dtype_bin(float): specify datatype for bin on load if need something
            other than uint16
        flatfield (np.ndarray):
            flatfield forcorrection factors

    Returns:
        DataCube
    """
    assert mem == "RAM", "read_arina does not support memory mapping"

    f = h5py.File(filename, "r")
    nimages = 0

    # Count the number of images in all datasets
    for dset in f["entry"]["data"]:
        nimages = nimages + f["entry"]["data"][dset].shape[0]
        height = f["entry"]["data"][dset].shape[1]
        width = f["entry"]["data"][dset].shape[2]
        dtype = f["entry"]["data"][dset].dtype

    width = width // binfactor
    height = height // binfactor

    assert (
        nimages % scan_width < 1e-6
    ), "scan_width must be integer multiple of x*y size"

    if dtype.type is np.uint32:
        print("Dataset is uint32 but will be converted to uint16")
        dtype = np.dtype(np.uint16)

    if dtype_bin:
        array_3D = np.empty((nimages, width, height), dtype=dtype_bin)
    else:
        array_3D = np.empty((nimages, width, height), dtype=dtype)

    image_index = 0

    if flatfield is None:
        correction_factors = 1
    else:
        # Avoid div by 0 errors -> pixel with value 0 will be set to meadian
        flatfield[flatfield == 0] = 1
        correction_factors = np.median(flatfield) / flatfield

    for dset in f["entry"]["data"]:
        image_index = _processDataSet(
            f["entry"]["data"][dset],
            image_index,
            array_3D,
            binfactor,
            correction_factors,
        )

    if f.__bool__():
        f.close()

    scan_height = int(nimages / scan_width)

    datacube = DataCube(
        np.flip(
            array_3D.reshape(
                scan_width, scan_height, array_3D.data.shape[1], array_3D.data.shape[2]
            ),
            0,
        )
    )

    return datacube


def _processDataSet(dset, start_index, array_3D, binfactor, correction_factors):
    image_index = start_index
    nimages_dset = dset.shape[0]

    for i in range(nimages_dset):
        if binfactor == 1:
            array_3D[image_index] = np.multiply(
                dset[i].astype(array_3D.dtype), correction_factors
            )
        else:
            array_3D[image_index] = bin2D(
                np.multiply(dset[i].astype(array_3D.dtype), correction_factors),
                binfactor,
            )

        image_index = image_index + 1
    return image_index
