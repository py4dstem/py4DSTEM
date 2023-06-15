import h5py
import numpy as np
from py4DSTEM.classes import DataCube

def read_arina(filename, scan_width):

    """
    Args:
        filename: str or Path Path to the file
        scan_width: x dimension of scan

    Returns:
            DataCube
    """
    f = h5py.File(filename, "r")
    nimages = 0

    # Count the number of images in all datasets
    for dset in f["entry"]["data"]:
        nimages = nimages + f["entry"]["data"][dset].shape[0]
        height = f["entry"]["data"][dset].shape[1]
        width = f["entry"]["data"][dset].shape[2]
        dtype = f["entry"]["data"][dset].dtype

    assert (
        nimages % scan_width < 1e-6
    ), "scan_width must be integer multiple of x*y size"

    if dtype.type is np.uint32:
        print("Dataset is uint32 but will be converted to uint16")
        dtype = np.dtype(np.uint16)

    array_3D = np.empty((nimages, width, height), dtype=dtype)

    image_index = 0

    for dset in f["entry"]["data"]:
        image_index = _processDataSet(
            f["entry"]["data"][dset], image_index, array_3D, scan_width
        )

    if f.__bool__():
        f.close()

    scan_height = int(nimages / scan_width)

    datacube = DataCube(
        array_3D.reshape(
            scan_width, scan_height, array_3D.data.shape[1], array_3D.data.shape[2]
        )
    )

    return datacube


def _processDataSet(dset, start_index, array_3D, scan_width):
    image_index = start_index
    nimages_dset = dset.shape[0]

    for i in range(nimages_dset):
        array_3D[image_index] = dset[i].astype(array_3D.dtype)

        image_index = image_index + 1

    return image_index
