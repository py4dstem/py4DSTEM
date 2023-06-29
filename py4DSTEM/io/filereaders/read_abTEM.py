import h5py
from py4DSTEM.classes import DataCube, VirtualDiffraction, VirtualImage


def read_abTEM(
    filename,
    mem="RAM",
    binfactor: int = 1,
):
    """
    File reader for abTEM datasets
    Args:
        filename: str with path to  file
        mem (str):  Must be "RAM" or "MEMMAP". Specifies how the data is
            loaded; "RAM" transfer the data from storage to RAM, while "MEMMAP"
            leaves the data in storage and creates a memory map which points to
            the diffraction patterns, allowing them to be retrieved individually
            from storage.
        binfactor (int): Diffraction space binning factor for bin-on-load.

    Returns:
        DataCube
    """
    assert mem == "RAM", "read_abTEM does not support memory mapping"
    assert binfactor == 1, "abTEM files can only be read at full resolution"

    with h5py.File(filename, "r") as f:
        datasets = {}
        for key in f.keys():
            datasets[key] = f.get(key)[()]

    data = datasets["array"]
    sampling = datasets["sampling"]
    units = datasets["units"]

    if len(data.shape) == 4:

        datacube = DataCube(data=data)

        datacube.calibration.set_R_pixel_size(sampling[0])
        if sampling[0] != sampling[1]:
            print(
                "Warning: py4DSTEM currently only handles uniform x,y sampling. Setting sampling with x calibration"
            )
        datacube.calibration.set_Q_pixel_size(sampling[2])
        if sampling[2] != sampling[3]:
            print(
                "Warning: py4DSTEM currently only handles uniform qx,qy sampling. Setting sampling with qx calibration"
            )

        if units[0] == b"\xc3\x85":
            datacube.calibration.set_R_pixel_units("A")
        else:
            datacube.calibration.set_R_pixel_units(units[0].decode("utf-8"))

        datacube.calibration.set_Q_pixel_units(units[2].decode("utf-8"))

        return datacube

    elif len(data.shape) == 2:
        if units[0] == b"mrad":
            diffraction = VirtualDiffraction(data=data)
            if sampling[0] != sampling[1]:
                print(
                    "Warning: py4DSTEM currently only handles uniform qx,qy sampling. Setting sampling with x calibration"
                )
            diffraction.calibration.set_Q_pixel_units(units[0].decode("utf-8"))
            diffraction.calibration.set_Q_pixel_size(sampling[0])
            return diffraction
        elif units[0] == b"\xc3\x85":
            image = VirtualImage(data=data)
            if sampling[0] != sampling[1]:
                print(
                    "Warning: py4DSTEM currently only handles uniform x,y sampling. Setting sampling with x calibration"
                )
            image.calibration.set_Q_pixel_units("A")
            image.calibration.set_Q_pixel_size(sampling[0])
            return image
