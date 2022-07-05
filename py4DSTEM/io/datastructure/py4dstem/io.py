# Functions for reading and writing subclasses of the base EMD types

import numpy as np
import h5py

from ..emd.io import Array_from_h5, Metadata_from_h5





# Calibration

# read
def Calibration_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Metadata representation, and
    if so loads and returns it as a Calibration instance. Otherwise,
    raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A Calibration instance
    """
    cal = Metadata_from_h5(group)
    cal = Calibration_from_Metadata(cal)
    return cal

def Calibration_from_Metadata(metadata):
    """
    Converts a Metadata instance to a Calibration instance.

    Accepts:
        metadata (Metadata)

    Returns:
        (Calibration)
    """
    from .calibration import Calibration
    p = metadata._params
    metadata.__class__ = Calibration
    metadata.__init__(
        name = metadata.name
    )
    metadata._params.update(p)

    return metadata






# DataCube

# read

def DataCube_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read
    mode.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't exist, or if
    it exists but does not have a rank of 4, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DataCube instance
    """
    datacube = Array_from_h5(group)
    datacube = DataCube_from_Array(datacube)
    return datacube

def DataCube_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    from .datacube import DataCube
    assert(array.rank == 4), "Array must have 4 dimensions"
    array.__class__ = DataCube
    array.__init__(
        data = array.data,
        name = array.name,
        R_pixel_size = [array.dims[0][1]-array.dims[0][0],
                        array.dims[1][1]-array.dims[1][0]],
        R_pixel_units = [array.dim_units[0],
                         array.dim_units[1]],
        Q_pixel_size = [array.dims[2][1]-array.dims[2][0],
                        array.dims[3][1]-array.dims[3][0]],
        Q_pixel_units = [array.dim_units[2],
                         array.dim_units[3]],
        slicelabels = array.slicelabels
    )
    return array





# DiffractionSlice

# read

def DiffractionSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a DiffractionSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DiffractionSlice instance
    """
    diffractionslice = Array_from_h5(group)
    diffractionslice = DiffractionSlice_from_Array(diffractionslice)
    return diffractionslice


def DiffractionSlice_from_Array(array):
    """
    Converts an Array to a DiffractionSlice.

    Accepts:
        array (Array)

    Returns:
        (DiffractionSlice)
    """
    from .diffractionslice import DiffractionSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = DiffractionSlice
    array.__init__(
        data = array.data,
        name = array.name,
        slicelabels = array.slicelabels
    )
    return array




# RealSlice

# Reading

def RealSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a RealSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A RealSlice instance
    """
    realslice = Array_from_h5(group)
    realslice = RealSlice_from_Array(realslice)
    return realslice


def RealSlice_from_Array(array):
    """
    Converts an Array to a RealSlice.

    Accepts:
        array (Array)

    Returns:
        (RealSlice)
    """
    from .realslice import RealSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = RealSlice
    array.__init__(
        data = array.data,
        name = array.name,
        slicelabels = array.slicelabels
    )
    return array



# DiffractionImage

# read

def DiffractionImage_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a DiffractionImage. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DiffractionImage instance
    """
    diffractionimage = Array_from_h5(group)
    diffractionimage = DiffractionImage_from_Array(diffractionimage)
    return diffractionimage


def DiffractionImage_from_Array(array):
    """
    Converts an Array to a DiffractionImage.

    Accepts:
        array (Array)

    Returns:
        (DiffractionImage)
    """
    from .diffractionimage import DiffractionImage
    assert(array.rank == 2), "Array must have 2 dimensions"

    # get diffraction image metadata
    try:
        md = array.metadata['diffractionimage']
        mode = md['mode']
        geo = md['geometry']
        shift_corr = md['shift_corr']
    except KeyError:
        er = "DiffractionImage metadata could not be found"
        raise Exception(er)


    # instantiate as a DiffractionImage
    array.__class__ = DiffractionImage
    array.__init__(
        data = array.data,
        name = array.name,
        mode = mode,
        geometry = geo,
        shift_corr = shift_corr
    )
    return array




# VirtualImage

# read

def VirtualImage_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a VirtualImage. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A VirtualImage instance
    """
    image = Array_from_h5(group)
    image = VirtualImage_from_Array(image)
    return image


def VirtualImage_from_Array(array):
    """
    Converts an Array to a VirtualImage.

    Accepts:
        array (Array)

    Returns:
        (VirtualImage)
    """
    from .virtualimage import VirtualImage
    assert(array.rank == 2), "Array must have 2 dimensions"

    # get diffraction image metadata
    try:
        md = array.metadata['virtualimage']
        mode = md['mode']
        geo = md['geometry']
        shift_corr = md['shift_corr']
        eager_compute = md['eager_compute']
    except KeyError:
        er = "VirtualImage metadata could not be found"
        raise Exception(er)


    # instantiate as a DiffractionImage
    array.__class__ = VirtualImage
    array.__init__(
        data = array.data,
        name = array.name,
        mode = mode,
        geometry = geo,
        shift_corr = shift_corr,
        eager_compute = eager_compute
    )
    return array



# Probe

# read

def Probe_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a Probe. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A Probe instance
    """
    probe = Array_from_h5(group)
    probe = Probe_from_Array(probe)
    return probe


def Probe_from_Array(array):
    """
    Converts an Array to a Probe.

    Accepts:
        array (Array)

    Returns:
        (Probe)
    """
    from .probe import Probe
    assert(array.rank == 2), "Array must have 2 dimensions"

    # get diffraction image metadata
    try:
        md = array.metadata['probe']
        kwargs = {}
        for k in md.keys:
            v = md[k]
            kwargs[k] = v
    except KeyError:
        er = "Probe metadata could not be found"
        raise Exception(er)


    # instantiate as a DiffractionImage
    array.__class__ = Probe
    array.__init__(
        data = array.data,
        name = array.name,
        **kwargs
    )
    return array





