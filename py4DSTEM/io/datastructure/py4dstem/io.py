# Functions for reading and writing subclasses of the base EMD types

import numpy as np
import h5py
from os.path import basename

from ..emd.io import Array_from_h5, Metadata_from_h5
from ..emd.io import PointList_from_h5
from ..emd.io import PointListArray_from_h5, PointListArray_to_h5
from ..emd.io import _write_metadata, _read_metadata



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
        R_pixel_size = array.dims[0][1]-array.dims[0][0],
        R_pixel_units = array.dim_units[0],
        Q_pixel_size = array.dims[2][1]-array.dims[2][0],
        Q_pixel_units = array.dim_units[2],
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



# QPoints

# Reading

def QPoints_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid QPoints instance, and if so
    loads and returns it. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A QPoints instance
    """
    qpoints = PointList_from_h5(group)
    qpoints = QPoints_from_PointList(qpoints)
    return qpoints


def QPoints_from_PointList(pointlist):
    """
    Converts an PointList to QPoints.

    Accepts:
        pointlist (PointList)

    Returns:
        (QPoints)
    """
    from .qpoints import QPoints
    pointlist.__class__ = QPoints
    pointlist.__init__(
        data = pointlist.data,
        name = pointlist.name,
    )
    return pointlist





# BraggVectors


# write
def BraggVectors_to_h5(
    braggvectors,
    group
    ):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    BraggVectors .name field nested inside the passed group, and saves
    the data there.

    Accepts:
        group (HDF5 group)
    """

    ## Write
    grp = group.create_group(braggvectors.name)
    grp.attrs.create("emd_group_type", 4) # this tag indicates a Custom type
    grp.attrs.create("py4dstem_class", braggvectors.__class__.__name__)

    # Ensure that the PointListArrays have the appropriate names
    braggvectors._v_uncal.name = "_v_uncal"

    # Add vectors
    PointListArray_to_h5(
        braggvectors._v_uncal,
        grp
    )
    try:
        braggvectors._v_cal.name = "_v_cal"
        PointListArray_to_h5(
            braggvectors._v_cal,
            grp
        )
    except AttributeError:
        pass

    # Add metadata
    _write_metadata(braggvectors, grp)


# read
def BraggVectors_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid BraggVectors object of this name exists inside
    this group, and if it does, loads and returns it. If it doesn't, raises
    an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A BraggVectors instance
    """
    from .braggvectors import BraggVectors

    er = f"Group {group} is not a valid BraggVectors group"
    assert("emd_group_type" in group.attrs.keys()), er
    assert(group.attrs["emd_group_type"] == 4), er


    # Get uncalibrated peak
    v_uncal = PointListArray_from_h5(group['_v_uncal'])

    # Get Qshape metadata
    try:
        grp_metadata = group['_metadata']
        Qshape = Metadata_from_h5(grp_metadata['braggvectors'])['Qshape']
    except KeyError:
        raise Exception("could not read Qshape")

    # Set up BraggVectors
    braggvectors = BraggVectors(
        v_uncal.shape,
        Qshape = Qshape,
        name = basename(group.name)
    )
    braggvectors._v_uncal = v_uncal

    # Add calibrated peaks, if they're there
    try:
        v_cal = PointListArray_from_h5(group['_v_cal'])
        braggvectors._v_cal = v_cal
    except KeyError:
        pass

    # Add remaining metadata
    _read_metadata(braggvectors, group)

    return braggvectors















