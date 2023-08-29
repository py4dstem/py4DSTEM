# Functions for reading and writing subclasses of the base EMD types

import numpy as np
import h5py
from os.path import basename

from py4DSTEM.io.legacy.legacy13.v13_emd_classes.io import (
    Array_from_h5,
    Metadata_from_h5,
)
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.io import PointList_from_h5
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.io import (
    PointListArray_from_h5,
    PointListArray_to_h5,
)
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.io import (
    _write_metadata,
    _read_metadata,
)


# Calibration


# read
def Calibration_from_h5(group: h5py.Group):
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
    Constructs a Calibration object with the dict entries of a Metadata object
    Accepts:
        metadata (Metadata)
    Returns:
        (Calibration)
    """
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.calibration import Calibration

    cal = Calibration(name=metadata.name)
    cal._params.update(metadata._params)

    return cal


# DataCube

# read


def DataCube_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.datacube import DataCube

    assert array.rank == 4, "Array must have 4 dimensions"
    array.__class__ = DataCube
    try:
        R_pixel_size = array.dims[0][1] - array.dims[0][0]
    except IndexError:
        R_pixel_size = 1
    try:
        Q_pixel_size = array.dims[2][1] - array.dims[2][0]
    except IndexError:
        Q_pixel_size = 1
    array.__init__(
        data=array.data,
        name=array.name,
        R_pixel_size=R_pixel_size,
        R_pixel_units=array.dim_units[0],
        Q_pixel_size=Q_pixel_size,
        Q_pixel_units=array.dim_units[2],
        slicelabels=array.slicelabels,
    )
    return array


# DiffractionSlice

# read


def DiffractionSlice_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.diffractionslice import (
        DiffractionSlice,
    )

    assert array.rank == 2, "Array must have 2 dimensions"
    array.__class__ = DiffractionSlice
    array.__init__(data=array.data, name=array.name, slicelabels=array.slicelabels)
    return array


# RealSlice

# read


def RealSlice_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.realslice import RealSlice

    assert array.rank == 2, "Array must have 2 dimensions"
    array.__class__ = RealSlice
    array.__init__(data=array.data, name=array.name, slicelabels=array.slicelabels)
    return array


# VirtualDiffraction

# read


def VirtualDiffraction_from_h5(group: h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a VirtualDiffraction. Otherwise, raises an exception.
    Accepts:
        group (HDF5 group)
    Returns:
        A VirtualDiffraction instance
    """
    virtualdiffraction = Array_from_h5(group)
    virtualdiffraction = VirtualDiffraction_from_Array(virtualdiffraction)
    return virtualdiffraction


def VirtualDiffraction_from_Array(array):
    """
    Converts an Array to a VirtualDiffraction.
    Accepts:
        array (Array)
    Returns:
        (VirtualDiffraction)
    """
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.virtualdiffraction import (
        VirtualDiffraction,
    )

    assert array.rank == 2, "Array must have 2 dimensions"

    # get diffraction image metadata
    try:
        md = array.metadata["virtualdiffraction"]
        method = md["method"]
        mode = md["mode"]
        geometry = md["geometry"]
        shift_center = md["shift_center"]
    except KeyError:
        print("Warning: VirtualDiffraction metadata could not be found")
        method = ""
        mode = ""
        geometry = ""
        shift_center = ""

    # instantiate as a DiffractionImage
    array.__class__ = VirtualDiffraction
    array.__init__(
        data=array.data,
        name=array.name,
        method=method,
        mode=mode,
        geometry=geometry,
        shift_center=shift_center,
    )
    return array


# VirtualImage

# read


def VirtualImage_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.virtualimage import (
        VirtualImage,
    )

    assert array.rank == 2, "Array must have 2 dimensions"

    # get diffraction image metadata
    try:
        md = array.metadata["virtualimage"]
        mode = md["mode"]
        geo = md["geometry"]
        centered = md._params.get("centered", None)
        calibrated = md._params.get("calibrated", None)
        shift_center = md._params.get("shift_center", None)
        dask = md._params.get("dask", None)
    except KeyError:
        er = "VirtualImage metadata could not be found"
        raise Exception(er)

    # instantiate as a DiffractionImage
    array.__class__ = VirtualImage
    array.__init__(
        data=array.data,
        name=array.name,
        mode=mode,
        geometry=geo,
        centered=centered,
        calibrated=calibrated,
        shift_center=shift_center,
        dask=dask,
    )
    return array


# Probe

# read


def Probe_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.probe import Probe

    assert array.rank == 2, "Array must have 2 dimensions"
    # get diffraction image metadata
    try:
        md = array.metadata["probe"]
        kwargs = {}
        for k in md.keys:
            v = md[k]
            kwargs[k] = v
    except KeyError:
        er = "Probe metadata could not be found"
        raise Exception(er)

    # instantiate as a DiffractionImage
    array.__class__ = Probe
    array.__init__(data=array.data, name=array.name, **kwargs)
    return array


# QPoints

# Reading


def QPoints_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.qpoints import QPoints

    pointlist.__class__ = QPoints
    pointlist.__init__(
        data=pointlist.data,
        name=pointlist.name,
    )
    return pointlist


# BraggVectors


# write
def BraggVectors_to_h5(braggvectors, group):
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
    grp.attrs.create("emd_group_type", 4)  # this tag indicates a Custom type
    grp.attrs.create("py4dstem_class", braggvectors.__class__.__name__)

    # Ensure that the PointListArrays have the appropriate names
    braggvectors._v_uncal.name = "_v_uncal"

    # Add vectors
    PointListArray_to_h5(braggvectors._v_uncal, grp)
    try:
        braggvectors._v_cal.name = "_v_cal"
        PointListArray_to_h5(braggvectors._v_cal, grp)
    except AttributeError:
        pass

    # Add metadata
    _write_metadata(braggvectors, grp)


# read
def BraggVectors_from_h5(group: h5py.Group):
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
    from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.braggvectors import (
        BraggVectors,
    )

    er = f"Group {group} is not a valid BraggVectors group"
    assert "emd_group_type" in group.attrs.keys(), er
    assert group.attrs["emd_group_type"] == 4, er

    # Get uncalibrated peak
    v_uncal = PointListArray_from_h5(group["_v_uncal"])

    # Get Qshape metadata
    try:
        grp_metadata = group["_metadata"]
        Qshape = Metadata_from_h5(grp_metadata["braggvectors"])["Qshape"]
    except KeyError:
        raise Exception("could not read Qshape")

    # Set up BraggVectors
    braggvectors = BraggVectors(v_uncal.shape, Qshape=Qshape, name=basename(group.name))
    braggvectors._v_uncal = v_uncal

    # Add calibrated peaks, if they're there
    try:
        v_cal = PointListArray_from_h5(group["_v_cal"])
        braggvectors._v_cal = v_cal
    except KeyError:
        pass

    # Add remaining metadata
    _read_metadata(braggvectors, group)

    return braggvectors
