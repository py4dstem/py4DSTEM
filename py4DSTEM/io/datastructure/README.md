# `py4DSTEM.io.datastructure` submodule

Starting in v13, we distinguish between the following types of datastructures,
each of which has a runtime representation (a Python class), and a persistent/storage
representation (an HDF5 specification):

DATA

I - Single data container classes, low level

    - Array-like data: EMD
    - Point-like data: PointList
    - Mixed point/array-like data, ragged arrays: PointListArray

II - Single data container classes, high level - inheriting from type I

    - DataCube (EMD)
    - RealSlice (EMD)
        - VirtualImage
        - StrainMap
        - DetectorImage
    - DiffractionSlice (EMD)
        - VirtualDiffraction
        - Probe
        - BVM
    - Lattice (PointList)

III - Composites (contain more than one of type I / II):

    - BraggPeaks
        - contains two PointListArrays, cal and uncal



METADATA

IV - Metadata (microscope and acquisition metadata)

V - Calibration (contains all calibration and coordinate system information)


