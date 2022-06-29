# `py4DSTEM.io.datastructure`

There are two abstraction levels of datastructures in py4DSTEM: emd and py4dstem
objects.


The emd classes are -

        0   Metadata
        1   Array
        2   PointList
        3   PointListArray

Each of this wraps a block of data with a .h5 read and write method.  Metadata
wraps non-nested dictionaries of string keys to strings, numbers, and small
arrays. Array wraps numpy array like syntax for a block of data which can be a
numpy ndarray, memmap, or h5py Dataset, with implicit EMD V1 style dim vectors
stored and accessible as numpy arrays. PointList wraps numpy structured arrays,
i.e. it supports key-like access to vectors along a single dimension.
PointListArray wraps 2D grids of PointLists.


We identify 3 types of py4dstem class types.

Type I classes inherit directly from an emd class.  The class, parent pairs are:

        Calibration                 Metadata
        DataCube                    Array
        DiffractionSlice            Array
        RealSlice                   Array

Type II classes inherit from type I classes.  The class, parent pairs are:

        Probe                       DiffractionSlice
        DiffractionImage            DiffractionSlice
        VirtualImage                RealSlice
        StainMap                    RealSlice

Type III classes are "composition, not inheritance" classes, as they say. The
class, attribute pairs are:

        BraggPeaks                  (two PointlistArray instances)







