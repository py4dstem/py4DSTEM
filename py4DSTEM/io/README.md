# `py4DSTEM.io`



This module includes:

- `read.py`: the general py4DSTEM file reader, which parses file extentions (and possibly tags) to determine the filetype, then passes the file to the appropriate reader
- `nonnative/`: sub-module containing functions to read non-native file formats
- `native/`: sub-module for reading, writing, appending, and copying native .h5 files.  This module's build / reads functionality ultimately calls .to_h5 and .from_h5 methods of the datastructures defined in the `datastructure/` module.
- `datastructure/`: sub-module containing classes which hold and allow access to blocks of data at run-time, and native .to_h5 and .from_h5 methods enabling easy reading/writing compatibility with HDF5
- `google_drive_downloader.py`: reads sample files from google drive




Below, we first describe
- the structure of the  `datastructure` module and its classes
- how these are combined to make files in the `native` module
- how to extend the code by adding a new datastructure class




## `py4DSTEM.io.datastructure`

There are two abstraction levels of datastructures in py4DSTEM, each contained in it's own submodule: emd, and py4dstem.


The emd classes are -

        0   Metadata
        1   Array
        2   PointList
        3   PointListArray

Each of these wraps a block of data with a .h5 read and write method.  Metadata
wraps non-nested dictionaries of string keys to strings, numbers, tuples, lists,
small arrays, or None. Array wraps numpy array like syntax for a block of data
which can be a numpy ndarray, memmap, or h5py Dataset, with implicit EMD V1
style dim vectors stored and accessible as numpy arrays. PointList wraps numpy
structured arrays, i.e. it supports key-like access to vectors along a single
dimension. PointListArray wraps 2D grids of PointLists.  These classes all
support numpy-like slicing.  The Array, PointList, and PointListArray classes
all support easy storage/retrieval of arbitrary numbers of their own Metadata
instances. The Array class holds N+1 dimensional arrays, where N is the dimension
of the dimensions with labels, units, and dim-vectors attached, and the final,
optional dimension allows stacking an arbitrary number of these N-D arrays
which all share common dim-vectors, and can be sliced into with string labels.

In addition to the four primary EMD classes noted above, there are two special
classes - Tree and Root - which exist to facilitate building nexted trees of
objects, which can be written to and loaded from files as a single unit. More
on this below.



py4dstem classes build on emd classes.  They are more diverse and are meant to
hold blocks or multiple blocks of data corresponding to a logical unit of 4D-STEM
data analysis, rather than a general Python data structure.

We identify 3 types of py4dstem class types.  Type I classes inherit directly
from an emd class.  The (class, parent) pairs are:

        DataCube                    Array
        Calibration                 Metadata
        DiffractionSlice            Array
        RealSlice                   Array

Data

Type II classes inherit from type I classes.  The (class, parent) pairs are:

        Probe                       DiffractionSlice
        DiffractionImage            DiffractionSlice
        VirtualImage                RealSlice
        StainMap                    RealSlice

Type III classes are "composition oveer inheritance" classes - i.e. they contain
one or more of the other class types as attributes. The (class, attribute) pairs
are:

        BraggPeaks                  (two PointlistArray instances)






