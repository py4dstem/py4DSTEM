# `py4DSTEM.io`

This module supports read/write functionality.  Below, we discuss

1. Functions - for your basic read/write needs
2. Classes - class objects for wrapping, reading, and writing different kinds of data
3. Trees - nesting object instances, in-program and in your HDF5
4. Metadata - three categories of metadata, and their handlings, are specified
5. Calibrations - are very special Metadata!
6. Customizing classes




## I/O Functions

The four top level I/O functions are briefly described here. For more details, see the function call signatures and docstrings.

- `import_file`: reads non-native file formats, either loading to disc or creating a memory map.

- `read`: loads data from a file written by this program. The native HDF5-based format consists of nested trees, with the possibility of data, metadata, and further child nodes (i.e. lower level HDF5 groups) at each node. `read` allows reading the data at a single node, the full tree underneath a node including the root node, and the full tree underneath a node while excluding the root node's data.  Object / HDF5 trees, and handling of metadata and calibration data, are discussed in the relevant sections below.

- `save`: saves data from a running instance of py4DSTEM into an HDF5 file. Data can be saved to a new file or appended to an existing file.  A single object instance can be saved, or a whole object tree can be saved.

- `print_h5_tree`: prints the group tree of a native HDF5 file to standard output




## Classes

This module includes a set of Python classes, each wrapping different kinds of data, each with its own metadata, each with `.to_h5` and `.from_h5` methods for moving between HDF5 groups and python instances.

The basic data and metadata classes are:
- `Metadata`: a wrapper for dictionary-like data (i.e. string keys are used to access small blocks of data)
- `Array`: a wrapper for array-like data (e.g. a numpy array, memmap, h5py Dataset) 
- `PointList`: a wrapper for numpy structured arrays (i.e. N points in an M-dimensional space, where the M dimensions are string named, indexable fields)
- `PointListArray`: a 2D grid of PointLists

Derivative classes are discussed in `Customizing classes`.
The two classes (Node, Tree) which enable nesting data for ease of reading/writing whole sets of many data types at a time, and are discussed in the `Trees` section, below.




## Trees

HDF5 files have a tree-like structure, with some collection of nodes nested underneath other nodes.
The class definitions here are designed such that an analogous graph relating data object instances can be created, often under-the-hood and with no user input, as new data is generated.
For instance, if a dataset is loaded, and then some processing is performed on the dataset to generate an image, a tree graph representing the original dataset as the root node and the image as a child node can be created in the background.
At the end of some processing pipeline, the tree can then be used to easily save all of the data created at once.
Similarly, when reading a native HDF5 file, it is possible to read a single node, or an entire tree structure of data object instances.
This is implemented in the Node and Tree classes.




## Metadata

TODO




## Calibrations

TODO




## Customizing Classes

TODO




















# The old README:

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
classes - Tree and Node - which exist to facilitate building nexted trees of
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
        VirtualDiffraction          DiffractionSlice
        VirtualImage                RealSlice
        StainMap                    RealSlice

Type III classes are "composition oveer inheritance" classes - i.e. they contain
one or more of the other class types as attributes. The (class, attribute) pairs
are:

        BraggPeaks                  (two PointlistArray instances)



### Adding a new class

To add a new py4dstem class:

- create a file py4DSTEM/io/datastructure/py4dstem/yourclass.py. Add your
    class there.
- if the class needs read/write functionality not already inherited from a
    parent class, add YouClass_to_h5 and/or YourClass_from_h5 functions to
    py4DSTEM/io/datastructure/py4dstem/io.py, then add .to_h5 and/or .from_h5
    methods to your class definition, importing and running the functions you
    just added to datastructure/py4dstem/io.py.
- modify py4DSTEM/io/native/read.py, adding YourClass to the class imports
    at the top of the file, then adding it to the dictionary in _get_class
    at the end of the file
- add your class to the py4DSTEM/io/datastructure/py4dstem/__init__.py file

Your class should now be read/writable with py4DSTEM.  To add your class
automoatically to a running filetree under a DataCube object using built-in
DataCube methods:

- go to py4DSTEM/io/datastructure/py4dstem/datacube.py, and find the list of
    function imports from datacube_fns at the beginning of the class definition.
    Add the name of your new datacube method here.
- open py4DSTEM/io/datastructure/py4dstem/datacube_fns.py, and add your function.
    Please note that this function should be a wrapper around functional code
    from the py4DSTEM.process module - no new computational code should go here.
    New datacube_fns.py functions should import a function from process inside
    of the new function definition rather than at the top of the file to avoid
    circular import errors, run the function, wrap the output into an instance
    of the new class, add the instance to the datacube's tree, and optionally
    return the answer if `returncalc` is True.
    
    





