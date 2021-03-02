# py4DSTEM sample code and tutorials

This subdirectory contain sample code providing both an introduction to the py4DSTEM package and various examples of its use.

The files here include:
- `quickstart.ipynb`
- `io_tutorial.ipynb`
- `classification_twinBoundary.ipynb`

----

## `quickstart.ipynb`

This notebook provides a quick intro to using py4DSTEM to write code to analyze 4D-STEM data.  It demonstrates loading data, performing some initial visualizations such as virtual imaging and displaying diffraction data, detecting and displaying Bragg disk positions, and saving output.


## `io_tutorial.ipynb`

This notebook provides more in-depth information about the read/write functionality of py4DSTEM, including reading non-native filetypes, and detailed handling of the native HDF5 format,m including browsing and loading data, saving new files, appending new data to existing files, copying files, removing or overwriting data from a file, metadata handling, and packaging multiple data trees within a single py4DSTEM .h5 files.


## `classification_twinBoundary.ipynb`

This notebook demonstrates classification of a 4DSTEM dataset containing a twin boundary.  After detecting the Bragg disks, the disk positions are used to identify scan positions corresponding to the two sides of the twin.




