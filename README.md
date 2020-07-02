# py4DSTEM: open source processing and analysis of 4D-STEM data
[![DOI](https://zenodo.org/badge/148587083.svg)](https://zenodo.org/badge/latestdoi/148587083)

py4DSTEM is a python tool for analysis of four-dimensional scanning transimission electron microscopy (4D-STEM) data.
It is open source software distributed under a GPLv3 license. It is free to use, alter, or build on, provided that any work derived from py4DSTEM is also kept free and open.



## What is 4D-STEM?

In a traditional STEM experiment, a beam of high energy electrons is focused to a very fine probe - on the order of or, often, smaller than the atomic lattice spacings - and rastered across the surface of the sample.
A two-dimensional image is then formed by populating the value of each pixel with the electron flux through a detector at the corresponding beam position.
In 4D-STEM a pixellated detector is used, such that a 2D image of the diffraction plane is acquired at every raster position of the electron beam.
A 4D-STEM scan thus results in a 4D data array.


4D-STEM data is information rich.
A datacube can be collapsed in real space to yield information comparable to nanobeam electron diffraction experiment, or in diffraction space to yield a variety of virtual images, corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to calculate the sample potential.
Structure in the diffracted halos of amorphous systems can be used to describe the short and medium range order.


For more information, see [https://arxiv.org/abs/2003.09523](https://arxiv.org/abs/2003.09523).



## Using py4DSTEM

This sections describes:
- Installation
- Running the GUI
- Running from a python interpretter
- Sample jupter notebooks
- Sample scripts
- For developers



### Installation

The recommended installation for py4DSTEM uses the Anaconda python distribution.
First, download and install Anaconda.  Instructions can be found at www.anaconda.com/download.
Then open a terminal and run

```
conda updata conda
conda create -n py4dstem
conda activate py4dstem
conda install pip
pip install py4dstem
```

In order, these commands
- ensure your installation of anaconda is up-to-date
- make a virtual environment - see below!
- enter the environment
- make sure your new environment talks nicely to pip, a tool for installing Python packages
- use pip to install py4DSTEM

Virtual environments are used to make sure packages that have different dependencies don't conflict with one another.
Because the directions above install py4DSTEM to its own virtual environment, each time you want to use py4DSTEM,
you'll need to activate this environment.
This is included in the directions below for running py4DSTEM, and assume you've named your virtual environment 'py4dstem'


### Running the GUI

From a terminal, run
```
conda activate py4dstem
py4dstem
```

### Running from a python interpreter

From any python interpreter inside the `py4dstem` conda environment, py4DSTEM can be imported in the usual way:

```
import py4DSTEM
```


### Sample code 

**As the base code is currently under construction, the sample code (Jupyter notebooks and scripts) have been temporarily taken down.  They'll be back soon.  We apologize for any inconvenience, and appreciate your patience!**



### For contributors

To contribute to py4DSTEM, first fork the repository.

Set up and activate a new environment, e.g. in anaconda use
```
conda create -n py4dstem_dev
conda activate py4dstem_dev
```
Next, navigate to the directory where you want to put the project, and type
```
git clone your_fork_url.git
```
where `your_fork_url` is the url to the fork you created. 
Navigate into the new py4DSTEM directory which contains the `setup.py` file, and run
```
conda install pip
pip install -e .
```
where the -e option installs the package in 'editable' mode, so that changes within this project directory will be reflected in py4DSTEM imports from Python interpreters within this environment.

Create a new branch, and make any edits.
To merge back in to the main repository, submit a pull request to the dev branch of github.com/py4dstem/py4DSTEM.





## More information

### Dependencies

* numpy
* scipy
* h5py
* ncempy
* pymatgen
* numba
* scikit-image
* scikit-learn
* PyQt5
* pyqtgraph
* qtconsole
* ipywidgets
* tqdm

### Optional dependencies

* ipyparallel
* dask


### Versioning

v. 0.9.8



### License

GNU GPLv3



### Acknowledgements

The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and development time which made this project possible.

