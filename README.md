# py4DSTEM: open source processing and analysis of 4D-STEM data
[![DOI](https://zenodo.org/badge/148587083.svg)](https://zenodo.org/badge/latestdoi/148587083)

The aim of py4DSTEM is to provide a comprehensive suite of tools for interacting with, visualizing, and analyzing 4DSTEM data.
Intended applications include generating virtual images; classification and segmentation; mapping orientation, crystallinity, and strain fields; and other analytical tools including ptychography and fluctuation electron microscopy.

py4DSTEM can be used at three distinct levels of interaction.
A graphical user interface facilites rapid data exploration, and live testing and tuning of analysis parameters.
For analysis requiring greater user control, py4DSTEM can be run and interfaced directly from the command line using a python 3 interpreter.
For large scale projects, py4DSTEM enables automated batch processing of many 4DSTEM datasets.

py4DSTEM is open source, copyleft software.
It is free to use, alter, or build on, provided that any work derived from py4DSTEM is also kept free and open.

## Quick overview

4DSTEM is a powerful, versatile, emerging technique in the field nanocharacterization.
This section provides a brief description of what 4D-STEM is, some of the challenges associated with 4D-STEM, and how py4DSTEM helps address these challenges.

### What is 4D-STEM?

In a Scanning Transmission Electron Microscopy Experiment (STEM), a beam of high energy electrons is focused to a very fine probe - on the order of or, often, smaller than the atomic lattice spacings - and rastered across the surface of the sample.
In traditional STEM, a (two dimensional) image is formed by populating the value of each pixel by the number of electrons (times a scaling factor) scattered into a detector at each beam position.
The geometry of the detector - it's size, shape, and position in the microscope's diffraction plane - determines which electrons are collected.
As a result, different detector geometries can give rise to rather different images, by varying which electron scattering processes dominate image contrast.
For instance, high-angle annular dark-field detectors collect only electrons scattered to high angles, and are popular because image contrast then generally scales monotonically with the projected potential of the sample.

4D-STEM stands for 4-Dimensional Scanning Transmission Electron Microscopy.
In 4D-STEM, the standard STEM detectors, which integrate all electrons scattered over a large region, are replaced with a pixellated detector, which instead detects the electron flux scattered to each angle in the diffraction plane.
While a typical STEM image therefore produces a single number for each position of the electron beam, a 4D-STEM dataset produces a two dimensional map of diffraction space intensities for each beam position.
The resulting four dimensional data hypercube can be collapsed in real space to yield information comparable to nanobeam electron diffraction experiment.
Alternatively, it can be collapsed in diffraction space to yield a variety of `virtual images', corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.

More information still can be extracted by coherently combining the real and reciprocal space pictures.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to deconvolve the electron beam shape from the sample structure, yielding the sample potential itself.
Variance in the data intensity can be used to extract correlation functions describing the short and medium range order and disorder.


### What are some of the challenges of analysis of 4D-STEM data?

In terms of hardware, 4D-STEM has been made possible by the advent of electron detectors with the speed and dynamic range necessary to capture complete diffraction patterns at each scan position at collection speeds fast enough that sample drift is not prohibitive.
In terms of data analysis, 4D-STEM is where the field of STEM butts heads with the big data problem.
A typical 4D-STEM scan can generate a gigabyte of data in under a minute, where the specific data rate depends on the detector - and will continue to increase with new hardware developements.
The capacity to handle terabytes of raw data from a single session is required.

The size and complexity of 4D-STEM data makes the initial stages of data screening and preprocessing both more challenging and more important.
The ability to quickly scan through data becomes non-trivial, as many possible 2D slices through a given 4D datacube are possible, and which are most relavant will vary on a case-by-case basis.
For these large datasets, compression without sacrificing useful information takes on increasing importance.

Analysis of 4D-STEM data can involve significant amounts of data processing.
A growing number of excellent studies devoted to untangling these complex datasets in the most useful ways can be found in the literature, demonstrating 4D-STEM based mapping of everything from strain to local magnetic fields to non-spectroscopic composition maps and much more.
At this stage, these works tend to operate on the scale of individual datasets, demonstrating the principles and requisite machinery for new forms of 4D-STEM data analysis.
However, maximizing the impact of these tools to answer the broadest possible array of scientifc questions requires accessibility, scalability, and reproducibility.
In light of the size of the data and the variety and complexity of approaches to its analyses, these represent significant challenges.


### How does py4DSTEM help?

py4DSTEM is here to help!




## Getting started

### Installation

py4DSTEM uses python 3, and the Anaconda python distribution is highly recommended.
Download and installation instructions for Anaconda can be found at www.anaconda.com/download.

Next, ensure the dependencies (see below) are installed in a python 3 environment.  In anaconda, you can use:

```
conda install hyperspy -c conda-forge
conda install h5py
conda install pyqtgraph
conda install PyQt5
pip install ncempy
```

Note that as ncempy is installed with pip rather than conda, anaconda users should make sure that pip is pointing to their anaconda installation (rather than some other python version).  You can check by calling `which pip`, and confirming that the output is a path somewhere inside your anaconda directory.

To get the py4DSTEM package, use the 'Clone or Download' link on this page to copy the py4DSTEM repository somewhere on your system.  From the command line, you can navigate the the directory where you'd like to put py4DSTEM, and run:

```
git clone https://github.com/bsavitzky/py4DSTEM.git
```

Finally, navigate to the py4DSTEM root directory and run the setup.py script by calling:

```
pip install .
```

That's it!

To install in "editable" mode, so changes you make to the source code are available without reinstalling, run:
```python
pip install -e .
```
(You do need to restart your IPython kernel for changes to take effect there.)

### Running py4DSTEM with the GUI

Ensure you are in the conda environment you installed in, and run:

```
py4dstem
```

### Running py4DSTEM from an python interpreter

In a python interpreter, py4DSTEM can now be imported in the usual way:

```
import py4DSTEM
```


### Dependencies

* hyperspy
* h5py
* pyqtgraph
* PyQt5


## Versioning


v. 0.6


## License

GNU GPLv3


## Acknowledgements

The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and developement time which made this project possible.

