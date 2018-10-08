# py4DSTEM: open source processing and analysis of 4D-STEM data

The aim of py4DSTEM is to provide a comprehensive suite of tools for interacting with, visualizing, and analyzing 4DSTEM data.
Intended applications include generating virtual images; classification and segmentation; mapping orientation, crystallinity, and strain fields; and other analytical tools including ptychography and fluctuation electron microscopy.

py4DSTEM is designed to be used at three possible levels of interaction.
A graphical user interface facilites rapid data exploration, and live testing and tuning of analysis parameters.
For analysis requiring greater user control, py4DSTEM can be run and interfaced directly from the command line using a python 3 interpreter.
For large scale projects, py4DSTEM enables automated batch processing of many 4DSTEM datasets.

py4DSTEM is open source, ``copy left'' software.


## Quick overview

4DSTEM is a powerful, versatile, emerging technique in the field nanocharacterization.

### What is 4D-STEM?

In a Scanning Transmission Electron Microscopy Experiment (STEM), a beam of high energy electrons is focused to a very fine probe -- on the order of or smaller than the atomic lattice spacings -- and rastered across the surface of the sample.
In traditional STEM, a (two dimensional) image is formed by populating the value of each pixel by the number of electrons (times a scaling factor) scattered into a detector at each beam position.
The geometry of the detector -- it's size, shape, and position in the microscope's diffraction plane -- determines which electrons are collected, and therefore what the primary image contrast mechanisms will be.
For instance, high-angle annular dark-field detectors collect only electrons scattered to high angles, and are popular because with these detectors image contrast generally scales monotonically with the projected potential of the sample.

4D-STEM stands for 4-Dimensional Scanning Transmission Electron Microscopy.
In 4D-STEM, the standard STEM detectors, which integrate all electrons scattered over a large region, are replaced with a pixellated detector, which instead detects the electron flux scattered to each angle in the diffraction plane.
While a STEM image therefore produces a single number for each position of the electron beam, a 4D-STEM dataset produces a two dimensional map of diffraction space intensities for each beam position.
The resulting four dimensional data  hypercube be collapsed in real space to yield information comparable to a position averaged nanobeam electron diffraction pattern.
Alternatively, it can be collapsed in diffraction space to yield a variety of `virtual images', corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.

More information still can be extracted by coherently combining the real and reciprocal space pictures.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to deconvolve the electron beam shape from the sample structure, yielding the sample potential itself.
Variance in the data intensity can be used to extract correlation functions describing the short and medium range order and disorder.


### What are some of the challenges of analysis of 4D STEM data?

In terms of hardware, 4D-STEM has been made possible by the advent of electron detectors with the speed and dynamic range necessary to capture complete diffraction patterns at each scan position fast enough that sample drift is not prohibitive.
In terms of data analysis, 4D-STEM is where the field of STEM butts heads with the big data problem.

Since the turn of the millenium, the availability of aberration corrected instruments has made STEM an increasingly invaluable tool in direct interrogation of matter at the atomic scale.
//A key advantage of STEM, as in all imaging methods, is the preeminence of local, as opposed to average, structure -- today, analysis of defects, polarization fields, compositional gradients, and so on are all accessible at the scale of individual atoms.
Individual images are frequently so information-rich as to warrant detailed and individually-tailored analysis.



### How does py4DSTEM help?

## Getting started

Installing and running the code

### Dependencies

hyperspy


## Versioning

v. 0.1

## License

GNU GPLv3
