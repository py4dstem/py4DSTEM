
> :warning: **py4DSTEM version 0.14 update** :warning: Warning: this is a major update and we expect some workflows to break.  You can still install previous versions of py4DSTEM [as discussed here](#legacyinstall)

> :warning: **Phase retrieval refactor version 0.14.9** :warning: Warning: The phase-retrieval modules in py4DSTEM (DPC, parallax, and ptychography) underwent a major refactor in version 0.14.9 and as such older tutorial notebooks will not work as expected. Notably, class names have been pruned to remove the trailing "Reconstruction" (`DPCReconstruction` -> `DPC` etc.), and regularization functions have dropped the `_iter` suffix (and are instead specified as boolean flags). We are working on updating the tutorial notebooks to reflect these changes. In the meantime, there's some more information in the relevant pull request [here](https://github.com/py4dstem/py4DSTEM/pull/597#issuecomment-1890325568).

![py4DSTEM logo](/images/py4DSTEM_logo.png)

**py4DSTEM** is an open source set of python tools for processing and analysis of four-dimensional scanning transmission electron microscopy (4D-STEM) data.
Additional information:

- [Installation instructions](#install)
- [The py4DSTEM documentation pages](https://py4dstem.readthedocs.io/en/latest/index.html).
- [Tutorials and example code](https://github.com/py4dstem/py4DSTEM_tutorials)
- [Want to get involved?](#Contributing)
- [Our open access py4DSTEM publication in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927621000477) describing this project and demonstrating a variety of applications.
- [Our open access 4D-STEM review in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927619000497) describing this project and demonstrating a variety of applications.



# What is 4D-STEM?

In a traditional STEM experiment, a beam of high energy electrons is focused to a very fine probe - on the order of, or even smaller than, the spacing between atoms - and rastered across the surface of the sample. A conventional two-dimensional STEM image is formed by populating the value of each pixel with the electron flux through a detector at the corresponding beam position. In 4D-STEM, a pixelated detector is used instead, where a 2D image of the diffracted probe is recorded at every rastered probe position. A 4D-STEM scan thus results in a 4D data array.


4D-STEM data is information rich.
A datacube can be collapsed in real space to yield information comparable to nanobeam electron diffraction experiment, or in diffraction space to yield a variety of virtual images, corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to calculate the sample potential.
Structure in the diffracted halos of amorphous systems can be used to describe the short and medium range order.

**py4DSTEM** supports many different modes of 4DSTEM analysis.
The tutorials, sample code, module, and function documentation all provide more detailed discussion on some of the analytical methods possible with this code.



<a id='install'></a>
# py4DSTEM Installation

[![PyPI version](https://badge.fury.io/py/py4dstem.svg)](https://badge.fury.io/py/py4dstem)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/py4dstem/badges/version.svg)](https://anaconda.org/conda-forge/py4dstem)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/py4dstem/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/py4dstem)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/py4dstem/badges/platforms.svg)](https://anaconda.org/conda-forge/py4dstem)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/py4dstem/badges/downloads.svg)](https://anaconda.org/conda-forge/py4dstem)

The recommended installation for **py4DSTEM** uses the Anaconda python distribution.
First, download and install Anaconda: www.anaconda.com/download. 
If you prefer a more lightweight conda client, you can instead install Miniconda: https://docs.conda.io/en/latest/miniconda.html.
Then open a conda terminal and run one of the following sets of commands to ensure everything is up-to-date and create a new environment for your py4DSTEM installation:

```
conda update conda
conda create -n py4dstem
conda activate py4dstem
conda install -c conda-forge py4dstem pymatgen jupyterlab
```

In order, these commands
- ensure your installation of anaconda is up-to-date
- make a virtual environment (see below)
- enter the environment
- install py4DSTEM, as well as pymatgen (used for crystal structure calculations) and JupyterLab (an interface for running Python notebooks like those in the [py4DSTEM tutorials repository](https://github.com/py4dstem/py4DSTEM_tutorials))


We've had some recent reports install of `conda` getting stuck trying to solve the environment using the above installation.  If you run into this problem, you can install py4DSTEM using `pip` instead of `conda` by running:

```
conda update conda
conda create -n py4dstem python=3.10 
conda activate py4dstem
pip install py4dstem pymatgen 
```

Both `conda` and `pip` are programs which manage package installations, i.e. make sure different codes you're installing which depend on one another are using mutually compatible versions.  Each has advantages and disadvantages; `pip` is a little more bare-bones, and we've seen this install work when `conda` doesn't.  If you also want to use Jupyterlab you can then use either `pip install jupyterlab` or `conda install jupyterlab`.

If you would prefer to install only the base modules of **py4DSTEM**, and skip pymategen and Jupterlab, you can instead run:

```
conda install -c conda-forge py4dstem
```

Finally, regardless of which of the above approaches you used, in Windows you should then also run:

```
conda install pywin32
```

which enables Python to talk to the Windows API.

Please note that virtual environments are used in the instructions above in order to make sure packages that have different dependencies don't conflict with one another.
Because these directions install py4DSTEM to its own virtual environment, each time you want to use py4DSTEM you'll need to activate this environment.
You can do this in the command line by running `conda activate py4dstem`, or, if you're using the Anaconda Navigator, by clicking on the Environments tab and then clicking on `py4dstem`.

Last - as of the version 0.14.4 update, we've had a few reports of problems upgrading to the newest version.  We're not sure what's causing the issue yet, but have found the new version can be installed successfully in these cases using a fresh Anaconda installation.


<a id='legacyinstall'></a>
## Legacy installations (version <0.14)

The latest version of py4DSTEM (v0.14) makes changes to the classes and functions which may not be compatible with code written for prior versions.
We are working to ensure better backwards-compatibility in the future.
For now, if you have code from earlier versions, you can either (1) install the legacy version of your choice, or (2) update legacy code to use the version 0.14 methods.
To update your code to the new syntax, check out the examples in the [py4DSTEM tutorials repository](https://github.com/py4dstem/py4DSTEM_tutorials) and the docstrings for the classes and functions you're using.
To install the legacy version of py4DSTEM of your choice, you can call

```
pip install py4dstem==0.XX.XX
```

substituting the desired version for `XX.XX`.  For instance, you can install the last version 13 release with

```
pip install py4dstem==0.13.17
```

or the last version 12 release with

```
pip install py4dstem==0.12.24
```




## Advanced installations - GPU acceleration and ML functionality

To install the py4dstem with AI/ML functionality, follow the steps below.
If you're using a machine with an Nvidia GPU and CUDA capability, run:

```
conda update conda
conda create -n py4dstem-aiml
conda activate py4dstem-aiml
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 cupy 
pip install "py4dstem[aiml-cuda]"
```

If your machine does not have a CUDA capable device, run
```
conda update conda
conda create -n py4dstem
conda activate py4dstem
conda install pip
pip install "py4dstem[aiml]"
```



# The py4DSTEM GUI

The py4DSTEM GUI data browser has been moved to a separate repository.
You can [find that repository here](https://github.com/py4dstem/py4D-browser).
You can install the GUI from the command line with:

```
pip install py4D-browser
```

The py4D-browser can then be launched from the command line by calling:

```
py4DGUI
```




# Running the code

The anaconda navigator can be used to launch various Python interfaces, including Jupyter Notebooks, JupyterLab, PyCharm, and others.

Once you're inside the conda environment where you installed py4DSTEM and you've launched an interface to the Python interpreter, you can import **py4DSTEM** to access all its modules and functions using `import py4DSTEM`.


## Example code and tutorials

At this point you'll need code, and data!
Sample code demonstrating a variety of workflows can be found in [the py4DSTEM tutorials repository](https://github.com/py4dstem/py4DSTEM_tutorials) in the `/notebooks` directory.
These sample files are provided as Jupyter notebooks.
Links to the data used in each notebook are provided in the intro cell of each notebook.




# More information
<a id='Contributing'></a>

## Contributing Guide 
We are grateful for your interest in contributing to py4DSTEM. There are many ways to contribute to py4DSTEM, including Reporting bugs, Submitting feature requests, Improving documentation and Developing new code

For more information checkout our [Contributors Guide](CONTRIBUTORS.md)


## Documentation

Our documentation pages are [available here](https://py4dstem.readthedocs.io/en/latest/index.html).



## Scientific papers which use **py4DSTEM**

See a list [here](docs/papers.md).




# Acknowledgements

If you use py4DSTEM for a scientific study, please cite [our open access py4DSTEM publication in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927621000477). You are also free to use the py4DSTEM [logo in PDF format](images/py4DSTEM_logo_54.pdf) or [logo in PNG format](images/py4DSTEM_logo_54_export.png) for presentations or posters.


[![TRI logo](/images/toyota_research_institute.png)](https://www.tri.global/)


The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and development time which made this project possible.

[![DOE logo](/images/DOE_logo.png)](https://www.energy.gov/science/bes/basic-energy-sciences/)

Additional funding has been provided by the US Department of Energy, Office of Science, Basic Energy Sciences.



# License

GNU GPLv3

**py4DSTEM** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from **py4DSTEM** is also kept free and open under a GPLv3 license.

