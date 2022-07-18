
> :warning: **py4DSTEM version 0.13 update**: Warning, this is a major update and we expect some workflows to break.  To install the previous version of py4DSTEM, please use the command line:
```
pip install py4dstem==0.12.23
```



![py4DSTEM logo](/images/py4DSTEM_logo.png)

**py4DSTEM** is an open source set of python tools for processing and analysis of four-dimensional scanning transmission electron microscopy (4D-STEM) data. Additional information:

- [Our open access py4DSTEM publication in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927621000477) describing this project and demonstrating a variety of applications.
- [The py4DSTEM documentation pages](https://py4dstem.readthedocs.io/en/latest/index.html)
- [Our open access 4D-STEM review in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927619000497) describing this project and demonstrating a variety of applications.



# What is 4D-STEM?

In a traditional STEM experiment, a beam of high energy electrons is focused to a very fine probe - on the order of, or even smaller than, the spacing between atoms - and rastered across the surface of the sample. A conventional two-dimensional STEM image is formed by populating the value of each pixel with the electron flux through a detector at the corresponding beam position. In 4D-STEM a pixelated detector is used instead, where a 2D image of the diffracted STEM probe is recorded at every raster position of the beam. A 4D-STEM scan thus results in a 4D data array.


4D-STEM data is information rich.
A datacube can be collapsed in real space to yield information comparable to nanobeam electron diffraction experiment, or in diffraction space to yield a variety of virtual images, corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to calculate the sample potential.
Structure in the diffracted halos of amorphous systems can be used to describe the short and medium range order.

py4DSTEM supports many different modes of 4DSTEM analysis.
The tutorials, sample code, module, and function documentation all provide more detailed discussion on some of the analytical methods possible with this code.




# py4DSTEM Installation

The recommended installation for py4DSTEM uses the Anaconda python distribution.
First, download and install Anaconda: www.anaconda.com/download. 
If you prefer a more lightweight conda client, we recomment Miniconda: https://docs.conda.io/en/latest/miniconda.html.
Then open a conda terminal and run one of the following sets of commands:

**For x86 CPUS e.g. INTEL, AMD processors**
```
conda update conda
conda create -n py4dstem python==3.8
conda activate py4dstem
conda install pip
pip install py4dstem
```
**For Apple Silicon CPUs e.g. M1, M1Pro, M1Max, M2 processors**
```
conda update conda
conda create -n py4dstem python==3.8
conda activate py4dstem
conda install pyqt hdf5
conda install pip
pip install py4dstem
```

**If you're running Windows, you should then also run:**

```
conda install pywin32
```

In order, these commands
- ensure your installation of anaconda is up-to-date
- make a virtual environment - see below!
- enter the environment
- make sure your new environment talks nicely to pip, a tool for installing Python packages
- use pip to install py4DSTEM
- on Windows: enable python to talk to the windows API

Please note that virtual environments are used in the instructions above, to make sure packages that have different dependencies don't conflict with one another.
Because these directions install py4DSTEM to its own virtual environment, each time you want to use py4DSTEM, you'll need to activate this environment.
You can do this in the command line with `conda activate py4dstem`, or, if you're using the Anaconda Navigator, by clicking on the Environments tab and then clicking on `py4dstem`.



# Advanced installation - ML functionality

To install the py4dstem with AI/ML functionality, follow the steps below.

If you are running on Linux/Unix machine with Nvidia GPU and CUDA capability, run

```
conda update conda
conda create -n py4dstem-aiml python=3.8 -y && conda activate py4dstem-aiml
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 cupy 
pip install "py4dstem[aiml-cuda]"
```

If you are running on Windows with Nvidia GPU and CUDA capability, run
```
conda update conda
conda create -n py4dstem-aiml python=3.8 -y && conda activate py4dstem-aiml
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 cupy 
pip install "py4dstem[aiml-cuda]"
```

If you are running on Windows without any CUDA capable device or any Mac OS X system, run
```
conda update conda
conda create -n py4dstem python==3.8
conda activate py4dstem
conda install pip
pip install "py4dstem[aiml]"
```



# Running the GUI

At this stage of development, most of the analyses possible with py4DSTEM are accomplished using the code in .py scripts or .ipynb jupyter notebooks -- discussed further immediately below.
Our intention is to support many of these analyses through the GUI eventually.
At present the primary utility of the GUI is for browsing and visualizing 4DSTEM data.
Stay tuned for further developments!

To open the GUI from a terminal, run
```
conda activate py4dstem
py4dstem
```

A filename can be passed as a command line argument to the GUI to open that file immediately:
```
conda activate py4dstem
py4dstem path/to/data/file.h5
```



# Running the code

The anaconda navigator can be used to launch various python interfaces, including Jupyter Notebooks, JupyterLab, PyCharm, and others.

From any python interpreter inside the `py4dstem` conda environment, you can import py4DSTEM to access all its modules and functions using `import py4DSTEM`.


At this point you'll need code, and data!
Sample code lives in the top level directory called `sample_code`.
To run these files, you can download this repository from github by clicking on the green 'Code' button, unzip the files, and place them somewhere on your system.
Then navigate to the `sample_code` directory on your local filesystem, choose a sample `.ipynb` or `.py` file, and try running it.


Sample datasets are provided [here](https://drive.google.com/drive/folders/1GmxF1ltY7hBU4d5ZK8INXjaRW5Etnw_4).
Links to individual datasets are provided in the sample code files which make use of them.
Once you've selected a file of sample code to run, find the link in that file to the dataset it uses, download and place it somewhere in your filesystem, then edit the filepath in the code to indicate where you've put that data.

The largest collection of py4DSTEM workflows can be found on the tutorial repo here: https://github.com/py4dstem/py4DSTEM_tutorials



# More information

## For contributors

Please see [here](https://gist.github.com/bsavitzky/8b1ee4c1244814940e7cff4500535dba).


## Scientific papers which use py4DSTEM

See a list [here](docs/papers.md).


## Dependencies

* numpy
* scipy
* h5py
* ncempy
* numba
* scikit-image
* scikit-learn
* PyQt5
* pyqtgraph
* qtconsole
* ipywidgets
* tqdm
* gdown


## Optional dependencies

* ipyparallel
* dask
* cupy
* pymatgen



## License

GNU GPLv3

**py4DSTEM** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from py4DSTEM is also kept free and open.



# Acknowledgements


[![TRI logo](/images/toyota_research_institute.png)](https://www.tri.global/)


The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and development time which made this project possible.

[![DOE logo](/images/DOE_logo.png)](https://www.energy.gov/science/bes/basic-energy-sciences/)

Additional funding has been provided by the US Department of Energy, OFfice of Science, Basic Energy Sciences.


