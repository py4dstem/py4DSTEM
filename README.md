![py4DSTEM logo](/images/py4DSTEM_logo.png)

**py4DSTEM** is an open source set of python tools for processing and analysis of four-dimensional scanning transmission electron microscopy (4D-STEM) data.

[![DOI](https://zenodo.org/badge/148587083.svg)](https://zenodo.org/badge/latestdoi/148587083)


## What is 4D-STEM?

In a traditional STEM experiment, a beam of high energy electrons is focused to a very fine probe - on the order of, or even smaller than, the spacing between atoms - and rastered across the surface of the sample. A conventional two-dimensional STEM image is formed by populating the value of each pixel with the electron flux through a detector at the corresponding beam position. In 4D-STEM a pixelated detector is used, where a 2D image of the diffracted STEM probe is recorded at every raster position of the beam. A 4D-STEM scan thus results in a 4D data array.


4D-STEM data is information rich.
A datacube can be collapsed in real space to yield information comparable to nanobeam electron diffraction experiment, or in diffraction space to yield a variety of virtual images, corresponding to both traditional STEM imaging modes as well as more exotic virtual imaging modalities.
The structure, symmetries, and spacings of Bragg disks can be used to extract spatially resolved maps of crystallinity, grain orientations, and lattice strain.
Redundant information in overlapping Bragg disks can be leveraged to calculate the sample potential.
Structure in the diffracted halos of amorphous systems can be used to describe the short and medium range order.


py4DSTEM supports many different modes of 4DSTEM analysis.
The tutorials, sample code, module, and function documentation all provide more detailed discussion on some of the analytical methods possible with this code.
More information can also be found at [https://arxiv.org/abs/2003.09523](https://arxiv.org/abs/2003.09523).




## Installation

The recommended installation for py4DSTEM uses the Anaconda python distribution.
First, download and install Anaconda. Instructions can be found at www.anaconda.com/download.
Then open a terminal and run

```
conda update conda
conda create -n py4dstem python==3.7
conda activate py4dstem
conda install pip
pip install py4dstem
```

If you're running Windows, you should then also run

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



## Running the GUI

At this stage of development, most of the analyses possible with py4DSTEM are accomplished using the code in .py scripts or .ipynb jupyter notebooks -- discussed further immediately below.
Our intention is to support many of these analyses through the GUI eventually.
At present the primary utility of the GUI is for browsing and visualizing 4DSTEM data.
Stay tuned for further developments!

To open the GUI from a terminal, run
```
conda activate py4dstem
py4dstem
```



## Running the code

The anaconda navigator can be used to launch various python interfaces, including Jupyter Notebooks, JupyterLab, PyCharm, and others.

From any python interpreter inside the `py4dstem` conda environment, you can import py4DSTEM to access all its modules and functions using `import py4DSTEM`.


At this point you'll need code, and data!
Sample code lives in the top level directory called `sample_code`.
To run these files, you can download this repository from github by clicking on the green 'Code' button, unzip the files, and place them somewhere on your system.
Then navigate to the `sample_code` directory on your local filesystem, choose a sample `.ipynb` or `.py` file, and try running it.


Sample datasets are provided [here](https://drive.google.com/drive/folders/1GmxF1ltY7hBU4d5ZK8INXjaRW5Etnw_4).
Links to individual datasets are provided in the sample code files which make use of them.
Once you've selected a file of sample code to run, find the link in that file to the dataset it uses, download and place it somewhere in your filesystem, then edit the filepath in the code to indicate where you've put that data.




## For contributors

Please see [here](https://gist.github.com/bsavitzky/8b1ee4c1244814940e7cff4500535dba).




## More information

### Dependencies

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
* requests


### Optional dependencies

* ipyparallel
* dask


### Versioning

v. 0.12.2



### License

GNU GPLv3

**py4DSTEM** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from py4DSTEM is also kept free and open.


### Acknowledgements

The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and development time which made this project possible.

![TRI logo](/images/toyota_research_institute.png)
