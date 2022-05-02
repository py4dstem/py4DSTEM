![py4DSTEM logo](/images/py4DSTEM_logo.png)

**py4DSTEM** is an open source set of python tools for processing and analysis of four-dimensional scanning transmission electron microscopy (4D-STEM) data.

For additional information beyond what's decribed below, please see:

- [the py4DSTEM documentation pages](https://py4dstem.readthedocs.io/en/latest/index.html)
- [our open access publication in Microscopy and Microanalysis](https://doi.org/10.1017/S1431927621000477) describing this project and demonstrating a variety of applications


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
conda create -n py4dstem python==3.8
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

## Advanced installation - ML functionality

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

A filename can be passed as a command line argument to the GUI to open that file immediately:
```
conda activate py4dstem
py4dstem path/to/data/file.h5
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
* gdown


### Optional dependencies

* ipyparallel
* dask




### License

GNU GPLv3

**py4DSTEM** is open source software distributed under a GPLv3 license.
It is free to use, alter, or build on, provided that any work derived from py4DSTEM is also kept free and open.



## Papers which have used py4DSTEM


### 2022

[Correlative image learning of chemo-mechanics in phase-transforming solids](https://www.nature.com/articles/s41563-021-01191-0), Nature Materials (2022)

[Correlative analysis of structure and chemistry of LixFePO4 platelets using 4D-STEM and X-ray ptychography](https://doi.org/10.1016/j.mattod.2021.10.031), Materials Today 52, 102 (2022).

[Visualizing Grain Statistics in MOCVD WSe2 through Four-Dimensional Scanning Transmission Electron Microscopy](https://doi.org/10.1021/acs.nanolett.1c04315), Nano Letters 22, 2578 (2022).

[Electric field control of chirality](https://doi.org/10.1126/sciadv.abj8030), Science Advances 8 (2022).

[Real-Time Interactive 4D-STEM Phase-Contrast Imaging From Electron Event Representation Data: Less computation with the right representation](https://doi.org/10.1109/MSP.2021.3120981),  IEEE Signal Processing Magazine 39, 25 (2022).

[Microstructural dependence of defect formation in iron-oxide thin films](https://doi.org/10.1016/j.apsusc.2022.152844), Applied Surface Science 589, 152844 (2022).

[Chemical and Structural Alterations in the Amorphous Structure of Obsidian due to Nanolites](https://doi.org/10.1017/S1431927621013957), 28, 289 (2022).

[Nanoscale characterization of crystalline and amorphous phases in silicon oxycarbide ceramics using 4D-STEM](https://doi.org/10.1016/j.matchar.2021.111512), Materials Characterization 181, 111512 (2021).

[Disentangling multiple scattering with deep learning: application to strain mapping from electron diffraction patterns](https://arxiv.org/abs/2202.00204), arXiv:2202.00204 (2022).



### 2021

[Cryoforged nanotwinned titanium with ultrahigh strength and ductility](https://doi.org/10.1126/science.abe7252), Science 16 373, 1363 (2021).

[Strain fields in twisted bilayer graphene](https://doi.org/10.1038/s41563-021-00973-w), Nature Materials 20, 956 (2021).

[Determination of Grain-Boundary Structure and Electrostatic Characteristics in a SrTiO3 Bicrystal by Four-Dimensional Electron Microscopy](https://doi.org/10.1021/acs.nanolett.1c02960), Nanoletters 21, 9138 (2021).

[Local Lattice Deformation of Tellurene Grain Boundaries by Four-Dimensional Electron Microscopy](https://pubs.acs.org/doi/10.1021/acs.jpcc.1c00308), Journal of Physical Chemistry C 125, 3396 (2021).

[Extreme mixing in nanoscale transition metal alloys](https://doi.org/10.1016/j.matt.2021.04.014), Matter 4, 2340 (2021).

[Multibeam Electron Diffraction](https://doi.org/10.1017/S1431927620024770), Microscopy and Microanalysis 27, 129 (2021).

[A Fast Algorithm for Scanning Transmission Electron Microscopy Imaging and 4D-STEM Diffraction Simulations](https://doi.org/10.1017/S1431927621012083), Microscopy and Microanalysis 27, 835 (2021).

[Fast Grain Mapping with Sub-Nanometer Resolution Using 4D-STEM with Grain Classification by Principal Component Analysis and Non-Negative Matrix Factorization](https://doi.org/10.1017/S1431927621011946), Microscopy and Microanalysis 27, 794

[Prismatic 2.0 – Simulation software for scanning and high resolution transmission electron microscopy (STEM and HRTEM)](https://doi.org/10.1016/j.micron.2021.103141), Micron 151, 103141 (2021).

[4D-STEM of Beam-Sensitive Materials](https://doi.org/10.1021/acs.accounts.1c00073), Accounts of Chemical Research 54, 2543 (2021).


### 2020


[Tilted fluctuation electron microscopy](https://doi.org/10.1063/5.0015532), Applied Physics Letters 117, 091903 (2020).

[4D-STEM elastic stress state characterisation of a TWIP steel nanotwin](https://arxiv.org/abs/2004.03982), arXiv:2004.03982





## Acknowledgements

The developers gratefully acknowledge the financial support of the Toyota Research Institute for the research and development time which made this project possible.

![TRI logo](/images/toyota_research_institute.png)
