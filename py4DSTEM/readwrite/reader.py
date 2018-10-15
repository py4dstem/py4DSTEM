# Reads 4D-STEM data with hyperspy, stores as a DataCube object

import h5py
import numpy as np
import hyperspy.api as hs
from ..process.datastructure.datacube import DataCube
from ..process.log import log

@log
def read_data(filename):
    """
    Takes a filename as input, and outputs a DataCube object.

    If filename is a .h5 file, read_data() checks if the file was written by py4DSTEM.  If it
    was, the metadata are read and saved directly.  Otherwise, the file is read with hyperspy,
    and metadata is scraped and saved from the hyperspy file.
    """
    print("Reading file {}...\n".format(filename))
    # Check if file was written by py4DSTEM
    try:
        h5_file = h5py.File(filename,'r')
        if is_py4DSTEMfile(h5_file):
            print("{} is a py4DSTEM HDF5 file.  Reading...".format(filename))
            R_Ny,R_Nx,Q_Ny,Q_Nx = h5_file['4D-STEM_data']['datacube']['datacube'].shape
            datacube = DataCube(data=h5_file['4D-STEM_data']['datacube']['datacube'].value,
                            R_Ny=R_Ny,R_Nx=R_Nx,Q_Ny=Q_Ny,Q_Nx=Q_Nx,filename=filename,
                            is_py4DSTEM_file=True, h5_file=h5_file)
            h5_file.close()
            return datacube
        else:
            h5_file.close()
    except IOError:
        pass

    # Use hyperspy
    print("{} is not a py4DSTEM file.  Reading with hyperspy...".format(filename))
    try:
        hyperspy_file = hs.load(filename)
        if len(hyperspy_file.data.shape)==3:
            R_N, Q_Ny, Q_Nx = hyperspy_file.data.shape
            R_Ny, R_Nx = R_N, 1
        elif len(hyperspy_file.data.shape)==4:
            R_Ny, R_Nx, Q_Ny, Q_Nx = hyperspy_file.data.shape
        else:
            print("Error: unexpected raw data shape of {}".format(hyperspy_file.data.shape))
            print("Initializing random datacube...")
            return DataCube(data=np.random.rand(100,512,512),
                            R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                            filename=filename,is_py4DSTEM_file=False)
        return DataCube(data=hyperspy_file.data, R_Ny=R_Ny, R_Nx=R_Nx, Q_Ny=Q_Ny, Q_Nx=Q_Nx,
                            original_metadata_shortlist=hyperspy_file.metadata,
                            original_metadata_all=hyperspy_file.original_metadata,
                            filename=filename,is_py4DSTEM_file=False)
    except Exception as err:
        print("Failed to load", err)
        print("Initializing random datacube...")
        return DataCube(data=np.random.rand(100,512,512),R_Ny=10,R_Nx=10,Q_Ny=512,Q_Nx=512,
                        filename=None,is_py4DSTEM_file=False)


def is_py4DSTEMfile(h5_file):
    if ('version_major' in h5_file.attrs) and ('version_minor' in h5_file.attrs) and ('4D-STEM_data' in h5_file.keys()):
        return True
    else:
        return False


def show_log(filename):
    """
    Takes a filename as input, determines if it was written by py4DSTEM, and if it was prints
    the file's log.
    """
    print("Reading log for file {}...\n".format(filename))
    # Check if file was written by py4DSTEM
    try:
        h5_file = h5py.File(filename,'r')
        if is_py4DSTEMfile(h5_file):
            try:
                log_group = h5_file['log']
                for i in range(len(log_group.keys())):
                    key = 'log_item_'+str(i)
                    show_log_item(i, log_group[key])
                h5_file.close()
            except KeyError:
                print("Log cannot be read - HDF5 file has no log group.")
                h5_file.close()
        else:
            h5_file.close()
    except IOError:
        print("Log cannot be read - file is not an HDF5 file.")

def show_log_item(index, log_item):
    time = log_item.attrs['time'].decode('utf-8')
    function = log_item.attrs['function'].decode('utf-8')
    version = log_item.attrs['version']

    print("*** Log index {}, at time {} ***".format(index, time))
    print("Function: \t{}".format(function))
    print("Inputs:")
    for key,value in log_item['inputs'].attrs.items():
        if type(value)==np.bytes_:
            print("\t\t{}\t{}".format(key,value.decode('utf-8')))
        else:
            print("\t\t{}\t{}".format(key,value))
    print("Version: \t{}\n".format(version))



############################### File structure ###############################
#
# /
# |--attr: version_major=0
# |--attr: version_minor=2
# |--grp: 4D-STEM_data
#             |
#             |--grp: datacube
#             |         |--attr: emd_group_type=1
#             |         |--data: datacube
#             |         |--data: dim1
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim2
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim3
#             |         |    |--attr: name="R_y"
#             |         |    |--attr: units="[n_m]"
#             |         |--data: dim4
#             |               |--attr: name="R_y"
#             |               |--attr: units="[n_m]"
#             |
#             |--grp: processing
#             |         |--# This will contain objects created and used during processing
#             |         |--# e.g. vacuum probe, shifts from scan coils, binary masks,
#             |         |--#      convolution kernels, etc.
#             |
#             |--grp: log
#             |         |-grp: log_item_1
#             |         |   |--attr: function="function"
#             |         |   |--grp: inputs
#             |         |   |    |--attr: input1=val1
#             |         |   |    |--attr: input2=val2
#             |         |   |    |--...
#             |         |   |
#             |         |   |--attr: version=0.1
#             |         |   |--attr: time="20181015_16:09:42"
#             |         |
#             |         |-grp: log_item_2
#             |         |-...
#             |
#             |--grp: metadata
#                       |--grp: original
#                       |   |--# Raw metadata from original files
#                       |
#                       |--grp: microscope
#                       |   |--# Acquisition parameters
#                       |   |--# Accelerating voltage, camera length, convergence angle, 
#                       |   |--# C2 aperture, spot size, exposure time, scan rotation angle,
#                       |   |--# scan shape, probe FWHM
#                       |
#                       |--grp: sample
#                       |   |--# Material, preparation
#                       |
#                       |--grp: user
#                       |   |--# Name, instituion, dept, contact email
#                       |
#                       |--grp: processing
#                       |   |--# original file name, processing perfomed - binning, cropping
#                       |   |--# Consider attaching logs
#                       |
#                       |--grp: calibration
#                       |   |--# R pixel size, K pixel size, R/K rotation offset
#                       |   |--# In case of duplicates here and in grp: microscope (e.g. pixel
#                       |   |--# sizes), quantities here are calculated from data rather than
#                       |   |--# being read from the instrument
#                       |
#



