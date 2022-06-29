# Sample workflow

# creates a tree of data objects 
# nested under a parent datacube
# while running a sample
# processing workflow
# (calibration)
# then saves the full tree 
# into an HDF5 file.

import py4DSTEM
from py4DSTEM.visualize import show
from os.path import exists
from os import remove
import numpy as np


# Set filepaths


# experimental aluminum dataset
# a same day/conditions vacuum scan
# a path to write to
filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
filepath_vacuum = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_141_vacuumScan/dataset_141.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"


# Load a datacube from a dm file
datacube = py4DSTEM.io.read(filepath_calibration_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])

print(f"Loaded a {datacube.data.shape} shaped datacube")


# Do some processing, adding objects
# to the dacube's tree as we go

print("w/ tree:")
print(datacube.tree)


# max/mean dp
dp_max = py4DSTEM.process.virtualimage.get_dp_max(datacube)
dp_mean = py4DSTEM.process.virtualimage.get_dp_mean(datacube)
dp_median = py4DSTEM.process.virtualimage.get_dp_median(datacube)
#show(datacube.tree['dp_max'])
#show(datacube.tree['dp_mean'])
#show(datacube.tree['dp_median'])

print(datacube.tree)
print()





## End workflow





## Test write/read


#datacube.tree['calibration'].tree['x'] = py4DSTEM.io.datastructure.Array(
#    data = np.ones((3,3)),
#    name = 'x'
#)
#print(datacube.tree['calibration'])
#print(datacube.tree['calibration/x'])

datacube.Q_pixel_size = 0.2
datacube.Q_pixel_units = "A^-1"
print(datacube.calibration)


# remove pre-existing files
#if exists(filepath_h5):
#    remove(filepath_h5)


# save a file
py4DSTEM.io.save(
    filepath_h5,
    datacube,
    tree = True,
    mode = 'o'
)

# print the file contents
py4DSTEM.io.print_h5_tree(
    filepath_h5
)

# load one object from the file tree
loaded_data_notree = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/',
    tree = True
)
print(loaded_data_notree)
print(loaded_data_notree.tree)
print(loaded_data_notree.tree['datacube/calibration'])




# save a file
py4DSTEM.io.save(
    filepath_h5,
    datacube,
    tree = 'noroot',
    mode = 'o'
)

# print the file contents
py4DSTEM.io.print_h5_tree(
    filepath_h5
)

# load one object from the file tree
loaded_data_notree = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/datacube/',
    tree = False
)
print("\nData loaded without tree:")
print(loaded_data_notree)
print(loaded_data_notree.tree)


# load the whole tree
loaded_data_tree = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/datacube',
    tree = True
)
print("\nData loaded with tree:")
print(loaded_data_tree)
print(loaded_data_tree.tree)
print(loaded_data_tree.tree['calibration'])


# load the tree but not its root
loaded_data_noroot = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/datacube',
    tree = 'noroot'
)
print("\nData loaded with tree but no root:")
print(loaded_data_noroot)
print(loaded_data_noroot['calibration'])


# ...play with / test combinations of save(tree= options


# save a file
py4DSTEM.io.save(
    filepath_h5,
    datacube,
    tree = True,
    mode = 'o'
)


datacube.tree.print()


# print the file contents
py4DSTEM.io.print_h5_tree(
    filepath_h5
)


# load the whole tree
loaded_data_tree = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment',
    tree = True
)
print("\nData loaded with tree:")
print(loaded_data_tree)
print(loaded_data_tree.tree)

loaded_data_tree.tree.print()



#### append

# append to a file
py4DSTEM.io.save(
    filepath_h5,
    datacube,
    tree = True,
    root = '4DSTEM_experiment/datacube',
    mode = 'a'
)


py4DSTEM.io.print_h5_tree(
    filepath_h5
)

data = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment'
)

data.tree.print()






# save a file
#py4DSTEM.io.save(
#    filepath_h5,
#    datacube,
#    tree = True,
#    mode = 'w'
#)

# print the file contents
#py4DSTEM.io.print_h5_tree(
#    filepath_h5
#)







##### fin
















#### Cont. workflow

# virtual imaging
#geometry_BF = {
#    'keys' : vals
#}
#geometry_ADF = {
#    'keys' : vals
#}
#datacube.get_virtual_image(**geometry_BF)
#datacube.get_virtual_image(**geometry_ADF)
#show(datacube.tree['BF'])
#show(datacube.tree['ADF'])
#
#
#print(datacube.tree)
#
#
## load a vacuum scan
## make a probe
## put the probe in datacube's tree
#
#filepath_vacuum = "filepath"
#datacube_vacuum = py4DSTEM.io.read(filepath_vacuum)
#
#datacube_vacuum.get_probe_ROI(lims)
#datacube_vacuum.get_probe_kernel(params)
#show(datacube_vacuum.tree['probe'].probe)
#show(datacube_vacuum.tree['probe'].kernel)
#py4DSTEM.visualize.show_probe_kernel(datacube_vacuum.tree['probe'].kernel)
#
#datacube.add_probe(datacube_vacuum.tree['probe'])
#show(datacube.tree['probe'].probe)
#show(datacube.tree['probe'].kernel)
#py4DSTEM.visualize.show_probe_kernel(datacube.tree['probe'].kernel)
#
#
## ...
## ...
## ...
#
#
#print(datacube)
#print(datacube.tree)   # show the tree
#
#
#
#
## Save the datacube with its tree
## Load datacube+tree from the .h5
#
#
#filepath_h5 = "~/Desktop/test.h5"
##py4DSTEM.io.save(
#    datacube,
#    filepath
#)
#datacube2 = py4DSTEM.io.read(filepath_h5)


