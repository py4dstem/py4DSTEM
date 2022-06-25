# Sample workflow

# creates a tree of data objects 
# nested under a parent datacube
# while running a sample
# processing workflow
# (calibration)
# then saves the full tree 
# into an HDF5 file.

import py4DSTEM
#from py4DSTEM.visualize import show
import numpy as np


# Set filepaths
#  - experimental aluminum dataset
#  - a same day/conditions vacuum scan
#  - a path to write to
filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
filepath_vacuum = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_141_vacuumScan/dataset_141.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"


# Load a datacube from a dm file
datacube = py4DSTEM.io.read(filepath_calibration_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])

print(f"Loaded a {datacube.data.shape} shaped datacube with tree:")
datacube.tree.print()



# Virtual diffraction

dp_max = py4DSTEM.process.virtualimage.get_dp_max(datacube)
dp_mean = py4DSTEM.process.virtualimage.get_dp_mean(datacube)
dp_median = py4DSTEM.process.virtualimage.get_dp_median(datacube)
#show(datacube.tree['dp_max'])
#show(datacube.tree['dp_mean'])
#show(datacube.tree['dp_median'])

print("After virtual diffraction")
datacube.tree.print()





# Virtual imaging

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


