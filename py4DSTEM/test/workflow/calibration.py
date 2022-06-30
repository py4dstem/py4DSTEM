# Calibration workflow

import py4DSTEM
from py4DSTEM.visualize import show
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

dp_max = datacube.get_dp_max()
dp_mean = datacube.get_dp_mean()
dp_median = datacube.get_dp_median()
#show(datacube.tree['dp_max'], scaling='log')
#show(datacube.tree['dp_mean'], scaling='log')
#show(datacube.tree['dp_median'], scaling='log')

print("After virtual diffraction")
datacube.tree.print()





# Virtual imaging

geometry_BF = (
    (432,432),
    30
)
geometry_ADF = (
    (432,432),
    (80,300)
)
#im_BF = py4DSTEM.process.virtualimage.get_virtual_image(
#    datacube,
#    geometry_BF,
#    name = 'vBF'
#)
#im_ADF = py4DSTEM.process.virtualimage.get_virtualimage(
#    datacube,
#    geometry_ADF,
#    name = 'vADF'
#)
#show(datacube.tree['vBF'])
#show(datacube.tree['vADF'])

#print("After virtual imaging")
#datacube.tree.print()




# Probe

## make a probe
## put the probe in datacube's tree

#datacube_vacuum = py4DSTEM.io.read(
#    filepath_vacuum,
#    name = 'datacube_vacuum'
#)
#print('Loaded a vacuum datacube:')
#print(datacube_vacuum)


#probe = py4DSTEM.process.probe.get_vacuum_probe(
#    datacube_vacuum,
#    ROI = (7,10,7,10))

#print(probe.calibration)

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
#print("After probe")
#datacube.tree.print()

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


