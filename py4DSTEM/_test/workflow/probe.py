# Calibration workflow

import py4DSTEM
from py4DSTEM.visualize import show
import numpy as np


# Set filepaths
#  - a vacuum scan
#  - a path to write to
filepath_vacuum = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_141_vacuumScan/dataset_141.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"



# Probe

datacube_vacuum = py4DSTEM.io.read(
    filepath_vacuum,
    name = 'datacube_vacuum'
)
print('Loaded a vacuum datacube:')
print(datacube_vacuum)

probe = datacube_vacuum.get_vacuum_probe(
    ROI = (7,10,7,10))


print(probe)
print(probe.metadata['probe'])

#show(datacube_vacuum.tree['probe'].probe)
#show(datacube_vacuum.tree['probe'].kernel)


# TODO
# goto .probe/probekernel.py
# change (probekernel -> kernel)
# make a wrapper function for the module
# connect the wrapper function to the Probe in py4dstem/probe.py
# test output

#probe.get_kernel(
#    mode = ''
#)

# TODO
#datacube.add_probe(
#    datacube_vacuum.tree['probe']
#)

#show(datacube.tree['probe'].probe)
#show(datacube.tree['probe'].kernel)
#py4DSTEM.visualize.show_probe_kernel(datacube.tree['probe'].kernel)

#print(probe)
#print(probe.calibration)




#print("After probe")
#datacube_vacuum.tree.print()




print('moew')


k = probe.get_kernel(
    mode = 'sigmoid',
    radii = (5,50),
)


print('meowo')


print(probe.metadata['kernel'])



py4DSTEM.io.save(
    filepath_h5,
    datacube_vacuum,
    tree = 'noroot',
    mode = 'o'
)
py4DSTEM.io.print_h5_tree(filepath_h5)

d = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/datacube_vacuum/probe',
    tree = False
)
print(d)
print(d.metadata['probe'])
print(d.metadata['kernel'])



