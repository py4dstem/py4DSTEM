# Disk detection

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
    data=datacube.data[0:10,0:10,:,:])

print(f"Loaded a {datacube.data.shape} shaped datacube with tree:")
datacube.tree.print()




# Virtual diffraction

dp_max = datacube.get_dp_max()
dp_mean = datacube.get_dp_mean()
dp_median = datacube.get_dp_median()


# Virtual imaging

geometry_BF = (
    (432,432),
    30
)
geometry_ADF = (
    (432,432),
    (80,300)
)
im_BF = datacube.get_virtual_image(
    mode = 'circle',
    geometry = geometry_BF,
    name = 'vBF'
)
im_ADF = datacube.get_virtual_image(
    mode = 'annulus',
    geometry = geometry_ADF,
    name = 'vADF'
)

# Probe

datacube_vacuum = py4DSTEM.io.read(
    filepath_vacuum,
    name = 'datacube_vacuum'
)

probe = datacube_vacuum.get_vacuum_probe(
    ROI = (7,10,7,10)
)

probe.get_kernel(
    mode = 'sigmoid',
    radii = (0,50)
)
datacube.add(probe)






# Disk detection


detect_params = {
    'minAbsoluteIntensity':0.65,
    'minPeakSpacing':20,
    'maxNumPeaks':20,
    'subpixel':'poly',
    'sigma':2,
    'edgeBoundary':20,
    'corrPower':1,
}


#### Select mode


# whole datacube
x = None

# one scan position
#x = 4,4

# several scan positions
#rxs = 0,3,5,0,5,8
#rys = 8,5,3,2,4,3
#x = rxs,rys

# mask specified scan positions
#x = np.zeros( datacube.Rshape, dtype=bool)
#x[3,3:6] = True



# Run
bragg = datacube.find_Bragg_disks(
    data = x,
    template = probe.kernel,
    **detect_params
)
print(bragg)





# Bragg vector map
#bvm = py4DSTEM.process.diskdetection.get_bragg_vector_map_raw(braggpeaks_raw,datacube.Q_Nx,datacube.Q_Ny)


bvm = py4DSTEM.process.diskdetection.get_bragg_vector_map_raw(
    bragg.v_uncal,
    datacube.Q_Nx,
    datacube.Q_Ny)

print(bvm)



# i/o


#py4DSTEM.io.save(
#    filepath_h5,
#    data = datacube,
#    mode = 'o',
#    tree = 'noroot'
#)
#py4DSTEM.io.print_h5_tree(filepath_h5)
#d = py4DSTEM.io.read(
##    filepath_h5,
#    root = '4DSTEM_experiment/datacube',
#    tree = 'noroot'
#)
#print(d)





