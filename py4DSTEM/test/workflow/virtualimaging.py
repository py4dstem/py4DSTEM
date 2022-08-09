# Virtual imaging workflow

import py4DSTEM
from py4DSTEM.visualize import show
import numpy as np


# Set filepaths
#  - experimental aluminum dataset
#  - a path to write to
filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"


# Load a datacube from a dm file
datacube = py4DSTEM.import_file(filepath_calibration_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])




# Virtual diffraction

#dp_max = datacube.get_dp_max()
#dp_mean = datacube.get_dp_mean()
#dp_median = datacube.get_dp_median()
#show(datacube.tree['dp_max'], scaling='log')
#show(datacube.tree['dp_mean'], scaling='log')
#show(datacube.tree['dp_median'], scaling='log')






# Virtual imaging

geometry_BF = (
    (432,432),
    30
)
geometry_ADF = (
    (432,432),
    (80,300)
)
im_BF = py4DSTEM.process.virtualimage.get_virtual_image(
    datacube,
    mode = 'circular',
    geometry = geometry_BF,
)
im_ADF = py4DSTEM.process.virtualimage.get_virtual_image(
    datacube,
    mode = 'annular',
    geometry = geometry_ADF,
)
#show(datacube.tree['vBF'])
#show(datacube.tree['vADF'])

print("After virtual imaging")
datacube.tree.print()



# mask detector

mask = np.zeros(datacube.Qshape, dtype=bool)
mask[30:35,30:35] = True

im_mask = py4DSTEM.process.virtualimage.get_virtual_image(
    datacube,
    mode = 'mask',
    geometry = mask,
)

