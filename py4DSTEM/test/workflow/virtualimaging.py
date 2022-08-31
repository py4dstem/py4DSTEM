# Virtual imaging workflow

import py4DSTEM
from py4DSTEM.visualize import show
import numpy as np


# Set filepaths
#  - experimental aluminum dataset
#  - a path to write to

#filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
#filepath_h5 = "/home/ben/Desktop/test.h5"
filepath_calibration_dm = "/Users/Ben/ben/data/py4DSTEM_sampleData/calibration_simulatedAuNanoplatelet/calibrationData_simulatedAuNanoplatelet_binned.h5"
filepath_h5 = "/Users/Ben/Desktop/test.h5"


# Load a datacube from a dm file
#datacube = py4DSTEM.import_file(filepath_calibration_dm)
#datacube = py4DSTEM.io.datastructure.DataCube(
#    data=datacube.data[0:15,0:20,:,:])
datacube = py4DSTEM.read(
    filepath_calibration_dm,
    data_id = "polyAu_4DSTEM"
    )
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
    (63,63),
    12
)
geometry_ADF = (
    (63,63),
    (40,150)
)
im_BF = datacube.get_virtual_image(
    mode = 'circular',
    geometry = geometry_BF,
)
im_ADF = datacube.get_virtual_image(
    mode = 'annular',
    geometry = geometry_ADF,
)

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



x0 = np.ones(datacube.Rshape)*63
y0 = np.ones(datacube.Rshape)*63
sy,sx = np.meshgrid(np.linspace(-5,5,datacube.Rshape[1]),np.linspace(-5,5,datacube.Rshape[0]))
x0 += sx
y0 += sy
datacube.calibration.set_origin((x0,y0))

# vis testing
m = datacube.get_virtual_image(
    mode = 'annular',
    #geometry = geometry_BF,
    geometry = ((0,0),(25,50)),
    centered = True,
    shift_center = True,
    return_mask = (0,19)
)
show(m)

