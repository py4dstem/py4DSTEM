import py4DSTEM
import numpy as np

# Make a 5D numpy array
shape = (8,8,64,64)
data = np.ones(shape)

datacube = py4DSTEM.io.datastructure.DataCube(
    data = data,
    name = 'test_datacube',
    R_pixel_size = 5,
    R_pixel_units = 'nm',
    Q_pixel_size = [0.01,0.02],
    Q_pixel_units = ['A^-1','nm^-1']
)

print(datacube)
print(datacube.calibration)



# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    datacube.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    dc = py4DSTEM.io.datastructure.DataCube.from_h5(grp['test_datacube'])


print(dc)
print(dc.calibration)





