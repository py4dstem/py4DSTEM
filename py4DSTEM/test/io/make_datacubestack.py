import py4DSTEM
import numpy as np

# Make a 5D numpy array
shape = (8,8,64,64,3)
data = np.ones(shape)

datacubestack = py4DSTEM.io.datastructure.DataCube(
    data = data,
    name = 'test_datacubestack',
    R_pixel_size = 5,
    R_pixel_units = 'nm',
    Q_pixel_size = [0.01,0.02],
    Q_pixel_units = ['A^-1','nm^-1'],
    slicelabels = [
        'datacube1',
        'datacube2',
        'datacube3'
    ]
)

print(datacubestack)

datacubestack.qunits = 'A^-1'
print(datacubestack.qunits)
print(datacubestack.dim_units)
print(datacubestack)

# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    datacubestack.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']

    datacubestack = py4DSTEM.io.datastructure.DataCube.from_h5(grp['test_datacubestack'])
    print(datacubestack)
    print(type(datacubestack))

    #ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_datacubestack')
    #print(ar)
    #print(datacube == ar)

print(datacubestack.__class__.__name__)
print(type(datacubestack.__class__.__name__))
print(str(datacubestack.__class__.__name__))


