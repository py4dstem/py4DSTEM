import py4DSTEM
import numpy as np

# Make a 3D numpy array
shape = (64,64,3)
data = np.ones(shape)

diffractionstack = py4DSTEM.io.datastructure.DiffractionSlice(
    data = data,
    name = 'test_diffractionstack',
    pixel_size = 3,
    pixel_units = 'A^-1',
    slicelabels = [
        'im',
        'a',
        'teapot'
    ]
)

print(diffractionstack)

print(diffractionstack.pixel_units)
print(diffractionstack.dim_units)
print(diffractionstack)

# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    diffractionstack.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']

    diffractionstack = py4DSTEM.io.datastructure.DiffractionSlice_from_h5(grp,'test_diffractionstack')
    print(diffractionstack)
    print(type(diffractionstack))

    #ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_diffractionstack')
    #print(ar)
    #print(diffraction == ar)

print(diffractionstack.__class__.__name__)
print(type(diffractionstack.__class__.__name__))
print(str(diffractionstack.__class__.__name__))


