import py4DSTEM
import numpy as np

# Make a 3D numpy array
shape = (64,64,3)
data = np.ones(shape)

realstack = py4DSTEM.io.datastructure.RealSlice(
    data = data,
    name = 'test_realstack',
    pixel_size = 3,
    pixel_units = 'nm',
    slicelabels = [
        'this',
        'being',
        'human'
    ]
)

print(realstack)

print(realstack.pixel_units)
print(realstack.dim_units)
print(realstack)

# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    realstack.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']

    realstack = py4DSTEM.io.datastructure.RealSlice.from_h5(grp['test_realstack'])
    print(realstack)
    print(type(realstack))

    #ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_realstack')
    #print(ar)
    #print(real == ar)

print(realstack.__class__.__name__)
print(type(realstack.__class__.__name__))
print(str(realstack.__class__.__name__))


