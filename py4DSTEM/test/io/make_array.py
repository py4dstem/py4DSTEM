import py4DSTEM
import numpy as np

# Make a 5D numpy array
shape = (3,4,5,6,7)
data = np.arange(np.prod(shape)).reshape(shape)

# Make the Array instance
py4dstem_array = py4DSTEM.io.datastructure.Array(
    data = data,
    name = 'test_array',
    units = 'intensity',
    dims = [
        5,
        [0,5],
        0.123,
        [0,2,3,4,5,10]
    ],
    dim_units = [
        'nm',
        'nm',
        'A^-1',
    ],
    dim_names = [
        'rx',
        'ry',
        'qx',
    ]
)


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/Users/Ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    py4dstem_array.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    array_names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'],
        'test_array')
    ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_array')
    ar2 = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_array')
    ar3 = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_array')

    print(array_names)
    print(exists)
    print(py4dstem_array)
    print(ar)

    ar.set_dim(1,[-25,25],'A')
    print(ar)


print()
print()
print()

ar2.name=''
ar3.name=''
print(ar2)
print(ar3)

with h5py.File(fp,'a') as f:
    grp = f['experiment']
    # write the array to the h5 file
    ar2.to_h5(grp)
    ar3.to_h5(grp)

with h5py.File(fp,'r') as f:
    grp = f['experiment']
    array_names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'])
    print(array_names)
    ar4 = py4DSTEM.io.datastructure.Array_from_h5(grp,'Array0')
    ar5 = py4DSTEM.io.datastructure.Array_from_h5(grp,'Array1')

print(ar4)
print(ar5)


