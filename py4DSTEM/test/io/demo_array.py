### Show the py4DSTEM Array class

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
        ['x','y'],
        [0,5],
        [0,0.123],
        [0,2,3,4,5,10,11]
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

print("## 1 ##")
print(py4dstem_array)
print()


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

#with h5py.File(fp,'w') as f:
#    group = f.create_group('experiment')
#    py4dstem_array.to_h5(group) # write the array to the h5 file


#with h5py.File(fp,'r') as f:
#    grp = f['experiment']
#    ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_array')

# ensure we have a new object
#assert(ar is not py4dstem_array)


# Show array after writing and reading
#print("## 2 ##")
#print(ar)
#print()


# Alter a dim vector and show
#ar.set_dim(1,[-25,25],'A')

#print("## 3 ##")
#print(ar)
#print()



