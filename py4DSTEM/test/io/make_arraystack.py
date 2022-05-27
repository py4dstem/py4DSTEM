import py4DSTEM
import numpy as np

# Make a 3D numpy array
shape = (10,10,4)
data = np.arange(np.prod(shape)).reshape(shape)

# Make the Array instance
py4dstem_arraystack = py4DSTEM.io.datastructure.Array(
    data = data,
    name = 'test_arraystack',
    units = 'intensity',
    dims = [
        5,
        [0,5]
    ],
    dim_units = [
        'nm',
        'nm'
    ],
    dim_names = [
        'rx',
        'ry'
    ],
    slicelabels = [
        'the',
        'cow',
        'jumped',
        'over'
    ]
)

print("__repr__:")
print(py4dstem_arraystack)

print()
print(".get_slice({element}).__repr__:")
print(py4dstem_arraystack.get_slice('cow'))
print()
print("[{element}].__repr__:")
print(py4dstem_arraystack['cow'])

print()
print(".labels and .labels._dict")
print(py4dstem_arraystack.slicelabels)
print(py4dstem_arraystack.slicelabels._dict)

print()
print(".labels and .labels._dict after element assignment")
py4dstem_arraystack.slicelabels[2] = 'meow'
print(py4dstem_arraystack.slicelabels)
print(py4dstem_arraystack.slicelabels._dict)



# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/Users/Ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    py4dstem_arraystack.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    array_names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'],
        'test_arraystack')
    ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_arraystack')

    print(array_names)
    print(exists)
    print(ar)

print(py4dstem_arraystack.data.shape)
print(ar.data.shape)


    #ar.set_dim(1,[-25,25],'A')
    #print(ar)



# Test slicing

print(ar['cow'])
print(ar[3,4,1])
print(ar[1:5,2:5,1])
print(ar['cow',3,4])
print(ar['cow',1:5,2:5])

