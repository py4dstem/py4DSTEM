import py4DSTEM
import numpy as np

# Make a 3D numpy array
shape = (10,10,4)
data = np.arange(np.prod(shape)).reshape(shape)

# Make the Array instance
py4dstem_arraystack = py4DSTEM.io.datastructure.ArrayStack(
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
    labels = [
        'the',
        'cow',
        'jumped',
        'over'
    ]
)

print("__repr__:")
print(py4dstem_arraystack)

print()
print(".get_data({element}).__repr__:")
print(py4dstem_arraystack.get_data('cow'))

print()
print(".labels and .labels._dict")
print(py4dstem_arraystack.labels)
print(py4dstem_arraystack.labels._dict)

print()
print(".labels and .labels._dict after element assignment")
py4dstem_arraystack.labels[2] = 'meow'
print(py4dstem_arraystack.labels)
print(py4dstem_arraystack.labels._dict)



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
    array_names = py4DSTEM.io.datastructure.find_Arrays(grp)
    exists = py4DSTEM.io.datastructure.Array_exists(grp,'test_arraystack')
    ar = py4DSTEM.io.datastructure.Array_from_h5(grp,'test_arraystack')

    print(array_names)
    print(exists)
    print(ar)

    #ar.set_dim(1,[-25,25],'A')
    #print(ar)


