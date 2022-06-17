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

metadata = py4DSTEM.io.datastructure.Metadata(name='some_metadata')
metadata.set_p('toast','butter')

py4dstem_array._metadata = metadata
print(py4dstem_array._metadata)



# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
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

    print(py4dstem_array)
    print(ar)



print()

print(ar._metadata)

#with h5py.File(fp,'a') as f:
#    grp = f['experiment']
#    # write the array to the h5 file
#    ar2.to_h5(grp)
#    ar3.to_h5(grp)

#with h5py.File(fp,'r') as f:
#    grp = f['experiment']
#    array_names = py4DSTEM.io.datastructure.find_EMD_groups(
#        grp,
#        py4DSTEM.io.datastructure.EMD_group_types['Array'])
#    print(array_names)
#    ar4 = py4DSTEM.io.datastructure.Array_from_h5(grp,'Array0')
#    ar5 = py4DSTEM.io.datastructure.Array_from_h5(grp,'Array1')

#print(ar4)
#print(ar5)


