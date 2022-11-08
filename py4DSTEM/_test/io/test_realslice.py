import py4DSTEM
import numpy as np

# Make a 2D numpy array
shape = (256,256)
data = np.arange(np.prod(shape)).reshape(shape)

# Make the Array instance
py4dstem_realslice = py4DSTEM.io.datastructure.RealSlice(
    data = data,
    name = 'test_realslice',
    pixel_size = 5,
    pixel_units = 'nm',
)


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    py4dstem_realslice.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    array_names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Array'],
        'test_realslice')
    ar = py4DSTEM.io.datastructure.RealSlice.from_h5(grp['test_realslice'])

    print(array_names)
    print(exists)
    print(py4dstem_realslice)
    print(ar)



