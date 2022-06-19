import py4DSTEM
import numpy as np

cal = py4DSTEM.io.datastructure.Calibration(name='calibration')
origin = np.ones((10,10,2))*5.2
cal.set_origin(origin)

print(cal)


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    print(cal.name)
    print('meow')
    cal.to_h5(group)
print('meowmeow')


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Metadata'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Metadata'],
        'calibration')
    cal1 = py4DSTEM.io.datastructure.Calibration.from_h5(grp['calibration'])

    print(names)
    print(exists)
    print(cal)
    print(cal1)


print()
print()

cal2 = cal1.copy()

print(cal2)




