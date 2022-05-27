import py4DSTEM
import numpy as np

cal = py4DSTEM.io.datastructure.Calibration((10,10,50,50))

origin = np.ones((10,10,2))*5.2
cal.set_origin(origin)

print(cal)


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/Users/Ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    cal.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_Calibration(grp)
    exists = py4DSTEM.io.datastructure.Calibration_exists(grp,'calibration')
    cal1 = py4DSTEM.io.datastructure.Calibration_from_h5(grp,'calibration')

    print(names)
    print(exists)
    print(cal)
    print(cal1)


print()
print()
print()

#pl2 = pl.copy(name='')
#pl3 = pl.copy(name='')






