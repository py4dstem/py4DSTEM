import py4DSTEM
import numpy as np

md = py4DSTEM.io.datastructure.Metadata(name='test_metadata')

print(md)

md['x'] = 10
md['y'] = 1.234
md['cows'] = 'arecute'
md['mice'] = 'arenice'
md['z'] = [1,2,3]
md['a'] = np.ones(5)
md['names'] = ['a','b','c']
md['xxx'] = np.ones((3,3))

print(md)
print(f"z = {md['z']}")
print(f"names = {md['names']}")

# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    md.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Metadata'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['Metadata'],
        'test_metadata')
    md1 = py4DSTEM.io.datastructure.Metadata.from_h5(grp['test_metadata'])

    print(names)
    print(exists)
    print(md)
    print(md1)


print()
print()

md2 = md1.copy()

print(md2)



print(f"z = {md2['z']}")
print(f"names = {md2['names']}")
print(md2['xxx'])

print(md2.keys)
