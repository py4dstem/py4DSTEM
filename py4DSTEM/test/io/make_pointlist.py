import py4DSTEM
import numpy as np

# Make a structured numpy array
dtype = [('qx',float),
         ('qy',float),
         ('I',float),
         ('h',int),
         ('k',int),
         ('l',int)]
data = np.zeros(100,dtype=dtype)

# Make the Array instance
pointlist = py4DSTEM.io.datastructure.PointList(
    data = data,
    name = 'test_pointlist'
)


# Add metadata
md = py4DSTEM.io.datastructure.Metadata()
md.set_p('cows','come home')
pointlist._metadata = md

print(pointlist)
print(pointlist._metadata)



# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    pointlist.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointList'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointList'],
        'test_pointlist')
    pl = py4DSTEM.io.datastructure.PointList_from_h5(grp,'test_pointlist')

    print(names)
    print(exists)
    print(pointlist)
    print(pl)

print(pl._metadata)

print()
print()
print()

pl2 = pl.copy(name='')
pl3 = pl.copy(name='')

print(pl2)
print(pl3)
print(pl3._metadata)

with h5py.File(fp,'a') as f:
    grp = f['experiment']
    # write the array to the h5 file
    pl2.to_h5(grp)
    pl3.to_h5(grp)

with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointList'])
    print(names)
    pl4 = py4DSTEM.io.datastructure.PointList_from_h5(grp,'PointList0')
    pl5 = py4DSTEM.io.datastructure.PointList_from_h5(grp,'PointList1')

print(pl4)
print(pl5)

pl6 = pl5.add_fields([('q',float),('cows',int)], name='with_cows')

print(pl6)


