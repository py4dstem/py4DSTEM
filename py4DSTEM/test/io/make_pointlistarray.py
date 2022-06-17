import py4DSTEM
import numpy as np

# Set up datatype and shape
dtype = [('qx',float),
         ('qy',float),
         ('I',float)]
shape = (20,30)


# Make the Array instance
pla = py4DSTEM.io.datastructure.PointListArray(
    dtype = dtype,
    shape = shape,
    name = 'test_pla'
)

for rx in range(pla.shape[0]):
    for ry in range(pla.shape[1]):
        p = pla.get_pointlist(rx,ry)
        p.append( np.zeros(10,dtype=dtype) )


md = py4DSTEM.io.datastructure.Metadata()
md.set_p('bingo', 'bongo')
pla.metadata = md

print(pla)
print(pla.metadata)


# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/home/ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    # write the array to the h5 file
    pla.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'])
    exists = py4DSTEM.io.datastructure.EMD_group_exists(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'],
        'test_pla')
    pla1 = py4DSTEM.io.datastructure.PointListArray_from_h5(grp,'test_pla')

    print(names)
    print(exists)
    print(pla)
    print(pla1)

print(pla1.metadata)

print()
print()
print()

pla2 = pla.copy(name='')
pla3 = pla.copy(name='')

print(pla2)
print(pla3)
print(pla3.metadata)

with h5py.File(fp,'a') as f:
    grp = f['experiment']
    # write the array to the h5 file
    pla2.to_h5(grp)
    pla3.to_h5(grp)

with h5py.File(fp,'r') as f:
    grp = f['experiment']
    names = py4DSTEM.io.datastructure.find_EMD_groups(
        grp,
        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'])
    print(names)
    pla4 = py4DSTEM.io.datastructure.PointListArray_from_h5(grp,'PointListArray0')
    pla5 = py4DSTEM.io.datastructure.PointListArray_from_h5(grp,'PointListArray1')

print(pla4)
print(pla5)

pla6 = pla5.add_fields([('q',float),('cows',int)], name='with_cows')

print(pla6)


# Test grabbing a pointlist
pl1 = pla.get_pointlist(3,4)
print(pl1)

# Test grabbing a pointlist with __getitem__ slicing
pl2 = pla6[3,4]
print(pl2)




