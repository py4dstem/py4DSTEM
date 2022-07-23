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
        p.add( np.zeros(0,dtype=dtype) )


md1 = py4DSTEM.io.datastructure.Metadata('elephants')
md1['bingo'] = 'bongo'
md2 = py4DSTEM.io.datastructure.Metadata('rhinos')
md2['dingo'] = 'dongo'

pla.metadata = md1
pla.metadata = md2

print(pla)
print(pla.metadata.keys())
print(pla.metadata['elephants'])
print(pla.metadata['rhinos'])




print('')
print(pla[2,3])
print(pla[2,3].data)


d = np.ones(
    10,
    dtype = dtype)
pla[2,3].add(d)
print(pla[2,3].data)


print('')
print('')
print(d)
d = py4DSTEM.io.datastructure.PointList(
    data = d
)
print(d)
pla[2,3] = d
print(pla[2,3])
print(pla[2,3].data)

print('')
print('')
print('')




# Write to and HDF5 file

fp = "/home/ben/Desktop/test.h5"

print(pla)
py4DSTEM.io.save(
    fp,
    data = pla,
    mode = 'o'
)
py4DSTEM.io.print_h5_tree( fp )
d = py4DSTEM.io.read(
    fp,
    root = '4DSTEM_experiment/test_pla'
)
print(d)





#with h5py.File(fp,'w') as f:
#    group = f.create_group('experiment')
#    # write the array to the h5 file
#    pla.to_h5(group)


#with h5py.File(fp,'r') as f:
#    grp = f['experiment']
#    names = py4DSTEM.io.datastructure.find_EMD_groups(
#        grp,
#        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'])
#    exists = py4DSTEM.io.datastructure.EMD_group_exists(
#        grp,
#        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'],
#        'test_pla')
#    pla1 = py4DSTEM.io.datastructure.PointListArray.from_h5(grp['test_pla'])
#
#    print(names)
#    print(exists)
#    print(pla)
#    print(pla1)

#print(pla1.metadata.keys())
#print(pla1.metadata['elephants'])
#print(pla1.metadata['rhinos'])

#print()
#print()
#print()

#pla2 = pla.copy(name='')
#pla3 = pla.copy(name='')

#print(pla2)
#print(pla3)
#print(pla3._metadata)

#with h5py.File(fp,'a') as f:
#    grp = f['experiment']
#    # write the array to the h5 file
#    pla2.to_h5(grp)
#    pla3.to_h5(grp)
#
#with h5py.File(fp,'r') as f:
#    grp = f['experiment']
#    names = py4DSTEM.io.datastructure.find_EMD_groups(
#        grp,
#        py4DSTEM.io.datastructure.EMD_group_types['PointListArray'])
#    print(names)
#    pla4 = py4DSTEM.io.datastructure.PointListArray.from_h5(grp['PointListArray0'])
#    pla5 = py4DSTEM.io.datastructure.PointListArray.from_h5(grp['PointListArray1'])

#print(pla4)
#print(pla5)

#pla6 = pla5.add_fields([('q',float),('cows',int)], name='with_cows')

#print(pla6)


# Test grabbing a pointlist
#pl1 = pla.get_pointlist(3,4)
#print(pl1)

# Test grabbing a pointlist with __getitem__ slicing
#pl2 = pla6[3,4]
#print(pl2)




