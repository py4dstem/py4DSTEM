import py4DSTEM
import numpy as np

# Make a 4D numpy array
shape = (8,8,64,64)
data = np.ones(shape)

datacube = py4DSTEM.io.datastructure.DataCube(
    data = data,
    name = 'test_datacube',
    rsize = 5,
    runits = 'nm',
    qsize = [0.01,0.1],
    qunits = ['A^-1','nm^-1']
)

print("## 1 ##")
print(datacube)
print()

datacube.qsize = 0.01
datacube.qunits = 'A^-1'
print("## 2 ##")
print(datacube)
print()



# Write to and HDF5 file

import h5py
from os.path import exists
from os import remove
fp = "/Users/Ben/Desktop/test.h5"
if exists(fp): remove(fp)

with h5py.File(fp,'w') as f:
    group = f.create_group('experiment')
    datacube.to_h5(group)


with h5py.File(fp,'r') as f:
    grp = f['experiment']
    new_datacube = py4DSTEM.io.datastructure.DataCube_from_h5(grp,'test_datacube')


print("## 3 ##")
print(new_datacube)
print()





