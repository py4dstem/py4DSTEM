# testing the default read/write behavior
# in order to support auto-read/write without specifying paths within the h5 file
# for simple cases with a single object

# Tests 4 use cases:

# (1) save one object, with nothing it its tree. then load it.
# (2) save one object, with stuff in its tree. then load it.
# (3) load from a file with one top-level object and some tree underneath it,
#     but load something other than the top level object
# (4) load from a file with multiple top level objects.


import py4DSTEM
import numpy as np
import os


### Prepare data

## Load a datacube and populate its tree with some objects

# Set filepath to a datacube
filepath_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"

# Load a datacube from a dm file
datacube = py4DSTEM.io.import_file(filepath_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])

# Virtual diffraction
dp_max = datacube.get_dp_max()
dp_mean = datacube.get_dp_mean()
dp_median = datacube.get_dp_median()

# Virtual imaging
geometry_BF = (
    (432,432),
    30
)
geometry_ADF = (
    (432,432),
    (80,300)
)
im_BF = datacube.get_virtual_image(
    mode = 'circle',
    geometry = geometry_BF,
    name = 'vBF'
)
im_ADF = datacube.get_virtual_image(
    mode = 'annulus',
    geometry = geometry_ADF,
    name = 'vADF'
)


## Prepare other objects without trees: a second datacube, and an array
datacube2 = py4DSTEM.io.DataCube(
    data = np.copy(datacube.data),
    name = 'datacube2'
)

array = py4DSTEM.io.Array(
    data = np.arange(6*7*8).reshape(6,7,8),
    name = 'some_array'
)



### Test io

filepath1 = "/home/ben/Desktop/test1.h5"
filepath2 = "/home/ben/Desktop/test2.h5"
filepath3 = "/home/ben/Desktop/test3.h5"
filepath4 = "/home/ben/Desktop/test4.h5"
for fp in (filepath1,filepath2,filepath3,filepath4):
    if os.path.exists(fp): os.remove(fp)


# Use case 1:

print('\n*** Use Case #1 ***\n')

# save a DataCube with nothing in its Tree
py4DSTEM.io.save(
    filepath1,
    datacube2,
)
# check contents
py4DSTEM.io.print_h5_tree(filepath1)
# read the object, without specifying a path inside the .h5
d = py4DSTEM.io.read(
    filepath1,
)
# check the output
print(d)
print(d.tree)




# Use case 2:

print('\n*** Use Case #2 ***\n')

# save a DataCube with data in its Tree
py4DSTEM.io.save(
    filepath2,
    datacube,
)
# check contents
py4DSTEM.io.print_h5_tree(filepath2)
# read the object, without specifying a path inside the .h5
d = py4DSTEM.io.read(
    filepath2,
)
# check the output
print(d)
print(d.tree)





# Use case 3:

print('\n*** Use Case #3 ***\n')

# save a DataCube with data in its Tree
py4DSTEM.io.save(
    filepath3,
    datacube,
)
# check contents
py4DSTEM.io.print_h5_tree(filepath3)
# read from a specifyied path inside the .h5
d = py4DSTEM.io.read(
    filepath3,
    root = '4DSTEM/datacube/vBF'
)
# check the output
print(d)
print(d.tree)






# Use case 4:

print('\n*** Use Case #4 ***\n')

# save a DataCube with multipl top level objects
py4DSTEM.io.save(
    filepath4,
    datacube,
)
py4DSTEM.io.save(
    filepath4,
    datacube2,
    mode = 'a'
)
py4DSTEM.io.save(
    filepath4,
    array,
    mode = 'a'
)
# check contents
py4DSTEM.io.print_h5_tree(filepath4)
# read from the file
d = py4DSTEM.io.read(
    filepath4,
    root = '4DSTEM/some_array/'
)
# check the output
print(d)
print(d.tree)






