import h5py
import numpy as np

fp = "/Users/Ben/Desktop/test.h5"
f = h5py.File(fp,'w')

grp = f.create_group('ponies/r/us')

shape = (2,3,4,5)
data = np.arange(np.prod(shape)).reshape(shape)
D = len(shape)

dset = grp.create_dataset(
    'meow',
    shape = shape,
    data = data,
    dtype = int
)


dset.attrs.create('x','')


print(dset.attrs['x'])





