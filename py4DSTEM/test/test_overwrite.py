import numpy as np
import py4DSTEM

fp = '/Users/Ben/Desktop/test.h5'

data_old = np.arange(100).reshape((10,10))
data_new = data_old+1

diffslice_old = py4DSTEM.datastructure.DiffractionSlice(
                    data=data_old,name="data")
diffslice_new = py4DSTEM.datastructure.DiffractionSlice(
                    data=data_new,name="data")

py4DSTEM.io.native.save(fp, diffslice_old, overwrite=True)
py4DSTEM.io.native.append(fp, diffslice_new, overwrite=True)




