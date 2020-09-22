import numpy as np
from os import remove as rm
from os.path import exists
from py4DSTEM.io.native import copy, remove, repack, append
from py4DSTEM.io import read

dpath = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/"
fp0 = dpath+'test.h5'
fp1 = dpath+'test2.h5'
fp2 = dpath+'test3.h5'

if exists(fp1): rm(fp1)
if exists(fp2): rm(fp2)

copy(fp0,fp1)
remove(fp1,data=[0,1,3])
copy(fp1,fp2)
repack(fp1)

data1 = read(fp0,data_id=0)
data2 = read(fp0,data_id=1)
data=[data1,data2]
append(fp1,data, overwrite=1)
data1.data=np.zeros((2,2,2,2))
append(fp1,data, overwrite=2)


