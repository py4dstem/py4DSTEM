from py4DSTEM.file.io.native import remove
from py4DSTEM.file.io import read

dpath = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/"
fp = dpath+'test.h5'

read(fp)
remove(fp,data=[0,1,4],d=True)
read(fp)


