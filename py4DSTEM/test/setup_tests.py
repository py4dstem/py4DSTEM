# Filepaths for py4DSTEM unittests

from pathlib import Path
dirpath = "/Users/Ben/Work/Data/py4DSTEM_sampleData/test"
fname_small = "small4DSTEMscan_10x10.h5"
fname_dm = "small4DSTEMscan_10x10.dm3"

fp_small = Path(dirpath) / Path(fname_small)
fp_dm = Path(dirpath) / Path(fname_dm)


