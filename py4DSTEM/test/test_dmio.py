# Filepaths
filepath_input = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.dm3"

import py4DSTEM

# Load the data
datacube = py4DSTEM.file.io.read(filepath_input)
#print(datacube)
print(datacube.metadata)


