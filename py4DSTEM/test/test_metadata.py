"""
test_metadata.py

######## DESCRIPTION ########
This script tests the metadata read/write functionality of py4DSTEM.

######## VERSION INFO ########
Last updated on 2019-11-15 with py4DSTEM version 0.9.18.
"""

### PARAMETERS ###

# Filepaths
filepath1 = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/testMetadata1.h5"
filepath2 = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/testMetadata2.h5"

# Metadata to set
Rpixsize = 10
Rpixunits = 'nm'
Qpixsize = 1.8
Qpixunits = 'A^-1'
accelerating_voltage = 300
accelerating_voltage_units = 'k_eV'

### END PARAMETERS ###


import py4DSTEM
from os.path import exists
from os import remove
import numpy as np

# Prepare files
# We want one existing file containing some data and one filepath that doesn't point to anything
dc = py4DSTEM.file.datastructure.DataCube(data=np.ones((3,3,3,3)),name='datacube')
py4DSTEM.file.io.native.save(filepath1,dc,overwrite=True)
if exists(filepath2):
    remove(filepath2)

# Make a metadata object
metadata = py4DSTEM.file.io.native.Metadata()

# Write some metadata
metadata.set_R_pixel_size(Rpixsize)
metadata.set_R_pixel_size_units(Rpixunits)
metadata.set_Q_pixel_size(Qpixsize)
metadata.set_Q_pixel_size_units(Qpixunits)

# Show the metadata
print(metadata.get_R_pixel_size())
print(metadata.get_R_pixel_size_units())
print(metadata.get_Q_pixel_size())
print(metadata.get_Q_pixel_size_units(where=True))

# Save metadata 
metadata.to_h5(filepath1)   # Add to an existing h5 file
metadata.to_h5(filepath2)   # Write to a new h5 file

# Read from the files and compare
metadata_from_h5_1 = py4DSTEM.file.io.native.Metadata().from_h5(filepath1)
metadata_from_h5_2 = py4DSTEM.file.io.native.Metadata().from_h5(filepath2)

print('Testing metadata read/write...')
assert (metadata.get_R_pixel_size() == \
        metadata_from_h5_1.get_R_pixel_size() == \
        metadata_from_h5_2.get_R_pixel_size())
assert (metadata.get_R_pixel_size_units() == \
        metadata_from_h5_1.get_R_pixel_size_units() == \
        metadata_from_h5_2.get_R_pixel_size_units())
print('All tests passed!')


