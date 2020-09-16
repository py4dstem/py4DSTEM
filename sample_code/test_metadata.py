"""
test_metadata.py

######## DESCRIPTION ########
This script tests the metadata read/write functionality of py4DSTEM, by

######## VERSION INFO ########
Last updated on 2019-11-15 with py4DSTEM version 0.9.18.
"""

### PARAMETERS ###

# Filepaths
filepath_input = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10.h5"
filepath_output = "/Users/Ben/Work/Data/py4DSTEM_sampleData/smallDataSets/small4DSTEMscan_10x10_metadataOnly.h5"

# Metadata to set
Rpixsize = 10
Rpixunits = 'nm'
Qpixsize = 1.8
Qpixunits = 'A^-1'
accelerating_voltage = '300'
accelerating_voltage_units = 'k_eV'

### END PARAMETERS ###


import py4DSTEM

metadata = py4DSTEM.file.io.native.Metadata()

# Write some metadata
metadata.set_R_pixel_size(Rpixsize)
metadata.set_R_pixel_size_units(Rpixunits)
metadata.set_Q_pixel_size(Qpixsize)
metadata.set_Q_pixel_size_units(Qpixunits)

print(metadata.get_R_pixel_size())
print(metadata.get_R_pixel_size_units())
print(metadata.get_Q_pixel_size())
print(metadata.get_Q_pixel_size_units(where=True))

metadata.to_h5(filepath_input)



