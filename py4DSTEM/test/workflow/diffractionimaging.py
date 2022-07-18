# Diffraction imaging




import py4DSTEM
from py4DSTEM.visualize import show
import numpy as np





# Set filepaths
#  - experimental aluminum dataset
#  - a path to write to
filepath_calibration_dm = "/media/AuxDriveB/Data/HadasSternlicht/conductive_polymers/500C/dataset_123_aluminumStandard/dataset_123.dm4"
filepath_h5 = "/home/ben/Desktop/test.h5"


# Load a datacube from a dm file
datacube = py4DSTEM.io.read(filepath_calibration_dm)
datacube = py4DSTEM.io.datastructure.DataCube(
    data=datacube.data[0:15,0:20,:,:])




# Set flags

mode = 'mean'
geometry_type = 'mask'
shift_corr = False



# Set function args

geo_dict = {
    'pts' : (np.array([2,3,5,8,9]),
             np.array([5,8,3,4,9])),
    'lims' : (2,5,2,8),
    'none' : None,
    'mask' : np.ones(datacube.Rshape, dtype=bool),
    'mask_float' : np.ones(datacube.Rshape, dtype=float),
}
geo = geo_dict[geometry_type]




# Run

dp = datacube.get_diffraction_image(
    name = 'diffraction_image',
    mode = mode,
    geometry = geo,
    shift_corr = shift_corr
)



# Test

print(dp)
print(dp.metadata)


py4DSTEM.io.save(
    filepath_h5,
    dp,
    mode = 'o'
)

py4DSTEM.io.print_h5_tree(filepath_h5, show_metadata=True)



dp_loaded = py4DSTEM.io.read(
    filepath_h5,
    root = '4DSTEM_experiment/diffraction_image',
    tree = False
)

print(dp_loaded)
print(dp_loaded.metadata)

#print(dp_loaded.metadata['diffractionimage']['geometry'])
#print(dp_loaded.metadata['diffractionimage']['mode'])



