# Test non-native file loading with `py4DSTEM.import_file`


import py4DSTEM
from os.path import join


# Set filepaths
filepath_dm = join(py4DSTEM._TESTPATH, "dm_test_file.dm3")

def test_import_dmfile():
    data = py4DSTEM.import_file( filepath_dm )
    assert isinstance(data, py4DSTEM.io.DataCube) # currently fails because this dm file is 3D, so the data reads as an Array and not a DataCube




