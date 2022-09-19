# Test non-native file loading with `py4DSTEM.import_file`

from py4DSTEM import import_file
from os.path import join
from download_test_data import filepath as dirpath


# Set filepaths
dirpath = join(dirpath, 'unit_test_data')
filepath_dm = join(dirpath, "dm_test_file.dm3")


def test_import_dmfile():
    data = import_file( filepath_dm )



