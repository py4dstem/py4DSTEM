import py4DSTEM
import emdfile
from os.path import join


# Set filepaths
filepath_dm = join(py4DSTEM._TESTPATH, "small_dm3.dm3")


def test_dmfile_3Darray():
    data = py4DSTEM.import_file( filepath_dm )
    assert isinstance(data, emdfile.Array)


# TODO
# def test_dmfile_4Darray():
# def test_dmfile_multiple_datablocks():

