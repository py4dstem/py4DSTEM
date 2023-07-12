import py4DSTEM
import emdfile
from os.path import join


# Set filepaths
filepath_dm4_datacube = join(py4DSTEM._TESTPATH, "small_datacube.dm4")
filepath_dm3_3Dstack = join(py4DSTEM._TESTPATH, "test_io/small_dm3_3Dstack.dm3")


def test_dmfile_datacube():
    data = py4DSTEM.import_file( filepath_dm4_datacube )
    assert isinstance(data, emdfile.Array)
    assert isinstance(data, py4DSTEM.DataCube)

def test_dmfile_3Darray():
    data = py4DSTEM.import_file( filepath_dm3_3Dstack )
    assert isinstance(data, emdfile.Array)


# TODO
# def test_dmfile_multiple_datablocks():
# def test_dmfile_2Darray

