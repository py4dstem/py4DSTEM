import py4DSTEM
import emdfile
from os.path import join


# Set filepaths
filepath = join(py4DSTEM._TESTPATH, "test_arina/STO_STEM_bench_20us_master.h5")


def test_read_arina():
    # read
    data = py4DSTEM.import_file(filepath)

    # check imported data
    assert isinstance(data, emdfile.Array)
    assert isinstance(data, py4DSTEM.DataCube)
