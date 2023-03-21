from py4DSTEM import read, DataCube, _TESTPATH
from os.path import join

path = join(_TESTPATH, 'simulatedAuNanoplatelet_binned_v0_9.h5')

def test_read_v0_9_noID():

    d = read(path)
    d
def test_read_v0_9_withID():

    d = read(path, data_id="polyAu_4DSTEM")
    assert(isinstance(d,DataCube))


