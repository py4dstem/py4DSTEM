from py4DSTEM import read, _TESTPATH
from os.path import join

path = join(_TESTPATH, 'simulatedAuNanoplatelet_binned_v0_9.h5')

def test_read_v0_9():

    d = read(path)
    d

