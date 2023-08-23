import py4DSTEM
import numpy as np

# filepath
from os import getcwd, remove
from os.path import join, exists
path = join(getcwd(),"test.h5")


def test_listwrite():

    # make two arrays
    ar1 = py4DSTEM.RealSlice(
        data = np.arange(24).reshape((2,3,4)),
        name = 'array1'
    )
    ar2 = py4DSTEM.RealSlice(
        data = np.arange(48).reshape((4,3,4)),
        name = 'array2'
    )

    # save them
    py4DSTEM.save(
        filepath = path,
        data = [ar1,ar2],
        mode = 'o'
    )

    # read them
    data1 = py4DSTEM.read(
        path,
        datapath = 'array1_root'
    )
    data2 = py4DSTEM.read(
        path,
        datapath = 'array2_root'
    )

    # check
    assert(np.array_equal(data1.data, ar1.data))
    assert(np.array_equal(data2.data, ar2.data))

    # delete the file
    if exists(path):
        remove(path)


