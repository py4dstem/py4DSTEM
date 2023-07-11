import py4DSTEM
import numpy as np


def test_attach():

    x = py4DSTEM.DiffractionSlice(np.ones((5,5)), name='x')
    y = py4DSTEM.DiffractionSlice(np.ones((5,5)), name='y')


    x.calibration.set_Q_pixel_size(50)
    y.calibration.set_Q_pixel_size(2)

    x.attach(y)

    assert('y' in x.treekeys)
    assert(x.calibration.get_Q_pixel_size() == 50)


