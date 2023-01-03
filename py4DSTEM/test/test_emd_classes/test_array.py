# test import

def test_Array():
    from py4DSTEM.io.datastructure.emd import Array
    import numpy as np

    shape = (3,4,5)
    d = np.arange(np.prod(shape)).reshape(shape)

    ar = Array(
        data = d
    )

