import py4DSTEM
import numpy as np
from os.path import join

# set filepath
path = py4DSTEM._TESTPATH + "/small_datacube.dm4"


class TestDataCube:
    # setup/teardown
    def setup_class(cls):
        # Read datacube
        datacube = py4DSTEM.import_file(path)
        cls.datacube = datacube

    # tests

    def test_binning_default_dtype(self):
        dtype = self.datacube.data.dtype
        assert dtype == np.uint16

        self.datacube.bin_Q(2)

        assert self.datacube.data.dtype == dtype

        new_dtype = np.uint32
        self.datacube.bin_Q(2, dtype=new_dtype)

        assert self.datacube.data.dtype == new_dtype
        assert self.datacube.data.dtype != dtype

        pass
