import py4DSTEM
import numpy as np
from os.path import join

# set filepath
path = join(py4DSTEM._TESTPATH, "small_dm3.dm3")




class TestDataCube:

    # setup/teardown
    def setup_class(cls):

        # Read sim Au datacube
        datacube = py4DSTEM.import_file(path)
        datacube = py4DSTEM.DataCube(
            data=datacube.data.reshape((10,10,512,512))
        )
        cls.datacube = datacube

    # tests

    def test_binning_default_dtype(self):

        dtype = self.datacube.data.dtype
        assert(dtype == np.uint16)

        self.datacube.bin_Q(2)

        assert(self.datacube.data.dtype == dtype)

        new_dtype = np.uint32
        self.datacube.bin_Q(2, dtype=new_dtype)

        assert(self.datacube.data.dtype == new_dtype)
        assert(self.datacube.data.dtype != dtype)

        pass

