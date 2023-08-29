import py4DSTEM
from py4DSTEM import Calibration
import numpy as np
from os import mkdir, remove, rmdir
from os.path import join, exists

# set filepaths
path_datacube = join(py4DSTEM._TESTPATH, "small_datacube.dm4")
path_3Darray = join(py4DSTEM._TESTPATH, "test_io/small_dm3_3Dstack.dm3")

path_out_dir = join(py4DSTEM._TESTPATH, "test_outputs")
path_out = join(path_out_dir, "test_calibration.h5")


class TestCalibration:
    # setup

    def setup_class(cls):
        if not exists(path_out_dir):
            mkdir(path_out_dir)

    def teardown_class(cls):
        if exists(path_out_dir):
            rmdir(path_out_dir)

    def teardown_method(self):
        if exists(path_out):
            remove(path_out)

    # test

    def test_imported_datacube_calibration(self):
        datacube = py4DSTEM.import_file(path_datacube)

        assert hasattr(datacube, "calibration")
        assert isinstance(datacube.calibration, Calibration)
        assert hasattr(datacube, "root")
        assert isinstance(datacube.root, py4DSTEM.Root)

    def test_instantiated_datacube_calibration(self):
        datacube = py4DSTEM.DataCube(data=np.ones((4, 8, 128, 128)))

        assert hasattr(datacube, "calibration")
        assert isinstance(datacube.calibration, Calibration)
        assert hasattr(datacube, "root")
        assert isinstance(datacube.root, py4DSTEM.Root)

        datacube.calibration.set_Q_pixel_size(10)

        py4DSTEM.save(path_out, datacube)

        new_datacube = py4DSTEM.read(path_out)

        assert hasattr(new_datacube, "calibration")
        assert isinstance(new_datacube.calibration, Calibration)
        assert hasattr(new_datacube, "root")
        assert isinstance(new_datacube.root, py4DSTEM.Root)

        assert new_datacube.calibration.get_Q_pixel_size() == 10
