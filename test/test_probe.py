import py4DSTEM
from py4DSTEM import Probe
import numpy as np
from os.path import join

# set filepath
path = py4DSTEM._TESTPATH + "/small_datacube.dm4"


class TestProbe:
    # setup/teardown
    def setup_class(cls):
        # Read datacube
        datacube = py4DSTEM.import_file(path)
        cls.datacube = datacube

    # tests

    def test_probe_gen_from_dp(self):
        p = Probe.from_vacuum_data(self.datacube[0, 0])
        assert isinstance(p, Probe)
        pass

    def test_probe_gen_from_stack(self):
        # get a 3D stack
        x, y = np.zeros(10).astype(int), np.arange(10).astype(int)
        data = self.datacube.data[x, y, :, :]
        # get the probe
        p = Probe.from_vacuum_data(data)
        assert isinstance(p, Probe)
        pass

    def test_probe_gen_from_datacube_ROI_1(self):
        ROI = np.zeros(self.datacube.Rshape, dtype=bool)
        ROI[3:7, 5:10] = True
        p = self.datacube.get_vacuum_probe(ROI)
        assert isinstance(p, Probe)

        self.datacube.tree()
        self.datacube.tree(True)
        _p = self.datacube.tree("probe")
        print(_p)

        assert p is self.datacube.tree("probe")
        pass

    def test_probe_gen_from_datacube_ROI_2(self):
        ROI = (3, 7, 5, 10)
        p = self.datacube.get_vacuum_probe(ROI)
        assert isinstance(p, Probe)
        assert p is self.datacube.tree("probe")
        pass
