import py4DSTEM
from os.path import join
from numpy import zeros

# set filepath
path = join(py4DSTEM._TESTPATH, "test_io/legacy_v0.9_simAuNanoplatelet_bin.h5")


class TestDiskDetectionWithCalibration:
    # setup/teardown
    def setup_class(cls):
        # Read sim Au datacube
        datacube = py4DSTEM.io.read(path, data_id="polyAu_4DSTEM")
        cls.datacube = datacube

        # prepare a probe
        mask = zeros(datacube.Rshape, dtype=bool)
        mask[28:33, 14:19] = 1
        probe = datacube.get_vacuum_probe(ROI=mask)
        alpha_pr, qx0_pr, qy0_pr = py4DSTEM.process.calibration.get_probe_size(
            probe.probe
        )
        probe.get_kernel(
            mode="sigmoid", origin=(qx0_pr, qy0_pr), radii=(alpha_pr, 2 * alpha_pr)
        )
        cls.probe = probe

        # Set disk detection parameters
        cls.detect_params = {
            "corrPower": 1.0,
            "sigma": 0,
            "edgeBoundary": 2,
            "minRelativeIntensity": 0,
            "minAbsoluteIntensity": 8,
            "minPeakSpacing": 4,
            "subpixel": "poly",
            "maxNumPeaks": 1000,
            #     'CUDA': True,
        }

    # tests

    def test_disk_detection(self):
        braggpeaks = self.datacube.find_Bragg_disks(
            template=self.probe.kernel,
            **self.detect_params,
        )

        # calibrate center

        # calibrate ellipse

        # calibrate pixel

        # calibrate rotation

        # show

        # save

        # load

        # check loaded data
        # check loaded cali
