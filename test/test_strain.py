import py4DSTEM
from py4DSTEM import StrainMap
from os.path import join
from numpy import zeros


# TODO add BVs file to test suite with calibration, remove disk detection here

# set filepath
path = join(py4DSTEM._TESTPATH,"test_io/legacy_v0.9_simAuNanoplatelet_bin.h5")



class TestStrainMap:

    # setup/teardown
    def setup_class(cls):

        # Read sim Au datacube
        datacube = py4DSTEM.io.read(
            path,
            data_id = 'polyAu_4DSTEM'
        )

        # prepare a probe
        mask = zeros(datacube.Rshape,dtype=bool)
        mask[28:33,14:19] = 1
        probe = datacube.get_vacuum_probe( ROI=mask )
        alpha_pr,qx0_pr,qy0_pr = py4DSTEM.process.calibration.get_probe_size( probe.probe )
        probe.get_kernel(
            mode='sigmoid',
            origin=(qx0_pr,qy0_pr),
            radii=(alpha_pr,2*alpha_pr)
        )

        # Set disk detection parameters
        detect_params = {
            'corrPower': 1.0,
            'sigma': 0,
            'edgeBoundary': 2,
            'minRelativeIntensity': 0,
            'minAbsoluteIntensity': 8,
            'minPeakSpacing': 4,
            'subpixel' : 'poly',
            'maxNumPeaks': 1000,
        #     'CUDA': True,
        }

        # find bragg peaks
        cls.braggpeaks = datacube.find_Bragg_disks(
            template = probe.kernel,
            **detect_params,
        )

        # set an arbitrary origin
        cls.braggpeaks.calibration.set_origin((0,0))



    # tests

    def test_strainmap_instantiation(self):

        strainmap = StrainMap(
            braggvectors = self.braggpeaks,
        )

        assert(isinstance(strainmap, StrainMap))


