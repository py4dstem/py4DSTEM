import py4DSTEM
from os.path import join, exists


path = join(py4DSTEM._TESTPATH, "test_io/legacy_v0.14.h5")


def _make_v14_test_file():
    # enforce v14
    assert py4DSTEM.__version__.split(".")[1] == "14", "no!"

    # Set filepaths
    filepath_data = join(
        py4DSTEM._TESTPATH, "test_io/legacy_v0.9_simAuNanoplatelet_bin.h5"
    )

    # Read sim Au datacube
    datacube = py4DSTEM.io.read(filepath_data, data_id="polyAu_4DSTEM")

    # # Virtual diffraction

    # Get mean and max DPs
    datacube.get_dp_mean()
    datacube.get_dp_max()

    # # Disk detection

    # find a vacuum region
    import numpy as np

    mask = np.zeros(datacube.Rshape, dtype=bool)
    mask[28:33, 14:19] = 1

    # generate a probe
    probe = datacube.get_vacuum_probe(ROI=mask)

    # Find the center and semiangle
    alpha, qx0, qy0 = py4DSTEM.process.probe.get_probe_size(probe.probe)

    # prepare the probe kernel
    kern = probe.get_kernel(mode="sigmoid", origin=(qx0, qy0), radii=(alpha, 2 * alpha))  # noqa: F841

    # Set disk detection parameters
    detect_params = {
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

    # compute
    braggpeaks = datacube.find_Bragg_disks(  # noqa: F841
        template=probe.kernel,
        **detect_params,
    )

    # # Virtual Imaging

    # set geometries
    geo_bf = ((qx0, qy0), alpha + 6)
    geo_df = ((qx0, qy0), (3 * alpha, 6 * alpha))

    # bright field
    datacube.get_virtual_image(
        mode="circle",
        geometry=geo_bf,
        name="bright_field",
    )

    # dark field
    datacube.get_virtual_image(mode="annulus", geometry=geo_df, name="dark_field")

    # # Write

    py4DSTEM.save(path, datacube, tree=None, mode="o")


class TestV14:
    # setup/teardown
    def setup_class(cls):
        if not (exists(path)):
            print("no test file for v14 found")
            if py4DSTEM.__version__.split(".")[1] == "14":
                print("v14 detected. writing new test file...")
                _make_v14_test_file()
            else:
                raise Exception(
                    f"No v14 testfile was found at path {path}, and a new one can't be written with this py4DSTEM version {py4DSTEM.__version__}"
                )

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_meowth(self):
        # py4DSTEM.print_h5_tree(path)
        data = py4DSTEM.read(path)
        data.tree()

        assert isinstance(data.tree("braggvectors"), py4DSTEM.BraggVectors)
        assert isinstance(data.tree("bright_field"), py4DSTEM.VirtualImage)
        assert isinstance(data.tree("dark_field"), py4DSTEM.VirtualImage)
        assert isinstance(data.tree("dp_max"), py4DSTEM.VirtualDiffraction)
        assert isinstance(data.tree("dp_mean"), py4DSTEM.VirtualDiffraction)
        assert isinstance(data.tree("probe"), py4DSTEM.Probe)

        pass
