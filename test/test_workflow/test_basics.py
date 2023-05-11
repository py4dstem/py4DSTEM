import py4DSTEM
from os.path import join

# set filepath
path = join(py4DSTEM._TESTPATH,"simulatedAuNanoplatelet_binned_v0_9.h5")



class TestBasics:

    # setup/teardown
    def setup_class(cls):

        # Read sim Au datacube
        datacube = py4DSTEM.io.read(
            path,
            data_id = 'polyAu_4DSTEM'
        )
        cls.datacube = datacube

        # get center and probe radius
        datacube.get_dp_mean()
        alpha, qx0, qy0 = datacube.get_probe_size()
        cls.alpha = alpha
        cls.qx0,cls.qy0 = qx0,qy0


    # tests

    def test_get_dp(self):
        dp = self.datacube[10,30]
        dp

    def test_show(self):
        dp = self.datacube[10,30]
        py4DSTEM.visualize.show(dp)

    # virtual diffraction and imaging

    def test_virt_diffraction(self):
        dp_mean = self.datacube.get_dp_mean()
        self.datacube.get_dp_max()


    def test_virt_imaging_bf(self):

        geo = ((self.qx0,self.qy0),self.alpha+3)

        # position detector
        self.datacube.position_detector(
            mode = 'circle',
            geometry = geo,
        )

        # compute
        self.datacube.get_virtual_image(
            mode = 'circle',
            geometry = geo,
            name = 'bright_field',
        )



    def test_virt_imaging_adf(self):

        geo = ((self.qx0,self.qy0),(3*self.alpha,6*self.alpha))

        # position detector
        self.datacube.position_detector(
            mode = 'annulus',
            geometry = geo,
        )

        # compute
        self.datacube.get_virtual_image(
            mode = 'annulus',
            geometry = geo,
            name = 'annular_dark_field',
        )




