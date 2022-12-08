import py4DSTEM
import unittest
from filepaths import fp_dm

class Test(unittest.TestCase):

    def setUp(self):
        self.datacube = py4DSTEM.io.read(fp_dm)

    def test_datacube(self):
        self.assertTrue(isinstance(self.datacube,
                        py4DSTEM.io.DataCube))
    def test_metadata(self):
        self.assertTrue(isinstance(self.datacube.metadata,
                        py4DSTEM.io.Metadata))

    def tearDown(self):
        del self.datacube


if __name__=='__main__':
    unittest.main()



