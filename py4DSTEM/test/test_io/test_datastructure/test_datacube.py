import unittest
import os
import h5py
import numpy as np

from tempfile import mkdtemp
from shutil import rmtree

from py4DSTEM.io.datastructure.datacube import get_datacube_from_grp

class Test(unittest.TestCase):

    def setUp(self):
        self.tmpdir = mkdtemp()
        self.h5_file = h5py.File(os.path.join(self.tmpdir, 'test.h5'), 'w')
        (self.h5_file
         .create_group('4DSTEM_experiment')
         .create_group('data')
         .create_group('datacubes')
         .create_group('datacube_0')
         .create_dataset('data', data=np.ones(shape=(2,2,3,3)), dtype='uint8')
        )
        self.datacube = self.h5_file['4DSTEM_experiment']['data']['datacubes']['datacube_0']
        self.datacube.attrs.create('emd_group_type', 1)
        dim1 = self.datacube.create_dataset('dim1', shape=(2,), dtype='uint8')
        dim1.attrs.create('name', np.string_("R_x"))
        dim1.attrs.create('units', np.string_("[pix]"))
        dim2 = self.datacube.create_dataset('dim2', shape=(2,), dtype='uint8')
        dim2.attrs.create('name', np.string_("R_y"))
        dim2.attrs.create('units', np.string_("[pix]"))
        dim3 = self.datacube.create_dataset('dim3', shape=(3,), dtype='uint8')
        dim3.attrs.create('name', np.string_("Q_x"))
        dim3.attrs.create('units', np.string_("[pix]"))
        dim4 = self.datacube.create_dataset('dim4', shape=(3,), dtype='uint8')
        dim4.attrs.create('name', np.string_("Q_y"))
        dim4.attrs.create('units', np.string_("[pix]"))
        

    def testH5Memmap(self):
        memmap_datacube = get_datacube_from_grp(self.datacube, mem='MEMMAP')
        ram_datacube = get_datacube_from_grp(self.datacube)
        self.assertTrue(np.array_equal(memmap_datacube.data, ram_datacube.data))
        self.assertTrue(memmap_datacube.name == ram_datacube.name)

    def tearDown(self):
        self.h5_file.close()
        rmtree(self.tmpdir)


if __name__=='__main__':
    unittest.main()