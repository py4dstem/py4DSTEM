import unittest
import os
import h5py
import numpy as np

from tempfile import mkdtemp
from shutil import rmtree

from py4DSTEM.io.datastructure.datacube import get_datacube_from_grp
from test_utils import create_EMD

class Test(unittest.TestCase):
    
    def setUp(self):
        self.tmpdir = mkdtemp()
        self.contiguous_h5file, self.contiguous_datacube = create_EMD('test.h5', self.tmpdir)
        self.chunked_h5file, self.chunked_datacube = create_EMD('test_chunked.h5', self.tmpdir, chunks=True)
        
    def test_contiguous_h5_memmap(self):
        memmap_datacube = get_datacube_from_grp(self.contiguous_datacube, mem='MEMMAP')
        ram_datacube = get_datacube_from_grp(self.contiguous_datacube)
        print(memmap_datacube.data, ram_datacube.data)
        self.assertTrue(np.array_equal(np.array(memmap_datacube.data), ram_datacube.data))
        self.assertTrue(memmap_datacube.name == ram_datacube.name)
        
    @unittest.skip("Not yet")
    def test_chunked_h5_memmap(self):
        memmap_datacube = get_datacube_from_grp(self.chunked_datacube, mem='MEMMAP')
        ram_datacube = get_datacube_from_grp(self.chunked_datacube)
        self.assertTrue(np.array_equal(memmap_datacube.data, ram_datacube.data))
        self.assertTrue(memmap_datacube.name == ram_datacube.name)

    def tearDown(self):
        self.contiguous_h5file.close()
        self.chunked_h5file.close()
        rmtree(self.tmpdir)


if __name__=='__main__':
    unittest.main()