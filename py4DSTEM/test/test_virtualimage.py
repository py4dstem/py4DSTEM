# Tests for virtual image module

import unittest
import os
import numpy as np
from py4DSTEM.file.io import read
from py4DSTEM.process.virtualimage import get_virtualimage_rect, get_virtualimage_circ, get_virtualimage_ann

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class TestVirtualImaging(unittest.TestCase):

    def setUp(self):

        # Params
        xmin,xmax = 206,306           # Rectangular detector shape
        ymin,ymax = 226,286
        x0,y0 = 240,270               # Center for circular and annular detectors
        R = 30                        # Circular detector radius
        Ri,Ro = 80,120                # Annular detector radii
        tolerance = 1e-6
        self.tolerance = tolerance

        # Filepaths
        fp_dc = THIS_DIR + "/testdata/virtualimage_test.dm3"
        fp_ans = THIS_DIR + "/testdata/virtualimage_test_ans.h5"

        # Load datacube
        dc = read(fp_dc)
        dc.set_scan_shape(10,10)

        # Get images
        self.vi_rect = get_virtualimage_rect(dc,xmin,xmax,ymin,ymax)
        self.vi_circ = get_virtualimage_circ(dc,x0,y0,R)
        self.vi_ann = get_virtualimage_ann(dc,x0,y0,Ri,Ro)

        # Load expected results
        ims = read(fp_ans)
        self.vi_rect_ans = ims.slices['rect']
        self.vi_circ_ans = ims.slices['circ']
        self.vi_ann_ans = ims.slices['ann']

    def test_rect_vi(self):
        self.assertTrue( np.sum(np.abs(self.vi_rect - self.vi_rect_ans)) < self.tolerance )

    def test_circ_vi(self):
        self.assertTrue( np.sum(np.abs(self.vi_circ - self.vi_circ_ans)) < self.tolerance )

    def test_ann_vi(self):
        self.assertTrue( np.sum(np.abs(self.vi_ann - self.vi_ann_ans)) < self.tolerance )




if __name__=="__main__":
    unittest.main()

