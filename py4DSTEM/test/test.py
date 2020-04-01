# Tests for virtual image module

import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
import py4DSTEM
from py4DSTEM.process.virtualimage import get_virtualimage_rect, get_virtualimage_circ, get_virtualimage_ann


class TestVirtualImaging(unittest.TestCase):

    def test_nothin(self):
        self.assertTrue(True)


if __name__=="__main__":

    unittest.main()



