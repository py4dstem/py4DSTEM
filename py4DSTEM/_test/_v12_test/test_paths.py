import unittest
import pathlib
from filepaths import *

class Test(unittest.TestCase):

    def setUp(self):
        self.a = 'a'

    def test(self):
        self.assertTrue(isinstance(fp_small,pathlib.PosixPath))

    def tearDown(self):
        pass


if __name__=='__main__':
    unittest.main()

