import unittest

class Test(unittest.TestCase):

    def setUp(self):
        self.a = 'a'

    def test(self):
        self.assertTrue(self.a == 'a')

    def tearDown(self):
        pass


if __name__=='__main__':
    unittest.main()

