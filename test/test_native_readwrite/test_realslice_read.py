# Test reading realslices in v13


import py4DSTEM
from os.path import join


# Set filepaths
filepath = join(py4DSTEM._TESTPATH, "YanAiming_bilayer_v01.h5")

def test_read_realslice():
    realslice = py4DSTEM.read(filepath, datapath='4DSTEM/Fit Data')





