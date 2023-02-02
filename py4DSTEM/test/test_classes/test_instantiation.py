import numpy as np
from py4DSTEM.io.classes.py4dstem import (
    DataCube,
    RealSlice,
    DiffractionSlice,
    VirtualDiffraction,
    VirtualImage,
    Probe,
    QPoints,
    BraggVectors
)


##;pdef test_DataCube():
##;p
##;p    shape = (5,6,12,12)
##;p    data = np.zeros(shape)
##;p    datacube = DataCube( data )
##;p    assert(isinstance(datacube,DataCube))
##;p
##;p    # test add method
##;p    datacube.get_dp_max

def test_RealSlice():

    shape = (5,12)
    data = np.zeros(shape)
    realslice = RealSlice( data )
    assert(isinstance(realslice,RealSlice))

def test_DiffractionSlice():

    shape = (15,16)
    data = np.zeros(shape)
    diffractionslice = DiffractionSlice( data )
    assert(isinstance(diffractionslice,DiffractionSlice))

def test_VirtualDiffraction():

    shape = (15,16)
    data = np.zeros(shape)
    virtualdiffraction = VirtualDiffraction( data )
    assert(isinstance(virtualdiffraction,VirtualDiffraction))

def test_VirtualImage():

    shape = (15,16)
    data = np.zeros(shape)
    virtualimage = VirtualImage( data )
    assert(isinstance(virtualimage,VirtualImage))

def test_Probe():

    shape = (15,16)
    data = np.zeros(shape)
    probe = Probe( data )
    assert(isinstance(probe,Probe))
    probe.get_kernel

def test_QPoints():

    length = 10
    dtype = [('qx',float),('qy',float),('intensity',float)]
    data = np.zeros(length,dtype=dtype)
    qpoints = QPoints( data )
    assert(isinstance(qpoints,QPoints))

def test_BraggVectors():

    Rshape = (12,13)
    Qshape = (50,50)
    braggvectors = BraggVectors(
        Rshape,
        Qshape
    )
    assert(isinstance(braggvectors,BraggVectors))



