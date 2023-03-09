import numpy as np
import py4DSTEM
emd = py4DSTEM.emd



def test_Node():

    # Root class instances should:
    # - have a name
    # - have a Tree
    # - know how to read/write to/from h5

    root = emd.Node()
    assert(isinstance(root,emd.Node))
    ##;passert(root.name == 'root')
    ##;passert(isinstance(root.tree, Tree))

    # h5io


def test_Metadata():

    # Metadata class instances should:
    # - TODO

    metadata = emd.Metadata()
    assert(isinstance(metadata,emd.Metadata))


def test_Array():

    # Array class instances should:
    # - TODO

    shape = (3,4,5)
    d = np.arange(np.prod(shape)).reshape(shape)

    ar = emd.Array(
        data = d
    )
    assert(isinstance(ar, emd.Array))


def test_PointList():

    # PointList class instances should:
    # - TODO

    dtype = [
        ('x',int),
        ('y',float)
    ]
    data = np.zeros(10,dtype=dtype)
    pointlist = emd.PointList(
        data=data
    )
    assert(isinstance(pointlist,emd.PointList))

def test_PointListArray():

    # PointListArray class instance should:
    # - TODO

    dtype = [
        ('x',int),
        ('y',float)
    ]
    shape = (5,5)
    pointlistarray = emd.PointListArray(
        dtype = dtype,
        shape = shape
    )
    assert(isinstance(pointlistarray,emd.PointListArray))








