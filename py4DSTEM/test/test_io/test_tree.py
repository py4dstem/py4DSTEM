import numpy as np

import py4DSTEM
from py4DSTEM.io.classes import Node



def test_tree():

    # instantiate a tree
    tree = py4DSTEM.io.classes.Tree()

    # make some EMD objects
    ar = py4DSTEM.io.classes.Array(
        data = np.array([[1,2],[3,4]])
    )
    pl = py4DSTEM.io.classes.PointList(
        data = np.zeros(
            5,
            dtype = [
                ('x',float),
                ('y',float)
            ]
        )
    )
    pl2 = py4DSTEM.io.classes.PointList(
        data = np.ones(
            4,
            dtype = [
                ('qx',float),
                ('qy',float)
            ]
        ),
        name = 'pointlist2'
    )
    pl3 = py4DSTEM.io.classes.PointList(
        data = np.ones(
            8,
            dtype = [
                ('r',float),
                ('theta',float)
            ]
        ),
        name = 'pointlist3'
    )
    pla = py4DSTEM.io.classes.PointListArray(
        shape = (3,4),
        dtype = [
            ('a',float),
            ('b',float)
        ]
    )
    md = py4DSTEM.io.classes.Metadata()
    md._params.update({
        'duck':'steve'
    })
    md2 = py4DSTEM.io.classes.Metadata(name='metadata2')
    md2._params.update({
        'duck':'steven'
    })
    md_root = py4DSTEM.io.classes.Metadata(name='metadata_root')
    md_root._params.update({
        'cow':'ben'
    })
    md_root2 = py4DSTEM.io.classes.Metadata(name='metadata_root2')
    md_root2._params.update({
        'cow':'benjamin'
    })
    root = py4DSTEM.io.classes.Root()

    # confirm that EMD objects >0 are Node instances, =0 are not
    assert(isinstance(ar,Node))
    assert(isinstance(pl,Node))
    assert(isinstance(pla,Node))
    assert(not(isinstance(md,Node)))
    assert(not(isinstance(md_root,Node)))

    # confirm objects have inherited Node attrs
    assert(hasattr(ar,"_tree"))
    assert(hasattr(ar,"_metadata"))
    assert(hasattr(pl,"_tree"))
    assert(hasattr(pl,"_metadata"))
    assert(hasattr(pla,"_tree"))
    assert(hasattr(pla,"_metadata"))

    # construct tree
    root.tree(ar)
    ar.tree(pla)
    pla.tree(pl)
    pla.tree(pl2)
    ar.metadata = md
    ar.metadata = md2
    root.tree(pl3)
    root.metadata = md_root
    root.metadata = md_root2

    # confirm that the data is where it should be
    assert(root.tree('array') == ar)
    assert(root.tree('array/pointlistarray') == pla)
    assert(pla.tree('pointlist') == pl)
    assert(pla.tree('pointlist2') == pl2)
    assert(ar.metadata['metadata'] == md)
    assert(ar.metadata['metadata2'] == md2)
    assert(root.tree('pointlist3') == pl3)
    assert(root.metadata['metadata_root'] == md_root)
    assert(root.metadata['metadata_root2'] == md_root2)

    # confirm that all nodes point to the same root node
    assert(root == pla.root == pl.root == pl2.root == pl3.root)





    # Write then read:
    # - root node
    # - some other single node
    # - whole tree from root
    # - some other whole subtree
    # - rootless tree from root
    # - rootless tree from some subtree

    # From the whole tree written from root, read:
    # - the root node
    # - the rootless tree

    # From the rootless tree, read:
    # - the (empty) root


