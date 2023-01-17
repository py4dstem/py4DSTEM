import numpy as np

import py4DSTEM
from py4DSTEM.io.classes import (
    Node,
    Root,
    Metadata,
    Array,
    PointList,
    PointListArray
)


class TestTree():

    # TODO: some setup method
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown(cls):
        pass

    def setup_method(self, method):
        pass

    def setup_method(self, method):
        pass




def test_tree():


    # make some EMD objects
    ar = Array(
        data = np.array([[1,2],[3,4]])
    )
    pl = PointList(
        data = np.zeros(
            5,
            dtype = [
                ('x',float),
                ('y',float)
            ]
        )
    )
    pl2 = PointList(
        data = np.ones(
            4,
            dtype = [
                ('qx',float),
                ('qy',float)
            ]
        ),
        name = 'pointlist2'
    )
    pl3 = PointList(
        data = np.ones(
            8,
            dtype = [
                ('r',float),
                ('theta',float)
            ]
        ),
        name = 'pointlist3'
    )
    pl4 = PointList(
        data = np.ones(
            7,
            dtype = [
                ('z',float),
                ('c',float)
            ]
        ),
        name = 'pointlist4'
    )
    pla = PointListArray(
        shape = (3,4),
        dtype = [
            ('a',float),
            ('b',float)
        ]
    )
    md = Metadata()
    md._params.update({
        'duck':'steve'
    })
    md2 = Metadata(name='metadata2')
    md2._params.update({
        'duck':'steven'
    })
    md3 = Metadata(name='metadata3')
    md3._params.update({
        'duck':'stevo'
    })
    md_root = Metadata(name='metadata_root')
    md_root._params.update({
        'cow':'ben'
    })
    md_root2 = Metadata(name='metadata_root2')
    md_root2._params.update({
        'cow':'benjamin'
    })
    root = Root()

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
    root.metadata = md_root
    root.metadata = md_root2
    root.tree(pl3)
    pl3.tree(pl4)
    pl4.metadata = md3
    root.tree(ar)
    ar.metadata = md
    ar.metadata = md2
    ar.tree(pla)
    pla.tree(pl)
    pla.tree(pl2)

    # structure should be:
    # root
    #   |-md_root
    #   |-md_root2
    #   |-pl3
    #   | |-pl4
    #   |   |-md3
    #   |-ar
    #     |-md
    #     |-md2
    #     |-pla
    #       |-pl
    #       |-pl2

    # confirm that the data is where it should be
    assert(root.tree('array') == ar)
    assert(root.tree('array/pointlistarray') == pla)
    assert(root.tree('array/pointlistarray/pointlist') == pl)
    assert(root.tree('pointlist3/pointlist4').metadata['metadata3'] == md3)
    assert(pla.tree('pointlist2') == pl2)
    assert(ar.metadata['metadata'] == md)
    assert(ar.metadata['metadata2'] == md2)
    assert(root.tree('pointlist3') == pl3)
    assert(root.metadata['metadata_root'] == md_root)
    assert(root.metadata['metadata_root2'] == md_root2)

    # confirm correct ._treepath attrs
    assert(ar._treepath == '/array')
    assert(pla._treepath == '/array/pointlistarray')
    assert(pl3._treepath == '/pointlist3')

    # confirm we can read from a tree with root paths
    assert(ar.tree('/array') == ar)
    assert(pl4.tree('/array') == ar)
    assert(pla.tree('/array/pointlistarray/pointlist') == pl)

    # confirm that all nodes point to the same root node
    assert(root == pla.root == pl.root == pl2.root == pl3.root == pl4.root)



    # CHOOSE ONE
    # cutting off a branch
    new_root = pl4.tree(cut=True)
    assert(pl4.root == new_root)
    assert(new_root.tree('pointlist4') == pl4)
    assert(new_root.metadata == root.metadata)

    # don't move metadata
    #new_root = pl4.tree(cut=False)
    #assert(len(new_root.metadata)==0)

    # copy metadata
    #new_root = pl4.tree(cut='copy')
    #assert(len(new_root.metadata)==2)
    #assert(new_root.metadata != root.metadata)

    # grafting
    #new_root = pl4.tree(cut=True)
    #pla.graft(pl4)
    #pla.tree(graft=pl4)
    #pla.tree(graft=(pl4,True))
    #assert(new_root.tree('pointlist4/pointlistarray/pointlist')==pl)





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



    # Append, then check correct writing of new data and metadata, root metadata


