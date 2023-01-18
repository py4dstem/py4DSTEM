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

    #@classmethod
    #def setup_class(cls):
    #    pass

    #@classmethod
    #def teardown(cls):
    #    pass

    def setup_method(self, method):
        self._build_tree()

    def teardown_method(self, method):
        self._del_tree()

    def _build_tree(self):
        """
        Helper function which builds the following tree of unique objects.

        Object Class          Name

        Root                  'root'
          |-Metadata          'metadata_root'
          |-Metadata          'metadata_root'
          |-PointList         'pointlist3'
          | |-PointList       'pointlist4'
          |   |-Metadata      'metadata3'
          |-Array             'array
            |-Metadata        'metadata'
            |-Metadata        'metadata2'
            |-PointListArray  'pointlistarray'
              |-PointList     'pointlist'
              |-PointList     'pointlist2'

        """
        # make some objects
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

        # assign data instances to variables in the class instance scope
        self.root = root
        self.md = md
        self.md2 = md2
        self.md3 = md3
        self.md_root = md_root
        self.md_root2 = md_root2
        self.ar = ar
        self.pl = pl
        self.pl2 = pl2
        self.pl3 = pl3
        self.pl4 = pl4
        self.pla = pla

        return

    def _del_tree(self):
        del(self.root)
        del(self.md)
        del(self.md2)
        del(self.md3)
        del(self.md_root)
        del(self.md_root2)
        del(self.ar)
        del(self.pl)
        del(self.pl2)
        del(self.pl3)
        del(self.pl4)
        del(self.pla)



    def test_nodes(self):
        """
        Confirm that nodes are Nodes, metadata are not nodes
        """
        # confirm that EMD objects >0 are Node instances, =0 are not
        assert(isinstance(self.ar,Node))
        assert(isinstance(self.pl,Node))
        assert(isinstance(self.pla,Node))
        assert(not(isinstance(self.md,Node)))
        assert(not(isinstance(self.md_root,Node)))

        # confirm objects have inherited Node attrs
        assert(hasattr(self.ar,"_tree"))
        assert(hasattr(self.ar,"_metadata"))
        assert(hasattr(self.pl,"_tree"))
        assert(hasattr(self.pl,"_metadata"))
        assert(hasattr(self.pla,"_tree"))
        assert(hasattr(self.pla,"_metadata"))
        assert(not(hasattr(self.md,"_tree")))
        assert(not(hasattr(self.md,"_metadata")))


    def test_structure(self):
        """
        Confirm that the structure of the tree is correct, and
        that objects can be retrieved as expected
        """
        # confirm that the data is where it should be
        # and is accessible via relative paths
        assert(self.root.tree('array') == self.ar)
        assert(self.root.tree('array/pointlistarray') == self.pla)
        assert(self.root.tree('array/pointlistarray/pointlist') == self.pl)
        assert(self.root.tree('pointlist3/pointlist4').metadata['metadata3'] == \
            self.md3)
        assert(self.pla.tree('pointlist2') == self.pl2)
        assert(self.ar.metadata['metadata'] == self.md)
        assert(self.ar.metadata['metadata2'] == self.md2)
        assert(self.root.tree('pointlist3') == self.pl3)
        assert(self.root.metadata['metadata_root'] == self.md_root)
        assert(self.root.metadata['metadata_root2'] == self.md_root2)

        # confirm data is accessible via root paths
        assert(self.ar.tree('/array') == self.ar)
        assert(self.pl4.tree('/array') == self.ar)
        assert(self.pla.tree('/array/pointlistarray/pointlist') == self.pl)

        # confirm correct ._treepath attrs
        assert(self.ar._treepath == '/array')
        assert(self.pla._treepath == '/array/pointlistarray')
        assert(self.pl3._treepath == '/pointlist3')

        # confirm that all nodes point to the same root node
        assert(self.root == self.pla.root == self.pl.root == self.pl2.root == \
            self.pl3.root == self.pl4.root)


    def test_branchcut(self):
        """
        Confirms correct branch cutting behavior
        for default metadata behavior (transfer root metadata)
        """
        # cutting off a branch
        new_root = self.pl4.tree(cut=True)
        assert(self.pl4.root == new_root)
        assert(new_root.tree('pointlist4') == self.pl4)
        assert(new_root.metadata == self.root.metadata)

    def test_branchcut_no_metadata(self):
        """
        Confirms correct branch cutting behavior
        without transferring old root metadata
        """
        # cutting off a branch
        new_root = self.pl4.tree(cut=False)
        assert(len(new_root.metadata)==0)

    def test_branchcut_copy_metadata(self):
        """
        Confirms correct branch cutting behavior
        when copying root metadata
        """
        # cutting off a branch
        new_root = self.pl4.tree(cut='copy')
        assert(len(new_root.metadata)==2)
        assert(new_root.metadata != self.root.metadata)

    def test_graft(self):
        """
        Confirms correct grafting behavior
        """
        # make a new tree to graft onto
        new_root = self.pl4.tree(cut=True)
        # graft from the old tree to the new one
        self.pla.tree(graft=self.pl4)
        #pla.graft(pl4) # alt syntax
        #pla.tree(graft=(pl4,True)) # alt syntax
        assert(new_root.tree('pointlist4/pointlistarray/pointlist') == self.pl)


class TestTreeIO():

    @classmethod
    def setup_class(cls):
        cls._build_tree(cls)
        cls._write_h5_single_node(cls)
        cls._write_h5_whole_tree(cls)
        cls._write_h5_subtree(cls)
        cls._write_h5_subtree_noroot(cls)
        cls._append_to_h5(cls)
        cls._append_over_h5(cls)

    @classmethod
    def teardown(cls):
        cls._clean_files(cls)

    #def setup_method(self, method):
    #def teardown_method(self, method):

    def _build_tree(self):
        """
        Helper function which builds the following tree of unique objects.

        Object Class          Name

        Root                  'root'
          |-Metadata          'metadata_root'
          |-Metadata          'metadata_root'
          |-PointList         'pointlist3'
          | |-PointList       'pointlist4'
          |   |-Metadata      'metadata3'
          |-Array             'array
            |-Metadata        'metadata'
            |-Metadata        'metadata2'
            |-PointListArray  'pointlistarray'
              |-PointList     'pointlist'
              |-PointList     'pointlist2'

        """
        # make some objects
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

        # assign data instances to variables in the class instance scope
        self.root = root
        self.md = md
        self.md2 = md2
        self.md3 = md3
        self.md_root = md_root
        self.md_root2 = md_root2
        self.ar = ar
        self.pl = pl
        self.pl2 = pl2
        self.pl3 = pl3
        self.pl4 = pl4
        self.pla = pla

        return

    def _write_h5_single_node(self):
        """
        Write a single node to file
        """
        # TODO
        pass

    def _write_h5_whole_tree(self):
        """
        Write the whole tree to file
        """
        # TODO
        pass

    def _write_h5_subtree(self):
        """
        Write a subtree to file
        """
        # TODO
        pass

    def _write_h5_subtree_noroot(self):
        """
        Write a subtree tree to file without it's root dataset
        """
        # TODO
        pass

    def _append_to_h5(self):
        """
        Append to an existing h5 file
        """
        # TODO
        pass

    def _append_over_h5(self):
        """
        Append to an existing h5 file, overwriting a redundantly named object
        """
        # TODO
        pass

    def _clean_files(self):
        """
        Delete h5 files which this test suite wrote
        """
        # TODO
        pass

    #def _del_tree(self):
    #    del(self.root)
    #    del(self.md)
    #    del(self.md2)
    #    del(self.md3)
    #    del(self.md_root)
    #    del(self.md_root2)
    #    del(self.ar)
    #    del(self.pl)
    #    del(self.pl2)
    #    del(self.pl3)
    #    del(self.pl4)
    #    del(self.pla)


    def test_io_single_node(self):
        """
        Write the whole tree, from the root node.
        Read it, and confirm objects are the same.
        """
        assert(isinstance(self.root,Root))
        # TODO NEXT
        pass

    def test_io_from_root(self):
        """
        Write the whole tree, from the root node.
        Read it, and confirm objects are the same.
        """
        pass


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


