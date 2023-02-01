import numpy as np
from os.path import join,exists
from os import remove
from numpy import array_equal
import h5py

import py4DSTEM
from py4DSTEM.emd import save,read
from py4DSTEM.emd.read import _is_EMD_file,_get_EMD_rootgroups
from py4DSTEM.emd.classes import (
    Node,
    Root,
    Metadata,
    Array,
    PointList,
    PointListArray
)

# Set paths
dirpath = py4DSTEM._TESTPATH
single_node_path = join(dirpath,"test_h5_single_node.h5")
unrooted_node_path = join(dirpath,"test_h5_unrooted_node.h5")
whole_tree_path = join(dirpath,"test_h5_whole_tree.h5")
subtree_path = join(dirpath,"test_h5_subtree.h5")
subtree_noroot_path = join(dirpath,"test_h5_subtree_noroot.h5")
subtree_append_to_path = join(dirpath,"test_h5_append_to.h5")
subtree_append_over_path = join(dirpath,"test_h5_append_over.h5")


class TreeBuilder():

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

        Also creates a single unrooted Array instance called 'unrooted_array'
        """
        # make some objects
        ar = Array(
            data = np.array([[1,2],[3,4]])
        )
        ar2 = Array(
            data = np.array([[10,20,30],[40,50,60]]),
            name = 'array'
        )
        ar_unrooted = Array(
            data = np.array([[5,6],[7,8]]),
            name = 'unrooted_array'
        )
        ar_unrooted.set_dim(
            0,
            dim = 0.25,
            units = 'ponies',
            name = 'pony_axis'
        )
        ar_unrooted.set_dim( 1, dim = [0,1.5] )
        ar_unrooted.set_dim_units(1,'cows')
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
        pl5 = PointList(
            data = np.ones(
                6,
                dtype = [
                    ('l',float),
                    ('m',float)
                ]
            ),
            name = 'pointlist5'
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
        root2 = Root()

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

        # construct smaller tree
        root2.tree(ar2)
        ar2.tree(pl5)

        # assign data instances to variables in the class instance scope
        self.root = root
        self.md = md
        self.md2 = md2
        self.md3 = md3
        self.md_root = md_root
        self.md_root2 = md_root2
        self.ar = ar
        self.ar2 = ar2
        self.ar_unrooted = ar_unrooted
        self.pl = pl
        self.pl2 = pl2
        self.pl3 = pl3
        self.pl4 = pl4
        self.pl5 = pl5
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
        del(self.ar_unrooted)
        del(self.pl)
        del(self.pl2)
        del(self.pl3)
        del(self.pl4)
        del(self.pla)







class TestTree(TreeBuilder):

    def setup_method(self, method):
        self._build_tree()

    def teardown_method(self, method):
        self._del_tree()


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
        assert(hasattr(self.ar,"_branch"))
        assert(hasattr(self.ar,"_metadata"))
        assert(hasattr(self.pl,"_branch"))
        assert(hasattr(self.pl,"_metadata"))
        assert(hasattr(self.pla,"_branch"))
        assert(hasattr(self.pla,"_metadata"))
        assert(not(hasattr(self.md,"_branch")))
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
        assert(new_root.tree('pointlist4/pointlistarray/pointlist') == self.pl)

    def test_graft2(self):
        """
        Confirms correct grafting behavior with alternative syntax
        """
        # make a new tree to graft onto
        new_root = self.pl4.tree(cut=True)
        # graft from the old tree to the new one
        self.pla.graft(self.pl4) # alt syntax
        assert(new_root.tree('pointlist4/pointlistarray/pointlist') == self.pl)

    def test_graft3(self):
        """
        Confirms correct grafting behavior with alternative syntax
        """
        # make a new tree to graft onto
        new_root = self.pl4.tree(cut=True)
        # graft from the old tree to the new one
        self.pla.tree(graft=(self.pl4,True)) # alt syntax
        assert(new_root.tree('pointlist4/pointlistarray/pointlist') == self.pl)



class TestTreeIO(TreeBuilder):

    @classmethod
    def setup_class(cls):
        cls._clear_files(cls)
        cls._build_tree(cls)
        cls._write_h5_single_node(cls)
        cls._write_h5_unrooted_node(cls)
        cls._write_h5_whole_tree(cls)
        cls._write_h5_subtree(cls)
        cls._write_h5_subtree_noroot(cls)
        cls._append_to_h5(cls)
        cls._append_over_h5(cls)

    @classmethod
    def teardown_class(cls):
        cls._clear_files(cls)

    def setup_method(self, method):
        self._build_tree()

    def teardown_method(self, method):
        self._del_tree()



    # Tests

    def test_file_validity(self):
        assert(_is_EMD_file(single_node_path))
        assert(_is_EMD_file(unrooted_node_path))
        assert(_is_EMD_file(whole_tree_path))
        assert(_is_EMD_file(subtree_path))
        assert(_is_EMD_file(subtree_noroot_path))
        assert(_is_EMD_file(subtree_append_to_path))
        assert(_is_EMD_file(subtree_append_over_path))


    def test_hdf5_group_paths(self):
        """
        Confirms that the _treepath is always the same as
        the group path without its root
        """
        # Open the file
        with h5py.File(whole_tree_path,'r') as f:

            # loop over nodes
            for node in (
                self.root,
                self.ar,
                self.pl,
                self.pl2,
                self.pl3,
                self.pl4,
                self.pla,
            ):
                assert( f[f'/{self.root.name}/' + node._treepath] )


    def test_single_node_io(self):
        """ Should contain a single array 'array' inside a single root 'root',
        and 'array' should match self.ar
        """
        # Load data
        loaded_data = read(
            single_node_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        lar = loaded_data.tree('array')
        assert(isinstance(lar,Array))

        # Did the root metadata load correctly
        lmd1 = lar.root.metadata['metadata_root']
        lmd2 = lar.root.metadata['metadata_root2']
        assert(lmd1 is not self.md_root)
        assert(lmd2 is not self.md_root2)
        assert(lmd1._params == self.md_root._params)
        assert(lmd2._params == self.md_root2._params)

        # Did the array load correctly?
        assert(lar is not self.ar)
        assert(lar.name == self.ar.name)
        assert(lar.units == self.ar.units)
        assert(array_equal(lar.data,self.ar.data))
        for i in range(len(lar.dims)):
            assert(array_equal(lar.dims[i],self.ar.dims[i]))
            assert(lar.dim_names[i] == self.ar.dim_names[i])
            assert(lar.dim_units[i] == self.ar.dim_units[i])
        pass



    def test_single_node_io_emdpath(self):
        """ Should contain a single array 'array' inside a single root 'root',
        and 'array' should match self.ar
        """
        # Load data
        loaded_data = read(
            single_node_path,
            emdpath = '/root/array'
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        lar = loaded_data.tree('array')
        assert(isinstance(lar,Array))

        # Did the root metadata load correctly
        lmd1 = lar.root.metadata['metadata_root']
        lmd2 = lar.root.metadata['metadata_root2']
        assert(lmd1 is not self.md_root)
        assert(lmd2 is not self.md_root2)
        assert(lmd1._params == self.md_root._params)
        assert(lmd2._params == self.md_root2._params)

        # Did the array load correctly?
        assert(lar is not self.ar)
        assert(lar.name == self.ar.name)
        assert(lar.units == self.ar.units)
        assert(array_equal(lar.data,self.ar.data))
        for i in range(len(lar.dims)):
            assert(array_equal(lar.dims[i],self.ar.dims[i]))
            assert(lar.dim_names[i] == self.ar.dim_names[i])
            assert(lar.dim_units[i] == self.ar.dim_units[i])
        pass


    def test_unrooted_node_io(self):
        """ Should contain a single array 'array' inside a single root 'root',
        and 'array' should match self.ar
        """
        # Load data
        loaded_data = read(
            unrooted_node_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'unrooted_array')
        lar = loaded_data.tree('unrooted_array')
        assert(isinstance(lar,Array))

        # Did the root metadata load correctly
        assert(len(lar.root.metadata.keys())==0)

        # Did the array load correctly?
        assert(lar is not self.ar_unrooted)
        assert(lar.name == self.ar_unrooted.name)
        assert(lar.units == self.ar_unrooted.units)
        assert(array_equal(lar.data,self.ar_unrooted.data))
        for i in range(len(lar.dims)):
            assert(array_equal(lar.dims[i],self.ar_unrooted.dims[i]))
            assert(lar.dim_names[i] == self.ar_unrooted.dim_names[i])
            assert(lar.dim_units[i] == self.ar_unrooted.dim_units[i])

        # Did the dim names/units pass through correctly?
        assert(lar.get_dim_name(0) == 'pony_axis')
        assert(lar.get_dim_units(0) == 'ponies')
        assert(lar.get_dim_name(1) == 'dim1')
        assert(lar.get_dim_units(1) == 'cows')



    def test_whole_tree_io(self):
        """ Should contain the full data tree:

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
        # Load data
        loaded_data = read(
            whole_tree_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lpl3 = loaded_data.tree('/pointlist3')
        lpl4 = loaded_data.tree('pointlist3/pointlist4')
        lmd3 = lpl4.metadata['metadata3']
        lar = loaded_data.tree('array')
        lmd = lar.metadata['metadata']
        lmd2 = lar.metadata['metadata2']
        lpla = loaded_data.tree('array/pointlistarray')
        lpl = loaded_data.tree('array/pointlistarray/pointlist')
        lpl2 = loaded_data.tree('array/pointlistarray/pointlist2')

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lpl3,PointList))
        assert(isinstance(lpl4,PointList))
        assert(isinstance(lmd3,Metadata))
        assert(isinstance(lar,Array))
        assert(isinstance(lmd,Metadata))
        assert(isinstance(lmd2,Metadata))
        assert(isinstance(lpla,PointListArray))
        assert(isinstance(lpl,PointList))
        assert(isinstance(lpl2,PointList))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lpl3 is not self.pl3)
        assert(lpl4 is not self.pl4)
        assert(lmd3 is not self.md3)
        assert(lar is not self.ar)
        assert(lmd is not self.md)
        assert(lmd2 is not self.md2)
        assert(lpla is not self.pla)
        assert(lpl is not self.pl)
        assert(lpl2 is not self.pl2)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        assert(array_equal(lpl3.data, self.pl3.data))
        assert(array_equal(lpl4.data, self.pl4.data))
        assert(lmd3._params == self.md3._params)
        assert(array_equal(lar.data, self.ar.data))
        assert(lmd._params == self.md._params)
        assert(lmd2._params == self.md2._params)
        assert(array_equal(lpl.data, self.pl.data))
        assert(array_equal(lpl2.data, self.pl2.data))
        for x in range(lpla.shape[0]):
            for y in range(lpla.shape[1]):
                assert(array_equal(lpla[x,y].data, self.pla[x,y].data))

        pass


    def test_whole_tree_io_treeIsFalse(self):
        """ Should contain only:

            Root                  'root'
              |-PointListArray    'pointlistarray'
        """
        # Load data
        loaded_data = read(
            whole_tree_path,
            emdpath = '/root/array/pointlistarray',
            tree = False
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lpla = loaded_data.tree('pointlistarray')

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lpla,PointListArray))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lpla is not self.pla)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        for x in range(lpla.shape[0]):
            for y in range(lpla.shape[1]):
                assert(array_equal(lpla[x,y].data, self.pla[x,y].data))

        pass


    def test_subtree_io(self):
        """ Tree should be root/pointlist3/pointlist4, plus
        metadata at root and pl4
        """
        # Load data
        loaded_data = read(
            subtree_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lpl3 = loaded_data.tree('/pointlist3')
        lpl4 = loaded_data.tree('pointlist3/pointlist4')
        lmd3 = lpl4.metadata['metadata3']

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lpl3,PointList))
        assert(isinstance(lpl4,PointList))
        assert(isinstance(lmd3,Metadata))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lpl3 is not self.pl3)
        assert(lpl4 is not self.pl4)
        assert(lmd3 is not self.md3)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        assert(array_equal(lpl3.data, self.pl3.data))
        assert(array_equal(lpl4.data, self.pl4.data))
        assert(lmd3._params == self.md3._params)

        pass



    def test_subtree_noroot_io(self):
        """ Tree should be root/pointlist4, plus
        metadata at root and pl4
        """
        # Load data
        loaded_data = read(
            subtree_noroot_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lpl4 = loaded_data.tree('pointlist4')
        lmd3 = lpl4.metadata['metadata3']

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lpl4,PointList))
        assert(isinstance(lmd3,Metadata))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lpl4 is not self.pl4)
        assert(lmd3 is not self.md3)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        assert(array_equal(lpl4.data, self.pl4.data))
        assert(lmd3._params == self.md3._params)

        pass


    def test_append_to_io(self):
        """ Tree should be
                root
                    |--array
                        |--pointlist5
        """
        # Load data
        loaded_data = read(
            subtree_append_to_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lar = loaded_data.tree('array')
        lmd = lar.metadata['metadata']
        lmd2 = lar.metadata['metadata2']
        lpl5 = loaded_data.tree('array/pointlist5')

        # Ensure the data that shouldn't be here, isn't
        assert('pointlist3' not in loaded_data._branch.keys())
        assert('array' in loaded_data._branch.keys())

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lar,Array))
        assert(isinstance(lmd,Metadata))
        assert(isinstance(lmd2,Metadata))
        assert(isinstance(lpl5,PointList))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lar is not self.ar)
        assert(lmd is not self.md)
        assert(lmd2 is not self.md2)
        assert(lpl5 is not self.pl5)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        assert(array_equal(lar.data, self.ar.data))
        assert(not(array_equal(lar.data, self.ar2.data))) # ensure we didn't overwrite
        assert(lmd._params == self.md._params)
        assert(lmd2._params == self.md2._params)
        assert(array_equal(lpl5.data, self.pl5.data))

        pass


    def test_append_over_io(self):
        """ Tree should be
                root
                    |--array
                        |--pointlist5
        where array should have no metadata and its data should
        be that of ar2 and not ar1
        """
        # Load data
        loaded_data = read(
            subtree_append_over_path
        )

        # Does the tree look as it should?
        assert(isinstance(loaded_data,Root))
        assert(loaded_data.name == 'root')

        lmd_root = loaded_data.metadata['metadata_root']
        lmd_root2 = loaded_data.metadata['metadata_root2']
        lar = loaded_data.tree('array')
        assert(len(lar.metadata)==0)
        lpl5 = loaded_data.tree('array/pointlist5')

        # Ensure the data that shouldn't be here, isn't
        assert('pointlist3' not in loaded_data._branch.keys())
        assert('array' in loaded_data._branch.keys())

        # Check types
        assert(isinstance(lmd_root,Metadata))
        assert(isinstance(lmd_root2,Metadata))
        assert(isinstance(lar,Array))
        assert(isinstance(lpl5,PointList))

        # New objects are distinct from old objects
        assert(lmd_root is not self.md_root)
        assert(lmd_root2 is not self.md_root2)
        assert(lar is not self.ar2)
        assert(lpl5 is not self.pl5)

        # New objects carry identical data to old objects
        assert(lmd_root._params == self.md_root._params)
        assert(lmd_root2._params == self.md_root2._params)
        assert(array_equal(lar.data, self.ar2.data))    # ensure we did overwrite
        assert(not(array_equal(lar.data, self.ar.data)))
        assert(array_equal(lpl5.data, self.pl5.data))

        pass






    # setup/teardown utilities

    def _write_h5_single_node(self):
        """
        Write a single node to file
        """
        save(
            single_node_path,
            self.ar,
            tree = False
        )

    def _write_h5_unrooted_node(self):
        """
        Write an unrooted node to file
        """
        save(
            unrooted_node_path,
            self.ar_unrooted
        )

    def _write_h5_whole_tree(self):
        """
        Write the whole tree to file
        """
        save(
            whole_tree_path,
            self.root,
            tree = True
        )

    def _write_h5_subtree(self):
        """
        Write a subtree to file
        """
        save(
            subtree_path,
            self.pl3,
            tree = True
        )

    def _write_h5_subtree_noroot(self):
        """
        Write a subtree tree to file without it's root dataset
        """
        save(
            subtree_noroot_path,
            self.pl3,
            tree = "noroot"
        )

    def _append_to_h5(self):
        """
        Append to an existing h5 file
        """
        save(
            subtree_append_to_path,
            self.ar,
            tree = False
        )
        save(
            subtree_append_to_path,
            self.ar2,
            tree = True,
            mode = 'a'
        )
        pass

    def _append_over_h5(self):
        """
        Append to an existing h5 file, overwriting a redundantly named object
        """
        save(
            subtree_append_over_path,
            self.ar,
            tree = False
        )
        save(
            subtree_append_over_path,
            self.ar2.root,
            tree = True,
            mode = 'ao'
        )
        pass

    def _clear_files(self):
        """
        Delete h5 files which this test suite wrote
        """
        paths = [
            single_node_path,
            unrooted_node_path,
            whole_tree_path,
            subtree_path,
            subtree_noroot_path,
            subtree_append_to_path,
            subtree_append_over_path
        ]
        for p in paths:
            if exists(p):
                remove(p)






