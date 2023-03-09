import numpy as np
from os.path import join,exists
from os import remove
from numpy import array_equal

import py4DSTEM
emd = py4DSTEM.emd
_TESTPATH = py4DSTEM._TESTPATH

# Set paths
dirpath = _TESTPATH
path_h5 = join(dirpath,"test.h5")





class TestEmd:

    ## Setup and teardown

    @classmethod
    def setup_class(cls):
        cls._clear_files(cls)
        cls._make_data(cls)

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        self._clear_files()

    def _make_data(self):
        """ Make
            - an array
            - an array with some 'slicelabels'
            #- a pointlist
            #- a pointlistarray
        """
        # arrays
        self.array = emd.Array(
            data = np.arange(np.prod((4,8))).reshape((4,8))
        )
        self.array2 = emd.Array(
            data = np.arange(np.prod((4,8,2))).reshape((4,8,2)),
            name = 'array2',
            units = 'meowths',
            dims = [
                [0,2],
                [0,2]
            ],
            dim_units = [
                'earthmiles',
                'pokemonmiles'
            ],
            dim_names = [
                'pokemonblue',
                'pokemonred'
            ],
            slicelabels = [
                'a',
                'b'
            ]
        )

        # pointlist
        # pointlistarray





    def _clear_files(self):
        """
        Delete h5 files which this test suite wrote
        """
        paths = [
            path_h5
        ]
        for p in paths:
            if exists(p):
                remove(p)





    ## Tests

    def test_array(self):
        """ Save then read as array, and compare its contents before/after
        """
        assert(isinstance(self.array,emd.Array))
        # save and read
        print(path_h5)
        emd.save(path_h5,self.array)
        root = emd.read(path_h5)
        new_array = root.tree('array')
        # check it's the same
        assert(isinstance(new_array,emd.Array))
        assert(array_equal(self.array.data,new_array.data))

    def test_array2(self):
        """ Save then read as array, and compare its contents before/after
        """
        assert(isinstance(self.array2,emd.Array))
        # save and read
        emd.save(path_h5,self.array2)
        root = emd.read(path_h5)
        new_array = root.tree('array2')
        # check it's the same
        assert(isinstance(new_array,emd.Array))
        assert(array_equal(self.array2.data,new_array.data))
        # check that metadata passed through
        assert(self.array2.name == new_array.name)
        assert(self.array2.units == new_array.units)
        assert(array_equal(self.array2.dims[0],new_array.dims[0]))
        assert(array_equal(self.array2.dims[1],new_array.dims[1]))
        for i in range(2):
            assert(self.array2.dim_units[i] == new_array.dim_units[i])
            assert(self.array2.dim_names[i] == new_array.dim_names[i])
            assert(self.array2.slicelabels[i] == new_array.slicelabels[i])

