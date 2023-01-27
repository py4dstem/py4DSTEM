import numpy as np
from os.path import join,exists
from os import remove
from numpy import array_equal

import py4DSTEM
from py4DSTEM.io import save,read
from py4DSTEM.io.read import _is_EMD_file,_get_EMD_rootgroups
from py4DSTEM.io.classes import (
    Node,
    Root,
    Metadata,
    Array,
    PointList,
    PointListArray
)
from py4DSTEM.io.classes.py4dstem import (
    DataCube,
    BraggVectors
)

# Set paths
dirpath = py4DSTEM._TESTPATH
path_dm3 = join(dirpath,"small_dm3.dm3")
path_h5 = join(dirpath,"test.h5")





class TestPy4dstem:

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
            - a datacube
            - a braggvectors instance with only v_uncal
            - a braggvectors instance with both PLAs
            - a diffractionslice
            - a realslice
            - a probe, with no kernel
            - a probe, with a kernel
            - a qpoints instance
            - a virtualdiffraction instance
            - a virtualimage instance
        """
        # datacube
        self.datacube = DataCube(
            data = np.arange(np.prod((4,5,6,7))).reshape((4,5,6,7))
        )
        # braggvectors
        self.braggvectors = BraggVectors(
            Rshape = (5,6),
            Qshape = (7,8)
        )
        for x in range(self.braggvectors.Rshape[0]):
            for y in range(self.braggvectors.Rshape[1]):
                L = int(4 * (np.sin(x*y)+1))
                self.braggvectors.vectors_uncal[x,y].add(
                    np.ones(L,dtype=self.braggvectors.vectors_uncal.dtype)
                )
        # braggvectors 2
        self.braggvectors2 = BraggVectors(
            Rshape = (5,6),
            Qshape = (7,8)
        )
        for x in range(self.braggvectors2.Rshape[0]):
            for y in range(self.braggvectors2.Rshape[1]):
                L = int(4 * (np.sin(x*y)+1))
                self.braggvectors2.vectors_uncal[x,y].add(
                    np.ones(L,dtype=self.braggvectors2.vectors_uncal.dtype)
                )
        self.braggvectors2._v_cal = self.braggvectors2._v_uncal.copy(name='_v_uncal')

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

    def test_datacube(self):
        """ Save then read a datacube, and compare its contents before/after
        """
        assert(isinstance(self.datacube,DataCube))
        # save and read
        save(path_h5,self.datacube)
        root = read(path_h5)
        new_datacube = root.tree('datacube')
        # check it's the same
        assert(isinstance(new_datacube,DataCube))
        assert(array_equal(self.datacube.data,new_datacube.data))


    def test_braggvectors(self):
        """ Save then read a BraggVectors instance, and compare contents before/after
        """
        assert(isinstance(self.braggvectors,BraggVectors))
        # save then read
        save(path_h5,self.braggvectors)
        root = read(path_h5)
        new_braggvectors = root.tree('braggvectors')
        # check it's the same
        assert(isinstance(new_braggvectors,BraggVectors))
        assert(new_braggvectors is not self.braggvectors)
        for x in range(new_braggvectors.shape[0]):
            for y in range(new_braggvectors.shape[1]):
                assert(array_equal(
                    new_braggvectors.v_uncal[x,y].data,
                    self.braggvectors.v_uncal[x,y].data))
        # check that _v_cal isn't there
        assert(not(hasattr(new_braggvectors,'_v_cal')))


    def test_braggvectors2(self):
        """ Save then read a BraggVectors instance, and compare contents before/after
        """
        assert(isinstance(self.braggvectors2,BraggVectors))
        # save then read
        save(path_h5,self.braggvectors2)
        root = read(path_h5)
        new_braggvectors = root.tree('braggvectors')
        # check it's the same
        assert(isinstance(new_braggvectors,BraggVectors))
        assert(new_braggvectors is not self.braggvectors2)
        for x in range(new_braggvectors.shape[0]):
            for y in range(new_braggvectors.shape[1]):
                assert(array_equal(
                    new_braggvectors.v_uncal[x,y].data,
                    self.braggvectors2.v_uncal[x,y].data))
        # check cal vectors are there
        assert(hasattr(new_braggvectors,'_v_cal'))
        # check it's the same
        for x in range(new_braggvectors.shape[0]):
            for y in range(new_braggvectors.shape[1]):
                assert(array_equal(
                    new_braggvectors.v[x,y].data,
                    self.braggvectors2.v[x,y].data))






