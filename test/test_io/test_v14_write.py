import numpy as np
from os.path import join,exists
from os import remove
from numpy import array_equal

import py4DSTEM
from py4DSTEM import save,read
emd = py4DSTEM.emd

from py4DSTEM.classes import (
    DataCube,
    BraggVectors,
    DiffractionSlice,
    VirtualDiffraction,
    Probe,
    RealSlice,
    VirtualImage,
    QPoints,
    Calibration
)

# Set paths
dirpath = py4DSTEM._TESTPATH
path_dm3 = join(dirpath,"small_dm3.dm3")
path_h5 = join(dirpath,"test.h5")




class TestSingleDataNodeIO:

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
        # calibration
        self.datacube.calibration.set_Q_pixel_size(0.062)
        self.datacube.calibration.set_Q_pixel_units("A^-1")
        self.datacube.calibration.set_R_pixel_size(2.8)
        self.datacube.calibration.set_R_pixel_units("nm")
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
        # diffractionslice
        self.diffractionslice = DiffractionSlice(
            data = np.arange(np.prod((4,8,2))).reshape((4,8,2)),
            slicelabels = ['a','b']
        )
        # realslice
        self.realslice = RealSlice(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
            slicelabels = ['x','y']
        )
        # virtualdiffraction instance
        self.virtualdiffraction = VirtualDiffraction(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
        )
        # virtualimage instance
        self.virtualimage = VirtualImage(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
        )
        # probe, with no kernel
        self.probe1 = Probe(
            data = np.arange(8*12).reshape((8,12))
        )
        # probe, with a kernel
        self.probe2 = Probe(
            data = np.arange(8*12*2).reshape((8,12,2))
        )
        # qpoints instance
        self.qpoints = QPoints(
            data = np.ones(10,
                dtype = [('qx',float),('qy',float),('intensity',float)]
            )
        )









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
        # test dim vectors
        assert(self.datacube.dim_names[0] == 'Rx')
        assert(self.datacube.dim_names[1] == 'Ry')
        assert(self.datacube.dim_names[2] == 'Qx')
        assert(self.datacube.dim_names[3] == 'Qy')
        assert(self.datacube.dim_units[0] == 'nm')
        assert(self.datacube.dim_units[1] == 'nm')
        assert(self.datacube.dim_units[2] == 'A^-1')
        assert(self.datacube.dim_units[3] == 'A^-1')
        assert(self.datacube.dims[0][1] == 2.8)
        assert(self.datacube.dims[2][1] == 0.062)
        # check the calibrations
        assert(self.datacube.calibration.get_Q_pixel_size() == 0.062)
        assert(self.datacube.calibration.get_Q_pixel_units() == "A^-1")
        # save and read
        save(path_h5,self.datacube)
        root = read(path_h5)
        new_datacube = root.tree('datacube')
        # check it's the same
        assert(isinstance(new_datacube,DataCube))
        assert(array_equal(self.datacube.data,new_datacube.data))
        assert(new_datacube.calibration.get_Q_pixel_size() == 0.062)
        assert(new_datacube.calibration.get_Q_pixel_units() == "A^-1")
        assert(new_datacube.dims[0][1] == 2.8)
        assert(new_datacube.dims[2][1] == 0.062)




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


    def test_diffractionslice(self):
        """ test diffractionslice io
        """
        assert(isinstance(self.diffractionslice,DiffractionSlice))
        # save and read
        save(path_h5,self.diffractionslice)
        root = read(path_h5)
        new_diffractionslice = root.tree('diffractionslice')
        # check it's the same
        assert(isinstance(new_diffractionslice,DiffractionSlice))
        assert(array_equal(self.diffractionslice.data,new_diffractionslice.data))
        assert(new_diffractionslice.slicelabels == self.diffractionslice.slicelabels)


    def test_realslice(self):
        """ test realslice io
        """
        assert(isinstance(self.realslice,RealSlice))
        # save and read
        save(path_h5,self.realslice)
        root = read(path_h5)
        new_realslice = root.tree('realslice')
        # check it's the same
        assert(isinstance(new_realslice,RealSlice))
        assert(array_equal(self.realslice.data,new_realslice.data))
        assert(new_realslice.slicelabels == self.realslice.slicelabels)

    def test_virtualdiffraction(self):
        """ test virtualdiffraction io
        """
        assert(isinstance(self.virtualdiffraction,VirtualDiffraction))
        # save and read
        save(path_h5,self.virtualdiffraction)
        root = read(path_h5)
        new_virtualdiffraction = root.tree('virtualdiffraction')
        # check it's the same
        assert(isinstance(new_virtualdiffraction,VirtualDiffraction))
        assert(array_equal(self.virtualdiffraction.data,new_virtualdiffraction.data))
        pass

    def test_virtualimage(self):
        """ test virtualimage io
        """
        assert(isinstance(self.virtualimage,VirtualImage))
        # save and read
        save(path_h5,self.virtualimage)
        root = read(path_h5)
        new_virtualimage = root.tree('virtualimage')
        # check it's the same
        assert(isinstance(new_virtualimage,VirtualImage))
        assert(array_equal(self.virtualimage.data,new_virtualimage.data))
        pass

    def test_probe1(self):
        """ test probe io
        """
        assert(isinstance(self.probe1,Probe))
        # check for an empty kernel array
        assert(array_equal(self.probe1.kernel,np.zeros_like(self.probe1.probe)))
        # save and read
        save(path_h5,self.probe1)
        root = read(path_h5)
        new_probe1 = root.tree('probe')
        # check it's the same
        assert(isinstance(new_probe1,Probe))
        assert(array_equal(self.probe1.data,new_probe1.data))
        pass

    def test_probe2(self):
        """ test probe io
        """
        assert(isinstance(self.probe2,Probe))
        # save and read
        save(path_h5,self.probe2)
        root = read(path_h5)
        new_probe2 = root.tree('probe')
        # check it's the same
        assert(isinstance(new_probe2,Probe))
        assert(array_equal(self.probe2.data,new_probe2.data))
        pass


    def test_qpoints(self):
        """ test qpoints io
        """
        assert(isinstance(self.qpoints,QPoints))
        # save and read
        save(path_h5,self.qpoints)
        root = read(path_h5)
        new_qpoints = root.tree('qpoints')
        # check it's the same
        assert(isinstance(new_qpoints,QPoints))
        assert(array_equal(self.qpoints.data,new_qpoints.data))
        pass






