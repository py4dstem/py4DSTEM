import numpy as np
from os.path import join,exists
from os import remove
from numpy import array_equal

import py4DSTEM
from py4DSTEM import save,read
import emdfile as emd

from py4DSTEM import (
    Calibration,
    DiffractionSlice,
    RealSlice,
    QPoints,
    DataCube,
    VirtualImage,
    VirtualDiffraction,
    BraggVectors,
    Probe
)

# Set paths
dirpath = py4DSTEM._TESTPATH
path_dm3 = join(dirpath,"test_io/small_dm3_3Dstack.dm3")
path_h5 = join(dirpath,"test.h5")


class TestDataCubeIO():

    def test_datacube_instantiation(self):
        """
        Instantiate a datacube and apply basic calibrations
        """
        datacube = DataCube(
            data = np.arange(np.prod((4,5,6,7))).reshape((4,5,6,7))
        )
        # calibration
        datacube.calibration.set_Q_pixel_size(0.062)
        datacube.calibration.set_Q_pixel_units("A^-1")
        datacube.calibration.set_R_pixel_size(2.8)
        datacube.calibration.set_R_pixel_units("nm")

        return datacube

    def test_datacube_io(self):
        """
        Instantiate, save, then read a datacube, and
        compare its contents before/after
        """
        datacube = self.test_datacube_instantiation()

        assert(isinstance(datacube,DataCube))
        # test dim vectors
        assert(datacube.dim_names[0] == 'Rx')
        assert(datacube.dim_names[1] == 'Ry')
        assert(datacube.dim_names[2] == 'Qx')
        assert(datacube.dim_names[3] == 'Qy')
        assert(datacube.dim_units[0] == 'nm')
        assert(datacube.dim_units[1] == 'nm')
        assert(datacube.dim_units[2] == 'A^-1')
        assert(datacube.dim_units[3] == 'A^-1')
        assert(datacube.dims[0][1] == 2.8)
        assert(datacube.dims[2][1] == 0.062)
        # check the calibrations
        assert(datacube.calibration.get_Q_pixel_size() == 0.062)
        assert(datacube.calibration.get_Q_pixel_units() == "A^-1")
        # save and read
        save(path_h5,datacube,mode='o')
        new_datacube = read(path_h5)
        # check it's the same
        assert(isinstance(new_datacube,DataCube))
        assert(array_equal(datacube.data,new_datacube.data))
        assert(new_datacube.calibration.get_Q_pixel_size() == 0.062)
        assert(new_datacube.calibration.get_Q_pixel_units() == "A^-1")
        assert(new_datacube.dims[0][1] == 2.8)
        assert(new_datacube.dims[2][1] == 0.062)



class TestBraggVectorsIO():

    def test_braggvectors_instantiation(self):
        """
        Instantiate a braggvectors instance
        """
        braggvectors = BraggVectors(
            Rshape = (5,6),
            Qshape = (7,8)
        )
        for x in range(braggvectors.Rshape[0]):
            for y in range(braggvectors.Rshape[1]):
                L = int(4 * (np.sin(x*y)+1))
                braggvectors._v_uncal[x,y].add(
                    np.ones(L,dtype=braggvectors._v_uncal.dtype)
                )
        return braggvectors


    def test_braggvectors_io(self):
        """ Save then read a BraggVectors instance, and compare contents before/after
        """
        braggvectors = self.test_braggvectors_instantiation()

        assert(isinstance(braggvectors,BraggVectors))
        # save then read
        save(path_h5,braggvectors,mode='o')
        new_braggvectors = read(path_h5)
        # check it's the same
        assert(isinstance(new_braggvectors,BraggVectors))
        assert(new_braggvectors is not braggvectors)
        for x in range(new_braggvectors.shape[0]):
            for y in range(new_braggvectors.shape[1]):
                assert(array_equal(
                    new_braggvectors._v_uncal[x,y].data,
                    braggvectors._v_uncal[x,y].data))


class TestSlices:


    # test instantiation

    def test_diffractionslice_instantiation(self):
        diffractionslice = DiffractionSlice(
            data = np.arange(np.prod((4,8,2))).reshape((4,8,2)),
            slicelabels = ['a','b']
        )
        return diffractionslice

    def test_realslice_instantiation(self):
        realslice = RealSlice(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
            slicelabels = ['x','y']
        )
        return realslice

    def test_virtualdiffraction_instantiation(self):
        virtualdiffraction = VirtualDiffraction(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
        )
        return virtualdiffraction

    def test_virtualimage_instantiation(self):
        virtualimage = VirtualImage(
            data = np.arange(np.prod((8,4,2))).reshape((8,4,2)),
        )
        return virtualimage

    def test_probe_instantiation(self):
        probe = Probe(
            data = np.arange(8*12).reshape((8,12))
        )
        # add a kernel
        probe.kernel = np.ones_like(probe.probe)
        # return
        return probe


    # test io

    def test_diffractionslice_io(self):
        """ test diffractionslice io
        """
        diffractionslice = self.test_diffractionslice_instantiation()
        assert(isinstance(diffractionslice,DiffractionSlice))
        # save and read
        save(path_h5,diffractionslice,mode='o')
        new_diffractionslice = read(path_h5)
        # check it's the same
        assert(isinstance(new_diffractionslice,DiffractionSlice))
        assert(array_equal(diffractionslice.data,new_diffractionslice.data))
        assert(diffractionslice.slicelabels == new_diffractionslice.slicelabels)


    def test_realslice_io(self):
        """ test realslice io
        """
        realslice = self.test_realslice_instantiation()
        assert(isinstance(realslice,RealSlice))
        # save and read
        save(path_h5,realslice,mode='o')
        rs = read(path_h5)
        # check it's the same
        assert(isinstance(rs,RealSlice))
        assert(array_equal(realslice.data,rs.data))
        assert(rs.slicelabels == realslice.slicelabels)

    def test_virtualdiffraction_io(self):
        """ test virtualdiffraction io
        """
        virtualdiffraction = self.test_virtualdiffraction_instantiation()
        assert(isinstance(virtualdiffraction,VirtualDiffraction))
        # save and read
        save(path_h5,virtualdiffraction,mode='o')
        vd = read(path_h5)
        # check it's the same
        assert(isinstance(vd,VirtualDiffraction))
        assert(array_equal(vd.data,virtualdiffraction.data))
        pass

    def test_virtualimage_io(self):
        """ test virtualimage io
        """
        virtualimage = self.test_virtualimage_instantiation()
        assert(isinstance(virtualimage,VirtualImage))
        # save and read
        save(path_h5,virtualimage,mode='o')
        virtIm = read(path_h5)
        # check it's the same
        assert(isinstance(virtIm,VirtualImage))
        assert(array_equal(virtualimage.data,virtIm.data))
        pass


    def test_probe1_io(self):
        """ test probe io
        """
        probe0 = self.test_probe_instantiation()
        assert(isinstance(probe0,Probe))
        # save and read
        save(path_h5,probe0,mode='o')
        probe = read(path_h5)
        # check it's the same
        assert(isinstance(probe,Probe))
        assert(array_equal(probe0.data,probe.data))
        pass





class TestPoints:

    def test_qpoints_instantiation(self):
        qpoints = QPoints(
            data = np.ones(10,
                dtype = [('qx',float),('qy',float),('intensity',float)]
            )
        )
        return qpoints


    def test_qpoints_io(self):
        """ test qpoints io
        """
        qpoints0 = self.test_qpoints_instantiation()
        assert(isinstance(qpoints0,QPoints))
        # save and read
        save(path_h5,qpoints0,mode='o')
        qpoints = read(path_h5)
        # check it's the same
        assert(isinstance(qpoints,QPoints))
        assert(array_equal(qpoints0.data,qpoints.data))
        pass


