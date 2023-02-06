from py4DSTEM.process.diffraction import Crystal
from py4DSTEM import _TESTPATH
from os.path import join


# Set filepaths
filepath_braggpeaks = join(_TESTPATH, "crystal/braggpeaks_cali.h5")
filepath_cif1 = join(_TESTPATH, "crystal/LCO.cif")
filepath_cif2 = join(_TESTPATH, "crystal/Li2MnO3.cif")
filepath_cif3 = join(_TESTPATH, "crystal/LiMn2O4.cif")




class TestCrystal:

    def setup_cls(self):
        pass

    def teardown_cls(self):
        pass

    def setup_method(self):
        pass

    def teardown_method(self):
        pass



    def test_instantiation_from_cif(self):

        crystal = Crystal.from_CIF(filepath_cif1)
        assert(isinstance(crystal,Crystal))

        pass


    def test_Crystal2(self):

        #crystal = Crystal( **args )
        #assert(isinstance(crystal,Crystal))

        pass



