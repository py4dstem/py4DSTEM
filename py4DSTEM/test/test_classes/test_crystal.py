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
        self.braggpeaks = read(
            filepath_braggpeaks,
            data_id='braggpeaks_cal_raw'
        )
        self.crystal = Crystal.from_CIF(filepath_cif1)
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


    def test_generate_diffraction_pattern(self):

        self.crystal.generate_diffraction_pattern(
            zone_axis_lattice = [1,1,2],
            sigma_excitation_error = 0.2
        )






