from py4DSTEM.process.diffraction import Crystal
from py4DSTEM.process.diffraction import CrystalPhase
from py4DSTEM import _TESTPATH,read
from os.path import join


# Set filepaths
filepath_braggpeaks = join(_TESTPATH, "crystal/braggdisks_cali.h5")
filepath_cif1 = join(_TESTPATH, "crystal/LCO.cif")
filepath_cif2 = join(_TESTPATH, "crystal/LiMn2O4.cif")




class TestCrystal:

    def setup_class(self):

        # get bragg peaks
        self.braggpeaks = read(
            filepath_braggpeaks,
            data_id='braggpeaks_cal_raw'
        )

        # make a Crystal
        self.crystal = Crystal.from_CIF(filepath_cif1)

        # get structure factors
        self.q, self.inten = self.crystal.calculate_structure_factors(
            k_max = 1.7,
            tol_structure_factor = 0.1,
            return_intensities = True
        )

        # set up the orientation plan
        self.crystal.orientation_plan(
            angle_step_zone_axis=1.0,
            angle_step_in_plane=5.0,
            accel_voltage=300e3,
            zone_axis_range='fiber',
            fiber_axis=self.crystal.hexagonal_to_lattice([1,0,-1,0]),
            fiber_angles=[5,90],
            intensity_power=2.5,
        )

    def teardown_class(self):
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

    def test_match_single_pattern(self):

        xind,yind = 30,40

        # match the pattern
        orientation = self.crystal.match_single_pattern(
            self.braggpeaks[xind,yind],
            num_matches_return=1,
            verbose=True
        )

        # compute the predicted peaks at this orientation
        braggpeaks_fit = self.crystal.generate_diffraction_pattern(
            orientation,
            sigma_excitation_error=0.02
        )


    def test_match_orientations(self):

        orientation_map = self.crystal.match_orientations(
            self.braggpeaks
        )






class TestPhaseMapping:

    def setup_class(self):

        # get bragg peaks
        self.braggpeaks = read(
            filepath_braggpeaks,
            data_id='braggpeaks_cal_raw'
        )

        # make Crystals
        self.crystal1 = Crystal.from_CIF(filepath_cif1)
        self.crystal2 = Crystal.from_CIF(filepath_cif2)

        # get structure factors
        self.crystal1.calculate_structure_factors(
            k_max = 1.7,
            tol_structure_factor = 0.1,
            return_intensities = False
        )
        self.crystal2.calculate_structure_factors(
            k_max = 1.7,
            tol_structure_factor = 0.1,
            return_intensities = False
        )

        # set up orientation plans
        self.crystal1.orientation_plan(
            angle_step_zone_axis=1.0,
            angle_step_in_plane=5.0,
            accel_voltage=300e3,
            zone_axis_range='fiber',
            fiber_axis=self.crystal1.hexagonal_to_lattice([1,0,-1,0]),
            fiber_angles=[5,90],
            intensity_power=2.5,
        )
        self.crystal2.orientation_plan(
            angle_step_zone_axis=1.0,
            angle_step_in_plane=5.0,
            accel_voltage=300e3,
            zone_axis_range='fiber',
            fiber_axis=[1,1,2],
            fiber_angles=[5,180],
            intensity_power=2.5,
        )

        # get orientation maps
        self.orientation_map1 = self.crystal1.match_orientations(
            self.braggpeaks
        )
        self.orientation_map2 = self.crystal2.match_orientations(
            self.braggpeaks
        )


    def teardown_class(self):
        pass

    def setup_method(self):
        pass

    def teardown_method(self):
        pass



    def test_crystal_phase(self):

        # make the CrystalPhase instance
        self.crystal_phase = CrystalPhase(
            name = 'pristine_spinel_phases',
            crystals = [
                self.crystal1,
                self.crystal2
            ]
        )

        # quantify the phases
        self.crystal_phase.quantify_phase(
            self.braggpeaks,
            tolerance_distance = 0.035,
            method = 'nnls',
            intensity_power = 0,
        )


