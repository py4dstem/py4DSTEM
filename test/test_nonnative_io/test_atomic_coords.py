import py4DSTEM
from py4DSTEM.process.diffraction import Crystal
from os.path import join
from pathlib import Path
from ase.io import read
from ase import Atoms
import numpy as np

# set filepaths
filepath_si_cif = join(py4DSTEM._TESTPATH, "Si.cif")
filepath_si_prismatic = join(py4DSTEM._TESTPATH, "Si.prismatic")

# occupancy for Si atoms loaded via cif
cif_occupancy = {
    "0": {"Si": 1},
    "1": {"Si": 1},
    "2": {"Si": 1},
    "3": {"Si": 1},
    "4": {"Si": 1},
    "5": {"Si": 1},
    "6": {"Si": 1},
    "7": {"Si": 1},
}
# occupancy for Si atoms loaded via prismatic
prismatic_occupancy = np.ones(8, dtype=np.float32)


def create_py4DSTEM_Si_crystal():
    """
    Create a py4DSTEM crystal using Si params for comparing to loading funcs
    """
    xtal = py4DSTEM.process.diffraction.Crystal(
        positions=[
            [0.25, 0.75, 0.25],
            [0.0, 0.0, 0.5],
            [0.25, 0.25, 0.75],
            [0.0, 0.5, 0.0],
            [0.75, 0.75, 0.75],
            [0.5, 0.0, 0.0],
            [0.75, 0.25, 0.25],
            [0.5, 0.5, 0.5],
        ],
        numbers=14,
        cell=5.44370237,
    )

    return xtal


def test_from_ase_pure():
    """
    Test Crystal.from_ase function with pure ASE object
    """
    # create a ASE Si crystal
    a = 5.44370237
    scaled_positions = [
        [0.25, 0.75, 0.25],
        [0.0, 0.0, 0.5],
        [0.25, 0.25, 0.75],
        [0.0, 0.5, 0.0],
        [0.75, 0.75, 0.75],
        [0.5, 0.0, 0.0],
        [0.75, 0.25, 0.25],
        [0.5, 0.5, 0.5],
    ]
    atoms = Atoms("Si" * 8, scaled_positions=scaled_positions, cell=[a, a, a], pbc=True)
    xtal = Crystal.from_ase(atoms)

    # create py4DSTEM Si Crystal
    xtal_py4dstem = create_py4DSTEM_Si_crystal()

    # Run Tests
    # check its a Crystal
    assert isinstance(xtal, Crystal), "xtal not an instance of Crystal"
    # occupancy should be None, as not provided
    assert xtal.occupancy is None, "occupancy is not None"
    # check the cells are close
    assert np.allclose(xtal.cell, xtal_py4dstem.cell), "unit cells do not match"
    # check the atom positions are close, all fractional coords
    # TODO this is not be the best way to check this...
    assert np.allclose(xtal.positions.sum(), xtal_py4dstem.positions.sum())


def test_from_ase_cif():
    """
    Test Crystal.from_ase function with ASE object loaded from cif file
    """
    # create a ASE Si crystal
    atoms = read(filepath_si_cif, format="cif")
    xtal = Crystal.from_ase(atoms)

    # create py4DSTEM Si Crystal
    xtal_py4dstem = create_py4DSTEM_Si_crystal()

    # Run Tests
    # check its a Crystal
    assert isinstance(xtal, Crystal), "xtal not an instance of Crystal"
    # occupancy should be the same as cif_occupancy
    assert xtal.occupancy == cif_occupancy, "occupancy is not correct"
    # check the cells are close
    assert np.allclose(xtal.cell, xtal_py4dstem.cell), "unit cells do not match"
    # check the atom positions are close, all fractional coords
    # TODO this is not be the best way to check this...
    assert np.allclose(xtal.positions.sum(), xtal_py4dstem.positions.sum())


def test_from_ase_prismatic():
    """
    Test Crystal.from_ase function with ASE object created from prismatic file
    """
    # create a ASE Si crystal
    atoms = read(filepath_si_prismatic, format="prismatic")
    xtal = Crystal.from_ase(atoms)

    # create py4DSTEM Si Crystal
    xtal_py4dstem = create_py4DSTEM_Si_crystal()

    # Run Tests
    # check its a Crystal
    assert isinstance(xtal, Crystal), "xtal not an instance of Crystal"
    # occupancy should be the same as  prismatic occupancy
    assert np.all(xtal.occupancy == prismatic_occupancy), "occupancy is not correct"
    # check the cells are close
    assert np.allclose(xtal.cell, xtal_py4dstem.cell), "unit cells do not match"
    # check the atom positions are close, all fractional coords
    # TODO this is not be the best way to check this...
    assert np.allclose(xtal.positions.sum(), xtal_py4dstem.positions.sum())


def test_from_cif():
    """
    Test Crystal.from_cif function
    """

    # load cif
    xtal = Crystal.from_cif(filepath_si_cif)
    # create py4DSTEM Si Crystal
    xtal_py4dstem = create_py4DSTEM_Si_crystal()

    # Run Tests
    # check its a Crystal
    assert isinstance(xtal, Crystal), "xtal not an instance of Crystal"
    # occupancy should not be == cif_occupancy
    assert xtal.occupancy == cif_occupancy, "occupancy is not correct"
    # check the cells are close
    assert np.allclose(xtal.cell, xtal_py4dstem.cell), "unit cells do not match"
    # check the atom positions are close, all fractional coords
    # TODO this is not be the best way to check this...
    assert np.allclose(xtal.positions.sum(), xtal_py4dstem.positions.sum())


def test_from_prismatic():
    """
    Test Crystal.from_prismatic function
    """

    # load cif
    xtal = Crystal.from_prismatic(filepath_si_prismatic)
    # create py4DSTEM Si Crystal
    xtal_py4dstem = create_py4DSTEM_Si_crystal()

    # Run Tests
    # check its a Crystal
    assert isinstance(xtal, Crystal), "xtal not an instance of Crystal"
    # occupancy should be == prismatic_occupancy
    assert xtal.occupancy is prismatic_occupancy, "occupancy is not correct"
    # check the cells are close
    assert np.allclose(xtal.cell, xtal_py4dstem.cell), "unit cells do not match"
    # check the atom positions are close, all fractional coords
    # TODO this is not be the best way to check this...
    assert np.allclose(xtal.positions.sum(), xtal_py4dstem.positions.sum())
