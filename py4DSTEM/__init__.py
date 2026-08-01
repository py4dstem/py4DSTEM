from py4DSTEM.version import __version__
from emdfile import tqdmnd

# matplotlib < 3.11 mathtext calls pyparsing's deprecated camelCase API
# (parseString / parseAll / resetCache) once per rendered math label, which
# floods plotting output with deprecation messages under pyparsing >= 3.3.
# Rebind those compatibility aliases to call the snake_case API directly with
# the renamed arguments, so the deprecated code paths are never executed.
# Upgrading matplotlib instead is not an option in hosted notebooks (Colab
# preimports matplotlib, and a mid-session upgrade leaves a broken half-old,
# half-new install). Skipped entirely on matplotlib >= 3.11.
import matplotlib as _matplotlib
import pyparsing as _pyparsing

if getattr(_matplotlib, "__version_info__", (99,)) < (3, 11) and hasattr(
    _pyparsing.ParserElement, "parse_string"
):

    def _parse_string_compat(self, instring, parseAll=False, *, parse_all=None):
        return self.parse_string(
            instring, parse_all=parseAll if parse_all is None else parse_all
        )

    _pyparsing.ParserElement.parseString = _parse_string_compat
    _pyparsing.ParserElement.resetCache = classmethod(lambda cls: cls.reset_cache())

from importlib.metadata import packages_distributions

is_package_lite = "py4DSTEM-lite" in packages_distributions()["py4DSTEM"]

### io

# substructure
from emdfile import (
    Node,
    Root,
    Metadata,
    Array,
    PointList,
    PointListArray,
    Custom,
    print_h5_tree,
)

_emd_hook = True

# structure
from py4DSTEM import io
from py4DSTEM.io import import_file, read, save

### basic data classes

# data
from py4DSTEM.data import (
    Data,
    Calibration,
    DiffractionSlice,
    RealSlice,
    QPoints,
)

# datacube
from py4DSTEM.datacube import DataCube, VirtualImage, VirtualDiffraction

### visualization

from py4DSTEM import visualize
from py4DSTEM.visualize import show, show_complex

### analysis classes

# braggvectors
from py4DSTEM.braggvectors import (
    Probe,
    BraggVectors,
    BraggVectorMap,
)

try:
    from py4DSTEM.process import classification
except (ImportError, ModuleNotFoundError) as exc:
    if not is_package_lite:
        raise exc

# diffraction
from py4DSTEM.process.diffraction import Crystal, Orientation

# ptycho
from py4DSTEM.process import phase

# polar
from py4DSTEM.process.polar import PolarDatacube

# strain
from py4DSTEM.process.strain.strain import StrainMap

try:
    from py4DSTEM.process import wholepatternfit
except (ImportError, ModuleNotFoundError) as exc:
    if not is_package_lite:
        raise exc


### more submodules
# TODO

from py4DSTEM import preprocess
from py4DSTEM import process

### utilities

# config
from py4DSTEM.utils.configuration_checker import check_config

# TODO - config .toml

# testing
from os.path import dirname, join

_TESTPATH = join(dirname(__file__), "../test/unit_test_data")
