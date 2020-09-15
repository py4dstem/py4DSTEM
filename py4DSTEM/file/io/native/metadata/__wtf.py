# Zoink!

from .parameters import Param
from .h5rw import h5write, h5read, h5append, h5info

class Metadata(Param):
    """
    An object for storing, reading, and writing metadata to/from .h5 files.
    """
    def __init__(self):
        """
        """
        self.microscope = 1
        self.sample = 2
        self.user = 3
        self.calibration = 4
        self.comments = 5






