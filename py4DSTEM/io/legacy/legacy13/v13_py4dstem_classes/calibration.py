# Defines the Calibration class, which stores calibration metadata

from typing import Optional
from py4DSTEM.io.legacy.legacy13.v13_emd_classes.metadata import Metadata


class Calibration(Metadata):
    """ """

    def __init__(
        self,
        name: Optional[str] = "calibration",
    ):
        """
        Args:
            name (optional, str):
        """
        Metadata.__init__(self, name=name)

        self.set_Q_pixel_size(1)
        self.set_R_pixel_size(1)
        self.set_Q_pixel_units("pixels")
        self.set_R_pixel_units("pixels")

    def set_Q_pixel_size(self, x):
        self._params["Q_pixel_size"] = x

    def get_Q_pixel_size(self):
        return self._get_value("Q_pixel_size")

    def set_R_pixel_size(self, x):
        self._params["R_pixel_size"] = x

    def get_R_pixel_size(self):
        return self._get_value("R_pixel_size")

    def set_Q_pixel_units(self, x):
        pix = ("pixels", "A^-1", "mrad")
        assert x in pix, f"{x} must be in {pix}"
        self._params["Q_pixel_units"] = x

    def get_Q_pixel_units(self):
        return self._get_value("Q_pixel_units")

    def set_R_pixel_units(self, x):
        self._params["R_pixel_units"] = x

    def get_R_pixel_units(self):
        return self._get_value("R_pixel_units")

    # HDF5 read/write

    # write inherited from Metadata

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import (
            Calibration_from_h5,
        )

        return Calibration_from_h5(group)


########## End of class ##########
