# Defines the Probe class, which stores vacuum probes
# and cross-correlation kernels derived from them

from py4DSTEM.io.legacy.legacy13.v13_emd_classes.array import Array, Metadata
from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.diffractionslice import (
    DiffractionSlice,
)

from typing import Optional, Union
import numpy as np
import h5py


class Probe(DiffractionSlice):
    """
    Stores a vacuum probe.
    """

    def __init__(self, data: np.ndarray, name: Optional[str] = "probe", **kwargs):
        """
        Accepts:
            data (2D or 3D np.ndarray): the vacuum probe, or
                the vacuum probe + kernel
            name (str): a name
        Returns:
            (Probe)
        """
        # if only the probe is passed, make space for the kernel
        if data.ndim == 2:
            data = np.dstack([data, np.zeros_like(data)])

        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self, name=name, data=data, slicelabels=["probe", "kernel"]
        )

        # Set metadata
        md = Metadata(name="probe")
        for k, v in kwargs.items():
            md[k] = v
        self.metadata = md

    ## properties

    @property
    def probe(self):
        return self.get_slice("probe").data

    @probe.setter
    def probe(self, x):
        assert x.shape == (self.data.shape[:2])
        self.data[:, :, 0] = x

    @property
    def kernel(self):
        return self.get_slice("kernel").data

    @kernel.setter
    def kernel(self, x):
        assert x.shape == (self.data.shape[:2])
        self.data[:, :, 1] = x

    # HDF5 read/write

    # write inherited from Array

    # read
    def from_h5(group):
        from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes.io import Probe_from_h5

        return Probe_from_h5(group)


############ END OF CLASS ###########
