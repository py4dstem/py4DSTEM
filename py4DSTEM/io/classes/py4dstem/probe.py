# Defines the Probe class, which stores vacuum probes
# and cross-correlation kernels derived from them

from py4DSTEM.io.classes.array import Array, Metadata
from py4DSTEM.io.classes.py4dstem.diffractionslice import DiffractionSlice

from typing import Optional,Union
import numpy as np
import h5py

class Probe(DiffractionSlice):
    """
    Stores a vacuum probe.
    """

    from py4DSTEM.io.classes.py4dstem.probe_fns import (
        get_kernel
    )

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'probe',
        **kwargs
        ):
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
            data = np.dstack([
                data,
                np.zeros_like(data)
            ])

        # initialize as a DiffractionSlice
        DiffractionSlice.__init__(
            self,
            name = name,
            data = data,
            slicelabels = [
                'probe',
                'kernel'
            ]
        )

        # Set metadata
        md = Metadata(name='probe')
        for k,v in kwargs.items():
            md[k] = v
        self.metadata = md



    ## properties

    @property
    def probe(self):
        return self.get_slice('probe').data
    @probe.setter
    def probe(self,x):
        assert(x.shape == (self.data.shape[:2]))
        self.data[:,:,0] = x
    @property
    def kernel(self):
        return self.get_slice('kernel').data
    @kernel.setter
    def kernel(self,x):
        assert(x.shape == (self.data.shape[:2]))
        self.data[:,:,1] = x





    # HDF5 i/o

    # write inherited from Array

    # read
    def from_h5(group):
        """
        Takes a valid group for an HDF5 file object which is open in
        read mode. Determines if it's a valid Array, and if so loads and
        returns it as a Probe. Otherwise, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A Probe instance
        """
        # Load from H5 as an Array
        probe = Array.from_h5(group)

        # Convert to a Probe

        assert(array.rank == 2), "Array must have 2 dimensions"

        # get diffraction image metadata
        try:
            md = array.metadata['probe']
            kwargs = {}
            for k in md.keys:
                v = md[k]
                kwargs[k] = v
        except KeyError:
            er = "Probe metadata could not be found"
            raise Exception(er)

        # instantiate as a Probe
        array.__class__ = Probe
        array.__init__(
            data = array.data,
            name = array.name,
            **kwargs
        )

        # Return
        return array






############ END OF CLASS ###########










