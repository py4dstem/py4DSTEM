
from py4DSTEM.data import DiffractionSlice, Data
from py4DSTEM.classes.methods import ProbeMethods

from typing import Optional
import numpy as np

class Probe(DiffractionSlice,ProbeMethods,Data):
    """
    Stores a vacuum probe.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'probe'
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
            data = np.stack([
                data,
                np.zeros_like(data)
            ])

        # initialize as a ProbeMethods instance
        super(ProbeMethods).__init__()

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


    ## properties

    @property
    def probe(self):
        return self.get_slice('probe').data
    @probe.setter
    def probe(self,x):
        assert(x.shape == (self.data.shape[1:]))
        self.data[0,:,:] = x
    @property
    def kernel(self):
        return self.get_slice('kernel').data
    @kernel.setter
    def kernel(self,x):
        assert(x.shape == (self.data.shape[1:]))
        self.data[1,:,:] = x




    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = DiffractionSlice._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
        }
        return args




