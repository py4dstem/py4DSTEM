# Defines the RealSlice class, which stores 2(+1)D real-space shaped data

from py4DSTEM.io.classes.array import Array

from typing import Optional,Union
import numpy as np
import h5py


# TODO: sync dim vectors with calibration
# TODO: add calibration class which these can all inherit from (class Data ?)


class RealSlice(Array):
    """
    Stores a real-space shaped 2D data array.
    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'realslice',
        units: Optional[str] = 'intensity',
        slicelabels: Optional[Union[bool,list]] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the realslice
            slicelabels(None or list): names for slices if this is a stack of
                realslices

        Returns:
            A new RealSlice instance
        """
        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = 'intensity',
            #dims = [
            #    pixel_size[0],
            #    pixel_size[1],
            #],
            #dim_units = [
            #    pixel_units[0],
            #    pixel_units[1],
            #],
            #dim_names = [
            #    'Rx',
            #    'Ry'
            #],
            slicelabels = slicelabels
        )


    # read
    @classmethod
    def _get_constructor_args(cls,group):
        """
        Returns a dictionary of args/values to pass to the class constructor
        """
        ar_constr_args = Array._get_constructor_args(group)
        args = {
            'data' : ar_constr_args['data'],
            'name' : ar_constr_args['name'],
            'units' : ar_constr_args['units'],
            'slicelabels' : ar_constr_args['slicelabels']
        }
        return args








