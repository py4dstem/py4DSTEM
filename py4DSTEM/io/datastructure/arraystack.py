# Defines the DataCube class, which stores 4D-STEM datacubes

from .array import Array, Array_from_h5

from typing import Optional,Union
import numpy as np
import h5py

class ArrayStack(Array):
    """
    Stores an (N+1)-dimensional array, where the first N dimensions are each
    Array objects, and the last dimension is a set of names, which are string-like
    keys enabling access to the N-D slices.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'arraystack',
        units: Optional[str] = '',
        dims: Optional[list] = None,
        dim_names: Optional[list] = None,
        dim_units: Optional[list] = None,
        labels: Optional[list] = None
        ):
        """
        Accepts:
            data (np.ndarray): the (N+1)-dimensional data
            name (str): the name of the Array
            units (str): units for the pixel values
            dims (list): calibration vectors for each of the first N axes of
                the array.  See the Array.__init__ docstring for more details.
            dim_units (list): the units for the calibration dim vectors. See
                the Array.__init__ docstring for more details.
            dim_names (list): labels for each of the dim vectors. See the
                Array.__init__ docstring for more details.
            labels (list): strings which label the final dimension of
                the data array

        Returns:
            A new ArrayStack instance
        """
        # Get number of Arrays
        self.depth = data.shape[-1]

        # Populate labels
        if labels is None:
            labels = np.arange(self.depth).astype(str)
        elif len(labels) < self.depth:
            labels = np.concatenate((labels,
                [f'array{i}' for i in range(len(labels),self.depth)]))
        else:
            labels = labels[:self.depth]
        self.labels = Labels(labels)

        # initialize as an Array
        Array.__init__(
            self,
            data = data,
            name = name,
            units = units,
            dims = dims,
            dim_units = dim_units,
            dim_names = dim_names
        )

        # Set labels dim
        self.set_dim(
            n = self.rank-1,
            dim = labels,
            units = 'arrays',
            name = 'labels'
        )


    def get_data(self,label,name=None):
        idx = self.labels._dict[label]
        return Array(
            data = self.data[..., idx],
            name = name if name is not None else self.name+'_'+label,
            units = self.units[:-1],
            dims = self.dims[:-1],
            dim_units = self.dim_units[:-1],
            dim_names = self.dim_names[:-1]
        )


    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A set of {self.depth} Arrays with {self.rank-1}-dimensions and shape {self.shape[:-1]}, called '{self.name}'"
        string += "\n"
        string += "\n" +space + "The labels are:"
        for label in self.labels:
            string += "\n" + space + f"    {label}"
        string += "\n"
        string += "\n"
        string += "\n" + space + "The Array dimensions are:"
        for n in range(self.rank-1):
            string += "\n"+space+f"    {self.dim_names[n]} = [{self.dims[n][0]},{self.dims[n][1]},...] {self.dim_units[n]}"
            if not self.dim_is_linear[n]:
                string += "  (*non-linear*)"
        string += "\n)"
        return string


# List subclass for accessing data slices with a dict
class Labels(list):
    def __init__(self,x=[]):
        list.__init__(self,x)
        self.setup_labels_dict()
    def __setitem__(self,idx,label):
        list.__setitem__(self,idx,label)
        self.setup_labels_dict()

    def setup_labels_dict(self):
        self._dict = {}
        for idx,label in enumerate(self):
            self._dict[label] = idx



# Reading

def Slice2D_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't, exist, or if
    it exists but does not have 4 dimensions, raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A DataCube instance
    """
    datacube = Array_from_h5(group, name)
    datacube = DataCube_from_Array(datacube)
    return datacube


def Slice2D_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    assert(array.rank in (2,3)), "Array must have 2 or 3 dimensions"
    array.__class__ = DataCube
    array.__init__(
        data = array.data,
        name = array.name,
        rsize = [array.dims[0][1]-array.dims[0][0],
                 array.dims[1][1]-array.dims[1][0]],
        runits = [array.dim_units[0],
                  array.dim_units[1]],
        qsize = [array.dims[2][1]-array.dims[2][0],
                 array.dims[3][1]-array.dims[3][0]],
        qunits = [array.dim_units[2],
                  array.dim_units[3]]
    )
    return array



