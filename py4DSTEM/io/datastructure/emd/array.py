# Defines the Array class, which stores any N-dimensional array-like data.
# Implements the EMD file standard - https://emdatasets.com/format

from typing import Optional,Union
import numpy as np
import h5py
from numbers import Number

from .tree import Tree
from .metadata import Metadata

class Array:
    """
    A class which stores any N-dimensional array-like data, plus basic metadata:
    a name and units, as well as calibrations for each axis of the array, and names
    and units for those axis calibrations.

    In the simplest usage, only a data array is passed:

    >>> ar = Array(np.ones((20,20,256,256)))

    will create an array instance whose data is the numpy array passed, and with
    automatically populated dimension calibrations in units of pixels.

    Additional arguments may be passed to populate the object metadata:

    >>> ar = Array(
    >>>     np.ones((20,20,256,256)),
    >>>     name = 'test_array',
    >>>     units = 'intensity',
    >>>     dims = [
    >>>         [0,5],
    >>>         [0,5],
    >>>         [0,0.01],
    >>>         [0,0.01]
    >>>     ],
    >>>     dim_units = [
    >>>         'nm',
    >>>         'nm',
    >>>         'A^-1',
    >>>         'A^-1'
    >>>     ],
    >>>     dim_names = [
    >>>         'rx',
    >>>         'ry',
    >>>         'qx',
    >>>         'qy'
    >>>     ],
    >>> )

    will create an array with a name and units for its data, where its first two
    dimensions are in units of nanometers, have pixel sizes of 5nm, and are
    described by the handles 'rx' and 'ry', and where its last two dimensions
    are in units of inverse Angstroms, have pixels sizes of 0.01A^-1, and are
    described by the handles 'qx' and 'qy'.

    Arrays in which the length of each pixel is non-constant are also
    supported.  For instance,

    >>> x = np.logspace(0,1,100)
    >>> y = np.sin(x)
    >>> ar = Array(
    >>>     y,
    >>>     dims = [
    >>>         x
    >>>     ]
    >>> )

    generates an array representing the values of the sine function sampled
    100 times along a logarithmic interval from 1 to 10. In this example,
    this data could then be plotted with, e.g.

    >>> plt.scatter(ar.dims[0], ar.data)

    If the `slicelabels` keyword is passed, the first N-1 dimensions of the
    array are treated normally, while the final dimension is used to represent
    distinct arrays which share a common shape and set of dim vectors.  Thus

    >>> ar = Array(
    >>>     np.ones((50,50,4)),
    >>>     name = 'test_array_stack',
    >>>     units = 'intensity',
    >>>     dims = [
    >>>         [0,2],
    >>>         [0,2]
    >>>     ],
    >>>     dim_units = [
    >>>         'nm',
    >>>         'nm'
    >>>     ],
    >>>     dim_names = [
    >>>         'rx',
    >>>         'ry'
    >>>     ],
    >>>     slicelabels = [
    >>>         'a',
    >>>         'b',
    >>>         'c',
    >>>         'd'
    >>>     ]
    >>> )

    will generate a single Array instance containing 4 arrays which each have
    a shape (50,50) and a common set of dim vectors ['rx','ry'], and which
    can be indexed into with the names assigned in `slicelabels` using

    >>> ar.get_slice('a')

    which will return a 2D (non-stack-like) Array instance with shape (50,50)
    and the dims assigned above.  The Array attribute .rank is equal to the
    number of dimensions for a non-stack-like Array, and is equal to N-1
    for stack-like arrays.

    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'array',
        units: Optional[str] = '',
        dims: Optional[list] = None,
        dim_names: Optional[list] = None,
        dim_units: Optional[list] = None,
        slicelabels = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the Array
            units (str): units for the pixel values
            dims (variable): calibration vectors for each of the axes of the data
                array.  Valid values for each element of the list are None,
                a number, a 2-element list/array, or an M-element list/array
                where M is the data array.  If None is passed, the dim will be
                populated with integer values starting at 0 and its units will
                be set to pixels.  If a number is passed, the dim is populated
                with a vector beginning at zero and increasing linearly by this
                step size.  If a 2-element list/array is passed, the dim is
                populated with a linear vector with these two numbers as the first
                two elements.  If a list/array of length M is passed, this is used
                as the dim vector, (and must therefore match this dimension's
                length). If dims recieves a list of fewer than N arguments for an
                N-dimensional data array, the extra dimensions are populated as if
                None were passed, using integer pixel values. If the `dims`
                parameter is not passed, all dim vectors are populated this way.
            dim_units (list): the units for the calibration dim vectors. If
                nothing is passed, dims vectors which have been populated
                automatically with integers corresponding to pixel numbers
                will be assigned units of 'pixels', and any other dim vectors
                will be assigned units of 'unknown'.  If a list with length <
                the array dimensions, the passed values are assumed to apply
                to the first N dimensions, and the remaining values are
                populated with 'pixels' or 'unknown' as above.
            dim_names (list): labels for each axis of the data array. Values
                which are not passed, following the same logic as described
                above, will be autopopulated with the name "dim#" where #
                is the axis number.
            slicelabels (None or True or list): if not None, must be True or a
                list of strings, indicating a "stack-like" array.  In this case,
                the first N-1 dimensions of the array are treated normally, in
                the sense of populating dims, dim_names, and dim_units, while the
                final dimension is treated distinctly: it indexes into
                distinct arrays which share a set of dimension attributes, and
                can be sliced into using the string labels from the `slicelabels`
                list, with the syntax array['label'] or array.get_slice('label').
                If `slicelabels` is `True` or is a list with length less than the
                final dimension length, unassigned dimensions are autopopulated
                with labels `array{i}`. The flag array.is_stack is set to True
                and the array.rank attribute is set to N-1.

        Returns:
            A new Array instance
        """
        self.data = data
        self.name = name
        self.units = units
        self.dims = dims
        self.dim_names = dim_names
        self.dim_units = dim_units

        self.shape = self.data.shape
        self.rank = self.data.ndim

        self.tree = Tree()
        if not hasattr(self, "_metadata"):
            self._metadata = {}

        # flag to help assign dim names and units
        dim_in_pixels = np.zeros(self.rank, dtype=bool)


        ## Handle array stacks

        if slicelabels is None:
            self.depth = 0
            self.is_stack = False

        else:
            self.depth = self.shape[-1]
            self.shape = self.shape[:-1]
            dim_in_pixels = dim_in_pixels[:-1]
            self.rank -= 1
            self.is_stack = True

            # Populate labels
            if slicelabels is True:
                slicelabels = [f'array{i}' for i in range(self.depth)]
            elif len(slicelabels) < self.depth:
                slicelabels = np.concatenate((slicelabels,
                    [f'array{i}' for i in range(len(slicelabels),self.depth)]))
            else:
                slicelabels = slicelabels[:self.depth]
            slicelabels = Labels(slicelabels)

        self.slicelabels = slicelabels


        ## Set dim vectors

        # if none were passed
        if self.dims is None:
            self.dims = [self._unpack_dim(1,self.shape[n]) for n in range(self.rank)]
            dim_in_pixels[:] = True

        # if some but not all were passed
        elif len(self.dims)<self.rank:
            _dims = self.dims
            N = len(_dims)
            self.dims = []
            for n in range(N):
                dim = self._unpack_dim(_dims[n],self.shape[n])
                self.dims.append(dim)
            for n in range(N,self.rank):
                self.dims.append(self._unpack_dim(1,self.shape[n]))
                dim_in_pixels[n] = True

        # if all were passed
        elif len(self.dims)==self.rank:
            _dims = self.dims
            self.dims = []
            for n in range(self.rank):
                dim = self._unpack_dim(_dims[n],self.shape[n])
                self.dims.append(dim)

        # otherwise
        else:
            raise Exception(f"too many dim vectors were passed - expected {self.rank}, received {len(self.dims)}")


        ## set dim vector names

        # if none were passed
        if self.dim_names is None:
            self.dim_names = [f"dim{n}" for n in range(self.rank)]

        # if some but not all were passed
        elif len(self.dim_names)<self.rank:
            N = len(self.dim_names)
            self.dim_names = [name for name in self.dim_names] + \
                             [f"dim{n}" for n in range(N,self.rank)]

        # if all were passed
        elif len(self.dim_names)==self.rank:
            pass

        # otherwise
        else:
            raise Exception(f"too many dim names were passed - expected {self.rank}, received {len(self.dim_names)}")


        ## set dim vector units

        # if none were passed
        if self.dim_units is None:
            self.dim_units = [['unknown','pixels'][i] for i in dim_in_pixels]

        # if some but not all were passed
        elif len(self.dim_units)<self.rank:
            N = len(self.dim_units)
            self.dim_units = [units for units in self.dim_units] + \
                             [['unknown','pixels'][dim_in_pixels[i]] for i in range(N,self.rank)]

        # if all were passed
        elif len(self.dim_units)==self.rank:
            pass

        # otherwise
        else:
            raise Exception(f"too many dim units were passed - expected {self.rank}, received {len(self.dim_units)}")




    #### Methods


    ## Slicing

    def get_slice(self,label,name=None):
        idx = self.slicelabels._dict[label]
        return Array(
            data = self.data[..., idx],
            name = name if name is not None else self.name+'_'+label,
            units = self.units[:-1],
            dims = self.dims[:-1],
            dim_units = self.dim_units[:-1],
            dim_names = self.dim_names[:-1]
        )

    def __getitem__(self,x):
        if isinstance(x,str):
            return self.get_slice(x)
        elif isinstance(x,tuple) and isinstance(x[0],str):
            return self.get_slice(x[0])[x[1:]]
        else:
            return self.data[x]


    ## Dim vectors

    def set_dim(
        self,
        n:int,
        dim:Union[list,np.ndarray],
        units:Optional[str]=None,
        name:Optional[str]=None
        ):
        """
        Sets the n'th dim vector, using `dim` as described in the Array
        documentation. If `units` and/or `name` are passed, sets these
        values for the n'th dim vector.

        Accepts:
            n (int): specifies which dim vector
            dim (list or array): length must be either 2, or equal to the
                length of the n'th axis of the data array
            units (Optional, str):
            name: (Optional, str):
        """
        length = self.shape[n]
        _dim = self._unpack_dim(dim,length)
        self.dims[n] = _dim
        if units is not None: self.dim_units[n] = units
        if name is not None: self.dim_names[n] = name



    @staticmethod
    def _unpack_dim(dim,length):
        """
        Given a dim vector as passed at instantiation and the expected length
        of this dimension of the array, this function checks the passed dim
        vector length, and checks the dim vector type.  For number-like dim-
        vectors:
            -if it is a number, turns it into the list [0,number] and proceeds
                as below
            -if it has length 2, linearly extends the vector to its full length
            -if it has length `length`, returns the vector as is
            -if it has any other length, raises an Exception.

        For string-like dim vectors, the length must match the array dimension
        length.

        Accepts:
            dim (list or array)
            length (int)

        Returns
            the unpacked dim vector
        """
        # Expand single numbers
        if isinstance(dim,Number):
            dim = [0,dim]

        N = len(dim)

        # for string dimensions:
        if not isinstance(dim[0],Number):
            assert(N == length), f"For non-numerical dims, the dim vector length must match the array dimension length. Recieved a dim vector of length {N} for an array dimension length of {length}."

        # For number-like dimensions:
        if N == length:
            return dim
        elif N == 2:
            start,step = dim[0],dim[1]-dim[0]
            stop = start + step*length
            return np.arange(start,stop,step)
        else:
            raise Exception(f"dim vector length must be either 2 or equal to the length of the corresponding array dimension; dim vector length was {dim} and the array dimension length was {length}")


    def _dim_is_linear(self,dim,length):
        """
        Returns True if a dim is linear, else returns False
        """
        dim_expanded = self._unpack_dim(dim[:2],length)
        return np.array_equal(dim,dim_expanded)



    # set up metadata property

    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self,x):
        assert(isinstance(x,Metadata))
        self._metadata[x.name] = x




    ## Representation to standard output

    def __repr__(self):

        if not self.is_stack:
            space = ' '*len(self.__class__.__name__)+'  '
            string = f"{self.__class__.__name__}( A {self.rank}-dimensional array of shape {self.shape} called '{self.name}',"
            string += "\n"+space+"with dimensions:"
            string += "\n"
            for n in range(self.rank):
                string += "\n"+space+f"{self.dim_names[n]} = [{self.dims[n][0]},{self.dims[n][1]},...] {self.dim_units[n]}"
            string += "\n)"

        else:
            space = ' '*len(self.__class__.__name__)+'  '
            string = f"{self.__class__.__name__}( A stack of {self.depth} Arrays with {self.rank}-dimensions and shape {self.shape}, called '{self.name}'"
            string += "\n"
            string += "\n" +space + "The labels are:"
            for label in self.slicelabels:
                string += "\n" + space + f"    {label}"
            string += "\n"
            string += "\n"
            string += "\n" + space + "The Array dimensions are:"
            for n in range(self.rank):
                string += "\n"+space+f"    {self.dim_names[n]} = [{self.dims[n][0]},{self.dims[n][1]},...] {self.dim_units[n]}"
                if not self._dim_is_linear(self.dims[n],self.shape[n]):
                    string += "  (*non-linear*)"
            string += "\n)"

        return string



    # HDF5 read/write

    def to_h5(self,group):
        from .io import Array_to_h5
        Array_to_h5(self,group)

    def from_h5(group):
        from .io import Array_from_h5
        return Array_from_h5(group)





########### END OF CLASS ###########


# List subclass for accessing data slices with a dict
class Labels(list):

    def __init__(self,x=[]):
        list.__init__(self,x)
        self.setup_labels_dict()

    def __setitem__(self,idx,label):
        label_old = self[idx]
        del(self._dict[label_old])
        list.__setitem__(self,idx,label)
        self._dict[label] = idx

    def setup_labels_dict(self):
        self._dict = {}
        for idx,label in enumerate(self):
            self._dict[label] = idx


