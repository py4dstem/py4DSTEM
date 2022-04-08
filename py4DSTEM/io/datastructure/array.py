# Defines the Array class, which stores any N-dimensional array-like data.
# Implements the EMD file standard - https://emdatasets.com/format

from typing import Optional,Union
import numpy as np
import h5py
from numbers import Number

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
    >>>     np.ones((20,20,256,256)
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

    will create an array with a name and units for its data, with its first two
    dimensions in units of nanometers, with each pixel having a size of 5nm, and
    described by the handles 'rx' and 'ry', here meant to represent the (x,y)
    position in real space, and with its last two dimensions in units of inverse
    Angstroms, with each pixel having a size of 0.01A^-1, and dsecribed by the
    handles 'qx' and 'qy', representing the (x,y) position in diffraction space.

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
    along a logarithmic interval from 1 to 10. In this example, this data
    could then be plotted with, e.g.

    >>> plt.scatter(ar.dims[0], ar.data)

    """
    def __init__(
        self,
        data: np.ndarray,
        name: Optional[str] = 'array',
        units: Optional[str] = '',
        dims: Optional[list] = None,
        dim_names: Optional[list] = None,
        dim_units: Optional[list] = None
        ):
        """
        Accepts:
            data (np.ndarray): the data
            name (str): the name of the Array
            units (str): units for the pixel values
            dims (list): calibration vectors for each of the axes of the data
                array.  Valid values for each element of the list are None,
                a number, a 2-element list/array, or an M-element list/array
                where M is the data array.  If None is passed, the dim will be
                populated with integer values starting at 0 and its units will
                be set to pixels.  If a number is passed, the dim is populated
                with a vector beginning at zero and increasing linearly by this
                step size.  If a 2-element list/array is passed, the dim is
                populated with a linear vector with these two numbers as the first
                two elements.  If a list/array of length M is passed, this is used
                as the dim vector.  If dims recieves a list of fewer than N
                arguments for an N-dimensional data array, the extra dimensions
                are populated as if None were passed, using integer pixel values.
                If the `dims` parameter is not passed, all dim vectors are
                populated this way.
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
        self.rank = len(self.shape)

        # flags to identify which dim vectors can be stored
        # with only two elements
        self.dim_is_linear = np.zeros(self.rank, dtype=bool)

        # flags to help assign dim names and units
        dim_in_pixels = np.zeros(self.rank, dtype=bool)

        # flag identifying a normal, non-stack array
        self._isstack = False


        ## Set dim vectors

        # if none were passed
        if self.dims is None:
            self.dims = [self._unpack_dim(1,self,shape[n])[0] for n in range(self.rank)]
            self.dim_is_linear[:] = True
            dim_in_pixels[:] = True

        # if some but not all were passed
        elif len(self.dims)<self.rank:
            _dims = self.dims
            N = len(_dims)
            self.dims = []
            for n in range(N):
                dim,flag = self._unpack_dim(_dims[n],self.shape[n])
                self.dims.append(dim)
                self.dim_is_linear[n] = flag
            for n in range(N,self.rank):
                self.dims.append(np.arange(self.shape[n]))
                self.dim_is_linear[n] = True
                dim_in_pixels[n] = True

        # if all were passed
        elif len(self.dims)==self.rank:
            _dims = self.dims
            self.dims = []
            for n in range(self.rank):
                dim,flag = self._unpack_dim(_dims[n],self.shape[n])
                self.dims.append(dim)
                self.dim_is_linear[n] = flag

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


    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A {self.rank}-dimensional array of shape {self.shape} called '{self.name}',"
        string += "\n"+space+"with dimensions:"
        string += "\n"
        for n in range(self.rank):
            string += "\n"+space+f"{self.dim_names[n]} = [{self.dims[n][0]},{self.dims[n][1]},...] {self.dim_units[n]}"
        string += "\n)"
        return string


    def set_dim(self,
                n:int,
                dim:Union[list,np.ndarray],
                units:Optional[str]=None,
                name:Optional[str]=None):
        """
        Sets the n'th dim vector, using `dim` as described in the Array documentation.
        If `units` and/or `name` are passed, sets these values for the n'th dim vector.

        Accepts:
            n (int): specifies which dim vector
            dim (list or array): length must be either 2, or equal to the length of
                the n'th axis of the data array
            units (Optional, str):
            name: (Optional, str):
        """
        length = self.shape[n]
        _dim,is_linear = self._unpack_dim(dim,length)
        self.dims[n] = _dim
        self.dim_is_linear[n] = is_linear
        if units is not None: self.dim_units[n] = units
        if name is not None: self.dim_names[n] = name



    @staticmethod
    def _unpack_dim(dim,length):
        """
        Given a dim vector as passed at instantiation and the expected length of this
        dimension of the array, this function checks the passed dim vector length, and
        checks the dim vector type.  For number-like dim-vectors:

        -if it is a number, turns it into the list [0,number] and proceeds as below

        -if it has length 2, linearly extends the vector to its full length

        -if it has length `length`, returns the vector as is

        -if it has any other length, raises an Exception.

        For string-like dim vectors, the length must match the array dimension length.

        Accepts:
            dim (list or array)
            length (int)

        Returns
            (2-tuple) the unpacked dim vector, and a boolean flag indicating if
            this vector was linearly extended
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
            return dim,False
        elif N == 2:
            start,step = dim[0],dim[1]-dim[0]
            stop = start + step*length
            return np.arange(start,stop,step),True
        else:
            raise Exception(f"dim vector length must be either 2 or equal to the length of the corresponding array dimension; dim vector length was {dim} and the array dimension length was {length}")



    # Writing to an HDF5 file

    def to_h5(self,group):
        """
        Takes a valid HDF5 group for an HDF5 file object which is open in write or append
        mode. Writes a new group with a name given by this Array's .name field nested
        inside the passed group, and saves the data there.

        If the Array has no name, it will be assigned the name "Array#" where # is the
        lowest available integer.  If the Array's name already exists here in this file,
        raises and exception.

        TODO: add overwite option.

        Accepts:
            group (HDF5 group)
        """
        # Detemine the name of the group
        if self.name == '':
            # Assign the name "Array#" for lowest available #
            keys = [k for k in group.keys() if k[:5]=='Array']
            i,found = -1,False
            while not found:
                i += 1
                found = ~np.any([int(k[5:])==i for k in keys])
            self.name = f"Array{i}"
        else:
            # Check if the name is already in the file
            if self.name in group.keys():
                # TODO add an overwrite option
                raise Exception(f"A group named {self.name} already exists in this file. Try using another name.")


        ## Write

        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",1) # this tag indicates an Array type object
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # add the data
        data = grp.create_dataset(
            "data",
            shape = self.shape,
            data = self.data,
            #dtype = type(self.data)
        )
        data.attrs.create('units',self.units) # save 'units' but not 'name' - 'name' is the group name

        # Determine if this is an arraystack
        # such that the last dim is a list of names
        normal_dims = self.rank
        if self._isstack:
            normal_dims -= 1

        # Add the normal dim vectors
        for n in range(normal_dims):

            # unpack info
            dim = self.dims[n]
            name = self.dim_names[n]
            units = self.dim_units[n]
            is_linear = self.dim_is_linear[n]

            # compress the dim vector if it's linear
            if is_linear:
                dim = dim[:2]

            # write
            dset = grp.create_dataset(
                f"dim{n}",
                data = dim
            )
            dset.attrs.create('name',name)
            dset.attrs.create('units',units)

        # Add stack dim vector, if present
        if self._isstack:
            n = self.rank-1
            name = '_labels_'
            dim = [s.encode('utf-8') for s in self.labels]

            # write
            dset = grp.create_dataset(
                f"dim{n}",
                data = dim
            )
            dset.attrs.create('name',name)



########### END OF CLASS ###########





