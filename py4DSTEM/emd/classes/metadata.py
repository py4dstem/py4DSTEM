import numpy as np
from numbers import Number
from typing import Optional
from os.path import basename



class Metadata:
    """
    Stores metadata in the form of a flat (non-nested) dictionary.
    Keys are arbitrary strings.  Values may be strings, numbers, arrays,
    or lists of the above types.

    Usage:

        >>> meta = Metadata()
        >>> meta['param'] = value
        >>> val = meta['param']

    If the parameter has not been set, the getter methods return None.
    """
    _emd_group_type = 'metadata'

    def __init__(
        self,
        name: Optional[str] = 'metadata',
        data: Optional[dict] = None
        ):
        """
         Args:
            name (Optional, string):
        """
        self.name = name
        self._params = {}

        if data is not None:
            assert(isinstance(data,dict)), f"`data` must be a dict, not type {type(data)}"
            self._params.update(data)


    ### __get/setitem__

    def __getitem__(self,x):
        return self._params[x]
    def __setitem__(self,k,v):
        self._params[k] = v


    @property
    def keys(self):
        return self._params.keys()


    def copy(self,name=None):
        """
        """
        if name is None: name = self.name+"_copy"
        md = Metadata(name=name)
        md._params.update(self._params)
        return md



    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A Metadata instance called '{self.name}', containing the following fields:"
        string += "\n"

        maxlen = 0
        for k in self._params.keys():
            if len(k)>maxlen: maxlen=len(k)

        for k,v in self._params.items():
            if isinstance(v,np.ndarray):
                v = f"{v.ndim}D-array"
            string += "\n"+space+f"{k}:{(maxlen-len(k)+3)*' '}{str(v)}"
        string += "\n)"

        return string



    # HDF5 i/o

    # write
    def to_h5(self,group):
        """
        Accepts an h5py Group which is open in write or append mode. Writes
        a new group with this object's name and saves its metadata in it.

        Accepts:
            group (h5py Group)
        """
        # Make a new group
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type","metadata")
        grp.attrs.create("python_class",self.__class__.__name__)

        # Save data
        for k,v in self._params.items():

            # None
            if v is None:
                v = "_None"
                v = np.string_(v)  # convert to byte string
                dset = grp.create_dataset(k, data=v)
                dset.attrs['type'] = np.string_('None')

            # strings
            elif isinstance(v, str):
                v = np.string_(v)  # convert to byte string
                dset = grp.create_dataset(k, data=v)
                dset.attrs['type'] = np.string_('string')

            # bools
            elif isinstance(v, bool):
                dset = grp.create_dataset(k, data=v, dtype=bool)
                dset.attrs['type'] = np.string_('bool')

            # numbers
            elif isinstance(v, Number):
                dset = grp.create_dataset(k, data=v, dtype=type(v))
                dset.attrs['type'] = np.string_('number')

            # arrays
            elif isinstance(v, np.ndarray):
                dset = grp.create_dataset(k, data=v, dtype=v.dtype)
                dset.attrs['type'] = np.string_('array')

            # tuples
            elif isinstance(v, tuple):

                # of numbers
                if isinstance(v[0], Number):
                    dset = grp.create_dataset(k, data=v)
                    dset.attrs['type'] = np.string_('tuple')

                # of tuples
                elif any([isinstance(v[i], tuple) for i in range(len(v))]):
                    dset_grp = grp.create_group(k)
                    dset_grp.attrs['type'] = np.string_('tuple_of_tuples')
                    dset_grp.attrs['length'] = len(v)
                    for i,x in enumerate(v):
                        dset_grp.create_dataset(
                            str(i),
                            data=x)

                # of arrays
                elif isinstance(v[0], np.ndarray):
                    dset_grp = grp.create_group(k)
                    dset_grp.attrs['type'] = np.string_('tuple_of_arrays')
                    dset_grp.attrs['length'] = len(v)
                    for i,ar in enumerate(v):
                        dset_grp.create_dataset(
                            str(i),
                            data=ar,
                            dtype=ar.dtype)

                # of strings
                elif isinstance(v[0], str):
                    dset_grp = grp.create_group(k)
                    dset_grp.attrs['type'] = np.string_('tuple_of_strings')
                    dset_grp.attrs['length'] = len(v)
                    for i,s in enumerate(v):
                        dset_grp.create_dataset(
                            str(i),
                            data=np.string_(s))

                else:
                    er = f"Metadata only supports writing tuples with numeric and array-like arguments; found type {type(v[0])}"
                    raise Exception(er)

            # lists
            elif isinstance(v, list):

                # of numbers
                if isinstance(v[0], Number):
                    dset = grp.create_dataset(k, data=v)
                    dset.attrs['type'] = np.string_('list')

                # of arrays
                elif isinstance(v[0], np.ndarray):
                    dset_grp = grp.create_group(k)
                    dset_grp.attrs['type'] = np.string_('list_of_arrays')
                    dset_grp.attrs['length'] = len(v)
                    for i,ar in enumerate(v):
                        dset_grp.create_dataset(
                            str(i),
                            data=ar,
                            dtype=ar.dtype)

                # of strings
                elif isinstance(v[0], str):
                    dset_grp = grp.create_group(k)
                    dset_grp.attrs['type'] = np.string_('list_of_strings')
                    dset_grp.attrs['length'] = len(v)
                    for i,s in enumerate(v):
                        dset_grp.create_dataset(
                            str(i),
                            data=np.string_(s))

                else:
                    er = f"Metadata only supports writing lists with numeric and array-like arguments; found type {type(v[0])}"
                    raise Exception(er)

            else:
                er = f"Metadata supports writing numbers, bools, strings, arrays, tuples of numbers or arrays, and lists of numbers or arrays. Found an unsupported type {type(v[0])}"
                raise Exception(er)


    # read
    @classmethod
    def from_h5(cls,group):
        """
        Accepts an h5py Group which is open in read mode, confirms that
        it represents an EMD MetadataDict group, then loads and returns it
        as a Metadata instance.

        Accepts:
            group (HDF5 group)

        Returns:
            (Metadata)
        """
        # Validate inputs
        er = f"Group {group} is not a valid EMD Metadata group"
        assert("emd_group_type" in group.attrs.keys()), er
        assert(group.attrs["emd_group_type"] == "metadata"), er

        # Get data
        data = {}
        for k,v in group.items():

            # get type
            try:
                t = group[k].attrs['type'].decode('utf-8')
            except KeyError:
                raise Exception(f"unrecognized Metadata value type {type(v)}")

            # None
            if t == 'None':
                v = None

            # strings
            elif t == 'string':
                v = v[...].item()
                v = v.decode('utf-8')
                v = v if v != "_None" else None

            # numbers
            elif t == 'number':
                v = v[...].item()

            # bools
            elif t == 'bool':
                v = v[...].item()

            # array
            elif t == 'array':
                v = np.array(v)

            # tuples of numbers
            elif t == 'tuple':
                v = tuple(v[...])

            # tuples of arrays
            elif t == 'tuple_of_arrays':
                L = group[k].attrs['length']
                tup = []
                for l in range(L):
                    tup.append(np.array(v[str(l)]))
                v = tuple(tup)

            # tuples of tuples
            elif t == 'tuple_of_tuples':
                L = group[k].attrs['length']
                tup = []
                for l in range(L):
                    x = v[str(l)][...]
                    if x.ndim == 0:
                        x = x.item()
                    else:
                        x = tuple(x)
                    tup.append(x)
                v = tuple(tup)

            # tuples of strings
            elif t == 'tuple_of_strings':
                L = group[k].attrs['length']
                tup = []
                for l in range(L):
                    s = v[str(l)][...].item().decode('utf-8')
                    tup.append(s)
                v = tuple(tup)

            # lists of numbers
            elif t == 'list':
                v = list(v[...])

            # lists of arrays
            elif t == 'list_of_arrays':
                L = group[k].attrs['length']
                _list = []
                for l in range(L):
                    _list.append(np.array(v[str(l)]))
                v = _list

            # list of strings
            elif t == 'list_of_strings':
                L = group[k].attrs['length']
                _list = []
                for l in range(L):
                    s = v[str(l)][...].item().decode('utf-8')
                    _list.append(s)
                v = _list

            else:
                raise Exception(f"unrecognized Metadata value type {t}")


            # add data
            data[k] = v


        # make Metadata instance, add data, and return
        md = cls(basename(group.name))
        md._params.update(data)
        return md



