import numpy as np
from numbers import Number
from typing import Optional
import h5py

from .ioutils import determine_group_name
from .ioutils import EMD_group_exists, EMD_group_types


class Metadata:
    """
    Stores metadata in the form of a flat (non-nested) dictionary.
    Keys are arbitrary strings.  Values may be strings, numbers, arrays,
    or lists of the above types.

    Usage:

        >>> c = Metadata()
        >>> c.set_p(p,v)
        >>> v = c.get_p(p)

    If the parameter has not been set, the getter methods return None
    The value of a parameter may be a number, representing the entire dataset,
    or a 2D typically (R_Nx,R_Ny)-shaped array, representing values at each
    detector pixel. For parameters with 2D array values,

        >>> c.get_p()

    will return the entire 2D array, and

        >>> c.get_p(rx,ry)

    will return the value of `p` at position `rx,ry`.
    """
    def __init__(
        self,
        name: Optional[str] ='metadata'
        ):
        """
         Args:
            name (Optional, string):
        """
        self.name = name

        # create parameter dictionary
        self._params = {}



    ### getter/setter methods

    def set_p(self,p,v):
        self._params[p] = v
    def get_p(self,p):
        return self._params[p]
    def get_keys(self):
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



    ## Writing to an HDF5 file

    def to_h5(self,group):
        """
        Takes a valid HDF5 group for an HDF5 file object which is open in write or append
        mode. Writes a new group with a name given by this Calibration instance's .name
        field nested inside the passed group, and saves the data there.

        If the Calibration instance has no name, it will be assigned the name
        Calibration"#" where # is the lowest available integer.  If the instance has a name
        which already exists here in this file, raises and exception.

        TODO: add overwite option.

        Accepts:
            group (HDF5 group)
        """

        # Detemine the name of the group
        # if current name is invalid, raises and exception
        # TODO: add overwrite option
        determine_group_name(self, group)

        ## Write
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",EMD_group_types['Metadata'])
        grp.attrs.create("py4dstem_class",self.__class__.__name__)

        # Save data
        for k,v in self._params.items():
            if isinstance(v,str): v = np.string_(v)
            grp.create_dataset(k, data=v)


## Read Calibration objects

def Metadata_from_h5(group:h5py.Group, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid Metadata object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A Metadata instance
    """
    er = f"No Metadata instance called {name} could be found in group {group} of this HDF5 file."
    assert(EMD_group_exists(
            group,
            EMD_group_types['Metadata'],
            name)), er
    grp = group[name]


    # Get metadata
    name = grp.name.split('/')[-1]

    # Get data
    data = {}
    for k,v in grp.items():
        v = v[...]
        if v.ndim==0:
            v=v.item()
            if isinstance(v,bytes):
                v = v.decode('utf-8')
        elif v.ndim==1:
            str_mask = [isinstance(v[i],bytes) for i in range(len(v))]
            if any(str_mask):
                inds = np.nonzero(str_mask)[0]
                for ind in inds:
                    v[ind] = v[ind].decode('utf-8')
        data[k] = v


    md = Metadata(name=name)
    md._params.update(data)

    return md



