# Defines the Custom class, which contains any number of EMD base
# classes as attributes, i.e. they are constructed by composition,
# rather than inheritance, of the other classes.  Custom classes are
# themselves EMD base classes and can therefore be nested in other custom
# classes themselves.

from typing import Optional,Union
import numpy as np
import h5py
from numbers import Number
from os.path import basename

from py4DSTEM.io.classes.tree import Node,Root
from py4DSTEM.io.classes.metadata import Metadata

class Custom(Node):
    """
    """
    _emd_group_type = 'custom'

    def __init__(self,name='custom'):
        Node.__init__(self,name=name)


    # write
    def to_h5(self,group):
        """
        Constructs an h5 group, adds metadata, and adds all attributes
        which point to EMD nodes.

        Accepts:
            group (h5py Group)

        Returns:
            (h5py Group) the new node's Group
        """
        # Construct the group and add metadata
        grp = Node.to_h5(self,group)

        # Add any attributes which are themselves emd nodes
        for k,v in vars(self).items():
            if isinstance(v,Node):
                if not isinstance(v,Root):
                    v.name = k
                    attr_grp = v.to_h5(grp)
                    attr_grp.attrs['emd_group_type'] = 'custom_' + \
                        attr_grp.attrs['emd_group_type']



    # read

    @classmethod
    def _get_constructor_args(cls,group):
        """
        Takes a group, and returns a dictionary of args/values
        to pass to the class constructor.

        This function needs to be written for any child classes;
        subclassing but failing to overwrite this method will
        throw a runtime error. You'll probably want to include
        {'name' : basename(group.name)}. To retrieve the data stored in
        a Python class instance attribute, use self._get_emd_attr_data.
        To populate subsequent to instance construction, add a
        _populate_instance method.
        """
        raise Exception(f"Custom class {cls} needs a `_get_constructor_args` method!")


    def _get_emd_attr_data(self,group):
        """ Loops through h5 groups under group, finding and returning
        a {name:instance} dictionary of {attribute name : class instance}
        pairs from all 'custom_' subgroups
        """
        groups = [g for g in group.keys() if isinstance(group[g],h5py.Group)]
        groups = [g for g in groups if 'emd_group_type' in group[g].attrs.keys()]
        groups = [g for g in groups if groups[g].attrs['emd_group_type']=='custom_']
        dic = {}
        for g in groups:
            name,grp = g,groups[g]
            __class__ = _get_class(grp)
            data = __class__.from_h5(grp)
            dic[name] = data
        return dic





