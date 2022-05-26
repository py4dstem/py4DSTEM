# Utility functions for datastructure i/o

import numpy as np

def determine_group_name(obj, group):
    """
    Takes an instance of a py4DSTEM data class and a valid HDF5 group for an HDF5
    file object which is open in write or append mode. Determines an appropriate
    name for store this object in the H5 file.

    If the object has no name, it will be assigned the name "{obj.__class__}#" where
    # is the lowest available integer.

    If the object has a name, checks to ensure that this name is not already in
    use.  If it is available, returns the name.  If it is not, raises an exception.

    TODO: add overwite option.

    Accepts:
        obj (an instance of a py4DSTEM data class)
        group (HDF5 group)

    Returns:
        (str) a valid group name
    """

    classname = obj.__class__.__name__

    # Detemine the name of the group
    if obj.name == '':
        # Assign the name "{obj.__class__}#" for lowest available #
        keys = [k for k in group.keys() if k[:len(classname)]==classname]
        i,found = -1,False
        while not found:
            i += 1
            found = ~np.any([int(k[len(classname):])==i for k in keys])
        obj.name = f"{classname}{i}"
    else:
        # Check if the name is already in the file
        if obj.name in group.keys():
            # TODO add an overwrite option
            raise Exception(f"A group named {obj.name} already exists in this file. Try using another name.")


