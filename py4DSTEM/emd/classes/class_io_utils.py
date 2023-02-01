
import inspect
import h5py



# Define the EMD group types

EMD_base_group_types = (
    "root",
    "metadatabundle",
    "metadata",
)
EMD_data_group_types = (
    "node",
    "array",
    "pointlist",
    "pointlistarray",
    "custom",
)
EMD_custom_group_types = tuple(
    ["custom_"+s for s in EMD_data_group_types]
)
EMD_group_types = EMD_base_group_types + EMD_data_group_types + EMD_custom_group_types




def _get_class(grp):
    """
    Function returning Class constructors from corresponding strings
    """
    from py4DSTEM.io_emd import classes
    # TODO - stitch py4dstem classes back in here
    #from py4DSTEM.io_py4dstem import classes as classes_py4dstem

    # Build lookup table for classes
    lookup = {}
    for name, obj in inspect.getmembers(classes):
        if inspect.isclass(obj):
            lookup[name] = obj

    # Get the class from the group tags and return
    try:
        classname = grp.attrs['python_class']
        __class__ = lookup[classname]
        return __class__
    except KeyError:
        return None
        #raise Exception(f"Unknown classname {classname}")





# Finding and validating EMD groups

def find_EMD_groups(group:h5py.Group, emd_group_type):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and finds all groups inside this group at its top level matching
    `emd_group_type`. Does not do a nested search. Returns the names of all
    groups found.

    Accepts:
        group (HDF5 group):
        emd_group_type (int)
    """
    keys = [k for k in group.keys() if "emd_group_type" in group[k].attrs.keys()]
    return [k for k in keys if group[k].attrs["emd_group_type"] == emd_group_type]


def EMD_group_exists(group:h5py.Group, emd_group_type, name:str):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if an object of this `emd_group_type` and name exists
    inside this group, and returns a boolean.

    Accepts:
        group (HDF5 group):
        emd_group_type (int):
        name (string):

    Returns:
        bool
    """
    if name in group.keys():
        if "emd_group_type" in group[name].attrs.keys():
            if group[name].attrs["emd_group_type"] == emd_group_type:
                return True
            return False
        return False
    return False






