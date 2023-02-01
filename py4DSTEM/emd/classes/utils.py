
import inspect



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
    # TODO - build hook for dependent package classes

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





