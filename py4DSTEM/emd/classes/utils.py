
import inspect
import types
import sys


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
    Returns a dictionary of Class constructors from corresponding strings
    """
    from py4DSTEM.emd import classes

    # Build lookup table for classes
    lookup = {}
    for name, obj in inspect.getmembers(classes):
        if inspect.isclass(obj):
            lookup[name] = obj

    # hook for dependent package classes
    for module in _get_dependent_packages():
        lookup_tmp = {}
        for name, obj in inspect.getmembers(module.classes):
            if inspect.isclass(obj):
                lookup_tmp[name] = obj
        lookup.update(lookup_tmp)

    # Get the class from the group tags and return
    try:
        classname = grp.attrs['python_class']
        __class__ = lookup[classname]
        return __class__
    except KeyError:
        raise Exception(f"Unknown classname {classname}")




def _get_dependent_packages():
    """
    Searches packages with the top level attribute "_emd_hook" = True.
    Returns a generator of all such packages
    """
    for module in sys.modules.values():
        if isinstance(module, types.ModuleType):
            if hasattr(module, "_emd_hook"):
                if module._emd_hook is True:
                    yield module



