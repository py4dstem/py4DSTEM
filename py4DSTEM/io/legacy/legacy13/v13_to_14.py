# Convert v13 to v14 classes


# v13 imports

from py4DSTEM.io.legacy.legacy13.v13_emd_classes import (
    Root as Root13,
    Metadata as Metadata13,
    Array as Array13,
    PointList as PointList13,
    PointListArray as PointListArray13
)
from py4DSTEM.io.legacy.legacy13.v13_py4dstem_classes import (
    Calibration as Calibration13,
    DataCube as DataCube13,
    DiffractionSlice as DiffractionSlice13,
    VirtualDiffraction as VirtualDiffraction13,
    RealSlice as RealSlice13,
    VirtualImage as VirtualImage13,
    Probe as Probe13,
    QPoints as QPoints13,
    BraggVectors as BraggVectors13
)



# v14 imports

from emdfile import (
    Root,
    Metadata,
    Array,
    PointList,
    PointListArray
)

from py4DSTEM.classes import (
    Calibration,
    DataCube,
    DiffractionSlice,
    VirtualDiffraction,
    RealSlice,
    VirtualImage,
    Probe,
    QPoints,
    BraggVectors
)



def v13_to_14(v13tree, v14tree):
    """
    Converts a v13 data tree to a v14 data tree
    """

    # if a list of root names was returned, pass it through
    if isinstance(v13tree, list):
        return v13tree

    # get the root and a node to grow from
    if isinstance(v13tree,Root13):
        root = startnode = _v13_to_14_cls(v13_tree)
    else:
        startnode = _v13_to_14_cls(v13tree)
        root = Root( name=startnode.name )
        root.tree(startnode)

    # populate tree
    _populate_tree(startnode)
    return startnode



def _populate_tree(node):
    for key in node.tree.keys():
        newnode = _v13_to_14_cls(node.tree[key])
        node.tree(newnode)
        _populate_tree(newnode)




def _v13_to_14_cls(obj):
    """
    Convert a single version 13 object instance to the equivalent version 14 object,
    including metadata.
    """

    assert(isinstance(obj, (
        Root13,
        Metadata13,
        Array13,
        PointList13,
        PointListArray13,
        Calibration13,
        DataCube13,
        DiffractionSlice13,
        VirtualDiffraction13,
        RealSlice13,
        VirtualImage13,
        Probe13,
        QPoints13,
        Braggvectors13
    ))), f"obj must be a v13 class instance, not type {type(obj)}"

    if isinstance(obj, Root13):
        pass

    elif isinstance(obj, Metadata13):
        pass

    elif isinstance(obj, Array13):
        pass

    elif isinstance(obj, PointList13):
        pass

    elif isinstance(obj, PointListArray13):
        pass

    elif isinstance(obj, Calibration13):
        pass

    elif isinstance(obj, DataCube13):
        pass

    elif isinstance(obj, DiffractionSlice13):
        pass

    elif isinstance(obj, VirtualDiffraction13):
        pass

    elif isinstance(obj, RealSlice13):
        pass

    elif isinstance(obj, VirtualImage13):
        pass

    elif isinstance(obj, Probe13):
        pass

    elif isinstance(obj, QPoints13):
        pass

    elif isinstance(obj, BraggVectors13):
        pass

    else:
        raise Exception(f"Unexpected object type {type(obj)}")



    # Handle metadata




