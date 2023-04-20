# Convert v13 to v14 classes

from emdfile import tqdmnd


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



def v13_to_14( v13tree ):
    """
    Converts a v13 data tree to a v14 data tree
    """

    # if a list of root names was returned, pass it through
    if isinstance(v13tree, list):
        return v13tree

    # make a root and fine the node to grow from
    if isinstance(v13tree,Root13):
        node = _v13_to_14_cls(v13tree)
    else:
        node = _v13_to_14_cls(v13tree)
        root = Root( name=node.name )
        root.tree(node)

    # populate tree
    _populate_tree(v13tree,node)
    return node



def _populate_tree(node13,node14):
    for key in node13.tree.keys():
        newnode13 = node13.tree[key]
        newnode14 = _v13_to_14_cls(newnode13)
        node14.tree(newnode14)
        _populate_tree(newnode13,newnode14)




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
        BraggVectors13
    ))), f"obj must be a v13 class instance, not type {type(obj)}"

    if isinstance(obj, Root13):
        x = Root( name=obj.name )

    elif isinstance(obj, Metadata13):
        x = Metadata( name=obj.name )
        x._params.update( obj._params )

    elif isinstance(obj, Array13):

        # prepare arguments
        args = {
            'name' : obj.name,
            'data' : obj.data
        }
        if hasattr(obj,'units'): args['units'] = obj.units
        if hasattr(obj,'dim_names'): args['dim_names'] = obj.dim_names
        if hasattr(obj,'dim_units'): args['dim_units'] = obj.dim_units
        if hasattr(obj,'slicelabels'): args['slicelabels'] = obj.slicelabels
        if hasattr(obj,'dims'):
            dims = []
            for dim in obj.dims:
                dims.append(dim)
            args['dims'] = dims

        # get the array
        x = Array(
            **args
        )

    elif isinstance(obj, PointList13):
        x = PointList(
            name = obj.name,
            data = obj.data
        )

    elif isinstance(obj, PointListArray13):
        x = PointListArray(
            name = obj.name,
            dtype = obj.dtype,
            shape = obj.shape
        )
        for idx,jdx in tqdmnd(
            x.shape[0],x.shape[1],
            desc='transferring PointListArray v13->14',
            unit='foolishness'):
            x[idx,jdx] = obj[idx,jdx]

    elif isinstance(obj, Calibration13):
        x = Calibration( name=obj.name )
        x._params.update( obj.params )

    elif isinstance(obj, DataCube13):
        x = DataCube(
            name = obj.name,
            data = obj.data,
            slicelabels = obj.slicelabels
        )

    elif isinstance(obj, DiffractionSlice13):
        x = DiffractionSlice(
            name = obj.name,
            data = obj.data,
            units = obj.units,
            slicelabels = obj.slicelabels
        )

    elif isinstance(obj, VirtualDiffraction13):
        x = VirtualDiffraction(
            name = obj.name,
            data = obj.data
        )

    elif isinstance(obj, RealSlice13):
        x = RealSlice(
            name = obj.name,
            data = obj.data,
            units = obj.units,
            slicelabels = obj.slicelabels
        )
        pass

    elif isinstance(obj, VirtualImage13):
        x = VirtualImage(
            name = obj.name,
            data = obj.data
        )
        pass

    elif isinstance(obj, Probe13):
        x = Probe(
            name = obj.name,
            data = obj.data
        )

    elif isinstance(obj, QPoints13):
        x = PointList(
            name = obj.name,
            data = obj.data
        )

    elif isinstance(obj, BraggVectors13):
        x = BraggVectors(
            name = obj.name,
            Rshape = obj.Rshape,
            Qshape = obj.Qshape
        )
        x._v_uncal = obj._v_uncal
        if hasattr(obj,'_v_cal'):
            x._v_cal = obj._v_cal

    else:
        raise Exception(f"Unexpected object type {type(obj)}")



    # Handle metadata
    if hasattr(obj,'metadata'):
        for key in obj.metadata.keys():
            md = obj.metadata[key]
            dm = Metadata( name=md.name )
            dm._params.update( md._params )
            x.metadata = dm



    # Return
    return x

