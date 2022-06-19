# Utility functions for datastructure i/o

import numpy as np
import h5py
from ...tqdmnd import tqdmnd



# Define the EMD group types

EMD_group_types = {
    'Metadata' : 0,
    'Array' : 1,
    'PointList' : 2,
    'PointListArray': 3
}








# Utility functions for finding and validating EMD groups

def find_EMD_groups(group:h5py.Group, emd_group_type):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and finds all groups inside this group at its top level matching `emd_group_type`.
    Does not do a nested search. Returns the names of all groups found.

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




# This needs tending ;p

def determine_group_name(obj, group):
    """
    Takes an instance of a Python class instance and a valid HDF5 group
    for a h5py File which is open in write or append mode. Determines
    an appropirate name to assign the instance as represented in the h5
    file inside this group.

    If the instance has a .name attribute and that name is not already
    represented as a group at this level, returns this name.
    the instance's .name attribute is a valid addition to the h5 group.
    If the name is a duplicate of one already in the group, changes the
    object's .name attribute.
    
    
    an
    appropriate name for storing this object in the H5 file.

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




# Read and write for base EMD types





####################


## METADATA

# write
def Metadata_to_h5(metadata,group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open
    in write or append mode. Writes a new group with a name given by
    this Metadata instance's .name field nested inside the passed
    group, and saves the data there.

    If the Metadata instance has no name, it will be assigned the
    name Metadata"#" where # is the lowest available integer.  If the
    instance has a name which already exists here in this file, raises
    and exception.

    TODO: add overwite option.

    Accepts:
        group (HDF5 group)
    """

    # Detemine the name of the group
    # if current name is invalid, raises and exception
    # TODO: add overwrite option
    determine_group_name(metadata, group)

    ## Write
    grp = group.create_group(metadata.name)
    grp.attrs.create("emd_group_type",EMD_group_types['Metadata'])
    grp.attrs.create("py4dstem_class",metadata.__class__.__name__)

    # Save data
    for k,v in metadata._params.items():
        if isinstance(v,str): v = np.string_(v)
        grp.create_dataset(k, data=v)


# read
def Metadata_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid Metadata object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A Metadata instance
    """
    from .metadata import Metadata
    from os.path import basename

    er = f"Group {group} is not a valid EMD Metadata group"
    assert("emd_group_type" in group.attrs.keys()), er
    assert(group.attrs["emd_group_type"] == EMD_group_types['Metadata']), er

    # Get data
    data = {}
    for k,v in group.items():
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

    md = Metadata(basename(group.name))
    md._params.update(data)

    return md





#### ARRAY

# write

def Array_to_h5(array,group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    Array's .name field nested inside the passed group, and saves the
    data there.

    If the Array has no name, it will be assigned the name "Array#" where
    # is the lowest available integer.  If the Array's name already exists
    here in this file, raises and exception.

    TODO: add overwite option.

    Accepts:
        group (HDF5 group)
    """

    # Detemine the name of the group
    # if current name is invalid, raises and exception
    # TODO: add overwrite option
    determine_group_name(array, group)


    ## Write

    grp = group.create_group(array.name)
    grp.attrs.create("emd_group_type",1) # this tag indicates an Array
    grp.attrs.create("py4dstem_class",array.__class__.__name__)

    # add the data
    data = grp.create_dataset(
        "data",
        shape = array.data.shape,
        data = array.data,
        #dtype = type(array.data)
    )
    data.attrs.create('units',array.units) # save 'units' but not 'name' - 'name' is the group name

    # Add the normal dim vectors
    for n in range(array.rank):

        # unpack info
        dim = array.dims[n]
        name = array.dim_names[n]
        units = array.dim_units[n]
        is_linear = array._dim_is_linear(dim,array.shape[n])

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
    if array.is_stack:
        n = array.rank
        name = '_labels_'
        dim = [s.encode('utf-8') for s in array.slicelabels]

        # write
        dset = grp.create_dataset(
            f"dim{n}",
            data = dim
        )
        dset.attrs.create('name',name)

    # Add metadata
    grp_metadata = grp.create_group('metadata')
    for name,md in array._metadata.items():
        array._metadata[name].name = name
        array._metadata[name].to_h5(grp_metadata)


## read

def Array_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode.
    Determines if this group represents an Array object and if it does, loads
    returns it. If it doesn't, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        An Array instance
    """
    from .array import Array
    from os.path import basename

    er = f"Group {group} is not a valid EMD Array group"
    assert("emd_group_type" in group.attrs.keys()), er
    assert(group.attrs["emd_group_type"] == EMD_group_types['Array']), er

    # get data
    dset = group['data']
    data = dset[:]
    units = dset.attrs['units']
    rank = len(data.shape)

    # determine if this is a stack array
    last_dim = group[f"dim{rank-1}"]
    if last_dim.attrs['name'] == '_labels_':
        is_stack = True
        normal_dims = rank-1
    else:
        is_stack = False
        normal_dims = rank

    # get dim vectors
    dims = []
    dim_units = []
    dim_names = []
    for n in range(normal_dims):
        dim_dset = group[f"dim{n}"]
        dims.append(dim_dset[:])
        dim_units.append(dim_dset.attrs['units'])
        dim_names.append(dim_dset.attrs['name'])

    # if it's a stack array, get the labels
    if is_stack:
        slicelabels = last_dim[:]
        slicelabels = [s.decode('utf-8') for s in slicelabels]
    else:
        slicelabels = None

    # make Array
    ar = Array(
        data = data,
        name = basename(group.name),
        units = units,
        dims = dims,
        dim_names = dim_names,
        dim_units = dim_units,
        slicelabels = slicelabels
    )

    # add metadata
    grp_metadata = group['metadata']
    for key in grp_metadata.keys():
        ar.metadata = Metadata_from_h5(grp_metadata[key])

    return ar





##### POINTLIST


# write
def PointList_to_h5(pointlist,group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    PointList's .name field nested inside the passed group, and saves
    the data there.

    If the PointList has no name, it will be assigned the name "PointList#"
    where # is the lowest available integer.  If the PointList's name
    already exists here in this file, raises and exception.

    TODO: add overwite option.

    Accepts:
        group (HDF5 group)
    """

    # Detemine the name of the group
    # if current name is invalid, raises and exception
    # TODO: add overwrite option
    determine_group_name(pointlist, group)

    ## Write
    grp = group.create_group(pointlist.name)
    grp.attrs.create("emd_group_type",2) # this tag indicates a PointList
    grp.attrs.create("py4dstem_class",pointlist.__class__.__name__)

    # Add data
    for f,t in zip(pointlist.fields,pointlist.types):
        group_current_field = grp.create_group(f)
        group_current_field.attrs.create("dtype", np.string_(t))
        group_current_field.create_dataset("data", data=pointlist.data[f])

    # Add metadata
    grp_metadata = grp.create_group('metadata')
    for name,md in pointlist._metadata.items():
        pointlist._metadata[name].name = name
        pointlist._metadata[name].to_h5(grp_metadata)


# read
def PointList_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid PointList object of this name exists inside
    this group, and if it does, loads and returns it. If it doesn't, raises
    an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A PointList instance
    """
    from .pointlist import PointList
    from os.path import basename

    er = f"Group {group} is not a valid EMD PointList group"
    assert("emd_group_type" in group.attrs.keys()), er
    assert(group.attrs["emd_group_type"] == EMD_group_types['PointList']), er


    # Get metadata
    fields = list(group.keys())
    if 'metadata' in fields:
        fields.remove('metadata')
    dtype = []
    for field in fields:
        curr_dtype = group[field].attrs["dtype"].decode('utf-8')
        dtype.append((field,curr_dtype))
    length = len(group[fields[0]+'/data'])

    # Get data
    data = np.zeros(length,dtype=dtype)
    if length > 0:
        for field in fields:
            data[field] = np.array(group[field+'/data'])

    # Make the PointList
    pl = PointList(
        data=data,
        name=basename(group.name))

    # Add additional metadata
    grp_metadata = group['metadata']
    for key in grp_metadata.keys():
        pl.metadata = Metadata_from_h5(grp_metadata[key])

    return pl



##### POINTLISTARRAY


# write
def PointListArray_to_h5(pointlistarray,group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    write or append mode. Writes a new group with a name given by this
    PointListArray's .name field nested inside the passed group, and
    saves the data there.

    If the PointListArray has no name, it will be assigned the name
    "PointListArray#" where # is the lowest available integer.  If the
    PointListArray's name already exists here in this file, raises and
    exception.

    TODO: add overwite option.

    Accepts:
        group (HDF5 group)
    """

    # Detemine the name of the group
    # if current name is invalid, raises and exception
    # TODO: add overwrite option
    determine_group_name(pointlistarray, group)

    ## Write
    grp = group.create_group(pointlistarray.name)
    grp.attrs.create("emd_group_type",3) # this tag indicates a PointListArray
    grp.attrs.create("py4dstem_class",pointlistarray.__class__.__name__)

    # Add data
    dtype = h5py.special_dtype(vlen=pointlistarray.dtype)
    dset = grp.create_dataset(
        "data",
        pointlistarray.shape,
        dtype
    )
    for (i,j) in tqdmnd(dset.shape[0],dset.shape[1]):
        dset[i,j] = pointlistarray.get_pointlist(i,j).data

    # Add metadata
    grp_metadata = grp.create_group('metadata')
    for name,md in pointlistarray._metadata.items():
        pointlistarray._metadata[name].name = name
        pointlistarray._metadata[name].to_h5(grp_metadata)


# read
def PointListArray_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read mode,
    and a name.  Determines if a valid PointListArray object of this name exists
    inside this group, and if it does, loads and returns it. If it doesn't,
    raises an exception.

    Accepts:
        group (HDF5 group)
        name (string)

    Returns:
        A PointListArray instance
    """
    from .pointlistarray import PointListArray
    from os.path import basename

    er = f"Group {group} is not a valid EMD PointListArray group"
    assert("emd_group_type" in group.attrs.keys()), er
    assert(group.attrs["emd_group_type"] == EMD_group_types['PointListArray']), er


    # Get data
    dset = group['data']
    shape = group['data'].shape
    dtype = group['data'][0,0].dtype
    pla = PointListArray(
        dtype=dtype,
        shape=shape,
        name=basename(group.name)
    )
    for (i,j) in tqdmnd(shape[0],shape[1],desc="Reading PointListArray",unit="PointList"):
        try:
            pla.get_pointlist(i,j).append(dset[i,j])
        except ValueError:
            pass

    # Add metadata
    grp_metadata = group['metadata']
    for key in grp_metadata.keys():
        pla.metadata = Metadata_from_h5(grp_metadata[key])

    return pla





## subclasses




## DATACUBE


#read 
def DataCube_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in read
    mode.  Determines if an Array object of this name exists inside this group,
    and if it does, loads and returns it as a DataCube. If it doesn't exist, or if
    it exists but does not have a rank of 4, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DataCube instance
    """
    datacube = Array_from_h5(group)
    datacube = DataCube_from_Array(datacube)
    return datacube


def DataCube_from_Array(array):
    """
    Converts an Array to a DataCube.

    Accepts:
        array (Array)

    Returns:
        datacube (DataCube)
    """
    from .datacube import DataCube
    assert(array.rank == 4), "Array must have 4 dimensions"
    array.__class__ = DataCube
    array.__init__(
        data = array.data,
        name = array.name,
        R_pixel_size = [array.dims[0][1]-array.dims[0][0],
                        array.dims[1][1]-array.dims[1][0]],
        R_pixel_units = [array.dim_units[0],
                         array.dim_units[1]],
        Q_pixel_size = [array.dims[2][1]-array.dims[2][0],
                        array.dims[3][1]-array.dims[3][0]],
        Q_pixel_units = [array.dim_units[2],
                         array.dim_units[3]],
        slicelabels = array.slicelabels
    )
    return array





######## DiffractionSlice

# Reading

def DiffractionSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a DiffractionSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A DiffractionSlice instance
    """
    diffractionslice = Array_from_h5(group)
    diffractionslice = DiffractionSlice_from_Array(diffractionslice)
    return diffractionslice


def DiffractionSlice_from_Array(array):
    """
    Converts an Array to a DiffractionSlice.

    Accepts:
        array (Array)

    Returns:
        (DiffractionSlice)
    """
    from .diffractionslice import DiffractionSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = DiffractionSlice
    array.__init__(
        data = array.data,
        name = array.name,
        pixel_size = [array.dims[0][1]-array.dims[0][0],
                      array.dims[1][1]-array.dims[1][0]],
        pixel_units = [array.dim_units[0],
                       array.dim_units[1]],
        slicelabels = array.slicelabels
    )
    return array




######## RealSlice

# Reading

def RealSlice_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Array, and if so loads and
    returns it as a RealSlice. Otherwise, raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A RealSlice instance
    """
    realslice = Array_from_h5(group, name)
    realslice = RealSlice_from_Array(realslice)
    return realslice


def RealSlice_from_Array(array):
    """
    Converts an Array to a RealSlice.

    Accepts:
        array (Array)

    Returns:
        (RealSlice)
    """
    from .realslice import RealSlice
    assert(array.rank == 2), "Array must have 2 dimensions"
    array.__class__ = RealSlice
    array.__init__(
        data = array.data,
        name = array.name,
        pixel_size = [array.dims[0][1]-array.dims[0][0],
                      array.dims[1][1]-array.dims[1][0]],
        pixel_units = [array.dim_units[0],
                       array.dim_units[1]],
        slicelabels = array.slicelabels
    )
    return array




######### Calibration

# read
def Calibration_from_h5(group:h5py.Group):
    """
    Takes a valid HDF5 group for an HDF5 file object which is open in
    read mode. Determines if it's a valid Metadata representation, and
    if so loads and returns it as a Calibration instance. Otherwise,
    raises an exception.

    Accepts:
        group (HDF5 group)

    Returns:
        A Calibration instance
    """
    cal = Metadata_from_h5(group)
    cal = Calibration_from_Metadata(cal)
    return cal

def Calibration_from_Metadata(metadata):
    """
    Converts a Metadata instance to a Calibration instance.

    Accepts:
        metadata (Metadata)

    Returns:
        (Calibration)
    """
    from .calibration import Calibration
    p = metadata._params
    metadata.__class__ = Calibration
    metadata.__init__(
        name = metadata.name
    )
    metadata._params.update(p)

    return metadata




