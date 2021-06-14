import h5py
from os.path import exists
from ...datastructure import Metadata
from .h5rw import h5write,h5read,h5append,h5info
from ..read import is_py4DSTEM_file,get_py4DSTEM_topgroups

def metadata_to_h5(fp, metadata, overwrite=False, topgroup='4DSTEM_experiment'):
    """
    Saves metadata to an existsing py4DSTEM .h5 file.

    If fp points to an existing .h5 file, and if no metadata is
    present, writes it there.
    If fp points to an existing .h5 file which already contains
    some metadata, and there are no conflicting metadata items,
    adds the new metadata.  If there are conflicts, overwrites
    iff overwrite=True, otherwise skips these items.
    If an existing .h5 file has more than one topgroup, the
    'topgroup' argument should be passed to specify which group
    to write to.
    If there is no file at fp, raises an error.  Writing metadata
    only to an otherwise empty .h5 can be accomplished with
        >>> io.save(filepath,metadata)
    """
    assert isinstance(metadata,Metadata)
    assert exists(fp)
    assert is_py4DSTEM_file(fp), "File found at path {} is not recognized as a py4DSTEM file.".format(fp)
    tgs = get_py4DSTEM_topgroups(fp)
    assert(topgroup in tgs)
    tg = topgroup

    with h5py.File(fp,'r+') as f:
        if 'metadata' not in f[tg]: f[tg].create_group('metadata')
        mgp = f[tg + '/metadata']
        if 'microscope' not in mgp: mgp.create_group('microscope')
        if 'sample' not in mgp: mgp.create_group('sample')
        if 'user' not in mgp: mgp.create_group('user')
        if 'calibration' not in mgp: mgp.create_group('calibration')
        if 'comments' not in mgp: mgp.create_group('comments')

    h5append(fp,tg+'/metadata/microscope',metadata.microscope)
    h5append(fp,tg+'/metadata/calibration',metadata.calibration)
    h5append(fp,tg+'/metadata/sample',metadata.sample)
    h5append(fp,tg+'/metadata/user',metadata.user)
    h5append(fp,tg+'/metadata/comments',metadata.comments)

    return

def metadata_from_h5(fp, topgroup='4DSTEM_experiment'):
    """
    Reads the metadata in a py4DSTEM formatted .h5 file and stores it in a
    Metadata instance.
    """
    assert is_py4DSTEM_file(fp), "File found at path {} is not recognized as a py4DSTEM file.".format(fp)
    tgs = get_py4DSTEM_topgroups(fp)
    assert(topgroup in tgs)
    tg = topgroup

    # Read the dictionaries
    microscope = h5read(fp,tg+'/metadata/microscope')
    calibration = h5read(fp,tg+'/metadata/calibration')
    sample = h5read(fp,tg+'/metadata/sample')
    user = h5read(fp,tg+'/metadata/user')
    comments = h5read(fp,tg+'/metadata/comments')

    # Put them into a Metadata instance
    metadata = Metadata()
    for key in microscope: metadata.microscope[key] = microscope[key]
    for key in calibration: metadata.calibration[key] = calibration[key]
    for key in sample: metadata.sample[key] = sample[key]
    for key in user: metadata.user[key] = user[key]
    for key in comments: metadata.comments[key] = comments[key]

    return metadata








