from os import rename
from os.path import exists, dirname, basename
from .copy import copy

def repack(filepath, topgroup='4DSTEM_experiment'):
    """
    Fully releases the storage space associated with any data blocks that have
    been 'removed' from this .h5 file but which are still allocated to the file.
    See the docstring for ``io.write.append`` for more info.

    Args:
        filepath: the filepath to an existing py4DSTEM .h5 file
        topgroup: the toplevel group
    """
    _filepath = dirname(filepath)+'_'+basename(filepath)
    while exists(_filepath):
        _filepath = dirname(_filepath)+'_'+basename(_filepath)
    copy(filepath,_filepath,topgroup_orig=topgroup,topgroup_new=topgroup,delete=True)
    rename(_filepath,filepath)


