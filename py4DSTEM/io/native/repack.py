from os import rename
from os.path import exists, dirname, basename
from .copy import copy

def repack(fp, topgroup='4DSTEM_experiment'):
    """
    Fully releases the storage space associated with any data blocks that have
    been 'removed' from this .h5 file but which are still allocated to the file.

    Accepts:
        fp             the filepath to an existing py4DSTEM .h5 file
        topgroup       The toplevel group
    """
    _fp = dirname(fp)+'_'+basename(fp)
    while exists(_fp):
        _fp = dirname(_fp)+'_'+basename(_fp)
    copy(fp,_fp,topgroup_orig=topgroup,topgroup_new=topgroup,delete=True)
    rename(_fp,fp)


