# Append additional DataObjects to an existing py4DSTEM formatted .h5 file.
# 
# See filestructure.txt for a description of the file structure.

from ._append import _append
from .repack import repack

def append(filepath, data, overwrite=0, topgroup='4DSTEM_experiment'):
    """
    Appends data to an existing py4DSTEM .h5 file at filepath.

    Note:
        The HDF5 format does not allow easy release of storage space - naively
        'deleting' an object removes associated links and names but maintains
        control of the associated datablocks. Consequently, when using this
        function to append a datablock with the same name as some existing block
        of data, the precise behavior must be specified using the ``overwrite``
        argument.

        Using `overwrite=1` performs a naive deletion of this sort.  This is likely
        fine on a one-off basis for small data blocks - e.g. a single 2D array -
        but will quickly lead to file size bloat if done repeatedly or done with
        larger data elements, e.g. DataCubes or PointListArrays.  If you want to
        do this and then decide to release the space later, you can do

            >>> io.native.repack(filepath)

        Using `overwrite=2` fully deletes any blocks of data being overwritten.
        To do so, it copies all the dataobjects we need to keep to a new file,
        then delete the original file and replace it with the new one.  When
        complete, this will have released the storage space, however note that
        performing the task will then require additional CPU and storage resources.

    Args:
        filepath: path to an existing py4DSTEM .h5 file to append to
        data: a single DataObject or a list of DataObjects or Metadata instance
        overwrite (int, optional): controls behavior when a dataobject with the same name
            as one already in the .h5 file is found. Must be in {0,1,2}.  Behaviors:

                * **0**: print an error message identifying any duplicate names, don't
                  save anything
                * **1**: remove the .h5 group that exists in the file and saves the new
                  object. Does NOT release the associated storage space in the file. See
                  the Note above.
                * **2**: remove the .h5 group that exists in the file and saves the new
                  obect.  DOES release the associated storage space in the file.  See
                  the Note above.

            If a Metadata instance is passed, this flag also controls behavior for
            conflicting metadata items; a 0 skips conflicting items, and 1 or 2
            overwrites them.
        topgroup (str): name of the h5 toplevel group containing the py4DSTEM file of
            interest
    """
    _append(filepath,data,overwrite,topgroup)
    if overwrite==2:
        repack(filepath,topgroup)

