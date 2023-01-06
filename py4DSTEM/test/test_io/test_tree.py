import py4DSTEM
from os.path import join

# Set filepaths
# TODO: add small 4D datacube
filepath_datacube = join(py4DSTEM._TESTPATH, "small_dm3.dm3")



def test_tree():

    data = py4DSTEM.import_file( filepath_datacube )

    # Make a tree of basic EMD objects

    # Write then read:
    # - root node
    # - some other single node
    # - whole tree from root
    # - some other whole subtree
    # - rootless tree from root
    # - rootless tree from some subtree

    # From the whole tree written from root, read:
    # - the root node
    # - the rootless tree

    # From the rootless tree, read:
    # - the (empty) root


