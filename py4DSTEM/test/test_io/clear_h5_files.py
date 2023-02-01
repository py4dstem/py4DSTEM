from os import remove
from os.path import exists,join

from py4DSTEM import _TESTPATH as dirpath


# Set paths
single_node_path = join(dirpath,"test_h5_single_node.h5")
unrooted_node_path = join(dirpath,"test_h5_unrooted_node.h5")
whole_tree_path = join(dirpath,"test_h5_whole_tree.h5")
subtree_path = join(dirpath,"test_h5_subtree.h5")
subtree_noroot_path = join(dirpath,"test_h5_subtree_noroot.h5")
subtree_append_to_path = join(dirpath,"test_h5_append_to.h5")
subtree_append_over_path = join(dirpath,"test_h5_append_over.h5")


paths = [
    single_node_path,
    unrooted_node_path,
    whole_tree_path,
    subtree_path,
    subtree_noroot_path,
    subtree_append_to_path,
    subtree_append_over_path
]
for p in paths:
    if exists(p):
        remove(p)


