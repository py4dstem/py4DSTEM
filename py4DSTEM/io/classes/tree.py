import numpy as np
from numbers import Number
from typing import Optional
import h5py
from os.path import basename

from py4DSTEM.io.classes import Metadata


class Node:
    """
    Adds the following functionality:

        # tree fn'ality
        self.show_tree(all=False)  # whole tree or downstream
        self.add_to_tree(Node(name='node'))
        self.get_from_tree('node1/node2')
        self.get_from_tree('/rootnode/node1/node2')

        # all the above syntax in one place:
        self.tree(
            # no args = show downstream
            all = False,  # True -> whole tree
            add = Node(name='node'), # add to tree
            get = 'node1/node2'  # leading '/' gets from root
        )

        # metadata fn'ality
        self.metadata = Metadata(name='x')
        md = self.metadata['x']


    Does not add read/write methods.
    """
    def __init__(
        self,
        name: Optional[str] = 'node'
        ):
        self.name = name
        self._tree = Tree()
        self._metadata = {}

    # tree methods
    def show_tree(self,root=False):
        assert(isinstance(root,bool))
        if not root:
            self._tree.print()
        else:
            self.root._tree.print()

    def add_to_tree(self,node):
        assert(isinstance(node,Node))
        assert(hasattr(self,"_root")), "This node has no root node, so can't add new nodes."
        node._root = self._root
        self._tree[node.name] = node

    def get_from_tree(self,name):
        if name[0] != '/':
            return self._tree[name]
        else:
            return self.root._tree[name]

    def tree(
        self,
        arg=False,
        ):
        if isinstance(arg,bool):
            self.show_tree(root=arg)
        elif isinstance(arg,Node):
            self.add_to_tree(arg)
        elif isinstance(arg,str):
            return self.get_from_tree(arg)
        else:
            raise Exception(f'invalid argument type passed to .tree() {type(arg)}')

    # setting a root node
    @property
    def root(self):
        return self._root
    @root.setter
    def root(self,x):
        assert(isinstance(x,Root))
        self._root = x




    # supports storage/retreival of any number of Metadata instances
    # store: `node.metadata = x` for some Metadata instance `x`
    # retrieve: `node.metadata[x.name]` for some string `x.name`
    @property
    def metadata(self):
        return self._metadata
    @metadata.setter
    def metadata(self,x):
        assert(isinstance(x,Metadata))
        self._metadata[x.name] = x


    # displays top level contents of the node
    def __repr__(self):

        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( A Node called '{self.name}', containing the following top-level objects in its tree:"
        string += "\n"
        for k,v in self._tree.items():
            string += "\n"+space+f"    {k} \t\t ({v.__class__.__name__})"
        string += "\n)"
        return string





class Tree:

    def __init__(self):
        self._dict = {}



    # Enables adding items to the Tree's dictionary
    # TODO: add value validation - must be an EMD object
    # TODO: validate the key by excluding '/' chars
    def __setitem__(self, key, value):
        assert(isinstance(value,Node)), f"only Node instances can be added to tree. not type {type(value)}"
        self._dict[key] = value


    # Enables retrieving items at top level,
    # or nested items using 'node1/node2' syntax
    def __getitem__(self,x):
        l = x.split('/')
        try:
            l.remove('')
            l.remove('')
        except ValueError:
            pass
        return self._getitem_from_list(l)

    def _getitem_from_list(self,x):
        if len(x) == 0:
            raise Exception("invalid slice value to tree")

        k = x.pop(0)
        er = f"{k} not found in tree - check keys"
        assert(k in self._dict.keys()), er

        if len(x) == 0:
            return self._dict[k]
        else:
            tree = self._dict[k]._tree
            return tree._getitem_from_list(x)



    # return the top level keys and items
    def keys(self):
        return self._dict.keys()
    def items(self):
        return self._dict.items()



    # print the full tree contents to screen
    def print(self):
        """
        Prints the tree contents to screen.
        """
        print('/')
        self._print_tree_to_screen(self)
        print('\n')

    def _print_tree_to_screen(self, tree, tablevel=0, linelevels=[]):
        """
        """
        if tablevel not in linelevels:
            linelevels.append(tablevel)
        keys = [k for k in tree.keys()]
        #keys = [k for k in keys if k != 'metadata']
        N = len(keys)
        for i,k in enumerate(keys):
            string = ''
            string += '|' if 0 in linelevels else ''
            for idx in range(tablevel):
                l = '|' if idx+1 in linelevels else ''
                string += '\t'+l
            #print(string)
            print(string+'--'+k)
            if i == N-1:
                linelevels.remove(tablevel)
            try:
                self._print_tree_to_screen(
                    tree[k].tree,
                    tablevel=tablevel+1,
                    linelevels=linelevels)
            except AttributeError:
                pass

        pass



    # Displays the top level objects in the Tree
    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( An object tree containing the following top-level object instances:"
        string += "\n"
        for k,v in self._tree.items():
            string += "\n"+space+f"    {k} \t\t ({v.__class__.__name__})"
        string += "\n)"
        return string






class Root(Node):
    """
    A tree node which can read/write to H5 and which has
    a ParentTree rather than a Tree
    """
    def __init__(
        self,
        name: Optional[str] ='root'
        ):
        """
         Args:
            name (Optional, string):
        """
        super().__init__()
        self.name = name
        self._tree = Tree()
        self._root = self

    @property
    def root(self):
        return self._root


    # HDF5 i/o

    # write
    def to_h5(self,group):
        """
        Takes a valid group for an HDF5 file object which is open
        in write or append mode. Writes a new group with this object's
        name and saves this object's data in it.

        Accepts:
            group (HDF5 group)
        """
        grp = group.create_group(self.name)
        grp.attrs.create("emd_group_type",EMD_group_types['Root'])
        grp.attrs.create("python_class",self.metadata.__class__.__name__)

    # read
    def from_h5(group):
        """
        Takes a valid HDF5 group for an HDF5 file object
        which is open in read mode.  Determines if a valid
        Root object of this name exists inside this group, and if it does,
        loads and returns it. If it doesn't, raises an exception.

        Accepts:
            group (HDF5 group)

        Returns:
            A Root instance
        """
        er = f"Group {group} is not a valid EMD Root group"
        assert("emd_group_type" in group.attrs.keys()), er
        assert(group.attrs["emd_group_type"] == EMD_group_types['Root']), er

        root = Root(basename(group.name))
        return root




# TODO

# Defines the ParentTree class, which inherits from emd.Tree, and
# adds the ability to track a parent datacube.

#from py4DSTEM.io.classes.array import Array
#from py4DSTEM.io.classes.py4dstem.calibration import Calibration
#from numpy import ndarray


##;pclass ParentTree(Tree):
##;p
##;p    def __init__(self, parent, calibration):
##;p        """
##;p        Creates a tree which is aware of and can point objects
##;p        added to it to it's parent and associated calibration.
##;p        `parent` is typically a DataCube, but need not be.
##;p        `calibration` should be a Calibration instance.
##;p        """
##;p        #assert(isinstance(calibration, Calibration))
##;p
##;p        Tree.__init__(self)
##;p        self._dict['calibration'] = calibration
##;p        self._parent = parent



    #def __setitem__(self, key, value):
    #    if isinstance(value, ndarray):
    #        value = Array(
    #            data = value,
    #            name = key
    #        )
    #    self._tree[key] = value
    #    value._parent = self._parent
    #    value.calibration = self._parent.calibration













