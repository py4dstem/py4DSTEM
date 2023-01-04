# Defines the Tree class, which stores the tree of child data
# class instances contained by a given class instance


class Tree:

    def __init__(self):
        self._tree = {}



    def __setitem__(self, key, value):
        self._tree[key] = value



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
        assert(k in self._tree.keys()), er

        if len(x) == 0:
            return self._tree[k]
        else:
            tree = self._tree[k].tree
            return tree._getitem_from_list(x)





    def __repr__(self):
        space = ' '*len(self.__class__.__name__)+'  '
        string = f"{self.__class__.__name__}( An object tree containing the following top-level object instances:"
        string += "\n"
        for k,v in self._tree.items():
            string += "\n"+space+f"    {k} \t\t ({v.__class__.__name__})"
        string += "\n)"
        return string

    def keys(self):
        return self._tree.keys()




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









