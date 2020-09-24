"""
A wrapper for tqdm (https://github.com/tqdm/tqdm) that supports
multidimensional iterators. Iterates over the Cartesian product
of the input iterators. Integers in the input are automagically
conveted to range(i) for convenience.
Examples:
for x,y in tqdmnd(range(20),range(10)):
    sleep(0.1)

for x,y in tqdmnd(20,10):
    sleep(0.1)
"""

from tqdm import tqdm

from itertools import product
from functools import reduce
from operator import mul
from numpy import integer

from collections.abc import Iterator

class nditer(Iterator):
    def __init__(self,*args):
        if len(args) > 1:
            self._it = product(*args)
        else:
            self._it = args[0]
        self._l = reduce(mul,[a.__len__() for a in args])
    def __iter__(self):
        return self._it.__iter__()
    def __next__(self):
        return self._it.__next__()
    def __len__(self):
        return self._l

def tqdmnd(*args,**kwargs):
    r = [range(i) if isinstance(i,(int,integer)) else i for i in args]
    return tqdm(nditer(*r),**kwargs)
