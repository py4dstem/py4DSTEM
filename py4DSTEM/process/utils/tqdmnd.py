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
    def __init__(self, *args):
        if len(args) > 1:
            self._it = product(*args)
        else:
            self._it = args[0]
        self._l = reduce(mul, [a.__len__() for a in args])

    def __iter__(self):
        return self._it.__iter__()

    def __next__(self):
        return self._it.__next__()

    def __len__(self):
        return self._l


def tqdmnd(*args, **kwargs):
    """
    An N-dimensional extension of tqdm providing an iterator and
    progress bar over the product of multiple iterators.

    Example Usage:
        for rx, ry in tqdmnd(datacube.R_Nx, datacube.R_Ny):
            data[rx,ry] = ...
    is equivalent to the following, while also providing a progress bar:
        for rx in range(datacube.R_Nx):
            for ry in range(datacube.R_Ny):
                data[rx,ry] = ...

    Accepts:
        *args:  Any number of iterators. The Cartesian product of these
                iterators is returned. Any integers `I` passed as arguments
                will be interpreted as `range(I)`. The input iterators must
                have a known length.
        **kwargs: keyword arguments passed through directly to tqdm.
                Full details are available at https://tqdm.github.io
                Some useful ones you'll encounter in py4DSTEM are:
                    disable (bool): hide the progress bar when True
                    keep (bool): delete the progress bar after completion when True
                    unit (str): pretty unit name for the display of iteration speed
                    unit_scale (bool): whether to scale the displayed units and add SI prefixes
                    desc (str): message displayed in front of the progress bar

    Returns:
        At each iteration, a tuple of indices is returned, corresponding to the 
        values of each input iterator (in the same order as the inputs). 
    """
    r = [range(i) if isinstance(i, (int, integer)) else i for i in args]
    return tqdm(nditer(*r), **kwargs)
