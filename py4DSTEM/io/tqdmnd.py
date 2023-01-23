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

        >>> for x,y in tqdmnd(5,6):
        >>>     <expression>

    is equivalent to

        >>> for x in range(5):
        >>>     for y in range(6):
        >>>         <expression>

    with a tqdmnd-style progress bar printed to standard output.

    Accepts:
        *args: Any number of integers or iterators. Each integer N
            is converted to a `range(N)` iterator. Then a loop is
            constructed from the Cartesian product of all iterables.
        **kwargs: keyword arguments passed through directly to tqdm.
            Full details are available at https://tqdm.github.io
            A few useful ones:
                disable (bool): if True, hide the progress bar
                keep (bool): if True, delete the progress bar after completion
                unit (str): unit name for the display of iteration speed
                unit_scale (bool): whether to scale the displayed units and add
                    SI prefixes
                desc (str): message displayed in front of the progress bar

    Returns:
        At each iteration, a tuple of indices is returned, corresponding to the
        values of each input iterator (in the same order as the inputs).
    """
    r = [range(i) if isinstance(i, (int, integer)) else i for i in args]
    return tqdm(nditer(*r), **kwargs)
