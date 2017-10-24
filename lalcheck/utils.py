from itertools import chain, combinations
from collections import defaultdict


class Bunch(dict):
    """
    Represents a bunch of data. Unlike a standard dict, the attribute notation
    can be used to set or get keys.
    """
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)
        self._hash = tuple(sorted(self.iteritems())).__hash__()

    def __hash__(self):
        return self._hash


class KeyCounter(object):
    """
    A dict which provides facilities for counting keys.
    """
    def __init__(self):
        self.dict = defaultdict(lambda: 0)

    def get_incr(self, item):
        val = self.dict[item]
        self.dict[item] += 1
        return val

    def incr(self, item):
        self.dict[item] += 1

    def __getitem__(self, item):
        return self.dict[item]


def powerset(iterable):
    """
    Returns the powerset of the given iterable as a frozenset of frozensets.
    """
    xs = list(iterable)
    res = chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))
    return frozenset(frozenset(x for x in tp) for tp in res)
