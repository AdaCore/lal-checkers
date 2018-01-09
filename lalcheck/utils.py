from itertools import chain, combinations
from collections import defaultdict
from funcy.calc import memoize


class Bunch(dict):
    """
    Represents a bunch of data. Unlike a standard dict, the attribute notation
    can be used to get keys. Once constructed, the object is immutable.
    """
    def __init__(self, **kw):
        kw['_hash'] = tuple(sorted(self.iteritems())).__hash__()
        dict.__init__(self, kw)
        self.__dict__.update(kw)

    def __setattr__(self, key, value):
        raise TypeError("Bunch does not support assignment")

    def __setitem__(self, key, value):
        raise TypeError("Bunch does not support assignment")

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


def partition(it, predicate):
    """
    Partitions an iterable into two lists, one containing all the elements
    that satisfy the given predicate, and one containing the others.

    :param iterable[T] it: The iterable to partition.
    :param T->bool predicate: The partition predicate.
    """
    res_true, res_false = [], []
    for x in it:
        (res_true if predicate(x) else res_false).append(x)
    return res_true, res_false


class Transformer(object):
    def __init__(self, fun):
        """
        Creates a new transformer using the given function.

        :param object -> (object | None) fun: The transforming function.
        """
        self._transform = fun

    def __or__(self, other):
        """
        Combines two transformer in a fallback fashion: apply the other
        transformer if the first one returned an invalid value.

        :param Transformer other: The transformer to combine with.
        :rtype: Transformer
        """
        @self.as_transformer
        def f(hint):
            x = self._transform(hint)
            return x if x is not None else other._transform(hint)

        return f

    def __and__(self, other):
        """
        Combines two transformer to form a single transformer that transforms
        pairs of values. The transformation succeeds if both individual
        transformer succeed.

        :param Transformer other: The transformer to combine with.
        :rtype: Transformer
        """
        @self.as_transformer
        def f(hint):
            a, b = hint
            new_a = self._transform(a)
            if new_a is not None:
                new_b = other._transform(b)
                if new_b is not None:
                    return new_a, new_b

            return None

        return f

    def __rshift__(self, other):
        """
        Combines two transformer in a chain fashion: use the other
        transformer on the value transformed by the first if this value is
        valid.

        :param Transformer other: The transformer to combine with.
        :rtype: Transformer
        """
        @self.as_transformer
        def f(hint):
            x = self._transform(hint)
            return None if x is None else other._transform(x)

        return f

    or_else = __or__
    and_then = __rshift__

    def lifted(self):
        """
        Turns a Transformer that works on values of type T to a transformer
        that works on values of type iterable[T].

        :rtype: Transformer
        """
        @self.as_transformer
        def f(hints):
            res = [self._transform(hint) for hint in hints]
            return res if all(x is not None for x in res) else None

        return f

    def get(self, x):
        """
        Forces transformation of the given argument.

        :param object x: The object to transform.
        :rtype: object
        :raise ValueError: if the transformation failed.
        """
        res = self._transform(x)
        if res is None:
            raise ValueError("Could not transform {}".format(x))
        return res

    @staticmethod
    def identity():
        @Transformer.as_transformer
        def f(x):
            return x
        return f

    @staticmethod
    def as_transformer(fun):
        """
        Constructs a transformer from a function.

        :param object -> (object | None) fun: The transforming function.
        :rtype: Transformer
        """
        return Transformer(fun)

    @staticmethod
    def from_transformer_builder(builder):
        """
        Constructs a transformer from a function that returns a transformer.

        :param () -> Transformer builder: A function that returns a
            transformer.

        :rtype: Transformer
        """
        return Transformer(lambda hint: builder()._transform(hint))

    @staticmethod
    def make_memoizing(transformer):
        """
        Constructs a transformer that memoizes transformed objects from an
        underlying transformer.

        :param Transformer transformer: The underlying transformer.

        :rtype: Transformer
        """
        return Transformer(memoize(transformer._transform))
