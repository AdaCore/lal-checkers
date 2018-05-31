from itertools import chain, combinations
from collections import defaultdict
from funcy.calc import memoize
import time


class Bunch(dict):
    """
    Represents a bunch of data. Unlike a standard dict, the attribute notation
    can be used to get keys. Once constructed, the object is immutable.
    """
    def __init__(self, **kw):
        dict.__init__(self, kw)
        kw['_hash'] = hash(tuple(sorted(kw.keys())))
        self.__dict__.update(kw)

    def __setattr__(self, key, value):
        raise TypeError("Bunch does not support assignment")

    def __setitem__(self, key, value):
        raise TypeError("Bunch does not support assignment")

    def copy(self, **kwargs):
        new_items = super(Bunch, self).copy()
        new_items.update(kwargs)
        return Bunch(**new_items)

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


def zip_dicts(a, b, all=True):
    """
    Returns a dictionary, which contains a key for each key present in either
    a or b / both a and b, depending on whether all is set to True or not.
    To each key is associated a pair which first element is the value
    associated to this key in the first dict (None if absent), and which
    second element is the value associated to this key in the second dict (None
    if absent).

    :param dict a: The first dict.

    :param dict b: The second dict.

    :param bool all: If True, the resulting dict's key set is the union of the
        key sets of the two given dicts, otherwise its intersection.

    :rtype: dict
    """
    op = frozenset.__or__ if all else frozenset.__and__
    return {
        k: (a.get(k, None), b.get(k, None))
        for k in op(frozenset(a.keys()), frozenset(b.keys()))
    }


def concat_dicts(a, b):
    return dict(a, **b)


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

    def catch(self, error_type):
        """
        Creates a transformer which catches errors of the given error type.
        If such an error is raising during transformation, this transformer
        will simply return None instead of propagating the error further.

        :param type error_type: The error type.
        :rtype: Transformer
        """
        @self.as_transformer
        def f(hint):
            try:
                return self._transform(hint)
            except error_type:
                return None

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
    def first_of(*others):
        """
        Given a list of transformers, creates a transformer which attempts to
        transform the given input with each transformer successively until
        an attempt succeeds.

        Note: is functionally equivalent to combining every transformer with
        the | operator.

        :param *Transformer others: The transformers to consider.
        :rtype: Transformer
        """
        @Transformer.as_transformer
        def f(hint):
            for other in others:
                x = other._transform(hint)
                if x is not None:
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
        memoized_builder = memoize(builder)
        return Transformer(lambda hint: memoized_builder()._transform(hint))

    @staticmethod
    def make_memoizing(transformer):
        """
        Constructs a transformer that memoizes transformed objects from an
        underlying transformer.

        :param Transformer transformer: The underlying transformer.

        :rtype: Transformer
        """
        return Transformer(memoize(transformer._transform))


class _StopWatch(object):
    @staticmethod
    def get_output_file():
        try:
            with open('profiler_config', 'r') as f:
                return f.readline()
        except IOError:
            return ''

    def __init__(self):
        self.timings = {}
        self.started = {}

    def register(self, name):
        self.timings[name] = 0

    def start(self, name):
        if name not in self.started:
            self.started[name] = [0,  time.clock()]
        else:
            self.started[name][0] += 1

    def stop(self, name):
        info = self.started[name]
        if info[0] == 0:
            self.timings[name] += time.clock() - info[1]
            del self.started[name]
        else:
            self.started[name][0] -= 1

    def inc(self, name, t):
        self.timings[name] += t

    def __del__(self):
        output_file = self.get_output_file()
        if len(output_file) == 0:
            return

        text = '\n---- Profiling Results ----\n\n' + '\n'.join(
            "Total time spent in {}: {} seconds.".format(name, t)
            for name, t in self.timings.iteritems()
        )

        if output_file == '<stdout>':
            print(text)
        else:
            with open(output_file, 'w') as f:
                f.write(text)


_stopwatch = _StopWatch()


def profile(use_name=None):
    if len(_stopwatch.get_output_file()) == 0:
        return lambda f: f

    def do(fun):
        f_name = fun.__name__ if use_name is None else use_name

        _stopwatch.register(f_name)

        def f(*args, **kwargs):
            _stopwatch.start(f_name)
            try:
                return fun(*args, **kwargs)
            finally:
                _stopwatch.stop(f_name)

        return f

    return do
