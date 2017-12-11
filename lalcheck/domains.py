"""
Provides some basic abstract domains.
"""

from utils import powerset
import itertools
import collections


class AbstractDomain(object):
    """
    Provides an interface for abstract domains, to guarantee the existence of
    basic operations.

    Abstract domains are also expected to provide a "bottom" element,
    which is less than any other element.
    Typically, abstract domains also provide a "top" element (but are not
    required to), which is greater than any other element.

    Note that some of the methods (e.g. is_empty), as well as the documentation
    of some methods are written assuming that we lie in the context of static
    analysis and abstract interpretation, where an abstract value typically
    represents a set of concrete values. Therefore, they may not make sense if
    the domain is used as a general purpose set, partial order, lattice, etc.

    Let's look at an example.

    The "Intervals(-2, 2)" domain can be used to represent subsets of the
    following set of concrete values: {-2, -1, 0, 1, 2}.
    - The "bottom" element represents the subset {}.
    - The "[0, 2]" element represents the subset {0, 1, 2}
    - The "top" element is [-2, 2] and represents the subset {-2, -1, 0, 1, 2}.
    Note how the subset {0, 2} does not have a perfect abstraction.
    Its best abstraction is "[0, 2]".
    """

    def build(self, *args):
        """
        Builds a new element of this abstract domain.
        """
        raise NotImplementedError

    def is_empty(self, x):
        """
        Returns True if the given element represents an empty set of
        concrete values.

        Often, the "bottom" element represents an empty set of
        concrete values, but this may not always be true.
        If it is not the case, this behavior should be overriden.
        """
        return self.eq(x, self.bottom)

    def join(self, a, b):
        """
        Returns the LUB (Least Upper Bound) of two elements belonging to this
        abstract domain.

        Typically, the join of two elements represents more concrete values
        than the union of the set of concrete values represented by
        each element.

        For example "Intervals(-2, 2).join((-2, -1), (1, 2))" returns
        "(-2, 2)", which represents {-2, -1, 0, 1, 2}. However, the union of
        {-2, -1} and {1, 2} is {-2, -1, 1, 2}, which does not contain 0.
        """
        raise NotImplementedError

    def meet(self, a, b):
        """
        Returns the GLB (Greatest Lower Bound) of two elements belonging to
        this abstract domain.
        """
        raise NotImplementedError

    def update(self, a, b, widen=False):
        """
        Returns an upper bound of the given elements.

        If widen is True, the given upper bound is such
        that applying a finite (preferably small) number of consecutive
        widening will return the "top" element of this domain.

        If widen is False, it will generally return the join of these
        two elements.
        """
        return self.join(a, b)

    def lt(self, a, b):
        """
        Returns True if the first element is less than the second element,
        that is, if the first element represents a smaller set of concrete
        values than the second one.
        """
        raise NotImplementedError

    def eq(self, a, b):
        """
        Returns True if the two elements are equal, that is, if they
        represent the same set of concrete values.
        """
        raise NotImplementedError

    def le(self, a, b):
        """
        Returns True if the first element is less than or equal to the second
        element, that is, if the first element represents a smaller set of
        concrete values than the second one, or the same one.
        """
        return self.eq(a, b) or self.lt(a, b)

    def gt(self, a, b):
        """
        Returns True if the first element is greater than the second element,
        that is, if the first element represents a bigger set of concrete
        values than the second one.
        """
        return self.lt(b, a)

    def ge(self, a, b):
        """
        Returns True if the first element is greater than or equal to the
        second element, that is, if the first element represents a bigger set
        of concrete values than the second one, or the same one.
        """
        return self.le(b, a)

    def lowest_among(self, xs):
        """
        Returns the lowest element among the given elements, that is, the
        element with represent the smallest amount of concrete values. It
        will raise "StopIteration" if such an element does not exist.
        """
        return next(
            x for x in xs if all(
                self.lt(x, y)
                for y in xs
                if not self.eq(x, y)
            )
        )

    def greatest_among(self, xs):
        """
        Returns the greatest element among the given elements, that is, the
        element with represent the biggest amount of concrete values. It
        will raise "StopIteration" if such an element does not exist.
        """
        return next(
            x for x in xs if all(
                self.gt(x, y)
                for y in xs
                if not self.eq(x, y)
            )
        )

    def generator(self):
        """
        Returns an iterable of all the abstract elements of this domain.
        """
        raise NotImplementedError

    def concretize(self, abstract):
        """
        Returns an iterable of concrete elements represented by the given
        abstract element.
        """
        raise NotImplementedError

    def abstract(self, concrete):
        """
        Returns the abstract element used to represent the given concrete
        element.
        """
        raise NotImplementedError


class Intervals(AbstractDomain):
    """
    An abstract domain used to represent sets of integers.
    """
    def __init__(self, m_inf, inf):
        """
        Constructs a new abstract domain of intervals in the range of the two
        given values, which are considered by said domain as
        -infinity and +infinity.
        """
        assert(m_inf <= inf)
        self.bottom = object()
        self.top = (m_inf, inf)

    def build(self, *args):
        """
        Creates a new interval.
        - If no argument is given, returns the "bottom" element representing
          the empty set of integers.
        - If one argument is given, returns the element representing
          the singleton set containing the given concrete value.
        - If two arguments are given, returns the element representing
          the set of integers that lie between these two values, provided
          these two values are inside this domain's range.
        """
        if len(args) == 0:
            return self.bottom
        elif len(args) == 1:
            return args[0], args[0]
        elif (len(args) == 2 and
              self.top[0] <= args[0] <= args[1] <= self.top[1]):
            return args[0], args[1]

    def left_unbounded(self, x):
        """
        Returns the element representing the set of integers that lie between
        -infinity (according to this domain) and the given value.
        """
        if x <= self.top[1]:
            return self.top[0], x

    def right_unbounded(self, x):
        """
        Returns the element representing the set of integers that lie between
        the given value and +infinity (according to this domain).
        """
        if x >= self.top[0]:
            return x, self.top[1]

    def join(self, a, b):
        if a == self.bottom:
            return b
        elif b == self.bottom:
            return a
        else:
            return min(a[0], b[0]), max(a[1], b[1])

    def meet(self, a, b):
        if a == self.bottom:
            return a
        elif b == self.bottom:
            return b
        elif a[1] < b[0] or b[1] < a[0]:
            return self.bottom
        else:
            return max(a[0], b[0]), min(a[1], b[1])

    def update(self, a, b, widen=False):
        if widen:
            return b if a == self.bottom else (
                a[0] if a[0] <= b[0] else self.top[0],
                a[1] if a[1] >= b[1] else self.top[1]
            )
        else:
            return self.join(a, b)

    def lt(self, a, b):
        if a == self.bottom:
            return b != self.bottom
        elif b == self.bottom:
            return False
        else:
            return (a[0] >= b[0] and a[1] <= b[1] and
                    (a[0] != b[0] or a[1] != b[1]))

    def eq(self, a, b):
        return a == b

    def le(self, a, b):
        return self.lt(a, b) or self.eq(a, b)

    def generator(self):
        dom_from, dom_to = self.top[0], self.top[1]
        for x_f in range(dom_from, dom_to + 1):
            for x_t in range(x_f, dom_to + 1):
                yield (x_f, x_t)

    def concretize(self, abstract):
        if abstract == self.bottom:
            return frozenset([])
        else:
            return frozenset(range(abstract[0], abstract[1] + 1))

    def abstract(self, concrete):
        return self.build(min(concrete), max(concrete))


class Product(AbstractDomain):
    """
    An abstract domain used to represent the cartesian product of
    sets of concrete values.
    """
    def __init__(self, *domains):
        """
        Constructs a new abstract domain from the given instances of abstract
        domains.
        """
        self.domains = list(domains)
        self.bottom = tuple(d.bottom for d in domains)
        self.top = tuple(d.top for d in domains)

    def build(self, *args):
        """
        Creates a new element representing the cartesian product of the given
        elements.
        """
        assert len(args) == len(self.domains)
        return tuple(args)

    def is_empty(self, x):
        # a cartesian product is empty iff any of its operand is empty.
        return any(
            domain.is_empty(v)
            for domain, v in zip(self.domains, x)
        )

    def join(self, a, b):
        return tuple(
            domain.join(x, y)
            for domain, x, y in zip(self.domains, a, b)
        )

    def meet(self, a, b):
        return tuple(
            domain.meet(x, y)
            for domain, x, y in zip(self.domains, a, b)
        )

    def update(self, a, b, widen=False):
        return tuple(
            domain.update(x, y, widen)
            for domain, x, y in zip(self.domains, a, b)
        )

    def lt(self, a, b):
        return all(
            domain.lt(x, y)
            for domain, x, y in zip(self.domains, a, b)
        )

    def eq(self, a, b):
        return all(
            domain.eq(x, y)
            for domain, x, y in zip(self.domains, a, b)
        )

    def generator(self):
        return itertools.product(*(
            dom.generator() for dom in self.domains
        ))

    def concretize(self, abstract):
        return frozenset(itertools.product(*(
            dom.concretize(x)
            for dom, x in zip(self.domains, abstract)
        )))

    def abstract(self, concrete):
        return tuple(
            dom.abstract(frozenset(concretes))
            for dom, concretes in zip(self.domains, zip(*concrete))
        )


class Set(AbstractDomain):
    """
    An abstract domain used to represent sets of sets of concrete values.
    Two sets of concrete values are considered equal according to the provided
    merge predicate. When two values are considered equal, they are joined.
    """
    def __init__(self, dom, merge_predicate, top):
        """
        Constructs a new abstract domain from the given abstract domain and the
        merge predicate. Its elements will represent sets of elements of the
        given abstract domain. Two elements will be considered equal and merged
        iff they satisfy the merge predicate or are equal according to inner
        domain. A top element for the domain must also be provided.
        """
        def actual_predicate(a, b):
            return (dom.eq(a, b) or
                    merge_predicate(a, b) or
                    merge_predicate(b, a))

        self.dom = dom
        self.merge_predicate = actual_predicate
        self.bottom = []
        self.top = top

    def build(self, elems):
        """
        Creates a new set which contains the given iterable of elements.
        """
        return self._reduce(elems)

    def _reduce(self, xs):
        """
        Reduces the given element, that is, merges its values that it
        considers equal according to the merge predicate.
        """

        return self._merge([], list(xs), self.dom.join)

    def _merge(self, a, b, merger):
        """
        Merges two instances of this domain together using the merge predicate.

        """
        if a is self.top or b is self.top:
            return self.top

        res = [x for x in a]
        changed = False

        for y in b:
            do_add = True
            for i, x in enumerate(res):
                if self.merge_predicate(x, y):
                    do_add = False
                    res[i] = merger(x, y)
                    if not self.dom.eq(x, res[i]):
                        changed = True
                    break

            if do_add:
                res.append(y)

        return self._merge([], res, merger) if changed else res

    def join(self, a, b):
        return self._merge(a, b, self.dom.join)

    def meet(self, a, b):
        return self._reduce(
            x for x in a if any(
                self.merge_predicate(x, y)
                for y in b
            )
        )

    def update(self, a, b, widen=False):
        return self._merge(
            a, b,
            lambda e_a, e_b: self.dom.update(e_a, e_b, widen)
        )

    def le(self, a, b):
        return all(any(
            self.merge_predicate(x, y)
            for y in b
        ) for x in a)

    def lt(self, a, b):
        return self.le(a, b) and any(any(
            not self.merge_predicate(x, y)
            for y in b
        ) for x in a)

    def eq(self, a, b):
        return self.le(a, b) and self.le(b, a)

    def generator(self):
        raise NotImplementedError

    def concretize(self, abstract):
        raise NotImplementedError

    def abstract(self, concrete):
        raise NotImplementedError


class FiniteLattice(AbstractDomain):
    """
    A general purpose finite lattice, to be constructed from a given
    "less than" relation.
    """
    @staticmethod
    def _relations_count(lts):
        """
        Returns the total number of elements in the relation given as a dict
        from element to iterable of elements.
        """
        return len(lts) + sum(len(e) for x, e in lts.iteritems())

    @staticmethod
    def _transitive_closure(lts):
        """
        Computes the transitive closure of the relation.
        """
        closed_lts = collections.defaultdict(set)
        closed_lts.update(lts)

        while True:
            init_size = FiniteLattice._relations_count(closed_lts)
            for k, lts in closed_lts.items():
                closed_lts[k].add(k)
                for r in lts:
                    closed_lts[r].add(r)
                    closed_lts[k].update(closed_lts[r])

            if FiniteLattice._relations_count(closed_lts) == init_size:
                return closed_lts

    @staticmethod
    def _inverse(lts):
        """
        Computes the inverse of the given relation.
        """
        inv_lts = collections.defaultdict(set)

        for k, lts in lts.items():
            for r in lts:
                inv_lts[r].add(k)

        return inv_lts

    @staticmethod
    def of_subsets(xs):
        """
        Constructor that can build a finite lattice from the given elements.
        The "less than" relation will simply be the "is subset of" relation.
        """
        sets = powerset(xs)
        return FiniteLattice({
            k: {v for v in sets if k.issubset(v)} for k in sets
        })

    def __init__(self, lts):
        """
        Constructs a new finite lattice from the given "less than" relation.
        The "bottom" and "top" elements are inferred automatically, which means
        that they must exist.
        """
        self.lts = FiniteLattice._transitive_closure(lts)
        self.inv_lts = FiniteLattice._inverse(self.lts)
        self.bottom = self.lowest_among(self.lts.keys())
        self.top = self.greatest_among(self.lts.keys())

    def build(self, elem):
        """
        Returns the given element
        """
        assert(elem in self.lts)
        return elem

    def is_empty(self, x):
        return len(x) == 0

    def join(self, a, b):
        return self.lowest_among(self.lts[a] & self.lts[b])

    def meet(self, a, b):
        return self.greatest_among(self.inv_lts[a] & self.inv_lts[b])

    def update(self, a, b, widen=False):
        return self.top if widen else self.join(a, b)

    def lt(self, a, b):
        return not self.eq(a, b) and b in self.lts[a]

    def eq(self, a, b):
        return a == b

    def generator(self):
        return self.lts[self.bottom]

    def concretize(self, abstract):
        return abstract

    def abstract(self, concretes):
        return self.build(concretes)


class FiniteSubsetLattice(AbstractDomain):
    """
    A general purpose finite lattice where elements represent subsets
    of a given set of elements. The "less than" relation is this inherited
    "is subset of" relation.

    Constructing a new instance of this domain does not have the overhead of
    computing the powerset of the given set, unlike FiniteLattice.of_subset.
    However, the different operations (may or) may not perform as fast.
    """
    def __init__(self, elems):
        self.bottom = frozenset()
        self.top = frozenset(elems)

    def build(self, elems):
        res = frozenset(elems)
        return res if res <= self.top else None

    def join(self, a, b):
        return a | b

    def meet(self, a, b):
        return a & b

    def update(self, a, b, widen=False):
        return self.top if widen else self.join(a, b)

    def lt(self, a, b):
        return a < b

    def eq(self, a, b):
        return a == b

    def generator(self):
        return powerset(self.top)

    def concretize(self, abstract):
        return abstract

    def abstract(self, concrete):
        return self.build(concrete)
