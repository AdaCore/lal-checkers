"""
Provides some basic abstract domains.
"""

from utils import powerset, zip_dicts
from domain_capabilities import Capability
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

    HasSplit = Capability.No
    HasConcretize = Capability.No

    def build(self, *args):
        """
        Builds a new element of this abstract domain.
        """
        raise NotImplementedError

    def is_empty(self, x):
        """
        Returns True if the given element represents an empty set of
        concrete values.
        """
        return self.size(x) == 0

    def size(self, x):
        """
        Returns the number of concrete values represented by the given element.
        """
        raise NotImplementedError

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

    def split(self, elem, separator):
        """
        Returns an iterable of disjoint elements of this abstract domain,
        such that all of them have an empty meet with the separator element,
        but the union of all the sets of concrete values represented by each
        elements is equal to the set difference between the set of concrete
        values represented by the original element and the set of concrete
        values represented by the separator.
        """
        raise NotImplementedError

    def touches(self, a, b):
        """
        Returns True if the two abstract elements touch each other, that is,
        their join describes the same set of concrete values as the union of
        sets of concrete values represented by the two given elements.
        """
        return self.size(self.join(a, b)) == self.size(a) + self.size(b)

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

    def str(self, x):
        """
        Returns a human-readable string representation of the given abstract
        element.
        """
        raise NotImplementedError


class Intervals(AbstractDomain):
    """
    An abstract domain used to represent sets of integers.
    """

    HasSplit = Capability.Yes
    HasConcretize = Capability.Yes

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
        else:
            raise NotImplementedError

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

    def is_empty(self, x):
        return x == self.bottom

    def size(self, x):
        return 0 if x == self.bottom else x[1] - x[0] + 1

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

    def split(self, elem, separator):
        if self.is_empty(self.meet(elem, separator)):
            return [elem]
        elif self.le(elem, separator):
            return []
        elif self.lt(separator, elem):
            if separator[0] == elem[0]:
                return [(separator[1] + 1, elem[1])]
            elif separator[1] == elem[1]:
                return [(elem[0], separator[0] - 1)]
            else:
                return [
                    (elem[0], separator[0] - 1),
                    (separator[1] + 1, elem[1])
                ]
        elif separator[0] <= elem[0]:
            return [(separator[1] + 1, elem[1])]
        else:
            return [(elem[0], separator[0] - 1)]

    def touches(self, a, b):
        return (a == self.bottom or b == self.bottom or
                a[1] == b[0] - 1 or a[0] == b[1] + 1)

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

    def str(self, x):
        return "[empty]" if x == self.bottom else "[{}, {}]".format(*x)


class Product(AbstractDomain):
    """
    An abstract domain used to represent the cartesian product of
    sets of concrete values.
    """

    HasSplit = Capability.IfAll(lambda self: self.domains, Capability.HasSplit)
    HasConcretize = Capability.IfAll(
        lambda self: self.domains, Capability.HasConcretize
    )

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

    def size(self, x):
        return reduce(
            lambda acc, dom_e: acc * (dom_e[0].size(dom_e[1])),
            zip(self.domains, x),
            1
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

    def split(self, elem, separator):
        def inner(elem, dimension):
            if dimension == len(elem):
                return []

            dom = self.domains[dimension]
            x_slice = elem[dimension]
            sep = separator[dimension]

            self_splits = [
                elem[:dimension] + (x_split,) + elem[dimension+1:]
                for x_split in dom.split(x_slice, sep)
            ]

            rest_splits = inner(
                elem[:dimension] + (sep,) + elem[dimension+1:],
                dimension + 1
            )

            return self_splits + rest_splits

        return inner(elem, 0)

    def touches(self, a, b):
        return any(
            self.domains[i].touches(a[i], b[i]) and all(
                self.domains[j].eq(a[j], b[j])
                for j in range(len(self.domains))
                if i != j
            )
            for i in range(len(self.domains))
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

    def str(self, x):
        return "({})".format(", ".join(
            dom.str(e) for dom, e in zip(self.domains, x)
        ))


class Powerset(AbstractDomain):
    """
    An abstract domain used to represent sets of sets of concrete values.
    Two sets of concrete values are considered equal according to the provided
    merge predicate. When two values are considered equal, they are joined.
    """

    HasSplit = Capability.IfSingle(lambda self: self.dom, Capability.HasSplit)

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

    def is_empty(self, x):
        return all(self.dom.is_empty(e) for e in x)

    def size(self, x):
        return sum((self.dom.size(e) for e in x), 0)

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
            self.dom.meet(x, y)
            for x in a
            for y in b
        )

    def update(self, a, b, widen=False):
        return self._merge(
            a, b,
            lambda e_a, e_b: self.dom.update(e_a, e_b, widen)
        )

    def le(self, a, b):
        return all(any(
            self.dom.le(x, y)
            for y in b
        ) for x in a)

    def lt(self, a, b):
        return self.le(a, b) and not self.le(b, a)

    def eq(self, a, b):
        return self.le(a, b) and self.le(b, a)

    def split(self, elem, separator):
        return self._reduce(
            e
            for x in elem
            for y in separator
            for e in self.dom.split(x, y)
        )

    def generator(self):
        raise NotImplementedError

    def concretize(self, abstract):
        raise NotImplementedError

    def abstract(self, concrete):
        raise NotImplementedError

    def str(self, x):
        return "{{{}}}".format(", ".join(
            sorted(self.dom.str(e) for e in x)
        ))


class FiniteLattice(AbstractDomain):
    """
    A general purpose finite lattice, to be constructed from a given
    "less than" relation.
    """

    HasSplit = Capability(lambda self: self.splitter is not None)
    HasConcretize = Capability.Yes

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
    def _subset_splitter(domain, elem, separator):
        without = elem - separator
        if without in domain.lts[domain.bottom]:
            return [without]
        else:
            return []

    @staticmethod
    def of_subsets(xs):
        """
        Constructor that can build a finite lattice from the given elements.
        The "less than" relation will simply be the "is subset of" relation.
        """
        sets = powerset(xs)
        return FiniteLattice({
            k: {v for v in sets if k.issubset(v)} for k in sets
        }, FiniteLattice._subset_splitter)

    def __init__(self, lts, splitter):
        """
        Constructs a new finite lattice from the given "less than" relation.
        The "bottom" and "top" elements are inferred automatically, which means
        that they must exist. The splitter function used to split elements
        must also be provided.
        """
        self.lts = FiniteLattice._transitive_closure(lts)
        self.inv_lts = FiniteLattice._inverse(self.lts)
        self.bottom = self.lowest_among(self.lts.keys())
        self.top = self.greatest_among(self.lts.keys())
        self.splitter = splitter

    def build(self, elem):
        """
        Returns the given element
        """
        assert(elem in self.lts)
        return elem

    def is_empty(self, x):
        return len(x) == 0

    def size(self, x):
        return len(x)

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

    def split(self, elem, separator):
        return self.splitter(self, elem, separator)

    def generator(self):
        return self.lts[self.bottom]

    def concretize(self, abstract):
        return abstract

    def abstract(self, concretes):
        return self.build(concretes)

    def str(self, x):
        return "{{{}}}".format(", ".join(sorted(str(e) for e in x)))


class FiniteSubsetLattice(AbstractDomain):
    """
    A general purpose finite lattice where elements represent subsets
    of a given set of elements. The "less than" relation is this inherited
    "is subset of" relation.

    Constructing a new instance of this domain does not have the overhead of
    computing the powerset of the given set, unlike FiniteLattice.of_subset.
    However, the different operations (may or) may not perform as fast.
    """

    HasSplit = Capability.Yes
    HasConcretize = Capability.Yes

    def __init__(self, elems):
        self.bottom = frozenset()
        self.top = frozenset(elems)

    def build(self, elems):
        res = frozenset(elems)
        return res if res <= self.top else None

    def size(self, x):
        return len(x)

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

    def split(self, elem, separator):
        return [elem - separator]

    def touches(self, a, b):
        return True

    def generator(self):
        return powerset(self.top)

    def concretize(self, abstract):
        return abstract

    def abstract(self, concrete):
        return self.build(concrete)

    def str(self, x):
        return "{{{}}}".format(", ".join(sorted(str(e) for e in x)))


class SparseArray(AbstractDomain):
    HasConcretize = Capability.Yes

    def __init__(self, index_dom, elem_dom, max_elems=0):
        """
        Creates a sparse array domain in which array elements use the given
        index domain and element domain. Additionally, a maximal amount of
        entries to keep track of in an array can specified.

        :param AbstractDomain index_dom: The index domain.
        :param AbstractDomain elem_dom: The element domain.
        :param int max_elems: The maximal amount of entries. Any integer <= 0
            means that there is no maximal amount of entries.
        """
        self.index_dom = index_dom
        self.elem_dom = elem_dom
        self.prod_dom = Product(index_dom, elem_dom)
        self.bottom = []
        self.top = [self.prod_dom.top]
        self.max_elems = max_elems

        if Capability.HasSplit(index_dom):
            self._join_elem = self._join_elem_precise
        else:
            self._join_elem = self._join_elem_imprecise

        # The singleton element representing the concrete empty array.
        # (not to mix up with the empty element (bottom) representing no
        # concrete array.
        self.empty = [(self.index_dom.bottom, self.elem_dom.top)]

    def build(self, elems):
        assert self.le(elems, self.top)
        return elems

    def is_empty(self, x):
        return len(x) == 0

    def _merge_element(self, array, e):
        """
        Merges the given element e with the given array, where all elements of
        the array INCLUDING e are disjoint. Here, "merge" means that we look
        for another element x in the array such that the join of x and e does
        not intersect with any other element of the array, so that we can
        safely replace x by the join of e and x. As a result:
        - The invariant stating that all elements of the array are disjoint is
          satisfied.
        - There is either exactly the same number of elements in the returned
          array as there are in "array", or exactly 1 (the top element if the
          routine failed).
        - We have included the information about "e" in "array" (at the cost
          of a loss of precision).

        :param list[object] array: The array in which to include e.
        :param object e: The element to include in the array.
        :return: A new array.
        :rtype: list[str]
        """
        for i in range(len(array)):
            i_x, x = array[i]
            index_join = self.index_dom.join(i_x, e[0])
            does_not_overlap = all(
                self.index_dom.is_empty(
                    self.index_dom.meet(index_join, array[j][0])
                )
                for j in range(i + 1, len(array))
            )

            if does_not_overlap:
                return (array[:i] +
                        [self.prod_dom.join(array[i], e)] +
                        array[i+1:])

        return self.top

    def normalized(self, array):
        for i, x in enumerate(array):
            for j in range(i + 1, len(array)):
                y = array[j]
                if self.prod_dom.touches(x, y):
                    return self.normalized(
                        array[:i] + array[i + 1:j] + array[j + 1:] + [
                            self.prod_dom.join(x, y)
                        ]
                    )

        # If the maximal amount of elements to store in the array is
        # reached, merge arbitrary elements together.
        while len(array) > self.max_elems > 0:
            x = array.pop(0)
            array = self._merge_element(array, x)

        return array

    def _join_elem_precise(self, x, elem):
        """
        Given an array x and an element elem, joins elem into x in a precise
        way. This means that when an existing element in the array overlaps
        with elem, the indices will be split apart so that two elements can
        hold their most precise representation.

        :param list[(object, object)] x: The sparse array.
        :param (object, object) elem: The element to join.
        :rtype: list[(object, object)]
        """
        # Filter out elements that would be absorbed anyway.
        x = [
            e for e in x
            if not self.prod_dom.le(e, elem)
        ]

        res = []
        left = [elem[0]]
        for idx, val in x:
            meet = self.index_dom.meet(idx, elem[0])

            if self.index_dom.is_empty(meet):
                res.append((idx, val))
            else:
                elem_join = self.elem_dom.join(val, elem[1])

                if self.elem_dom.eq(elem_join, val):
                    res.append((idx, val))
                else:
                    splits = self.index_dom.split(idx, meet)
                    res.extend([(split, val) for split in splits])
                    res.append((meet, elem_join))

                left = [
                    split
                    for l in left
                    for split in self.index_dom.split(l, idx)
                ]

        res.extend([(l, elem[1]) for l in left])
        return res

    def _join_elem_imprecise(self, x, elem):
        """
        Given an array x and an element elem, joins elem into x in a imprecise
        way. This means that when an existing element in the array overlaps
        with elem, the indices will be joined and a single element will contain
        the most conservative representation of both elements.

        :param list[(object, object)] x: The sparse array.
        :param (object, object) elem: The element to join.
        :rtype: list[(object, object)]
        """
        # Filter out elements that would be absorbed anyway.
        x = [
            e for e in x
            if not self.prod_dom.le(e, elem)
        ]

        res = []
        join = elem
        for e in x:
            meet = self.index_dom.meet(e[0], elem[0])

            if self.index_dom.is_empty(meet):
                res.append(e)
            else:
                join = self.prod_dom.join(e, join)

        return res + [join]

    def join(self, a, b):
        return self.normalized(reduce(self._join_elem, b, a))

    def meet(self, a, b):
        res = []
        for e_a in a:
            for e_b in b:
                meet = self.prod_dom.meet(e_a, e_b)
                if not self.prod_dom.is_empty(meet):
                    res.append(meet)
        return res

    def update(self, a, b, widen=False):
        if widen:
            if self.eq(a, b):
                return a
            else:
                # todo: we could be more precise on the widening by first
                # trying to detect if a specific element of the array could be
                # widened.
                return self.top
        else:
            return self.join(a, b)

    def le(self, a, b):
        return all(
            any(
                self.prod_dom.le(x, y)
                for y in b
            )
            for x in a
        )

    def lt(self, a, b):
        return self.le(a, b) and not self.eq(a, b)

    def _has_value_at(self, matrix, index, value):
        meets = [
            (meet, v)
            for i, v in matrix
            for meet in (self.index_dom.meet(i, index),)
            if not self.index_dom.is_empty(meet)
        ]

        indices_count = sum(self.index_dom.size(i) for i, _ in meets)
        if indices_count != self.index_dom.size(index):
            return False

        return all(self.elem_dom.eq(v, value) for _, v in meets)

    def eq(self, a, b):
        return (all(self._has_value_at(b, i, v) for i, v in a) and
                all(self._has_value_at(a, i, v) for i, v in b))

    def split(self, x, separator):
        raise NotImplementedError

    def generator(self):
        def index_overlaps(a, b):
            return not self.index_dom.is_empty(self.index_dom.meet(a, b))

        def gen_not_overlapping_indices(array):
            for index in self.index_dom.generator():
                if not any(index_overlaps(index, x[0]) for x in array):
                    yield index

        def gen_arrays_of_size(array, size):
            if size == 0:
                yield array
            else:
                for not_overlapping in gen_not_overlapping_indices(array):
                    for elem in self.elem_dom.generator():
                        inced_size = array + [(not_overlapping, elem)]

                        nexts = gen_arrays_of_size(inced_size, size - 1)
                        for n in nexts:
                            yield n

        i = 0
        while True:
            arrays = gen_arrays_of_size([], i)
            first = next(arrays, None)
            if first is None:
                break

            yield first
            for array in arrays:
                yield array

            i += 1

    def concretize(self, abstract):
        def gen_arrays(abstract, i=0, gened=frozenset()):
            if len(abstract) == i:
                yield gened
            else:
                concr_idx = self.index_dom.concretize(abstract[i][0])
                concr_val = self.elem_dom.concretize(abstract[i][1])

                parts = (
                    frozenset(zip(concr_idx, values))
                    for values in itertools.product(*(
                        concr_val
                        for _ in range(len(concr_idx))
                    ))
                )

                for part in parts:
                    for rest in gen_arrays(abstract, i + 1, gened | part):
                        yield rest

        return frozenset(gen_arrays(abstract))

    def abstract(self, concrete):
        def abstract_one(array):
            return [
                (
                    self.index_dom.abstract(frozenset([idx])),
                    self.elem_dom.abstract(frozenset([val]))
                )
                for idx, val in array
            ]

        return reduce(self.join, [abstract_one(x) for x in concrete])

    def str(self, x):
        return "{{{}}}".format(", ".join(sorted(
            "{}: {}".format(
                self.index_dom.str(idx),
                self.elem_dom.str(elem)
            )
            for idx, elem in x)
        ))


class AccessPathsLattice(AbstractDomain):
    """
    Abstract domain that represents an access path. Its elements are
    sentences of a simple access path language. For example:
     - ProductGet(Address(3, A), 1, A_1) for some product domain A represents
       the access to the component at index 1 of the element located at
       address 3 in the abstract memory representation.
     - NonNull() represents any access path that is not null.
    """
    HasSplit = Capability.Yes

    class NullDeref(LookupError):
        """
        The exception to throw in case a null dereference occurs.
        """
        pass

    class TopValue(LookupError):
        """
        The exception to throw in case a dereference on the top value occurs.
        """
        pass

    class BottomValue(LookupError):
        """
        The exception to throw in case a dereference on the bottom value
        occurs.
        """
        pass

    class AccessPath(object):
        """
        Base class for access paths expressions.
        """
        def size(self):
            """
            Returns the size of this access path.
            :rtype: int
            """
            raise NotImplementedError

        def access(self, state):
            """
            Returns the value stored in the memory at this access path.
            :param object state: The memory state.
            :rtype: object
            """
            raise NotImplementedError

        def inv_access(self, state, value):
            """
            Assigns the given value to the location in the memory at this
            access path.
            :param object state: The memory state.
            :param object value: The value to assign.
            """
            raise NotImplementedError

        def update(self, state, value):
            """
            See inv_access.
            """
            return self.inv_access(state, value)

        def __or__(self, other):
            """
            Returns the abstract join with another access path.
            :type other: AccessPathsLattice.AccessPath
            :rtype: AccessPathsLattice.AccessPath
            """
            raise NotImplementedError

        def __and__(self, other):
            """
            Returns the abstract meet with another access path.
            :type other: AccessPathsLattice.AccessPath
            :rtype: AccessPathsLattice.AccessPath
            """
            raise NotImplementedError

        def __eq__(self, other):
            """
            Returns true iff self represents the same concrete access paths as
            other.
            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            raise NotImplementedError

        def __lt__(self, other):
            """
            Returns true iff self represents a strict subset of the concrete
            access paths represented by other.
            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            raise NotImplementedError

        def __le__(self, other):
            """
            Returns true iff self represents a subset of the concrete access
            paths represented by other.
            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            return self < other or self == other

        def __gt__(self, other):
            """
            Returns true iff self represents a strict superset of the concrete
            access paths represented by other.
            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            return other < self

        def __ge__(self, other):
            """
            Returns true iff self represents a superset of the concrete access
            paths represented by other.
            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            return other <= self

        def split(self, separator):
            """
            Splits this access path using the given separator.
            See AbstractDomain.split for more details.
            :rtype separator: AccessPathsLattice.AccessPath
            :rtype: list[AccessPathsLattice.AccessPath]
            """
            raise NotImplementedError

        def touches(self, other):
            """
            Returns true if this access path touches the other access path.
            For example, NonNull() touches Null() because their concretizations
            does not overlap and their join AllPath() does not include elements
            that where not already in NonNull() or Null().

            :type other: AccessPathsLattice.AccessPath
            :rtype: bool
            """
            # todo: rename touches to adjacent
            raise NotImplementedError

        def __hash__(self):
            raise NotImplementedError

    class AllPath(AccessPath):
        """
        Represents all access paths.
        """
        def __init__(self):
            pass

        def size(self):
            return float('inf')

        def access(self, state):
            raise AccessPathsLattice.TopValue

        def inv_access(self, state, value):
            raise AccessPathsLattice.TopValue

        def __or__(self, other):
            return self

        def __and__(self, other):
            return other

        def __eq__(self, other):
            return isinstance(other, AccessPathsLattice.AllPath)

        def __lt__(self, other):
            return False

        def split(self, separator):
            if isinstance(separator, AccessPathsLattice.NonNull):
                return [AccessPathsLattice.Null()]
            elif isinstance(separator, AccessPathsLattice.Null):
                return [AccessPathsLattice.NonNull()]
            elif isinstance(separator, AccessPathsLattice.AllPath):
                return []
            else:
                return [self]

        def touches(self, other):
            return False

        def __hash__(self):
            return hash(())

        def __str__(self):
            return "[all-path]"

    class Null(AccessPath):
        """
        Represents the null access path.
        """
        def __init__(self):
            pass

        def size(self):
            return 1

        def access(self, state):
            raise AccessPathsLattice.NullDeref

        def inv_access(self, state, value):
            raise AccessPathsLattice.NullDeref

        def __or__(self, other):
            if (isinstance(other, AccessPathsLattice.Null) or
                    isinstance(other, AccessPathsLattice.BottomValue)):
                return self
            else:
                return AccessPathsLattice.AllPath()

        def __and__(self, other):
            if (isinstance(other, AccessPathsLattice.Null) or
                    isinstance(other, AccessPathsLattice.AllPath)):
                return self
            else:
                return AccessPathsLattice.NoPath()

        def __eq__(self, other):
            return isinstance(other, AccessPathsLattice.Null)

        def __lt__(self, other):
            return isinstance(other, AccessPathsLattice.AllPath)

        def split(self, separator):
            if (isinstance(separator, AccessPathsLattice.Null) or
                    isinstance(separator, AccessPathsLattice.AllPath)):
                return []
            else:
                return [self]

        def touches(self, other):
            return isinstance(other, AccessPathsLattice.NonNull)

        def __hash__(self):
            return hash(())

        def __str__(self):
            return "null"

    class NonNull(AccessPath):
        """
        Represents all access paths except the null access path.
        """
        def __init__(self):
            pass

        def size(self):
            return float('inf')

        def access(self, state):
            raise AccessPathsLattice.TopValue

        def inv_access(self, state, value):
            raise AccessPathsLattice.TopValue

        def __or__(self, other):
            if (isinstance(other, AccessPathsLattice.Null) or
                    isinstance(other, AccessPathsLattice.AllPath)):
                return AccessPathsLattice.AllPath()
            else:
                return self

        def __and__(self, other):
            if (isinstance(other, AccessPathsLattice.NonNull) or
                    isinstance(other, AccessPathsLattice.AllPath)):
                return self
            elif (isinstance(other, AccessPathsLattice.Address) or
                  isinstance(other, AccessPathsLattice.ProductGet)):
                return other
            else:
                return AccessPathsLattice.NoPath()

        def __eq__(self, other):
            return isinstance(other, AccessPathsLattice.NonNull)

        def __lt__(self, other):
            return isinstance(other, AccessPathsLattice.AllPath)

        def split(self, separator):
            if (isinstance(separator, AccessPathsLattice.AllPath) or
                    isinstance(separator, AccessPathsLattice.NonNull)):
                return []
            else:
                return [self]

        def touches(self, other):
            return isinstance(other, AccessPathsLattice.Null)

        def __hash__(self):
            return hash(())

        def __str__(self):
            return "[non-null]"

    class Address(AccessPath):
        """
        Represents the access path to a precise memory location.
        """
        def __init__(self, val, dom):
            """
            :param int val: The address of the access path.
            :param AbstractDomain dom: The domain of the elements living at
                this address.
            """
            self.val = val
            self.dom = dom

        def size(self):
            return 1

        def access(self, state):
            if self.val in state[0]:
                return state[0][self.val][1]
            else:
                raise AccessPathsLattice.TopValue

        def inv_access(self, state, value):
            state[0][self.val] = (self.dom, value)

        def __or__(self, other):
            if self <= other:
                return other
            elif other < self:
                return self
            elif (isinstance(other, AccessPathsLattice.Address) or
                  isinstance(other, AccessPathsLattice.ProductGet)):
                return AccessPathsLattice.NonNull()
            elif isinstance(other, AccessPathsLattice.Null):
                return AccessPathsLattice.AllPath()
            else:
                return self

        def __and__(self, other):
            if self <= other:
                return self
            elif other < self:
                return other
            else:
                return AccessPathsLattice.NoPath()

        def __eq__(self, other):
            return (isinstance(other, AccessPathsLattice.Address) and
                    self.val == other.val and self.dom == other.dom)

        def __lt__(self, other):
            if (isinstance(other, AccessPathsLattice.NonNull) or
                    isinstance(other, AccessPathsLattice.AllPath)):
                return True
            return False

        def split(self, separator):
            if (isinstance(separator, AccessPathsLattice.NonNull) or
                    isinstance(separator, AccessPathsLattice.AllPath) or
                    self == separator):
                return []
            else:
                return [self]

        def touches(self, other):
            return False

        def __hash__(self):
            return hash((self.val, self.dom))

        def __str__(self):
            return "0x{}".format(format(self.val, '08x'))

    class Subprogram(AccessPath):
        """
        Represents the access path to a subprogram.
        """
        def __init__(self, name, interface, defs, capture_paths):
            """
            :param object name: The object identifying the subprogram accessed.
            :param CallInterface interface: The call interface of the
                subprogram accessed.
            :param (function, function) defs: The forward and backward
                implementations of the subprogram.
            :param list[AccessPath] capture_paths: The access paths to the
                captured variables.
            """
            self.name = name
            self.interface = interface
            self.defs = defs
            self.vars = capture_paths

        def size(self):
            return 1

        def access(self, state):
            raise NotImplementedError

        def inv_access(self, state):
            raise NotImplementedError

        def __or__(self, other):
            if self <= other:
                return other
            elif other < self:
                return self
            elif isinstance(other, AccessPathsLattice.Null):
                return AccessPathsLattice.AllPath()
            else:
                return AccessPathsLattice.NonNull()

        def __and__(self, other):
            if self <= other:
                return self
            elif other < self:
                return other
            else:
                return AccessPathsLattice.NoPath()

        def __lt__(self, other):
            return (isinstance(other, AccessPathsLattice.NonNull) or
                    isinstance(other, AccessPathsLattice.AllPath))

        def __eq__(self, other):
            return (
                isinstance(other, AccessPathsLattice.Subprogram) and
                self.name == other.name and
                all(a == b for a, b in zip(self.vars, other.vars))
            )

        def split(self, separator):
            if (isinstance(separator, AccessPathsLattice.NonNull) or
                    isinstance(separator, AccessPathsLattice.AllPath) or
                    self == separator):
                return []
            else:
                return [self]

        def touches(self, other):
            return False

        def __hash__(self):
            return hash((self.name, self.vars))

        def __str__(self):
            return "Subprogram {} capturing ({})".format(
                self.name, ", ".join(str(x) for x in self.vars)
            )

    class ProductGet(AccessPath):
        """
        Represents the access path to a specific component of another access
        path.
        """
        def __init__(self, prefix, component, dom):
            """
            :param AccessPathsLattice.AccessPath prefix: The access path of
                the element which component is being taken.
            :param int component: The index of the component.
            :param AbstractDomain dom: The domain of the component elements.
            """
            self.prefix = prefix
            self.component = component
            self.dom = dom

        def size(self):
            return self.prefix.size()

        def access(self, state):
            return self.prefix.access(state)[self.component]

        def inv_access(self, state, value):
            top = self.prefix.dom.top
            self.prefix.inv_access(
                state,
                top[:self.component] + (value,) + top[self.component+1:]
            )

        def update(self, state, value):
            prod = self.prefix.access(state)
            self.prefix.inv_access(
                state,
                prod[:self.component] + (value,) + prod[self.component+1:]
            )

        def __or__(self, other):
            if self <= other:
                return other
            elif other < self:
                return self
            elif (isinstance(other, AccessPathsLattice.Address) or
                  isinstance(other, AccessPathsLattice.ProductGet)):
                return AccessPathsLattice.NonNull()
            elif isinstance(other, AccessPathsLattice.Null):
                return AccessPathsLattice.AllPath()
            else:
                return self

        def __and__(self, other):
            if self <= other:
                return self
            elif other < self:
                return other
            else:
                return AccessPathsLattice.NoPath()

        def __eq__(self, other):
            return (isinstance(other, AccessPathsLattice.ProductGet) and
                    self.prefix == other.prefix and
                    self.component == other.component and
                    self.dom == other.dom)

        def __lt__(self, other):
            if (isinstance(other, AccessPathsLattice.NonNull) or
                    isinstance(other, AccessPathsLattice.AllPath)):
                return True
            elif (isinstance(other, AccessPathsLattice.ProductGet) and
                  self.component == other.component and
                  self.dom == other.dom):
                return self.prefix < other.prefix
            return False

        def split(self, separator):
            if (isinstance(separator, AccessPathsLattice.NonNull) or
                    isinstance(separator, AccessPathsLattice.AllPath) or
                    self == separator):
                return []
            else:
                return [self]

        def touches(self, other):
            return False

        def __hash__(self):
            return hash((self.prefix, self.component, self.dom))

        def __str__(self):
            return "Get_{}({})".format(self.component, self.prefix)

    class NoPath(AccessPath):
        """
        Represents no access path.
        """
        def __init__(self):
            pass

        def size(self):
            return 0

        def access(self, state):
            raise AccessPathsLattice.BottomValue

        def inv_access(self, state, value):
            raise AccessPathsLattice.BottomValue

        def __or__(self, other):
            return other

        def __and__(self, other):
            return self

        def __eq__(self, other):
            return isinstance(other, AccessPathsLattice.NoPath)

        def __hash__(self):
            return hash(())

        def split(self, separator):
            return []

        def touches(self, other):
            return True

        def __lt__(self, other):
            return not isinstance(other, AccessPathsLattice.NoPath)

        def __str__(self):
            return "[no-path]"

    def __init__(self):
        self.bottom = AccessPathsLattice.NoPath()
        self.top = AccessPathsLattice.AllPath()

    def build(self, *args):
        raise NotImplementedError

    def size(self, x):
        return x.size()

    def join(self, a, b):
        return a | b

    def meet(self, a, b):
        return a & b

    def update(self, a, b, widen=False):
        return self.join(a, b)

    def lt(self, a, b):
        return a < b

    def eq(self, a, b):
        return a == b

    def le(self, a, b):
        return a <= b

    def split(self, elem, separator):
        return elem.split(separator)

    def touches(self, a, b):
        return a.touches(b)

    def generator(self):
        raise NotImplementedError

    def concretize(self, abstract):
        raise NotImplementedError

    def abstract(self, concrete):
        raise NotImplementedError

    def str(self, x):
        return str(x)


class RandomAccessMemory(AbstractDomain):
    def __init__(self):
        self.bottom = object()
        self.top = ({}, 0)

    def build(self, args):
        assert isinstance(args, dict)
        return args, 0

    def size(self, x):
        return 0 if x == self.bottom or any(
            dom.is_empty(elem) for dom, elem in x[0].values()
        ) else float('inf')

    def join(self, a, b):
        if a == self.bottom:
            return b
        elif b == self.bottom:
            return a
        elif a[1] != b[1]:
            raise NotImplementedError
        else:
            res = {}
            dct = zip_dicts(a[0], b[0], False)
            for k, ((x_dom, x_elem), (y_dom, y_elem)) in dct.iteritems():
                if x_dom == y_dom:
                    res[k] = (x_dom, x_dom.join(x_elem, y_elem))
            return res, a[1]

    def meet(self, a, b):
        if a == self.bottom or b == self.bottom:
            return self.bottom
        elif a[1] != b[1]:
            raise NotImplementedError
        else:
            res = {}
            dct = zip_dicts(a[0], b[0], True)

            for k, (x, y) in dct.iteritems():
                if x is None:
                    res[k] = y
                elif y is None:
                    res[k] = x
                elif x[0] == y[0]:
                    res[k] = (x[0], x[0].meet(x[1], y[1]))
                else:
                    return self.bottom

            return res, a[1]

    def update(self, a, b, widen=False):
        if widen:
            if a == self.bottom:
                return b
            elif b == self.bottom:
                return a
            elif a[1] != b[1]:
                raise NotImplementedError
            else:
                res = {}
                dct = zip_dicts(a[0], b[0], False)
                for k, ((x_dom, x_elem), (y_dom, y_elem)) in dct.iteritems():
                    if x_dom == y_dom:
                        res[k] = (x_dom, x_dom.update(x_elem, y_elem, True))
                return res, a[1]
        else:
            return self.join(a, b)

    def lt(self, a, b):
        return self.le(a, b) and not self.eq(a, b)

    def eq(self, a, b):
        if a == self.bottom:
            return b == self.bottom
        elif a[1] != b[1]:
            return NotImplementedError
        else:
            return len(a[0]) == len(b[0]) and all(
                (i in b[0] and
                 b[0][i][0] == x_dom and
                 x_dom.eq(x_elem, b[0][i][1]))
                for i, (x_dom, x_elem) in a[0].iteritems()
            )

    def le(self, a, b):
        if b == self.top:
            return a != self.top
        elif a == self.top:
            return False
        elif a[1] != b[1]:
            return NotImplementedError
        else:
            return all(
                x_dom == b[0][i][0] and x_dom.le(x_elem, b[0][i][1])
                if i in b[0] else True
                for i, (x_dom, x_elem) in a[0].iteritems()
            )

    def split(self, elem, separator):
        raise NotImplementedError

    def touches(self, a, b):
        raise NotImplementedError

    def generator(self):
        raise NotImplementedError

    def concretize(self, abstract):
        raise NotImplementedError

    def abstract(self, concrete):
        raise NotImplementedError

    def str(self, x):
        return "({}, {})".format(
            {
                i: x[0][i][0].str(x[0][i][1])
                for i in sorted(x[0].keys())
            },
            x[1]
        )


class Universe(AbstractDomain):
    def __init__(self):
        self.top = object()
        self.bottom = object()

    def build(self):
        return self.top

    def size(self, x):
        return 0 if x == self.bottom else float('inf')

    def join(self, a, b):
        return (self.bottom
                if a == self.bottom and b == self.bottom
                else self.top)

    def meet(self, a, b):
        return (self.bottom
                if a == self.bottom or b == self.bottom
                else self.top)

    def lt(self, a, b):
        return a == self.bottom and b == self.top

    def eq(self, a, b):
        return a == b

    def le(self, a, b):
        return self.lt(a, b) or self.eq(a, b)

    def split(self, elem, separator):
        raise NotImplementedError

    def touches(self, a, b):
        return a == self.bottom or b == self.bottom

    def generator(self):
        raise NotImplementedError

    def concretize(self, abstract):
        raise NotImplementedError

    def abstract(self, concrete):
        raise NotImplementedError

    def str(self, x):
        return "[any]" if x == self.top else "[none]"
