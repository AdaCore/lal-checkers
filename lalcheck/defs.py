"""
Provides the Definitions class, which instances can be used to hold the set of
operations that are well defined in the context of a program or a set of
programs.

Also provides a collection of predefined types and operations on these types.
"""

from collections import defaultdict
from lalcheck import domains


def interval_add_no_wraparound(domain):
    """
    Given an interval domain, returns a function which, given two sets of
    integers represented by elements of this interval domain, returns the
    smallest interval which contains all possible results of adding integers
    of each set in a pairwise manner.

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        if domain.eq(x, domain.bot) or domain.eq(y, domain.bot):
            return domain.bot

        frm, to = x[0] + y[0], x[1] + y[1]
        if frm >= domain.top[0] and to <= domain.top[1]:
            return frm, to
        else:
            return domain.top

    return do


def interval_sub_no_wraparound(domain):
    """
    Given an interval domain, returns a function which, given two sets of
    integers represented by elements of this interval domain, returns the
    smallest interval which contains all possible results of subtracting
    integers of each set in a pairwise manner.

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        if domain.eq(x, domain.bot) or domain.eq(y, domain.bot):
            return domain.bot

        frm, to = x[0] - y[1], x[1] - y[0]
        if frm >= domain.top[0] and to <= domain.top[1]:
            return frm, to
        else:
            return domain.top

    return do


def interval_inverse(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by an element of this interval domain, returns the
    smallest interval which contains the inverses of all the integers in the
    given interval.
    """
    def do(x):
        if domain.eq(x, domain.bot):
            return domain.bot

        a, b = -x[0], -x[1]
        return domain.build(min(a, b), max(a, b))

    return do


def interval_eq(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing equality between integers of each set in a pairwise
    manner.
    """
    def do(x, y):
        if domain.eq(x, domain.bot) or domain.eq(y, domain.bot):
            return bool_none
        elif x[0] == y[0] == x[1] == y[1]:
            return bool_true
        elif domain.eq(domain.meet(x, y), domain.bot):
            return bool_false
        else:
            return bool_both

    return do


def interval_neq(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing inequality between integers of each set in a pairwise
    manner.
    """
    eq = interval_eq(domain)

    def do(x, y):
        return boolean_not(eq(x, y))

    return do


def interval_lt(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is less than" between integers of each set in a
    pairwise manner.
    """
    def do(x, y):
        if domain.eq(x, domain.bot) or domain.eq(y, domain.bot):
            return bool_none
        elif x[1] < y[0]:
            return bool_true
        elif x[0] >= y[1]:
            return bool_false
        else:
            return bool_both

    return do


def interval_le(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is less than or equal" between integers of each set
    in a pairwise manner.
    """
    lt = interval_lt(domain)
    eq = interval_eq(domain)

    def do(x, y):
        return boolean_or(lt(x, y), eq(x, y))

    return do


def interval_gt(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is greater than" between integers of each set in a
    pairwise manner.
    """
    lt = interval_lt(domain)

    def do(x, y):
        return lt(y, x)

    return do


def interval_ge(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by an element of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is greater than or equal" between integers of each
    set in a pairwise manner.
    """
    le = interval_le(domain)

    def do(x, y):
        return le(y, x)

    return do


def boolean_not(x):
    """
    Given a set of booleans, returns the smallest set of boolean values
    containing the result of negating each boolean value of the given set.
    """
    if Boolean.eq(x, bool_both) or Boolean.eq(x, bool_none):
        return x
    elif Boolean.eq(x, bool_true):
        return bool_false
    else:
        return bool_true


def boolean_and(x, y):
    """
    Given two sets of booleans, returns the smallest set of boolean values
    containing all possible results of applying conjunction between booleans
    of each set in a pairwise manner.
    """
    if Boolean.eq(x, bool_none) or Boolean.eq(y, bool_none):
        return bool_none
    elif Boolean.eq(x, bool_false) or Boolean.eq(y, bool_false):
        return bool_false
    elif Boolean.eq(x, bool_true) and Boolean.eq(y, bool_true):
        return bool_true
    else:
        return bool_both


def boolean_or(x, y):
    """
    Given two sets of booleans, returns the smallest set of boolean values
    containing all possible results of applying disjunction between booleans
    of each set in a pairwise manner.
    """
    if Boolean.eq(x, bool_none) or Boolean.eq(y, bool_none):
        return bool_none
    elif Boolean.eq(x, bool_true) or Boolean.eq(y, bool_true):
        return bool_true
    elif Boolean.eq(x, bool_false) and Boolean.eq(y, bool_false):
        return bool_false
    else:
        return bool_both


Boolean = domains.FiniteLattice.of_subsets({'True', 'False'})
Int32 = domains.Intervals(-2147483648, 2147483647)

bool_none = Boolean.build({})
bool_false = Boolean.build({'False'})
bool_true = Boolean.build({'True'})
bool_both = Boolean.build({'True', 'False'})


class Definitions(object):
    """
    Provides facilities for storing operations between domain elements.
    """
    def __init__(self):
        """
        Creates a Definitions object containing no operation.
        """
        self.ops = defaultdict(dict)

    def register(self, name, tps, fun):
        """
        Registers a new operation of the given name, acting on the given
        types tps, and which semantics are given by fun.
        """
        self.ops[name][tps] = fun

    def lookup(self, name, tps):
        """
        Finds the semantics of the operation of the given name, which acts
        on the given types.
        """
        return self.ops[name][tps]

    def register_new_interval_int(self, dom):
        """
        Registers a new set of operations acting on the given interval domain.
        Defines the basic comparison and arithmetic operations.
        """
        self.register('<', (dom, dom, Boolean), interval_lt(dom))
        self.register('<=', (dom, dom, Boolean), interval_le(dom))
        self.register('==', (dom, dom, Boolean), interval_eq(dom))
        self.register('!=', (dom, dom, Boolean), interval_neq(dom))
        self.register('>=', (dom, dom, Boolean), interval_ge(dom))
        self.register('>', (dom, dom, Boolean), interval_gt(dom))

        self.register('+', (dom, dom, dom), interval_add_no_wraparound(dom))
        self.register('-', (dom, dom, dom), interval_sub_no_wraparound(dom))

        self.register('-', (dom, dom), interval_inverse(dom))
        return self

    @staticmethod
    def default():
        """
        Creates a Definitions object containing operations that act on the
        universal Boolean type.
        """
        defs = Definitions()
        defs.register('!', (Boolean, Boolean), boolean_not)
        defs.register('&&', (Boolean, Boolean, Boolean), boolean_and)
        defs.register('||', (Boolean, Boolean, Boolean), boolean_or)
        return defs
