"""
Provides a collection of useful operations on interval domains.
"""


from lalcheck import domains
import boolean_ops


def interval_add_no_wraparound(domain):
    """
    Given an interval domain, returns a function which, given two sets of
    integers represented by elements of this interval domain, returns the
    smallest interval which contains all possible results of adding integers
    of each set in a pairwise manner.

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return domain.bottom

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
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return domain.bottom

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
        if domain.eq(x, domain.bottom):
            return domain.bottom

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
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return boolean_ops.bool_none
        elif x[0] == y[0] == x[1] == y[1]:
            return boolean_ops.bool_true
        elif domain.eq(domain.meet(x, y), domain.bottom):
            return boolean_ops.bool_false
        else:
            return boolean_ops.bool_both

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
        return boolean_ops.boolean_not(eq(x, y))

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
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return boolean_ops.bool_none
        elif x[1] < y[0]:
            return boolean_ops.bool_true
        elif x[0] >= y[1]:
            return boolean_ops.bool_false
        else:
            return boolean_ops.bool_both

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
        return boolean_ops.boolean_or(lt(x, y), eq(x, y))

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


def interval_lit(domain):
    """
    Given an interval domain, returns a function which returns the smallest
    interval containing the given concrete value.
    """
    def do(lit):
        return domain.build(lit)

    return do


Int32 = domains.Intervals(-2147483648, 2147483647)
