"""
Provides the universal "Boolean" domain as well as a collection of
common useful operations.
"""


from lalcheck import domains
from lalcheck.constants import lits


Boolean = domains.FiniteLattice.of_subsets({lits.True, lits.False})

none = Boolean.build(frozenset([]))
false = Boolean.build(frozenset([lits.False]))
true = Boolean.build(frozenset([lits.True]))
both = Boolean.build(frozenset([lits.True, lits.False]))


def not_(x):
    """
    Given a set of booleans, returns the smallest set of boolean values
    containing the result of negating each boolean value of the given set.
    """
    if Boolean.eq(x, both) or Boolean.eq(x, none):
        return x
    elif Boolean.eq(x, true):
        return false
    else:
        return true


def and_(x, y):
    """
    Given two sets of booleans, returns the smallest set of boolean values
    containing all possible results of applying conjunction between booleans
    of each set in a pairwise manner.
    """
    if Boolean.eq(x, none) or Boolean.eq(y, none):
        return none
    elif Boolean.eq(x, false) or Boolean.eq(y, false):
        return false
    elif Boolean.eq(x, true) and Boolean.eq(y, true):
        return true
    else:
        return both


def or_(x, y):
    """
    Given two sets of booleans, returns the smallest set of boolean values
    containing all possible results of applying disjunction between booleans
    of each set in a pairwise manner.
    """
    if Boolean.eq(x, none) or Boolean.eq(y, none):
        return none
    elif Boolean.eq(x, true) or Boolean.eq(y, true):
        return true
    elif Boolean.eq(x, false) and Boolean.eq(y, false):
        return false
    else:
        return both


def inv_not(res, e_constr):
    """"
    The inverse of the not function, which result is constrained by the second
    argument.
    """
    if Boolean.is_empty(e_constr) or Boolean.eq(res, none):
        return None

    if Boolean.eq(res, both):
        return e_constr

    ret = not_(res)
    return ret if Boolean.le(ret, e_constr) else None


def inv_and(res, l_constr, r_constr):
    """
    The inverse of the and function, which results are constrained by the
    second and third arguments.
    """
    if (Boolean.is_empty(l_constr) or Boolean.is_empty(r_constr) or
            Boolean.eq(res, none)):
        return None

    if Boolean.eq(res, true):
        if Boolean.le(true, l_constr) and Boolean.le(true, r_constr):
            return true, true
    elif Boolean.eq(res, false):
        if Boolean.eq(false, l_constr) or Boolean.eq(false, r_constr):
            return l_constr, r_constr
        elif Boolean.eq(true, l_constr) and Boolean.le(false, r_constr):
            return true, false
        elif Boolean.eq(true, r_constr) and Boolean.le(false, l_constr):
            return false, true
        elif Boolean.eq(both, l_constr) and Boolean.eq(both, r_constr):
            return both, both
    else:
        return l_constr, r_constr


def inv_or(res, l_constr, r_constr):
    """
    The inverse of the or function, which results are constrained by the second
    and third arguments.
    """
    if (Boolean.is_empty(l_constr) or Boolean.is_empty(r_constr) or
            Boolean.eq(res, none)):
        return None

    if Boolean.eq(res, true):
        if Boolean.eq(true, l_constr) or Boolean.eq(true, r_constr):
            return l_constr, r_constr
        elif Boolean.eq(false, l_constr) and Boolean.le(true, r_constr):
            return false, true
        elif Boolean.eq(false, r_constr) and Boolean.le(true, l_constr):
            return true, false
        elif Boolean.eq(both, l_constr) and Boolean.eq(both, r_constr):
            return both, both
    elif Boolean.eq(res, false):
        if Boolean.le(false, l_constr) and Boolean.le(false, r_constr):
            return false, false
    else:
        return l_constr, r_constr


def lit(val):
    """
    Returns the smallest abstract element containing the given concrete
    boolean value.
    """
    if val == lits.True:
        return true
    elif val == lits.False:
        return false
