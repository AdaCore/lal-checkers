"""
Provides the universal "Boolean" domain as well as a collection of
common useful operations.
"""


from lalcheck import domains


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


def lit(val):
    """
    Returns the smallest abstract element containing the given concrete
    boolean value.
    """
    if val == 'True':
        return true
    elif val == 'False':
        return false


Boolean = domains.FiniteLattice.of_subsets({'True', 'False'})

none = Boolean.build(frozenset([]))
false = Boolean.build(frozenset(['False']))
true = Boolean.build(frozenset(['True']))
both = Boolean.build(frozenset(['True', 'False']))
