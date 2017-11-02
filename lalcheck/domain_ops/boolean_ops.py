"""
Provides the universal "Boolean" domain as well as a collection of
common useful operations.
"""


from lalcheck import domains


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


def boolean_lit(lit):
    """
    Returns the smallest abstract element containing the given concrete
    boolean value.
    """
    if lit == 'True':
        return bool_true
    elif lit == 'False':
        return bool_false


Boolean = domains.FiniteLattice.of_subsets({'True', 'False'})

bool_none = Boolean.build(frozenset([]))
bool_false = Boolean.build(frozenset(['False']))
bool_true = Boolean.build(frozenset(['True']))
bool_both = Boolean.build(frozenset(['True', 'False']))
