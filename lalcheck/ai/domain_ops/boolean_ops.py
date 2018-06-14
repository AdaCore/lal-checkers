"""
Provides the universal "Boolean" domain as well as a collection of
common useful operations.
"""


from lalcheck.ai import domains

from lalcheck.ai.constants import lits

Boolean = domains.FiniteLattice.of_subsets({lits.TRUE, lits.FALSE})

none = Boolean.build(frozenset([]))
false = Boolean.build(frozenset([lits.FALSE]))
true = Boolean.build(frozenset([lits.TRUE]))
both = Boolean.build(frozenset([lits.TRUE, lits.FALSE]))


def not_(x):
    """
    :param frozenset[str] x: A set of booleans represented by an element of
        the Boolean domain.

    :return: The smallest set of boolean values containing the
        result of negating each boolean value in x, represented by an element
        of the Boolean domain.

    :rtype: frozenset[str]
    """
    if Boolean.eq(x, both) or Boolean.eq(x, none):
        return x
    elif Boolean.eq(x, true):
        return false
    else:
        return true


def and_(x, y):
    """
    :param frozenset[str] x: A set of booleans represented by an element of
        the Boolean domain.

    :param frozenset[str] y: A set of booleans represented by an element of
        the Boolean domain.

    :return: The smallest set of boolean values containing all possible
        results of applying conjunction between booleans of each set in a
        pairwise manner, represented by an element of the Boolean domain.

    :rtype: frozenset[str]
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
    :param frozenset[str] x: A set of booleans represented by an element of
        the Boolean domain.

    :param frozenset[str] y: A set of booleans represented by an element of
        the Boolean domain.

    :return: The smallest set of boolean values containing all possible
        results of applying disjunction between booleans of each set in a
        pairwise manner, represented by an element of the Boolean domain.

    :rtype: frozenset[str]
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
    :param frozenset[str] res: A set of booleans corresponding to an output of
        the not operation, represented by an element of the Boolean domain.

    :param frozenset[str] e_constr: A constraint on the input value of the not
        operation, as an element of the Boolean domain.

    :return: A set of booleans describing all the possible inputs of the not
        operation which can result in the given output, represented by an
        element of the Boolean domain. Returns None if the constraint cannot
        be satisfied.

    :rtype: frozenset[str] | None
    """
    if Boolean.is_empty(e_constr) or Boolean.eq(res, none):
        return None

    if Boolean.eq(res, both):
        return e_constr

    ret = not_(res)
    return ret if Boolean.le(ret, e_constr) else None


def inv_and(res, l_constr, r_constr):
    """
    :param frozenset[str] res: A set of booleans corresponding to an output of
        a conjunction, represented by an element of the Boolean domain.

    :param frozenset[str] l_constr: A constraint on the left input value of
        a conjunction, as an element of the Boolean domain.

    :param frozenset[str] r_constr: A constraint on the right input value of
        a conjunction, as an element of the Boolean domain.

    :return: Two sets of booleans describing all the possible inputs of a
        conjunction operation which can result in the given output,
        represented by an element of the Boolean domain. Returns None if the
        constraints cannot be satisfied.

    :rtype: (frozenset[str], frozenset[str]) | None
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
    :param frozenset[str] res: A set of booleans corresponding to an output of
        a disjunction, represented by an element of the Boolean domain.

    :param frozenset[str] l_constr: A constraint on the left input value of
        a disjunction, as an element of the Boolean domain.

    :param frozenset[str] r_constr: A constraint on the right input value of
        a disjunction, as an element of the Boolean domain.

    :return: Two sets of booleans describing all the possible inputs of a
        disjunction operation which can result in the given output,
        represented by an element of the Boolean domain. Returns None if the
        constraints cannot be satisfied.

    :rtype: (frozenset[str], frozenset[str]) | None
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
    :param str val: A concrete element of the Boolean domain, i.e. 'True' or
        'False'.

    :return: The singleton set of booleans containing the given concrete
        boolean value, represented by an element of the Boolean domain.

    :rtype: frozenset[str]
    """
    if val == lits.TRUE:
        return true
    elif val == lits.FALSE:
        return false
