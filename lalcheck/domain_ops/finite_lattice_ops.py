"""
Provides a collection of common useful operations on finite lattices.

NOTE: these operations only make sense if the elements of the finite lattices
correspond to sets of concrete values.
"""


import boolean_ops


def eq(domain):
    """"
    Given an finite lattice domain, returns a function which, given two sets of
    concrete values represented by elements of this domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing equality between concrete values of each set in a
    pairwise manner.
    """
    def do(x, y):
        if domain.is_empty(x) or domain.is_empty(y):
            return boolean_ops.none
        elif all(all(a == b for a in x) for b in y):
            return boolean_ops.true
        elif not any(any(a == b for a in x) for b in y):
            return boolean_ops.false
        else:
            return boolean_ops.both

    return do


def neq(domain):
    """"
    Given an finite lattice domain, returns a function which, given two sets of
    concrete values represented by elements of this domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing inequality between concrete values of each set in a
    pairwise manner.
    """

    do_eq = eq(domain)

    def do(x, y):
        return boolean_ops.not_(do_eq(x, y))

    return do


def inv_eq(domain):
    """
    Given a finite lattice domain, returns a function which computes the
    inverse of the eq function, which results are constrained by its
    second and third arguments.
    """
    def do(res, l_constr, r_constr):
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            meet = domain.meet(l_constr, r_constr)
            return None if domain.is_empty(meet) else (meet, meet)
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            if len(l_constr) == 1:
                if not len(domain.meet(l_constr, r_constr)) == 0:
                    r_constr = frozenset(r_constr) - frozenset(l_constr)
            elif len(r_constr) == 1:
                if not len(domain.meet(l_constr, r_constr)) == 0:
                    l_constr = frozenset(l_constr) - frozenset(r_constr)

            if domain.is_empty(l_constr) or domain.is_empty(r_constr):
                return None
            else:
                return l_constr, r_constr
        else:
            return l_constr, r_constr

    return do


def inv_neq(domain):
    """
    Given a finite lattice domain, returns a function which computes the
    inverse of the neq function, which results are constrained by its
    second and third arguments.
    """

    do_inv_eq = inv_eq(domain)

    def do(res, l_constr, r_constr):
        return do_inv_eq(boolean_ops.not_(res), l_constr, r_constr)

    return do


def lit(domain):
    """
    Given a finite lattice domain, returns a function which, given a concrete
    value, returns the smallest element in the domain containing it.
    """
    def do(val):
        return domain.build(frozenset([val]))

    return do
