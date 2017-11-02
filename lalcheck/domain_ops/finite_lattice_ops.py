"""
Provides a collection of common useful operations on finite lattices.
"""


import boolean_ops


def finite_lattice_eq(domain):
    """"
    Given an finite lattice domain, returns a function which, given two sets of
    concrete values represented by elements of this domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing equality between concrete values of each set in a
    pairwise manner.
    """
    def do(x, y):
        if domain.is_empty(x) or domain.is_empty(y):
            return boolean_ops.bool_none
        elif all(all(a == b for a in x) for b in y):
            return boolean_ops.bool_true
        elif not any(any(a == b for a in x) for b in y):
            return boolean_ops.bool_false
        else:
            return boolean_ops.bool_both

    return do


def finite_lattice_lit(domain):
    """
    Given a finite lattice domain, returns a function which, given a concrete
    value, returns the smallest element in the domain containing it.
    """
    def do(lit):
        return domain.build(frozenset([lit]))

    return do
