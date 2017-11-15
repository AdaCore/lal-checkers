"""
Provides a collection of common useful operations on finite lattices.

NOTE: these operations only make sense if the elements of the finite lattices
correspond to sets of concrete values.
"""


import boolean_ops


def eq(domain):
    """
    :param lalcheck.domains.FiniteLattice domain: A finite lattice domain.

    :return: A function which performs the equality operation.

    :rtype: (frozenset[object], frozenset[object]) -> frozenset[str]
    """
    def do(x, y):
        """
        :param frozenset[object] x: A set of concrete elements, represented by
            an element of the abstract domain.

        :param frozenset[object] y: A set of concrete elements, represented by
            an element of the abstract domain.

        :return: A set of boolean values which contains all the possible
            boolean values that can result from testing equality between
            concrete values of each given set in a pairwise manner.

        :rtype: frozenset[str]
        """
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
    """
    :param lalcheck.domains.FiniteLattice domain: A finite lattice domain.

    :return: A function which performs the inequality operation.

    :rtype: (frozenset[object], frozenset[object]) -> frozenset[str]
    """

    do_eq = eq(domain)

    def do(x, y):
        """
        :param frozenset[object] x: A set of concrete elements, represented by
            an element of the abstract domain.

        :param frozenset[object] y: A set of concrete elements, represented by
            an element of the abstract domain.

        :return: A set of boolean values which contains all the possible
            boolean values that can result from testing inequality between
            concrete values of each given set in a pairwise manner.

        :rtype: frozenset[str]
        """
        return boolean_ops.not_(do_eq(x, y))

    return do


def inv_eq(domain):
    """"
    :param lalcheck.domains.FiniteLattice domain: A finite lattice domain.

    :return: A function which performs the inverse of the equality operation.

    :rtype: (frozenset[str], frozenset[object], frozenset[object])
                -> ((frozenset[object], frozenset[object]) | None)
    """
    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the equality operation, represented by an element of the
            Boolean domain.

        :param frozenset[object] l_constr: A constraint on the left input
            value of the equality operation, as an element of the finite
            lattice domain.

        :param frozenset[object] r_constr: A constraint on the right input
            value of the equality operation, as an element of the finite
            lattice domain.

        :return: Two sets of concrete values describing all the possible inputs
            of the equality operation which can result in the given output,
            represented by elements of the finite lattice domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: (frozenset[object], frozenset[object]) | None
        """
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
    """"
    :param lalcheck.domains.FiniteLattice domain: A finite lattice domain.

    :return: A function which performs the inverse of the inequality operation.

    :rtype: (frozenset[str], frozenset[object], frozenset[object])
                -> ((frozenset[object], frozenset[object]) | None)
    """

    do_inv_eq = inv_eq(domain)

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the inequality operation, represented by an element of the
            Boolean domain.

        :param frozenset[object] l_constr: A constraint on the left input
            value of the inequality operation, as an element of the finite
            lattice domain.

        :param frozenset[object] r_constr: A constraint on the right input
            value of the inequality operation, as an element of the finite
            lattice domain.

        :return: Two sets of concrete values describing all the possible inputs
            of the inequality operation which can result in the given output,
            represented by elements of the finite lattice domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: (frozenset[object], frozenset[object]) | None
        """
        return do_inv_eq(boolean_ops.not_(res), l_constr, r_constr)

    return do


def lit(domain):
    """
    :param lalcheck.domains.FiniteLattice domain: The finite lattice domain.

    :return: A function which can be used to build singleton elements of the
        given domain.

    :rtype: (object) -> frozenset[object]
    """
    def do(val):
        """
        :param object val: The concrete value to represent.

        :return: The singleton set containing the given value

        :rtype: frozenset[object]
        """
        return domain.build(frozenset([val]))

    return do
