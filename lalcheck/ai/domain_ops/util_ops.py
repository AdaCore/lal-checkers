"""
Contains some generic domain operations.
"""

import boolean_ops


def included(domain):
    """
    Returns a function which tests whether some concrete elements represented
    by its first arguments are included or not in the concretization of its
    second argument.

    :param lalcheck.ai.domains.AbstractDomain domain: The abstract domain
        on which to test inclusion of elements.

    :rtype: (object, object) -> frozenset
    """
    def do(x, y):
        """
        Returns a set of booleans as an element of the abstract boolean domain,
        such that True is included in that set if at least one concrete element
        of x is included in the concretization of y. False is included in that
        set if at least one element of x is not included in the concretization
        of y.

        :type x: object
        :type y: object
        :rtype: frozenset
        """
        if domain.is_empty(x):
            return boolean_ops.none

        meet = domain.meet(x, y)

        if domain.eq(meet, x):
            return boolean_ops.true
        elif domain.is_empty(meet):
            return boolean_ops.false
        else:
            return boolean_ops.both

    return do


def inv_included(domain):
    """
    Returns a function which computes the inverse of the inclusion test.

    :param lalcheck.ai.domains.AbstractDomain domain: The abstract domain
        on which to test inclusion of elements.

    :rtype: (frozenset, object, object) -> (object, object)
    """
    def do(res, x_constr, y_constr):
        """
        Given an expected result of the inclusion test and two constraints
        on the arguments, returns sets of possible input arguments as element
        of the given abstract domain such that the inclusion test would return
        the given booleans.

        :param frozenset res: A set of booleans as an element of the abstract
            boolean domain.

        :param object x_constr:
        :param object y_constr:
        :rtype: (object, object)
        """
        if (domain.is_empty(res) or domain.is_empty(x_constr) or
                domain.is_empty(y_constr)):
            return None

        # only implemented precisely for True. In the other cases, the
        # constraints are returned without further refinement.
        if res == boolean_ops.true:
            return domain.meet(x_constr, y_constr), y_constr

        return x_constr, y_constr

    return do
