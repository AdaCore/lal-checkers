"""
Provides a collection of common useful operations on product domains.
"""


import boolean_ops


def eq(elem_eq_defs):
    """
    :param list[(object, object)->frozenset[str]] elem_eq_defs: The equality
        function for each element of the product.

    :return: A function which tests the equality between elements of the
        product domain.

    :rtype: (tuple, tuple) -> frozenset[str]
    """
    def do(x, y):
        """
        :param tuple x: A set of concrete tuples, represented by an element
            of the abstract domain.

        :param tuple y: A set of concrete tuples, represented by an element
            of the abstract domain.

        :return: A set of boolean values which contains all the possible
            boolean values that can result from testing equality between
            the tuples represented by given abstract elements in a pairwise
            manner.

        :rtype: frozenset[str]
        """
        elem_res = [
            elem_eq_def(a, b)
            for elem_eq_def, a, b in zip(elem_eq_defs, x, y)
        ]
        return reduce(boolean_ops.and_, elem_res)

    return do


def neq(elem_eq_defs):
    """
    :param list[(object, object)->frozenset[str]] elem_eq_defs: The equality
        function for each element of the product.

    :return: A function which tests the inequality between elements of the
        product domain.

    :rtype: (tuple, tuple) -> frozenset[str]
    """
    do_eq = eq(elem_eq_defs)

    def do(x, y):
        """
        :param tuple x: A set of concrete tuples, represented by an element
            of the abstract domain.

        :param tuple y: A set of concrete tuples, represented by an element
            of the abstract domain.

        :return: A set of boolean values which contains all the possible
            boolean values that can result from testing inequality between
            the tuples represented by given abstract elements in a pairwise
            manner.

        :rtype: frozenset[str]
        """
        return boolean_ops.not_(do_eq(x, y))

    return do


def inv_eq(domain, elem_inv_eq_defs, elem_eq_defs):
    """"
    :param lalcheck.domains.Product domain: A product domain.

    :param list[function] elem_inv_eq_defs: The inverse of the inequality
        function for each element of the product.

    :param list[(object, object)->frozenset[str]] elem_eq_defs: The equality
        function for each element of the product.

    :return: A function which performs the inverse of the equality operation.

    :rtype: (frozenset[str], tuple, tuple)
                -> ((frozenset[object], frozenset[object]) | None)
    """

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the equality operation, represented by an element of the
            Boolean domain.

        :param tuple l_constr: A constraint on the left input value of the
            equality operation, as an element of the product domain.

        :param tuple r_constr: A constraint on the right input value of the
            equality operation, as an element of the product domain.

        :return: Two sets of concrete values describing all the possible inputs
            of the equality operation which can result in the given output,
            represented by elements of the product domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: (frozenset[object], frozenset[object]) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            expected = [boolean_ops.true] * len(domain.domains)
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            eq_tests = [
                e_eq_def(a, b)
                for e_eq_def, a, b in zip(elem_eq_defs, l_constr, r_constr)
            ]
            true_test_count = sum(
                1 for x in eq_tests
                if boolean_ops.Boolean.eq(x, boolean_ops.true)
            )

            if true_test_count == len(eq_tests):
                return None
            elif true_test_count == len(eq_tests) - 1:
                expected = [
                    boolean_ops.true
                    if boolean_ops.Boolean.eq(x, boolean_ops.true)
                    else boolean_ops.false
                    for x in eq_tests
                ]
            else:
                expected = [boolean_ops.both] * len(domain.domains)
        else:  # both
            return l_constr, r_constr

        zipped_ret = [
            elem_inv_eq_def(e_res, e_l_constr, e_r_constr)
            for elem_inv_eq_def, e_res, e_l_constr, e_r_constr in zip(
                elem_inv_eq_defs, expected, l_constr, r_constr
            )
        ]

        if any(x is None for x in zipped_ret):
            return None

        return (
            domain.build(*(l for l, _ in zipped_ret)),
            domain.build(*(r for _, r in zipped_ret))
        )

    return do


def inv_neq(domain, elem_inv_eq_defs, elem_eq_defs):
    """"
    :param lalcheck.domains.Product domain: A product domain.

    :param list[function] elem_inv_eq_defs: The inverse of the inequality
        function for each element of the product.

    :param list[(object, object)->frozenset[str]] elem_eq_defs: The equality
        function for each element of the product.

    :return: A function which performs the inverse of the inequality operation.

    :rtype: (frozenset[str], tuple, tuple)
                -> ((frozenset[object], frozenset[object]) | None)
    """
    do_inv_eq = inv_eq(domain, elem_inv_eq_defs, elem_eq_defs)

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the inequality operation, represented by an element of the
            Boolean domain.

        :param tuple l_constr: A constraint on the left input value of the
            inequality operation, as an element of the product domain.

        :param tuple r_constr: A constraint on the right input value of the
            inequality operation, as an element of the product domain.

        :return: Two sets of concrete values describing all the possible inputs
            of the inequality operation which can result in the given output,
            represented by elements of the product domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: (frozenset[object], frozenset[object]) | None
        """
        return do_inv_eq(boolean_ops.not_(res), l_constr, r_constr)

    return do


def lit(_):
    raise NotImplementedError
