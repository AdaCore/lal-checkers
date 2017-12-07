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


def construct(domain):
    """
    :param lalcheck.domains.Product domain: The product domain.
    :return: A function which can construct instances of the product domain.
    :rtype: *object -> tuple
    """
    def do(*x):
        """
        :param *object x: The values of each component.
        :return: An instance of the product domain using the given values.
        :rtype: tuple
        """
        return domain.build(*x)

    return do


def getter(n):
    """
    :param int n: The index of the field to build a getter for.

    :return: A function which gets the nth component of a given instance
        of a product domain.

    :rtype: tuple -> object
    """
    def do(x):
        """
        :param tuple x: An instance of a product domain.
        :return: The nth component of the given instance.
        """
        return x[n]

    return do


def updater(n):
    """
    :param int n: The index of the field to build an updater for.

    :return: A function which updates the nth component of a given instance
        of a product domain.

    :rtype: (tuple, object) -> tuple
    """
    def do(x, new):
        """
        :param tuple x: An instance of a product domain.

        :param object new: A value to update the nth component of the given
            instance with.

        :return: The same instance, with the updated field.

        :rtype: tuple
        """
        return tuple(
            new if i == n else old
            for i, old in enumerate(x)
        )

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


def inv_construct(domain):
    """
    :param lalcheck.domains.Product domain: The product domain.

    :return: A function which performs the inverse of the "construct"
        operation.

    :rtype: (tuple, *object) -> (tuple[object] | None)
    """
    def do(res, *constr):
        """
        :param tuple res: The expected output, as an instance of a product
            domain.

        :param object* constr: The constraints on the inputs of the
            constructor.

        :return: The set of inputs that could construct the given instance
            of the product domain.

        :rtype: tuple[object] | None
        """
        return domain.meet(res, constr)

    return do


def inv_getter(domain, n):
    """
    :param lalcheck.domains.Product domain: The product domain.

    :param int n: The index of the domain component for which to build the
        inverse getter.

    :return: A function which performs the inverse of a get operation.

    :rtype: (object, tuple) -> tuple
    """
    def do(res, constr):
        """
        :param object res: The expected output of the get operation.

        :param tuple constr: The constraint of the input tuples.

        :return: A set of tuple that could give the expected output when
            a get operation on the nth component is performed.

        :rtype: tuple | None
        """
        biggest = tuple(
            res if i == n else e_dom.top
            for i, e_dom in enumerate(domain.domains)
        )

        meet = domain.meet(biggest, constr)

        if domain.is_empty(meet):
            return None

        return meet

    return do


def inv_updater(domain, n):
    """
    :param lalcheck.domains.Product domain: The product domain.

    :param int n: The index of the domain component for which to build the
        inverse updater.

    :return: A function which performs the inverse of an update operation.

    :rtype: (object, tuple, object) -> (tuple, object)
    """
    def do(res, tuple_constr, elem_constr):
        """
        :param tuple res: The expected updated tuple.

        :param tuple tuple_constr: A constraint on the set of tuples to be
            updated.

        :param object elem_constr: A constraint on the set of values which
            to update the tuples with.

        :return: The set of tuples that could be updated into the expected one,
            as well as the set of elements that could be used to update the
            original tuples into the expected ones.

        :rtype: (tuple, object) | None
        """
        raise NotImplementedError

    return do


def lit(_):
    raise NotImplementedError
