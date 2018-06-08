"""
Provides a collection of useful operations on interval domains.
"""


from ai import domains
import boolean_ops


def add_no_wraparound(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the addition between two intervals.

    :rtype: ((int, int), (int, int)) -> (int, int)

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of integers which contains all the possible values
            that can result from the addition between concrete values of
            each given set in a pairwise manner, represented by an element
            of the interval domain.

        :rtype: (int, int)
        """
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return domain.bottom

        frm, to = x[0] + y[0], x[1] + y[1]
        if frm >= domain.top[0] and to <= domain.top[1]:
            return frm, to
        else:
            return domain.top

    return do


def sub_no_wraparound(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the subtraction between two intervals.

    :rtype: ((int, int), (int, int)) -> (int, int)

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of integers which contains all the possible values
            that can result from the subtraction between concrete values of
            each given set in a pairwise manner, represented by an element
            of the interval domain.

        :rtype: (int, int)
        """
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return domain.bottom

        frm, to = x[0] - y[1], x[1] - y[0]
        if frm >= domain.top[0] and to <= domain.top[1]:
            return frm, to
        else:
            return domain.top

    return do


def negate(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the negation of an interval.

    :rtype: ((int, int)) -> (int, int)
    """
    def do(x):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of integers which contains all the possible values
            that can result from the negating the concrete values of the given
            set of integers, represented by an element of the interval domain.

        :rtype: (int, int)
        """
        if domain.eq(x, domain.bottom):
            return domain.bottom

        return domain.build(-x[1], -x[0])

    return do


def eq(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the equality between two intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing equality between concrete values of
            each given set in a pairwise manner, represented by an element
            of the Boolean domain.

        :rtype: frozenset[str]
        """
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return boolean_ops.none
        elif x[0] == y[0] == x[1] == y[1]:
            return boolean_ops.true
        elif domain.eq(domain.meet(x, y), domain.bottom):
            return boolean_ops.false
        else:
            return boolean_ops.both

    return do


def neq(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inequality between two intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    do_eq = eq(domain)

    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing inequality between concrete values of
            each given set in a pairwise manner, represented by an element
            of the Boolean domain.

        :rtype: frozenset[str]
        """
        return boolean_ops.not_(do_eq(x, y))

    return do


def lt(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the "less than" between two intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing "is less than" between concrete values
            of each given set in a pairwise manner, represented by an element
            of the Boolean domain.

        :rtype: frozenset[str]
        """
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return boolean_ops.none
        elif x[1] < y[0]:
            return boolean_ops.true
        elif x[0] >= y[1]:
            return boolean_ops.false
        else:
            return boolean_ops.both

    return do


def le(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the "less than or equal" between two
        intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    do_lt = lt(domain)
    do_eq = eq(domain)

    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing "is less than or equal to" between
            concrete values of each given set in a pairwise manner, represented
            by an element of the Boolean domain.

        :rtype: frozenset[str]
        """
        return boolean_ops.or_(do_lt(x, y), do_eq(x, y))

    return do


def gt(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the "greater than" between two
        intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    do_lt = lt(domain)

    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing "is greater than" between concrete
            values of each given set in a pairwise manner, represented by an
            element of the Boolean domain.

        :rtype: frozenset[str]
        """
        return do_lt(y, x)

    return do


def ge(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the "greater than or equal" between two
        intervals.

    :rtype: ((int, int), (int, int)) -> frozenset[str]
    """
    do_le = le(domain)

    def do(x, y):
        """
        :param (int, int) x: A set of concrete integers, represented by
            an element of the interval domain.

        :param (int, int) y: A set of concrete integers, represented by
            an element of the interval domain.

        :return: A set of booleans which contains all the possible values
            that can result from testing "is greater than or equal to" between
            concrete values of each given set in a pairwise manner, represented
            by an element of the Boolean domain.

        :rtype: frozenset[str]
        """
        return do_le(y, x)

    return do


def inv_add_no_wraparound(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the addition operation
        between two intervals.

    :rtype: ((int, int), (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """
    def do(res, l_constr, r_constr):
        """
        :param (int, int) res: A set of integers corresponding to an output
            of the addition operation, represented by an element of the
            interval domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the addition operation, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the addition operation, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the addition operation which can result in the given output,
            represented by elements of the interval domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                domain.is_empty(res)):
            return None

        # Compute the largest possible left operand interval such that adding
        # to it the right constraint gives the desired result.
        p_l = res[0] - r_constr[1], res[1] - r_constr[0]

        # Compute the largest possible right operand interval such that adding
        # to it the left constraint gives the desired result.
        p_r = res[0] - l_constr[1], res[1] - l_constr[0]

        x = domain.meet(p_l, l_constr)
        y = domain.meet(p_r, r_constr)

        if domain.is_empty(x) or domain.is_empty(y):
            return None

        return x, y

    return do


def inv_sub_no_wraparound(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the subtraction operation
        between two intervals.

    :rtype: ((int, int), (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """
    def do(res, l_constr, r_constr):
        """
        :param (int, int) res: A set of integers corresponding to an output
            of the subtraction operation, represented by an element of the
            interval domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the subtraction operation, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the subtraction operation, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the subtraction operation which can result in the given output,
            represented by elements of the interval domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                domain.is_empty(res)):
            return None

        # Compute the largest possible left operand interval such that
        # subtracting from it the right constraint gives the desired result.
        p_l = res[0] + r_constr[0], res[1] + r_constr[1]

        # Compute the largest possible right operand interval such that
        # subtracting from it the left constraint gives the desired result.
        p_r = l_constr[0] - res[1], l_constr[1] - res[0]

        x = domain.meet(p_l, l_constr)
        y = domain.meet(p_r, r_constr)

        if domain.is_empty(x) or domain.is_empty(y):
            return None

        return x, y

    return do


def inv_inverse(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the "inverse" operation
        on an interval.

    :rtype: ((int, int), (int, int)) -> ((int, int) | None)
    """
    do_inverse = negate(domain)

    def do(res, constr):
        """
        :param (int, int) res: A set of integers corresponding to an output
            of the inverse operation, represented by an element of the
            interval domain.

        :param (int, int) constr: A constraint on the input value of the
            inverse operation, as an element of the interval domain.

        :return: A set of integers describing all the possible inputs
            of the inverse operation which can result in the given output,
            represented by an element of the interval domain. Returns None
            if the constraint cannot be satisfied.

        :rtype: (int, int) | None
        """
        ret = domain.meet(constr, do_inverse(res))
        if domain.is_empty(ret):
            return None
        return ret

    return do


def inv_eq(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the equality test
        between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """
    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the equality test, represented by an element of the Boolean
            domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the equality test, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the equality test, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the equality test which can result in the given output,
            represented by elements of the interval domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            meet = domain.meet(l_constr, r_constr)
            if domain.is_empty(meet):
                return None
            return meet, meet
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            meet = domain.meet(l_constr, r_constr)
            if domain.is_empty(meet):
                return l_constr, r_constr
            elif meet[0] == meet[1]:
                (x_f, x_t), (y_f, y_t) = l_constr, r_constr
                if x_f == x_t == y_f == y_t:
                    return None
                elif x_f == x_t == y_f:
                    return l_constr, (y_f + 1, y_t)
                elif x_f == x_t == y_t:
                    return l_constr, (y_f, y_t - 1)
                elif y_f == y_t == x_f:
                    return (x_f + 1, x_t), r_constr
                elif y_f == y_t == x_t:
                    return (x_f, x_t - 1), r_constr
            return l_constr, r_constr
        elif boolean_ops.Boolean.eq(res, boolean_ops.both):
            return l_constr, r_constr

    return do


def inv_neq(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the inequality test
        between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """
    do_inv_eq = inv_eq(domain)

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the inequality test, represented by an element of the Boolean
            domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the inequality test, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the inequality test, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the inequality test which can result in the given output,
            represented by elements of the interval domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        return do_inv_eq(boolean_ops.not_(res), l_constr, r_constr)

    return do


def inv_lt(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the "is less than" test
        between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """
    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the "is less than" test, represented by an element of the
            Boolean domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the "is less than" test, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the "is less than" test, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the "is less than"  test which can result in the given output,
            represented by elements of the interval domain. Returns None
            if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        x_f, x_t = l_constr[0], l_constr[1]
        y_f, y_t = r_constr[0], r_constr[1]

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            x_f, x_t = x_f, min(y_t - 1, x_t)
            y_f, y_t = max(x_f + 1, y_f), y_t
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            x_f, x_t = max(x_f, y_f), x_t
            y_f, y_t = y_f, min(x_t, y_t)
        elif boolean_ops.Boolean.eq(res, boolean_ops.both):
            return l_constr, r_constr

        if (x_f > x_t) or (y_f > y_t):
            return None
        else:
            return (x_f, x_t), (y_f, y_t)

    return do


def inv_le(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the "is less than or
        equal" test between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the "is less than or equal" test, represented by an element of
            the Boolean domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the "is less than or equal" test, as an element of the interval
            domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the "is less than or equal" test, as an element of the interval
            domain.

        :return: Two sets of integers describing all the possible inputs
            of the "is less than or equal" test which can result in the given
            output, represented by elements of the interval domain. Returns
            None if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        if (domain.is_empty(l_constr) or domain.is_empty(r_constr) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        x_f, x_t = l_constr[0], l_constr[1]
        y_f, y_t = r_constr[0], r_constr[1]

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            x_f, x_t = x_f, min(y_t, x_t)
            y_f, y_t = max(x_f, y_f), y_t
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            x_f, x_t = max(y_f + 1, x_f), x_t
            y_f, y_t = y_f, min(x_t - 1, y_t)
        elif boolean_ops.Boolean.eq(res, boolean_ops.both):
            return l_constr, r_constr

        if (x_f > x_t) or (y_f > y_t):
            return None
        else:
            return (x_f, x_t), (y_f, y_t)

    return do


def inv_gt(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the "is greater than"
        test between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """

    do_inv_le = inv_le(domain)

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the "is greater than" test, represented by an element of the
            Boolean domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the "is greater than" test, as an element of the interval domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the "is greater than" test, as an element of the interval domain.

        :return: Two sets of integers describing all the possible inputs
            of the "is greater than" test which can result in the given
            output, represented by elements of the interval domain. Returns
            None if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        return do_inv_le(boolean_ops.not_(res), l_constr, r_constr)

    return do


def inv_ge(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which computes the inverse of the "is greater than or
        equal" test between two intervals.

    :rtype: (frozenset[str], (int, int), (int, int))
                -> (((int, int), (int, int)) | None)
    """

    do_inv_lt = inv_lt(domain)

    def do(res, l_constr, r_constr):
        """
        :param frozenset[str] res: A set of booleans corresponding to an output
            of the "is greater than or equal" test, represented by an element
            of the Boolean domain.

        :param (int, int) l_constr: A constraint on the left input value of
            the "is greater than or equal" test, as an element of the interval
            domain.

        :param (int, int) l_constr: A constraint on the right input value of
            the "is greater than or equal" test, as an element of the interval
            domain.

        :return: Two sets of integers describing all the possible inputs
            of the "is greater than or equal" test which can result in the
            given output, represented by elements of the interval domain.
            Returns None if the constraints cannot be satisfied.

        :rtype: ((int, int), (int, int)) | None
        """
        return do_inv_lt(boolean_ops.not_(res), l_constr, r_constr)

    return do


def lit(domain):
    """
    :param lalcheck.domains.Intervals domain: An intervals domain.

    :return: A function which can be used to build singleton elements of the
        given interval domain.

    :rtype: (int) -> (int, int)
    """
    def do(val):
        """
        :param int val: The concrete integer to represent.

        :return: The singleton interval containing the given value

        :rtype: (int, int)
        """
        return domain.build(val)

    return do


Int32 = domains.Intervals(-2147483648, 2147483647)
