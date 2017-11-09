"""
Provides a collection of useful operations on interval domains.
"""


from lalcheck import domains
import boolean_ops


def add_no_wraparound(domain):
    """
    Given an interval domain, returns a function which, given two sets of
    integers represented by elements of this interval domain, returns the
    smallest interval which contains all possible results of adding integers
    of each set in a pairwise manner.

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
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
    Given an interval domain, returns a function which, given two sets of
    integers represented by elements of this interval domain, returns the
    smallest interval which contains all possible results of subtracting
    integers of each set in a pairwise manner.

    Note that overflow is represented by an unknown result (-inf, inf).
    """
    def do(x, y):
        if domain.eq(x, domain.bottom) or domain.eq(y, domain.bottom):
            return domain.bottom

        frm, to = x[0] - y[1], x[1] - y[0]
        if frm >= domain.top[0] and to <= domain.top[1]:
            return frm, to
        else:
            return domain.top

    return do


def inverse(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by an element of this interval domain, returns the
    smallest interval which contains the inverses of all the integers in the
    given interval.
    """
    def do(x):
        if domain.eq(x, domain.bottom):
            return domain.bottom

        a, b = -x[0], -x[1]
        return domain.build(min(a, b), max(a, b))

    return do


def eq(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing equality between integers of each set in a pairwise
    manner.
    """
    def do(x, y):
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
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing inequality between integers of each set in a pairwise
    manner.
    """
    do_eq = eq(domain)

    def do(x, y):
        return boolean_ops.not_(do_eq(x, y))

    return do


def lt(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is less than" between integers of each set in a
    pairwise manner.
    """
    def do(x, y):
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
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is less than or equal" between integers of each set
    in a pairwise manner.
    """
    do_lt = lt(domain)
    do_eq = eq(domain)

    def do(x, y):
        return boolean_ops.or_(do_lt(x, y), do_eq(x, y))

    return do


def gt(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by elements of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is greater than" between integers of each set in a
    pairwise manner.
    """
    do_lt = lt(domain)

    def do(x, y):
        return do_lt(y, x)

    return do


def ge(domain):
    """
    Given an interval domain, returns a function which, given a set of
    integers represented by an element of this interval domain, returns the
    smallest set which contains all the possible boolean values that can
    result from testing "is greater than or equal" between integers of each
    set in a pairwise manner.
    """
    do_le = le(domain)

    def do(x, y):
        return do_le(y, x)

    return do


def inv_add_no_wraparound(domain):
    """
    Given an interval domain, returns a function which computes the inverse of
    the add_no_wraparound function, which results are constrained by its
    second and third arguments.
    """

    def do(res, l_constr, r_constr):
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
    Given an interval domain, returns a function which computes the inverse of
    the sub_no_wraparound function, which results are constrained by its
    second and third arguments.
    """

    def do(res, l_constr, r_constr):
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
    Given an interval domain, returns a function which computes the inverse of
    the inverse function, which results are constrained by its second argument.
    """

    do_inverse = inverse(domain)

    def do(res, constr):
        ret = domain.meet(constr, do_inverse(res))
        if domain.is_empty(ret):
            return None
        return ret

    return do


def inv_eq(domain):
    """
    Given an interval domain, returns a function which computes the inverse of
    the eq function, which results are constrained by its second and third
    arguments.
    """

    def do(res, l_constr, r_constr):
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
    Given an interval domain, returns a function which computes the inverse of
    the neq function, which results are constrained by its second and third
    arguments.
    """

    do_inv_eq = inv_eq(domain)

    def do(res, l_constr, r_constr):
        return do_inv_eq(boolean_ops.not_(res), l_constr, r_constr)

    return do


def inv_lt(domain):
    """
    Given an interval domain, returns a function which computes the inverse of
    the lt function, which results are constrained by its second and third
    arguments.
    """

    def do(res, l_constr, r_constr):
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
    Given an interval domain, returns a function which computes the inverse of
    the le function, which results are constrained by its second and third
    arguments.
    """

    def do(res, l_constr, r_constr):
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
    Given an interval domain, returns a function which computes the inverse of
    the gt function, which results are constrained by its second and third
    arguments.
    """

    do_inv_le = inv_le(domain)

    def do(res, l_constr, r_constr):
        return do_inv_le(boolean_ops.not_(res), l_constr, r_constr)

    return do


def inv_ge(domain):
    """
    Given an interval domain, returns a function which computes the inverse of
    the ge function, which results are constrained by its second and third
    arguments.
    """

    do_inv_lt = inv_lt(domain)

    def do(res, l_constr, r_constr):
        return do_inv_lt(boolean_ops.not_(res), l_constr, r_constr)

    return do


def lit(domain):
    """
    Given an interval domain, returns a function which returns the smallest
    interval containing the given concrete value.
    """
    def do(val):
        return domain.build(val)

    return do


Int32 = domains.Intervals(-2147483648, 2147483647)
