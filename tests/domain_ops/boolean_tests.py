from lalcheck.domain_ops import boolean_ops
from itertools import product


all_elems = [
    boolean_ops.true,
    boolean_ops.false,
    boolean_ops.both,
    boolean_ops.none
]

debug = False


def _to_concrete(bool):
    if bool is boolean_ops.true:
        return {True}
    elif bool is boolean_ops.false:
        return {False}
    elif bool is boolean_ops.both:
        return {True, False}
    else:
        return set()


def _to_abstract(bools):
    if True in bools and False in bools:
        return boolean_ops.both
    elif True in bools:
        return boolean_ops.true
    elif False in bools:
        return boolean_ops.false
    else:
        return boolean_ops.none


class BinaryInverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a binary
    operation on the boolean domain.
    """
    def test_results(self):
        return all_elems

    def concrete_inverse(self, x, y, res):
        raise NotImplementedError

    def abstract_inverse(self, x, y, res):
        raise NotImplementedError

    def run(self):
        for expected in self.test_results():
            for x in all_elems:
                xs = _to_concrete(x)
                for y in all_elems:
                    ys = _to_concrete(y)
                    res = self.concrete_inverse(xs, ys, expected)

                    if len(res) == 0:
                        res_exact = None
                    else:
                        lhs = {x[0] for x in res}
                        rhs = {x[1] for x in res}
                        lhs = _to_abstract(lhs)
                        rhs = _to_abstract(rhs)
                        res_exact = lhs, rhs

                    res_short = self.abstract_inverse(x, y, expected)

                    if debug:
                        print(
                            expected,
                            x, y,
                            res_exact, res_short
                        )

                    assert (res_exact is None) == (res_short is None)
                    assert res_exact is None or (
                        boolean_ops.Boolean.eq(res_short[0], res_exact[0]) and
                        boolean_ops.Boolean.eq(res_short[1], res_exact[1])
                    )


class UnaryInverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a unary
    operation on the boolean domain.
    """
    def test_results(self):
        return all_elems

    def concrete_inverse(self, x, res):
        raise NotImplementedError

    def abstract_inverse(self, x, res):
        raise NotImplementedError

    def run(self):
        for expected in self.test_results():
            for x in all_elems:
                xs = _to_concrete(x)
                res = self.concrete_inverse(xs, expected)

                if len(res) == 0:
                    res_exact = None
                else:
                    res_exact = _to_abstract(res)

                res_short = self.abstract_inverse(x, expected)

                if debug:
                    print(
                        expected,
                        x,
                        res_exact, res_short
                    )

                assert (res_exact is None) == (res_short is None)
                assert res_exact is None or (
                    boolean_ops.Boolean.eq(res_short, res_exact)
                )


class AndInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "and" binary operation.
    """
    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a and b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if not (a and b)]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return boolean_ops.inv_and(res, x, y)


class OrInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "or" binary operation.
    """
    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a or b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if not (a or b)]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return boolean_ops.inv_or(res, x, y)


class NotInverseTest(UnaryInverseOperationTest):
    """
    Tests the inverse of the "not" unary operation.
    """
    def concrete_inverse(self, x, res):
        if res == boolean_ops.true:
            return [a for a in x if not a]
        elif res == boolean_ops.false:
            return [a for a in x if a]
        elif res == boolean_ops.both:
            return x
        else:
            return []

    def abstract_inverse(self, x, res):
        return boolean_ops.inv_not(res, x)


AndInverseTest().run()
OrInverseTest().run()
NotInverseTest().run()
