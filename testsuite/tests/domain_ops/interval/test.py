from lalcheck.ai import domains
from lalcheck.ai.domain_ops import boolean_ops, interval_ops
from itertools import product

test_dom = domains.Intervals(-4, 4)


class BinaryInverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a binary
    operation on an interval domain.
    """
    def __init__(self, domain, debug):
        self.domain = domain
        self.debug = debug

    def test_results(self):
        raise NotImplementedError

    def concrete_inverse(self, x, y, res):
        raise NotImplementedError

    def abstract_inverse(self, x, y, res):
        raise NotImplementedError

    def run(self):
        for expected in self.test_results():
            for x in self.domain.generator():
                xs = self.domain.concretize(x)
                for y in self.domain.generator():
                    ys = self.domain.concretize(y)
                    res = self.concrete_inverse(xs, ys, expected)

                    if len(res) == 0:
                        res_exact = None
                    else:
                        res_exact = tuple(
                            self.domain.abstract(frozenset(x[i] for x in res))
                            for i in range(2)
                        )

                    res_short = self.abstract_inverse(x, y, expected)

                    if self.debug:
                        print(expected, x, y, res_exact, res_short)

                    assert (res_exact is None) == (res_short is None)
                    assert res_exact is None or (
                        self.domain.eq(res_short[0], res_exact[0]) and
                        self.domain.eq(res_short[1], res_exact[1])
                    )


class UnaryInverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a unary
    operation on an interval domain.
    """
    def __init__(self, domain, debug):
        self.domain = domain
        self.debug = debug

    def test_results(self):
        raise NotImplementedError

    def concrete_inverse(self, x, res):
        raise NotImplementedError

    def abstract_inverse(self, x, res):
        raise NotImplementedError

    def run(self):
        for expected in self.test_results():
            for x in self.domain.generator():
                xs = self.domain.concretize(x)
                res = self.concrete_inverse(xs, expected)

                if len(res) == 0:
                    res_exact = None
                else:
                    res_exact = self.domain.abstract(frozenset(res))

                res_short = self.abstract_inverse(x, expected)

                if self.debug:
                    print(expected, x, res_exact, res_short)

                assert (res_exact is None) == (res_short is None)
                assert res_exact is None or (
                    self.domain.eq(res_short, res_exact)
                )


class AddInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "addition" binary operation.
    """
    def __init__(self, debug=False):
        super(AddInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_add_no_wraparound(self.domain)

    def test_results(self):
        yield (-3, 3)
        yield (3, 3)
        yield (2, 4)
        yield (-4, -4)
        yield self.domain.bottom

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if self.domain.eq(res, self.domain.bottom):
            return []
        return [(a, b) for a, b in p if res[0] <= a + b <= res[1]]

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class SubInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "subtraction" binary operation.
    """
    def __init__(self, debug=False):
        super(SubInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_sub_no_wraparound(self.domain)

    def test_results(self):
        yield (-3, 3)
        yield (3, 3)
        yield (2, 4)
        yield (-4, -4)
        yield self.domain.bottom

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if self.domain.eq(res, self.domain.bottom):
            return []
        return [(a, b) for a, b in p if res[0] <= a - b <= res[1]]

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class InverseInverseOperation(UnaryInverseOperationTest):
    """
    Tests the inverse of the "inverse" unary operation.
    """
    def __init__(self, debug=False):
        super(InverseInverseOperation, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_inverse(self.domain)

    def test_results(self):
        yield (-2, 2)
        yield (0, 3)
        yield self.domain.bottom

    def concrete_inverse(self, x, res):
        if self.domain.eq(res, self.domain.bottom):
            return []
        return [e for e in x if res[0] <= -e <= res[1]]

    def abstract_inverse(self, x, res):
        return self.inv(res, x)


class EqualsToInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(EqualsToInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_eq(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a == b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a != b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class NotEqualsToInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "not equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(NotEqualsToInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_neq(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a != b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a == b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class LessThanInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "less than" binary operation.
    """
    def __init__(self, debug=False):
        super(LessThanInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_lt(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a < b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a >= b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class LessThanOrEqualsToInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "less than or equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(LessThanOrEqualsToInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_le(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a <= b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a > b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class GreaterThanInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "greater than" binary operation.
    """
    def __init__(self, debug=False):
        super(GreaterThanInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_gt(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a > b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a <= b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


class GreaterThanOrEqualsToInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "greater than or equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(GreaterThanOrEqualsToInverseTest, self).__init__(test_dom, debug)
        self.inv = interval_ops.inv_ge(self.domain)

    def test_results(self):
        yield boolean_ops.true
        yield boolean_ops.false
        yield boolean_ops.both
        yield boolean_ops.none

    def concrete_inverse(self, x, y, res):
        p = list(product(x, y))
        if res == boolean_ops.true:
            return [(a, b) for a, b in p if a >= b]
        elif res == boolean_ops.false:
            return [(a, b) for a, b in p if a < b]
        elif res == boolean_ops.both:
            return p
        else:
            return []

    def abstract_inverse(self, x, y, res):
        return self.inv(res, x, y)


AddInverseTest().run()
SubInverseTest().run()
InverseInverseOperation().run()
EqualsToInverseTest().run()
NotEqualsToInverseTest().run()
LessThanInverseTest().run()
LessThanOrEqualsToInverseTest().run()
GreaterThanInverseTest().run()
GreaterThanOrEqualsToInverseTest().run()
