from lalcheck import domains
from lalcheck.domain_ops import boolean_ops, finite_lattice_ops
from itertools import product


test_dom = domains.FiniteLattice.of_subsets({1, 2, 3, 4})
debug = False


class BinaryInverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a binary
    operation on a finite lattice domain.
    """
    def __init__(self, domain):
        self.domain = domain

    def test_results(self):
        raise NotImplementedError

    def concrete_inverse(self, x, y, res):
        raise NotImplementedError

    def abstract_inverse(self, x, y, res):
        raise NotImplementedError

    def run(self):
        all_elems = self.domain.lts[self.domain.bottom]
        for expected in self.test_results():
            for x in all_elems:
                for y in all_elems:
                    res = self.concrete_inverse(x, y, expected)

                    if len(res) == 0:
                        res_exact = None
                    else:
                        lhs = frozenset({x[0] for x in res})
                        rhs = frozenset({x[1] for x in res})
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
                        self.domain.eq(res_short[0], res_exact[0]) and
                        self.domain.eq(res_short[1], res_exact[1])
                    )


class EqualsToInverseTest(BinaryInverseOperationTest):
    """
    Tests the inverse of the "equals to" binary operation.
    """
    def __init__(self):
        super(EqualsToInverseTest, self).__init__(test_dom)
        self.inv = finite_lattice_ops.inv_eq(self.domain)

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
    def __init__(self):
        super(NotEqualsToInverseTest, self).__init__(test_dom)
        self.inv = finite_lattice_ops.inv_neq(self.domain)

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


EqualsToInverseTest().run()
NotEqualsToInverseTest().run()
