from lalcheck.ai import domains
from lalcheck.ai.domain_ops import (
    boolean_ops, interval_ops, finite_lattice_ops, product_ops
)
from itertools import product


first_elem = domains.Intervals(-2, 2)
second_elem = boolean_ops.Boolean

test_dom = domains.Product(first_elem, second_elem)

elem_eqs = [
    interval_ops.eq(first_elem),
    finite_lattice_ops.eq(second_elem)
]

elem_inv_eqs = [
    interval_ops.inv_eq(first_elem),
    finite_lattice_ops.inv_eq(second_elem)
]

elem_inv_neqs = [
    interval_ops.inv_neq(first_elem),
    finite_lattice_ops.inv_neq(second_elem)
]


class InverseOperationTest(object):
    """
    Abstract test class. Can be inherited to test the inverse of a binary
    operation on a sparse array domain.
    """
    def __init__(self, doms, debug):
        self.domains = doms
        self.debug = debug

    def test_results(self):
        raise NotImplementedError

    def concrete_inverse(self, *args):
        raise NotImplementedError

    def abstract_inverse(self, *args):
        raise NotImplementedError

    def run(self):
        for expected in self.test_results():
            for x in product(*(dom.generator() for dom in self.domains)):
                xs = [dom.concretize(e) for dom, e in zip(self.domains, x)]

                abs_args = x + (expected,)
                concr_args = xs + [expected]

                concr_res = self.concrete_inverse(*concr_args)

                if len(concr_res) == 0:
                    res_exact = None
                elif len(self.domains) > 1:
                    res_exact = [
                        dom.abstract(frozenset(x[i] for x in concr_res))
                        for i, dom in enumerate(self.domains)
                    ]
                else:
                    res_exact = self.domains[0].abstract(frozenset(concr_res))

                abstr_res = self.abstract_inverse(*abs_args)

                if self.debug:
                    print(expected, x, res_exact, abstr_res)

                assert (res_exact is None) == (abstr_res is None)
                assert res_exact is None or all(
                    dom.eq(abstr_res[i], res_exact[i])
                    for i, dom in enumerate(self.domains)
                )


class EqualsToInverseTest(InverseOperationTest):
    """
    Tests the inverse of the "equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(EqualsToInverseTest, self).__init__(
            [test_dom, test_dom],
            debug
        )
        self.inv = product_ops.inv_eq(test_dom, elem_inv_eqs, elem_eqs)

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


class NotEqualsToInverseTest(InverseOperationTest):
    """
    Tests the inverse of the "not equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(NotEqualsToInverseTest, self).__init__(
            [test_dom, test_dom],
            debug
        )
        self.inv = product_ops.inv_neq(test_dom, elem_inv_eqs, elem_eqs)

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


class GetInverseTest(InverseOperationTest):
    def __init__(self, n, debug=False):
        super(GetInverseTest, self).__init__(
            [test_dom],
            debug
        )
        self.n = n
        self.inv = product_ops.inv_getter(test_dom, n)

    def test_results(self):
        return test_dom.domains[self.n].generator()

    def concrete_inverse(self, tuples, res):
        concr_res = test_dom.domains[self.n].concretize(res)
        return [
            x for x in tuples if x[self.n] in concr_res
        ]

    def abstract_inverse(self, tuples, res):
        return self.inv(res, tuples)


class UpdateInverseTest(InverseOperationTest):
    def __init__(self, n, debug=False):
        super(UpdateInverseTest, self).__init__(
            [test_dom, test_dom.domains[n]],
            debug
        )
        self.n = n
        self.inv = product_ops.inv_updater(test_dom, n)

    def test_results(self):
        # return test_dom.generator()
        # todo
        return iter(())

    def _updated(self, tple, val):
        return tuple(
            val if i == self.n else old
            for i, old in enumerate(tple)
        )

    def concrete_inverse(self, tuples, vals, res):
        concr_res = test_dom.concretize(res)
        p = product(tuples, vals)
        return [
            (tple, val)
            for (tple, val) in p
            if self._updated(tple, val) in concr_res
        ]

    def abstract_inverse(self, tuples, vals, res):
        return self.inv(res, tuples, vals)


EqualsToInverseTest().run()
NotEqualsToInverseTest().run()
GetInverseTest(0).run()
GetInverseTest(1).run()
UpdateInverseTest(0, True).run()
UpdateInverseTest(1).run()
