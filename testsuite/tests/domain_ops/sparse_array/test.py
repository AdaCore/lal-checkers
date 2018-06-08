from ai import domains
from ai.domain_ops import sparse_array_ops
from itertools import product


test_dom = domains.SparseArray(
    domains.Intervals(0, 2),
    domains.Intervals(0, 2)
)
all_elems = list(test_dom.generator())


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
                else:
                    res_exact = [
                        dom.abstract(frozenset(x[i] for x in concr_res))
                        for i, dom in enumerate(self.domains)
                    ]

                abstr_res = self.abstract_inverse(*abs_args)

                if self.debug:
                    print(expected, x, res_exact, abstr_res)

                assert (res_exact is None) == (abstr_res is None)
                assert res_exact is None or all(
                    dom.eq(abstr_res[i], res_exact[i])
                    for i, dom in enumerate(self.domains)
                )


class GetInverseTest(InverseOperationTest):
    """
    Tests the inverse of the "equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(GetInverseTest, self).__init__(
            [test_dom, test_dom.index_dom],
            debug
        )
        self.inv = sparse_array_ops.inv_get(test_dom)

    def test_results(self):
        yield (0, 0)
        yield (0, 2)

    @staticmethod
    def _get(array, index):
        try:
            return next(
                x[1]
                for x in array
                if index == x[0]
            )
        except StopIteration:
            return None

    def concrete_inverse(self, arrays, indices, res):
        p = list(product(arrays, indices))
        res_concr = test_dom.elem_dom.concretize(res)
        return [
            (a, i)
            for a, i in p
            if self._get(a, i) in res_concr
        ]

    def abstract_inverse(self, arrays, indices, res):
        return self.inv(res, arrays, indices)


class UpdatedInverseTest(InverseOperationTest):
    """
    Tests the inverse of the "equals to" binary operation.
    """
    def __init__(self, debug=False):
        super(UpdatedInverseTest, self).__init__(
            [test_dom, test_dom.elem_dom, test_dom.index_dom],
            debug
        )
        self.inv = sparse_array_ops.inv_updated(test_dom)

    def test_results(self):
        # yield [((0, 2), (0, 2))]
        # todo
        return iter(())

    @staticmethod
    def _updated(array, val, index):
        return frozenset(
            (x[0], val if x[0] == index else x[1])
            for x in array
        )

    def concrete_inverse(self, arrays, vals, indices, res):
        p = list(product(arrays, vals, indices))
        res_concr = test_dom.concretize(res)
        return [
            (a, v, i)
            for a, v, i in p
            if self._updated(a, v, i) in res_concr
        ]

    def abstract_inverse(self, arrays, vals, indices, res):
        return self.inv(res, arrays, vals, indices)


GetInverseTest().run()
UpdatedInverseTest(True).run()
