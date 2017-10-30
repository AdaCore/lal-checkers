"""
Provides the Definitions class, which instances can be used to hold the set of
operations that are well defined in the context of a program or a set of
programs.
"""

from collections import defaultdict
from domain_ops.boolean_ops import (
    Boolean, boolean_not, boolean_or, boolean_and
)
from domain_ops.interval_ops import (
    interval_add_no_wraparound, interval_sub_no_wraparound,
    interval_lt, interval_le, interval_eq, interval_neq,
    interval_ge, interval_gt, interval_inverse
)


class Definitions(object):
    """
    Provides facilities for storing operations between domain elements.
    """
    def __init__(self):
        """
        Creates a Definitions object containing no operation.
        """
        self.ops = defaultdict(dict)

    def register(self, name, doms, fun):
        """
        Registers a new operation of the given name, acting on the given
        domains doms, and which semantics are given by fun.
        """
        self.ops[name][doms] = fun

    def lookup(self, name, doms):
        """
        Finds the semantics of the operation of the given name, which acts
        on the given domains.
        """
        return self.ops[name][doms]

    def register_new_interval_int(self, dom):
        """
        Registers a new set of operations acting on the given interval domain.
        Defines the basic comparison and arithmetic operations.
        """
        self.register('<', (dom, dom, Boolean), interval_lt(dom))
        self.register('<=', (dom, dom, Boolean), interval_le(dom))
        self.register('==', (dom, dom, Boolean), interval_eq(dom))
        self.register('!=', (dom, dom, Boolean), interval_neq(dom))
        self.register('>=', (dom, dom, Boolean), interval_ge(dom))
        self.register('>', (dom, dom, Boolean), interval_gt(dom))

        self.register('+', (dom, dom, dom), interval_add_no_wraparound(dom))
        self.register('-', (dom, dom, dom), interval_sub_no_wraparound(dom))

        self.register('-', (dom, dom), interval_inverse(dom))
        return self

    @staticmethod
    def default():
        """
        Creates a Definitions object containing operations that act on the
        universal Boolean domain.
        """
        defs = Definitions()
        defs.register('!', (Boolean, Boolean), boolean_not)
        defs.register('&&', (Boolean, Boolean, Boolean), boolean_and)
        defs.register('||', (Boolean, Boolean, Boolean), boolean_or)
        return defs
