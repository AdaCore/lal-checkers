#! /usr/bin/env python

"""
This script will detect comparison and arithmetic operations that have operands
which are syntactically identical in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)
from checkers.syntactic.checker import Checker

import libadalang as lal


def same_tokens(left, right):
    """
    Returns whether left and right contain tokens that are structurally
    equivalent with regards to kind and contained text.

    :rtype: bool
    """
    return len(left) == len(right) and all(
        le.kind == ri.kind and le.text == ri.text
        for le, ri in zip(left, right)
    )


def has_same_operands(binop):
    """
    Checks whether binop has the same operands syntactically.

    :type binop: lal.BinOp
    :rtype: bool
    """
    return same_tokens(list(binop.f_left.tokens), list(binop.f_right.tokens))


def interesting_oper(op):
    """
    Predicate that returns whether op is an operator that is interesting in the
    context of this script.

    :rtype: bool
    """
    return not op.is_a(lal.OpMult, lal.OpPlus, lal.OpDoubleDot,
                       lal.OpPow, lal.OpConcat)


class SameOperandsChecker(Checker):
    @classmethod
    def name(cls):
        return "SAME_OPERANDS"

    def run(self, unit):
        for binop in unit.root.findall(lal.BinOp):
            if interesting_oper(binop.f_op) and has_same_operands(binop):
                self.report(binop, 'left and right operands of "{}" are'
                            ' identical'.format(binop.f_op.text))


if __name__ == '__main__':
    SameOperandsChecker.build_and_run()
