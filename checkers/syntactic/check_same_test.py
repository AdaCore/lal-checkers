#! /usr/bin/env python

"""
This script will detect syntactically identical expressions which are chained
together in a chain of logical operators in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

from checkers.syntactic.checker import Checker
import libadalang as lal


def list_tests(ifnode):
    """
    List all the tests of `ifnode`.

    :type ifnode: lal.IfStmt|lal.IfExpr
    """
    return [ifnode.f_cond_expr] + [
        f.f_cond_expr for f in ifnode.f_alternatives
    ]


def tokens_text(node):
    return tuple((t.kind, t.text) for t in node.tokens)


def has_same_tests(expr):
    """
    For an if-statement or an if-expression, checks whether any combination of
    its tests are syntactically equivalent. If duplicate operands are found,
    return them.

    :rtype: lal.Expr|None
    """
    tests = {}
    all_tests = list_tests(expr)
    if len(all_tests) > 1:
        for test in all_tests:
            tokens = tokens_text(test)
            if tokens in tests:
                return (tests[tokens], test)
            tests[tokens] = test


class SameTestChecker(Checker):
    @classmethod
    def name(cls):
        return "SAME_TEST"

    def run(self, unit):
        for ifnode in unit.root.findall((lal.IfStmt, lal.IfExpr)):
            res = has_same_tests(ifnode)
            if res is not None:
                fst_test, snd_test = res
                fst_line = fst_test.sloc_range.start.line
                self.report(snd_test, 'duplicate test with line {}'.format(
                    fst_line
                ))


if __name__ == '__main__':
    SameTestChecker.build_and_run()
