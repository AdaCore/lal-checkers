#! /usr/bin/env python

"""
This script will detect syntactically identical expressions which are chained
together in a chain of logical operators in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal
from lalcheck.ai.utils import dataclass, map_nonable
from lalcheck.checkers.support.checker import (
    SyntacticChecker, DiagnosticPosition, syntactic_checker_keepalive
)
from lalcheck.checkers.support.components import AnalysisUnit
from lalcheck.checkers.support.kinds import TestAlwaysFalse
from lalcheck.checkers.support.utils import tokens_info

from lalcheck.tools.scheduler import Task, Requirement


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        fst_line = diag[0].sloc_range.start.line
        return (
            DiagnosticPosition.from_node(diag[1]),
            "test duplicated with line {}".format(fst_line),
            TestAlwaysFalse,
            cls.HIGH
        )


def find_same_tests(unit):
    syntactic_checker_keepalive(unit)

    def list_tests(ifnode):
        """
        List all the tests of `ifnode`.

        :type ifnode: lal.IfStmt|lal.IfExpr
        """
        return [ifnode.f_cond_expr] + [
            f.f_cond_expr for f in ifnode.f_alternatives
        ]

    def has_same_tests(expr):
        """
        For an if-statement or an if-expression, checks whether any
        combination of its tests are syntactically equivalent. If duplicate
        operands are found, return them.

        :rtype: lal.Expr|None
        """
        tests = {}
        all_tests = list_tests(expr)
        if len(all_tests) > 1:
            for test in all_tests:
                tokens = tokens_info(test)
                if tokens in tests:
                    return (tests[tokens], test)
                tests[tokens] = test

    diags = []
    for ifnode in unit.root.findall((lal.IfStmt, lal.IfExpr)):
        res = has_same_tests(ifnode)
        if res is not None:
            diags.append(res)

    return Results(diags)


@Requirement.as_requirement
def SameTests(provider_config, files):
    return [SameTestFinder(
        provider_config, files
    )]


@dataclass
class SameTestFinder(Task):
    def __init__(self, provider_config, files):
        self.provider_config = provider_config
        self.files = files

    def requires(self):
        return {
            'unit_{}'.format(i): AnalysisUnit(self.provider_config, f)
            for i, f in enumerate(self.files)
        }

    def provides(self):
        return {
            'res': SameTests(
                self.provider_config,
                self.files
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        return {
            'res': map_nonable(find_same_tests, units)
        }


class SameTestChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "same_test"

    @classmethod
    def description(cls):
        return ("Reports message of kind '{}' when an if statement/"
                "expression contains several syntactically equivalent "
                "conditions.").format(TestAlwaysFalse.name())

    @classmethod
    def kinds(cls):
        return [TestAlwaysFalse]

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(SameTests)(*args, **kwargs)


checker = SameTestChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
