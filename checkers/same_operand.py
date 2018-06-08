#! /usr/bin/env python

"""
This script will detect comparison and arithmetic operations that have operands
which are syntactically identical in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal
from tools.scheduler import Task, Requirement
from checkers.support.components import AnalysisUnit
from lalcheck.utils import dataclass
from checkers.support.checker import SyntacticChecker


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        return (
            diag,
            'left and right operands of "{}" are identical'.format(
                diag.f_op.text
            ),
            SameOperandsChecker.name()
        )


def find_same_operands(unit):
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
        return same_tokens(list(binop.f_left.tokens),
                           list(binop.f_right.tokens))

    def interesting_oper(op):
        """
        Predicate that returns whether op is an operator that is interesting
        in the context of this script.

        :rtype: bool
        """
        return not op.is_a(lal.OpMult, lal.OpPlus, lal.OpDoubleDot,
                           lal.OpPow, lal.OpConcat)

    diags = []
    for binop in unit.root.findall(lal.BinOp):
        if interesting_oper(binop.f_op) and has_same_operands(binop):
            diags.append(binop)

    return Results(diags)


@Requirement.as_requirement
def SameOperands(provider_config, files):
    return [SameOperandsFinder(
        provider_config, files
    )]


@dataclass
class SameOperandsFinder(Task):
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
            'res': SameOperands(
                self.provider_config,
                self.files
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        return {
            'res': [find_same_operands(unit) for unit in units]
        }


class SameOperandsChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "same_operands_checker"

    @classmethod
    def description(cls):
        return ("Finds logical or arithmetic expressions in which the two "
                "operands are syntactically identical.")

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(SameOperands)(*args, **kwargs)


checker = SameOperandsChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
