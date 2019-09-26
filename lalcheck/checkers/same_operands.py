#! /usr/bin/env python

"""
This script will detect comparison and arithmetic operations that have operands
which are syntactically identical in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal
from lalcheck.ai.utils import dataclass, map_nonable
from lalcheck.checkers.support.checker import (
    SyntacticChecker, DiagnosticPosition, syntactic_checker_keepalive
)
from lalcheck.checkers.support.components import AnalysisUnit
from lalcheck.checkers.support.kinds import SameOperands as KindSameOperands
from lalcheck.checkers.support.utils import relevant_tokens

from lalcheck.tools.scheduler import Task, Requirement


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        return (
            DiagnosticPosition.from_node(diag),
            "operands of '{}' are identical".format(
                diag.f_op.text
            ),
            KindSameOperands,
            cls.HIGH
        )


def find_same_operands(unit):
    syntactic_checker_keepalive(unit)

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
        return same_tokens(relevant_tokens(binop.f_left),
                           relevant_tokens(binop.f_right))

    def interesting_oper(op):
        """
        Predicate that returns whether op is an operator that is interesting
        in the context of this script.

        :rtype: bool
        """
        return not op.is_a(lal.OpMult, lal.OpPlus, lal.OpDoubleDot,
                           lal.OpPow, lal.OpConcat, lal.OpAndThen,
                           lal.OpOrElse, lal.OpAnd, lal.OpOr, lal.OpXor)

    def is_float_type(decl):
        """
        Returns true if the given type declaration is the declaration of the
        float type.

        :param lal.TypeDecl decl: The type decl to check.
        :rtype: bool
        """
        return decl.p_is_float_type()

    def is_simple_nan_check(binop):
        """
        Predicate that returns whether the binary operation is thought of
        being a check that a float variable is NaN.

        :param lal.BinOp binop: The binary operation to check.
        :rtype: bool
        """
        if binop.f_op.is_a(lal.OpEq, lal.OpNeq):
            lhs = binop.f_left
            if lhs.is_a(lal.Name):  # right operand is the same.
                try:
                    lhs_type = lhs.p_expression_type
                    if lhs_type is not None:
                        return is_float_type(lhs_type)
                except lal.PropertyError:
                    pass

        return False

    diags = []
    for binop in unit.root.findall(lal.BinOp):
        if interesting_oper(binop.f_op) and has_same_operands(binop):
            if not is_simple_nan_check(binop):
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
            'res': map_nonable(find_same_operands, units)
        }


class SameOperandsChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "same_operands"

    @classmethod
    def description(cls):
        return ("Reports message of kind '{}' when an arithmetic expression "
                "has the same two operands. This checker filters out "
                "irrelevant operators like '+', '*', etc. as well as float "
                "inequality").format(KindSameOperands.name())

    @classmethod
    def kinds(cls):
        return [KindSameOperands]

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(SameOperands)(*args, **kwargs)


checker = SameOperandsChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
