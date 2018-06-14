#! /usr/bin/env python

"""
This script will detect syntactically identical expressions which are chained
together in a chain of logical operators in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import SyntacticChecker
from lalcheck.checkers.support.components import AnalysisUnit
from lalcheck.checkers.support.utils import same_as_parent

from lalcheck.tools.scheduler import Task, Requirement


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        fst_line = diag[0].sloc_range.start.line
        return (
            diag[1],
            'duplicate operand with line {}'.format(fst_line),
            SameLogicChecker.name()
        )


def find_same_logic(unit):
    def list_operands(binop):
        """
        List all the sub-operands of `binop`, as long as they have the same
        operator as `binop`.

        :type binop: lal.BinOp
        """

        def list_sub_operands(expr, op):
            """
            Accumulate sub-operands of `expr`, provided `expr` is a binary
            operator that has `op` as an operator.

            :type expr: lal.Expr
            :type op: lal.Op
            """
            if expr.is_a(lal.BinOp) and type(expr.f_op) is type(op):
                return (list_sub_operands(expr.f_left, op)
                        + list_sub_operands(expr.f_right, op))
            else:
                return [expr]

        op = binop.f_op
        return (list_sub_operands(binop.f_left, op)
                + list_sub_operands(binop.f_right, op))

    def is_bool_literal(expr):
        """
        Predicate to check whether `expr` is a boolean literal.
        """
        return (expr.is_a(lal.Identifier)
                and expr.text.lower() in ['true', 'false'])

    def tokens_text(node):
        return tuple((t.kind, t.text) for t in node.tokens)

    def has_same_operands(expr):
        """
        For a logic relation, checks whether any combination of its
        sub-operands are syntactically equivalent. If duplicate operands are
        found, return them.

        :rtype: lal.Expr|None
        """
        ops = {}
        all_ops = list_operands(expr)
        if len(all_ops) > 1:
            for op in all_ops:
                tokens = tokens_text(op)
                if tokens in ops:
                    return (ops[tokens], op)
                ops[tokens] = op

    def interesting_oper(op):
        """
        Check that op is a relational operator, which are the operators that
        interrest us in the context of this script.

        :rtype: bool
        """
        return op.is_a(lal.OpAnd, lal.OpOr, lal.OpAndThen,
                       lal.OpOrElse, lal.OpXor)

    diags = []
    for binop in unit.root.findall(lal.BinOp):
        if interesting_oper(binop.f_op) and not same_as_parent(binop):
            res = has_same_operands(binop)
            if res is not None:
                diags.append(res)

    return Results(diags)


@Requirement.as_requirement
def SameLogics(provider_config, files):
    return [SameLogicFinder(
        provider_config, files
    )]


@dataclass
class SameLogicFinder(Task):
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
            'res': SameLogics(
                self.provider_config,
                self.files
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        return {
            'res': [find_same_logic(unit) for unit in units]
        }


class SameLogicChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "same_logic_checker"

    @classmethod
    def description(cls):
        return ("Finds chains of boolean operators which contain syntactically"
                " identical expressions.")

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(SameLogics)(*args, **kwargs)


checker = SameLogicChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
