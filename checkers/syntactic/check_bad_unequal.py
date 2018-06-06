#! /usr/bin/env python

"""
This script will detect syntactically identical expressions which are chained
together in a chain of logical operators in the input Ada sources.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal

from checkers.support.checker import SyntacticChecker
from checkers.support.components import AnalysisUnit
from checkers.support.utils import same_as_parent
from lalcheck.utils import dataclass
from tools.scheduler import Task, Requirement


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_message(cls, diag):
        op, fst_val, snd_val = diag
        return (
            'expression is always true, "{}" is always different '
            'from {} or {}'.format(op.text, fst_val.text, snd_val.text)
        )

    @classmethod
    def diag_position(cls, diag):
        return diag[0]


def find_bad_unequals(unit):
    def is_literal(expr):
        if expr.is_a(lal.Identifier):
            try:
                ref = expr.p_xref
                return (ref is not None
                        and ref.p_basic_decl.is_a(lal.EnumLiteralDecl))
            except lal.PropertyError:
                pass
        return isinstance(expr, (lal.CharLiteral,
                                 lal.StringLiteral,
                                 lal.IntLiteral,
                                 lal.NullLiteral))

    def list_left_unequal_operands(binop):
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
            if isinstance(expr, lal.BinOp) and type(expr.f_op) is type(op):
                return (list_sub_operands(expr.f_left, op)
                        + list_sub_operands(expr.f_right, op))

            elif (isinstance(expr, lal.BinOp)
                  and isinstance(expr.f_op, lal.OpNeq)
                  and is_literal(expr.f_right)):
                return [(expr.f_left, expr.f_right)]

            else:
                return []

        op = binop.f_op
        return (list_sub_operands(binop.f_left, op)
                + list_sub_operands(binop.f_right, op))

    def tokens_text(node):
        return tuple((t.kind, t.text) for t in node.tokens)

    def has_same_operands(expr):
        """
        For a logic relation, checks whether any combination of its
        sub-operands are syntactically equivalent. If a duplicate operand is
        found, return it.

        :rtype: lal.Expr|None
        """
        ops = {}
        all_ops = list_left_unequal_operands(expr)
        if len(all_ops) > 1:
            for op in all_ops:
                (op_left, op_right) = op
                tokens = tokens_text(op_left)
                if tokens in ops:
                    return (op_left, ops[tokens], op_right)
                ops[tokens] = op_right

    def interesting_oper(op):
        """
        Check that op is a relational operator, which are the operators that
        interrest us in the context of this script.

        :rtype: bool
        """
        return isinstance(op, (lal.OpOr, lal.OpOrElse))

    diags = []
    for binop in unit.root.findall(lal.BinOp):
        if interesting_oper(binop.f_op) and not same_as_parent(binop):
            res = has_same_operands(binop)
            if res is not None:
                diags.append(res)

    return Results(diags)


@Requirement.as_requirement
def BadUnequals(project_config, files):
    return [BadUnequalFinder(
        project_config, files
    )]


@dataclass
class BadUnequalFinder(Task):
    def __init__(self, project_config, files):
        self.project_config = project_config
        self.files = files

    def requires(self):
        return {
            'unit_{}'.format(i): AnalysisUnit(self.project_config, f)
            for i, f in enumerate(self.files)
        }

    def provides(self):
        return {
            'res': BadUnequals(
                self.project_config,
                self.files
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        return {
            'res': [find_bad_unequals(unit) for unit in units]
        }


class BadUnequalChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "bad_unequal_checker"

    @classmethod
    def description(cls):
        return ("Finds disjunctions containing 'not equal' binary operators "
                "that have one operand in common.")

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(BadUnequals)(*args, **kwargs)


checker = BadUnequalChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
