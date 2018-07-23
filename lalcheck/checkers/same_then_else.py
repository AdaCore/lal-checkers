#! /usr/bin/env python

"""
This script will detect syntactically identical blocks in an if- or case-
statement or expression. It implements heuristics to avoid flagging valid uses
of code duplication.

Hence no message is issued in the following cases:
- if the duplicated block has less than 10 tokens
- if the duplicated block has fewer tokens than the test for this block
- if the duplicated block is the "else" part in an if-statement or an
  if-expression, and it duplicates a block not directly before the "else"
  part.
"""

from __future__ import (absolute_import, division, print_function)

import libadalang as lal
from lalcheck.ai.utils import dataclass, map_nonable
from lalcheck.checkers.support.checker import (
    SyntacticChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AnalysisUnit
from lalcheck.checkers.support.utils import relevant_tokens, tokens_info

from lalcheck.tools.scheduler import Task, Requirement


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        fst_line = diag[0].sloc_range.start.line
        return (
            DiagnosticPosition.from_node(diag[1]),
            'duplicate code with line {}'.format(fst_line),
            SameThenElseChecker.name(),
            cls.HIGH
        )


def find_same_then_elses(unit):
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

    def have_same_tokens(left, right):
        """
        Returns whether left and right nodes contain tokens that are
        structurally equivalent with regards to kind and contained text.

        :rtype: bool
        """
        return same_tokens(relevant_tokens(left), relevant_tokens(right))

    def list_blocks(node):
        """
        List all the sub-blocks of `node` that should be considered for
        duplicate checking. This is where we filter blocks based on heuristics.

        :type node: lal.IfStmt lal.IfExpr lal.CaseStmt lal.CaseExpr
        """

        def num_tokens(node):
            return len(relevant_tokens(node))

        def select_if_block(block, test):
            # Only report blocks of length greater than 10 tokens, and if the
            # length of the test leading to the block is no greater than the
            # length of the block. Otherwise, not sharing the blocks might be
            # better coding style.
            len_test = num_tokens(test)
            return num_tokens(block) > max(10, len_test)

        def select_case_block(block):
            # Only report blocks of length greater than 10 tokens
            return num_tokens(block) > 10

        def last_block_before_else(node):
            """
            Return the last block of code before the else part, in an
            if-statement or an if-expression.
            """
            if isinstance(node, lal.IfStmt):
                if len(node.f_alternatives) == 0:
                    return node.f_then_stmts
                else:
                    return node.f_alternatives[-1].f_stmts
            else:
                if len(node.f_alternatives) == 0:
                    return node.f_then_expr
                else:
                    return node.f_alternatives[-1].f_then_expr

        if isinstance(node, lal.IfStmt):
            blocks = []
            if select_if_block(node.f_then_stmts, node.f_cond_expr):
                blocks += [node.f_then_stmts]
            blocks += [sub.f_stmts for sub in node.f_alternatives
                       if select_if_block(sub.f_stmts, sub.f_cond_expr)]
            # Only return the else block if it is the same as the block
            # preceding it. Otherwise, there may be valid reasons for code
            # duplication, that have to do with the order of evaluation of
            # tests in an if-statement.
            if (node.f_else_stmts and
                    have_same_tokens(node.f_else_stmts,
                                     last_block_before_else(node))):
                blocks += [node.f_else_stmts]

        elif isinstance(node, lal.IfExpr):
            blocks = []
            if select_if_block(node.f_then_expr, node.f_cond_expr):
                blocks += [node.f_then_expr]
            blocks += [sub.f_then_expr for sub in node.f_alternatives
                       if select_if_block(sub.f_then_expr, sub.f_cond_expr)]
            # Only return the else block if it is the same as the block
            # preceding it. Otherwise, there may be valid reasons for code
            # duplication, that have to do with the order of evaluation of
            # tests in an if-expression.
            if (node.f_else_expr and
                    have_same_tokens(node.f_else_expr,
                                     last_block_before_else(node))):
                blocks += [node.f_else_expr]

        elif isinstance(node, lal.CaseStmt):
            blocks = [sub.f_stmts for sub in node.f_alternatives
                      if select_case_block(sub.f_stmts)]

        elif isinstance(node, lal.CaseExpr):
            blocks = [sub.f_expr for sub in node.f_cases
                      if select_case_block(sub.f_expr)]

        else:
            assert False

        return blocks

    def has_same_blocks(node):
        """
        For an if- or case- statement or expression, checks whether any
        combination of its sub-blocks are syntactically equivalent. If some
        duplicate operands are found, return them.

        :rtype: lal.Expr|None
        """
        blocks = {}
        duplicates = []
        for block in list_blocks(node):
            tokens = tokens_info(block)
            if tokens in blocks:
                duplicates.append((blocks[tokens], block))
            else:
                blocks[tokens] = block
        return duplicates

    diags = []
    for b in unit.root.findall((lal.IfStmt, lal.IfExpr, lal.CaseStmt,
                                lal.CaseExpr)):
        duplicates = has_same_blocks(b)
        for duplicate in duplicates:
            diags.append(duplicate)

    return Results(diags)


@Requirement.as_requirement
def SameThenElses(provider_config, files):
    return [SameThenElseFinder(
        provider_config, files
    )]


@dataclass
class SameThenElseFinder(Task):
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
            'res': SameThenElses(
                self.provider_config,
                self.files
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        return {
            'res': map_nonable(find_same_then_elses, units)
        }


class SameThenElseChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "same then else"

    @classmethod
    def description(cls):
        return ("Finds if statements/expressions in which multiple "
                "alternatives contain a syntactically equivalent body.")

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(SameThenElses)(*args, **kwargs)


checker = SameThenElseChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
