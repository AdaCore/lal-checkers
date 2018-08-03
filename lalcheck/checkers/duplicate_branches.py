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
    SyntacticChecker, DiagnosticPosition, create_best_provider
)
from lalcheck.checkers.support.components import AnalysisUnit
from lalcheck.checkers.support.kinds import DuplicateCode
from lalcheck.checkers.support.utils import relevant_tokens, tokens_info

from lalcheck.tools.scheduler import Task, Requirement

import argparse
from functools import partial
from collections import namedtuple


class Results(SyntacticChecker.Results):
    def __init__(self, diags):
        super(Results, self).__init__(diags)

    @classmethod
    def diag_report(cls, diag):
        fst_line = diag[0].sloc_range.start.line
        return (
            DiagnosticPosition.from_node(diag[1]),
            'duplicate code with line {}'.format(fst_line),
            DuplicateCode,
            cls.HIGH
        )


CheckerConfig = namedtuple('CheckerConfig', (
    'size_threshold',
    'newline_factor',
    'min_duplicates',
    'do_ifs',
    'do_cases'
))


def find_duplicate_branches(config, unit):
    """
    Find duplicate branches in if/case statements/expressions.

    :param CheckerConfig config: The configuration of the checker.
    :param lal.AnalysisUnit unit: The analysis unit on which to perform the
        check.
    :rtype: Results
    """
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

        def block_size(node):
            newlines = node.text.count('\n')
            return num_tokens(node) * (1 + config.newline_factor * newlines)

        def select_if_block(block, test):
            # Only report blocks of length greater than 10 tokens, and if the
            # length of the test leading to the block is no greater than the
            # length of the block. Otherwise, not sharing the blocks might be
            # better coding style.
            len_test = num_tokens(test)
            return block_size(block) > max(config.size_threshold, len_test)

        def select_case_block(block):
            # Only report blocks of length greater than 10 tokens
            return block_size(block) > config.size_threshold

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

        if isinstance(node, lal.IfStmt) and config.do_ifs:
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

        elif isinstance(node, lal.IfExpr) and config.do_ifs:
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

        elif isinstance(node, lal.CaseStmt) and config.do_cases:
            blocks = [sub.f_stmts for sub in node.f_alternatives
                      if select_case_block(sub.f_stmts)]

        elif isinstance(node, lal.CaseExpr) and config.do_cases:
            blocks = [sub.f_expr for sub in node.f_cases
                      if select_case_block(sub.f_expr)]

        else:
            blocks = []

        return blocks

    def has_same_blocks(node):
        """
        For an if- or case- statement or expression, checks whether any
        combination of its sub-blocks are syntactically equivalent. If some
        duplicate operands are found, return them.

        :rtype: lal.Expr|None
        """
        all_blocks = list_blocks(node)
        tokens_to_block = {}
        duplicates = []
        for block in all_blocks:
            tokens = tokens_info(block)
            if tokens in tokens_to_block:
                duplicates.append((tokens_to_block[tokens], block))
            else:
                tokens_to_block[tokens] = block

        min_duplicates = (len(all_blocks) if config.min_duplicates == -1
                          else config.min_duplicates)
        return duplicates if len(duplicates) >= min_duplicates else []

    diags = []
    for b in unit.root.findall((lal.IfStmt, lal.IfExpr, lal.CaseStmt,
                                lal.CaseExpr)):
        duplicates = has_same_blocks(b)
        for duplicate in duplicates:
            diags.append(duplicate)

    return Results(diags)


@Requirement.as_requirement
def DuplicateBranches(provider_config, files, checker_config):
    return [DuplicateBranchesFinder(
        provider_config, files, checker_config
    )]


@dataclass
class DuplicateBranchesFinder(Task):
    def __init__(self, provider_config, files, checker_config):
        self.provider_config = provider_config
        self.files = files
        self.checker_config = checker_config

    def requires(self):
        return {
            'unit_{}'.format(i): AnalysisUnit(self.provider_config, f)
            for i, f in enumerate(self.files)
        }

    def provides(self):
        return {
            'res': DuplicateBranches(
                self.provider_config,
                self.files,
                self.checker_config
            )
        }

    def run(self, **kwargs):
        units = kwargs.values()
        checker_func = partial(find_duplicate_branches, self.checker_config)
        return {
            'res': map_nonable(checker_func, units)
        }


class DuplicateBranchesChecker(SyntacticChecker):
    @classmethod
    def name(cls):
        return "duplicate branches"

    @classmethod
    def description(cls):
        return ("Finds if/case statements/expressions in which multiple "
                "alternatives contain a syntactically equivalent body.")

    @classmethod
    def kinds(cls):
        return [DuplicateCode]

    @classmethod
    def create_requirement(cls, project_file, scenario_vars, filenames, args):
        arg_values = cls.get_arg_parser().parse_args(args)

        return DuplicateBranches(
            create_best_provider(project_file, scenario_vars, filenames),
            tuple(filenames),
            CheckerConfig(
                size_threshold=arg_values.size_threshold,
                newline_factor=arg_values.newline_factor,
                min_duplicates=arg_values.min_duplicates,
                do_ifs=not arg_values.only_cases,
                do_cases=not arg_values.only_ifs
            )
        )

    @classmethod
    def get_arg_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--size-threshold', type=int, default=0,
                            help="The minimal amount of code that must be in "
                                 "common between two branches. By default, "
                                 "it is only based on the number of tokens, "
                                 "but newlines can be taken into account "
                                 "using --newline-factor.")
        parser.add_argument('--newline-factor', type=float, default=0,
                            help="To be used with --tokens-threshold. Adjusts "
                                 "the weight of duplicate code that spans "
                                 "multiple lines. The token threshold is "
                                 "compared against \"Tokens * "
                                 "(1 + Newline-Factor * Newlines)\"")

        parser.add_argument('--min-duplicates', type=int, default=-1,
                            help="Only report when at least 'min-duplicates' "
                                 "branches in the if/case statement are "
                                 "considered duplicates. Use -1 to report "
                                 "only when ALL branches are duplicates.")

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--only-ifs', action='store_true',
                           help="Only consider if statements/expressions.")
        group.add_argument('--only-cases', action='store_true',
                           help="Only consider case statements/expressions.")
        return parser


checker = DuplicateBranchesChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
