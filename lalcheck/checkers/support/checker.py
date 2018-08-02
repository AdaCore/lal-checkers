import argparse
from components import (
    ProjectProvider, AutoProvider, ModelConfig
)
from utils import closest_enclosing
import libadalang as lal
from funcy.calc import memoize


def create_best_provider(project_file, scenario_vars, filenames):
    if project_file is None:
        return AutoProvider(tuple(filenames))
    else:
        return ProjectProvider(
            project_file,
            tuple(scenario_vars.iteritems())
        )


class DiagnosticPosition(object):
    """
    Holds the position of a diagnostic.
    """
    def __init__(self, filename, start, end,
                 proc_name, proc_filename, proc_start, proc_end):
        """
        :param str filename: File in which the diagnostic was found

        :param (int, int) start: Line, column of the first character where
            the problem was found.

        :param (int, int) end: Line, column of the last character where
            the problem was found.

        :param str|None proc_name: Name of the procedure in which the
            diagnostic was found.

        :param str|None proc_filename: File in which that procedure lives.

        :param (int, int)|None proc_start: Line, column of the first character
            of that procedure.

        :param (int, int)|None proc_end: Line, column of the last character of
            that procedure.
        """
        self.filename = filename
        self.start = start
        self.end = end
        self.proc_name = proc_name
        self.proc_filename = proc_filename
        self.proc_start = proc_start
        self.proc_end = proc_end

    @staticmethod
    def from_node(ada_node):
        """
        Creates a position from a libadalang AdaNode.
        :param lal.AdaNode ada_node: The node from which to extract the
            position.
        """
        proc = closest_enclosing(ada_node, lal.SubpBody)
        if proc is None:
            proc_name = None
            proc_filename = None
            proc_start = None
            proc_end = None
        else:
            proc_name = proc.f_subp_spec.f_subp_name.text
            proc_filename = proc.unit.filename
            proc_start = (proc.sloc_range.start.line,
                          proc.sloc_range.start.column)
            proc_end = (proc.sloc_range.end.line,
                        proc.sloc_range.end.column)

        return DiagnosticPosition(
            ada_node.unit.filename,
            (ada_node.sloc_range.start.line, ada_node.sloc_range.start.column),
            (ada_node.sloc_range.end.line, ada_node.sloc_range.end.column),
            proc_name, proc_filename, proc_start, proc_end
        )


class CheckerResults(object):
    HIGH = 'high'
    LOW = 'low'

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        # Workaround PA24-023 (libadalang issue):
        # We force the reporting of all diagnostic so that memoization
        # takes place. This means that the reporting will keep working
        # after the analysis context is freed.
        # Todo: remove after PA24-023 is solved
        for diag in diagnostics:
            self.memoized_diag_report(diag)

    @classmethod
    @memoize
    def memoized_diag_report(cls, diag):
        """
        Temporary workaround PA24-023, wraps diag_report with a memoization
        mechanism.

        :param object diag: The diagnostic to report
        :rtype: (DiagnosticPosition, str, str, str)
        """
        return cls.diag_report(diag)

    @classmethod
    def diag_report(cls, diag):
        """
        Given a diagnostic, returns the position of the error, a message, an
        error flag and a gravity indication.

        :param object diag: The diagnostic.
        :rtype: (DiagnosticPosition, str, str, str)
        """
        raise NotImplementedError

    @staticmethod
    def gravity(is_precise):
        return CheckerResults.HIGH if is_precise else CheckerResults.LOW


class Checker(object):
    @classmethod
    def name(cls):
        """
        Returns the name of the checker
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def description(cls):
        """
        Returns a short description of the checker.
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        """
        Returns the requirement of this checker.
        :rtype: lalcheck.tools.scheduler.Requirement
        """
        raise NotImplementedError

    @classmethod
    def get_arg_parser(cls):
        """
        Returns the argument parser used by this checker.
        :rtype: argparse.ArgumentParser
        """
        raise NotImplementedError


class AbstractSemanticsChecker(Checker):
    class Results(CheckerResults):
        def __init__(self, analysis_results, diagnostics):
            super(AbstractSemanticsChecker.Results, self).__init__(diagnostics)
            self.analysis_results = analysis_results

        @classmethod
        def diag_report(cls, diag):
            raise NotImplementedError

    @staticmethod
    def requirement_creator(requirement_class):
        parser = AbstractSemanticsChecker.get_arg_parser()

        def create_requirement(project_file, scenario_vars, filenames, args):
            arg_values = parser.parse_args(args)

            return requirement_class(
                create_best_provider(project_file, scenario_vars, filenames),
                ModelConfig(arg_values.typer,
                            arg_values.type_interpreter,
                            arg_values.call_strategy,
                            arg_values.merge_predicate),
                tuple(filenames)
            )

        return create_requirement

    @classmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    def description(cls):
        raise NotImplementedError

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_arg_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--typer', default="default_robust")
        parser.add_argument('--type-interpreter', default="default")
        parser.add_argument('--call-strategy', default="unknown")
        parser.add_argument('--merge-predicate', default='always')
        return parser


class SyntacticChecker(Checker):
    class Results(CheckerResults):
        def __init__(self, diagnostics):
            super(SyntacticChecker.Results, self).__init__(diagnostics)

        @classmethod
        def diag_report(cls, diag):
            raise NotImplementedError

    @staticmethod
    def requirement_creator(requirement_class):
        def create_requirement(project_file, scenario_vars, filenames, args):
            return requirement_class(
                create_best_provider(project_file, scenario_vars, filenames),
                tuple(filenames)
            )

        return create_requirement

    @classmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    def description(cls):
        raise NotImplementedError

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_arg_parser(cls):
        return argparse.ArgumentParser()
