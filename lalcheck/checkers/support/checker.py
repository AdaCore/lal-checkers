import argparse
from components import (
    ProjectProvider, AutoProvider, ModelConfig
)


def create_best_provider(project_file, scenario_vars, filenames):
    if project_file is None:
        return AutoProvider(tuple(filenames))
    else:
        return ProjectProvider(
            project_file,
            tuple(scenario_vars.iteritems())
        )


class CheckerResults(object):
    HIGH = 'high'
    LOW = 'low'

    @classmethod
    def diag_report(cls, diag):
        """
        Given a diagnostic, returns the position of the error, a message and
        an error flag.

        :param object diag: The diagnostic.
        :rtype: (lal.AdaNode, str, str)
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
    def create_requirement(cls, *args):
        """
        Returns the requirement of this checker.
        :param *object args: The arguments received from the command line.
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
            self.analysis_results = analysis_results
            self.diagnostics = diagnostics

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
    def create_requirement(cls, *args):
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
            self.diagnostics = diagnostics

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
    def create_requirement(cls, *args):
        raise NotImplementedError

    @classmethod
    def get_arg_parser(cls):
        return argparse.ArgumentParser()
