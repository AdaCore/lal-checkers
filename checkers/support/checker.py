import argparse
from components import ProjectConfig, ModelConfig


class CheckerResults(object):
    @classmethod
    def diag_report(cls, diag):
        """
        Given a diagnostic, returns the position of the error, a message and
        an error flag.

        :param object diag: The diagnostic.
        :rtype: (lal.AdaNode, str, str)
        """
        raise NotImplementedError


class Checker(object):
    @classmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    def description(cls):
        raise NotImplementedError

    @classmethod
    def create_requirement(cls, *args):
        raise NotImplementedError


class AbstractSemanticsChecker(Checker):
    parser = argparse.ArgumentParser()
    parser.add_argument('--typer', default="default_robust")
    parser.add_argument('--type-interpreter', default="default")
    parser.add_argument('--call-strategy', default="unknown")
    parser.add_argument('--merge-predicate', default='always')

    class Results(CheckerResults):
        def __init__(self, analysis_results, diagnostics):
            self.analysis_results = analysis_results
            self.diagnostics = diagnostics

        @classmethod
        def diag_report(cls, diag):
            raise NotImplementedError

    def __init__(self, args):
        pass

    @staticmethod
    def requirement_creator(requirement_class):
        parser = AbstractSemanticsChecker.parser

        def create_requirement(project_file, scenario_vars, filenames, args):
            arg_values = parser.parse_args(args)
            return requirement_class(
                ProjectConfig(project_file, tuple(scenario_vars.iteritems())),
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


class SyntacticChecker(Checker):
    parser = argparse.ArgumentParser()

    class Results(CheckerResults):
        def __init__(self, diagnostics):
            self.diagnostics = diagnostics

        @classmethod
        def diag_report(cls, diag):
            raise NotImplementedError

    def __init__(self):
        pass

    @staticmethod
    def requirement_creator(requirement_class):
        def create_requirement(project_file, scenario_vars, filenames, args):
            return requirement_class(
                ProjectConfig(project_file, tuple(scenario_vars.iteritems())),
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
