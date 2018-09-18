from predetermined_test import PredeterminedTestChecker
from support.checker import AbstractSemanticsChecker
from support.kinds import TestAlwaysTrue


class TestAlwaysTrueChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "test_always_true"

    @classmethod
    def description(cls):
        return ("Reports a message of kind '{}' when a test always evaluates "
                "to 'True'").format(TestAlwaysTrue.name())

    @classmethod
    def kinds(cls):
        return [TestAlwaysTrue]

    @classmethod
    def create_requirement(cls, project_file, scenario_vars, filenames, args):
        return PredeterminedTestChecker.create_requirement(
            project_file, scenario_vars, filenames, ["--ignore-always-false"]
        )


checker = TestAlwaysTrueChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
