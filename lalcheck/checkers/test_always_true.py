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
    def create_requirement(cls, provider_config, analysis_files, args):
        return PredeterminedTestChecker.create_requirement(
            provider_config, analysis_files, ["--ignore-always-false"]
        )


checker = TestAlwaysTrueChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
