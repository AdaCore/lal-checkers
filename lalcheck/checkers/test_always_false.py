from predetermined_test import PredeterminedTestChecker
from support.checker import AbstractSemanticsChecker
from support.kinds import TestAlwaysFalse


class TestAlwaysFalseChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "test_always_false"

    @classmethod
    def description(cls):
        return ("Reports a message of kind '{}' when a test always evaluates "
                "to 'False'").format(TestAlwaysFalse.name())

    @classmethod
    def kinds(cls):
        return [TestAlwaysFalse]

    @classmethod
    def create_requirement(cls, provider_config, analysis_files, args):
        return PredeterminedTestChecker.create_requirement(
            provider_config, analysis_files, ["--ignore-always-true"]
        )


checker = TestAlwaysFalseChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
