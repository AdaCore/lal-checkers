from lalcheck.ai.constants import lits
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import PredeterminedCheck
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AbstractSemantics
from lalcheck.checkers.support.kinds import AlwaysTrue as KindAlwaysTrue
from lalcheck.checkers.support.utils import (
    collect_assumes_with_purpose, eval_expr_at
)

from lalcheck.tools.scheduler import Task, Requirement


class Results(AbstractSemanticsChecker.Results):
    """
    Contains the results of the tests always true checker.
    """
    def __init__(self, sem_analysis, tests_alway_true):
        super(Results, self).__init__(sem_analysis, tests_alway_true)

    @classmethod
    def diag_report(cls, diag):
        trace, cond = diag

        return (
            DiagnosticPosition.from_node(cond),
            "test always true",
            KindAlwaysTrue,
            cls.HIGH
        )


def check_tests_always_true(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_tests_always_true(analysis)


def find_tests_always_true(analysis):
    # Collect assume statements that have a PredeterminedCheck purpose.
    predetermined_checks = collect_assumes_with_purpose(
        analysis.cfg,
        PredeterminedCheck
    )

    # Use the semantic analysis to evaluate at those program points the
    # corresponding conditions being tested.
    if_conds_values = [
        (purpose.condition, eval_expr_at(analysis, node, check_expr))
        for node, check_expr, purpose in predetermined_checks
    ]

    # Finally, keep those that are TRUE and only TRUE.
    # Store the program trace and the condition expression.
    always_trues = [
        (
            frozenset(trace for trace, _ in trace_values),
            condition
        )
        for condition, trace_values in if_conds_values
        if len(trace_values) > 0
        if all(
            lits.TRUE in value and len(value) == 1
            for _, value in trace_values
        )
    ]

    return Results(analysis, always_trues)


@Requirement.as_requirement
def TestsAlwaysTrue(provider_config, model_config, files):
    return [TestsAlwaysTrueFinder(
        provider_config, model_config, files
    )]


@dataclass
class TestsAlwaysTrueFinder(Task):
    def __init__(self, provider_config, model_config, files):
        self.provider_config = provider_config
        self.model_config = model_config
        self.files = files

    def requires(self):
        return {
            'sem_{}'.format(i): AbstractSemantics(
                self.provider_config,
                self.model_config,
                self.files,
                f
            )
            for i, f in enumerate(self.files)
        }

    def provides(self):
        return {
            'res': TestsAlwaysTrue(
                self.provider_config,
                self.model_config,
                self.files
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_tests_always_true(analysis)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class TestsAlwaysTrueChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "always true"

    @classmethod
    def description(cls):
        return "Finds conditions that are always satisfied"

    @classmethod
    def kinds(cls):
        return [KindAlwaysTrue]

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(TestsAlwaysTrue)(*args, **kwargs)


checker = TestsAlwaysTrueChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
