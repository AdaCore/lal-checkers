from lalcheck.ai.constants import lits
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import PredeterminedCheck
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AbstractSemantics
from lalcheck.checkers.support.kinds import PredeterminedExpression
from lalcheck.checkers.support.utils import (
    collect_assumes_with_purpose, eval_expr_at
)

from lalcheck.tools.scheduler import Task, Requirement


class Results(AbstractSemanticsChecker.Results):
    """
    Contains the results of the predetermined tests checker.
    """
    def __init__(self, sem_analysis, predetermined_tests):
        super(Results, self).__init__(sem_analysis, predetermined_tests)

    @classmethod
    def diag_report(cls, diag):
        trace, cond, always_true = diag

        return (
            DiagnosticPosition.from_node(cond),
            "test always {}".format("true" if always_true else "false"),
            PredeterminedExpression,
            cls.HIGH
        )


def check_predetermined_tests(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_predetermined_tests(analysis)


def find_predetermined_tests(analysis):
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

    # Finally, keep those that are TRUE and only TRUE or FALSE and only FALSE.
    # Store the program trace and the condition expression.
    predermined_tests = [
        (
            frozenset(trace for trace, _ in trace_values),
            condition,
            lit == lits.TRUE
        )
        for condition, trace_values in if_conds_values
        for lit in (lits.TRUE, lits.FALSE)
        if len(trace_values) > 0
        if all(
            lit in value and len(value) == 1
            for _, value in trace_values
        )
    ]

    return Results(analysis, predermined_tests)


@Requirement.as_requirement
def PredeterminedTests(provider_config, model_config, files):
    return [PredeterminedTestFinder(
        provider_config, model_config, files
    )]


@dataclass
class PredeterminedTestFinder(Task):
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
            'res': PredeterminedTests(
                self.provider_config,
                self.model_config,
                self.files
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_predetermined_tests(analysis)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class PredeterminedTestChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "predetermined tests"

    @classmethod
    def description(cls):
        return "Finds conditions that are always or never satisfied"

    @classmethod
    def kinds(cls):
        return [PredeterminedExpression]

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(PredeterminedTests)(*args, **kwargs)


checker = PredeterminedTestChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
