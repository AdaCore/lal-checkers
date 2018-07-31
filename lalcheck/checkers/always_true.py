import lalcheck.ai.irs.basic.tree as irt
from lalcheck.ai.constants import lits
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import PredeterminedCheck
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AbstractSemantics

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
            TestsAlwaysTrueChecker.name(),
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
    # Retrieve nodes in the CFG that correspond to program statements.
    nodes_with_ast = (
        (node, node.data.node)
        for node in analysis.cfg.nodes
        if 'node' in node.data
    )

    # Collect those that are assume statements and that have a 'purpose' tag
    # which indicates that this assume statement was added to check
    # predetermined conditions.
    predetermined_checks = (
        (node, ast_node.expr, ast_node.data.purpose.condition)
        for node, ast_node in nodes_with_ast
        if isinstance(ast_node, irt.AssumeStmt)
        if PredeterminedCheck.is_purpose_of(ast_node)
    )

    # Use the semantic analysis to evaluate at those program points the
    # corresponding conditions being tested.
    if_conds_values = [
        (condition, [
            (frozenset(trace) | {node}, value)
            for anc in analysis.cfg.ancestors(node)
            for trace, value in analysis.eval_at(anc, check_expr).iteritems()
            if value is not None
        ])
        for node, check_expr, condition in predetermined_checks
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
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(TestsAlwaysTrue)(*args, **kwargs)


checker = TestsAlwaysTrueChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
