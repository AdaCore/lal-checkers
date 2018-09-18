from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import PredeterminedCheck
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition, create_best_provider
)
from lalcheck.checkers.support.components import AbstractSemantics, ModelConfig
from lalcheck.checkers.support.kinds import TestAlwaysTrue, TestAlwaysFalse
from lalcheck.checkers.support.utils import (
    collect_assumes_with_purpose, orig_text_matches, eval_expr_at,
    format_text_for_output
)

from lalcheck.tools.scheduler import Task, Requirement
from collections import namedtuple


CheckerConfig = namedtuple('CheckerConfig', ('lits',))


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
            "'{}' is always {}".format(
                format_text_for_output(cond.text),
                "true" if always_true else "false"
            ),
            TestAlwaysTrue if always_true else TestAlwaysFalse,
            cls.HIGH
        )


def check_predetermined_tests(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_predetermined_tests(analysis, CheckerConfig(
        lits=(True, False)
    ))


def _contains(node, other):
    """
    Given two nodes a and b, returns True if b appears inside a, as one of its
    (possibly indirect) children. Returns False if a is b.

    :param lal.AdaNode node: The node in which to search.
    :param lal.AdaNode other: The node to find.
    :rtype: bool
    """
    return node is not other and node.find(lambda x: x == other)


def find_predetermined_tests(analysis, config):
    # Collect assume statements that have a PredeterminedCheck purpose.
    predetermined_checks = collect_assumes_with_purpose(
        analysis.cfg,
        PredeterminedCheck
    )

    # Use the semantic analysis to evaluate at those program points the
    # corresponding conditions being tested. Filter out conditions explicitly
    # set to True of False by users.
    if_conds_values = [
        (purpose.condition, eval_expr_at(analysis, node, check_expr))
        for node, check_expr, purpose in predetermined_checks
        if not orig_text_matches(check_expr, ('true', 'false'))
    ]

    # Finally, keep those that are TRUE and only TRUE or FALSE and only FALSE.
    # Store the program trace and the condition expression.
    predermined_tests = [
        (
            frozenset(trace for trace, _ in trace_values),
            condition,
            lit
        )
        for condition, trace_values in if_conds_values
        for lit in config.lits
        if len(trace_values) > 0
        if all(
            lit in value and len(value) == 1
            for _, value in trace_values
        )
    ]

    # Filter out redundant ones. For example in "if False and then B", two
    # messages are generated: one for the sub-condition "False" and one for the
    # enclosing condition "False and then B", which is trivially also false.
    # This can be worked around by removing messages of nodes on which a child
    # node already has a message.

    filtered_predetermined_tests = [
        test
        for test in predermined_tests
        if not any(
            _contains(test[1], cond)
            for _, cond, _ in predermined_tests
        )
    ]

    return Results(analysis, filtered_predetermined_tests)


@Requirement.as_requirement
def PredeterminedTests(provider_config, model_config, files, config):
    return [PredeterminedTestFinder(
        provider_config, model_config, files, config
    )]


@dataclass
class PredeterminedTestFinder(Task):
    def __init__(self, provider_config, model_config, files, config):
        self.provider_config = provider_config
        self.model_config = model_config
        self.files = files
        self.config = config

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
                self.files,
                self.config
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_predetermined_tests(analysis, self.config)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class PredeterminedTestChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "predetermined_test"

    @classmethod
    def description(cls):
        return ("Reports a message of kind '{}' (resp. '{}') when a test "
                "always evaluates to 'True' (resp. 'False')").format(
            TestAlwaysTrue.name(), TestAlwaysFalse.name()
        )

    @classmethod
    def kinds(cls):
        return [TestAlwaysTrue, TestAlwaysFalse]

    @classmethod
    def create_requirement(cls, project_file, scenario_vars, filenames, args):
        arg_values = cls.get_arg_parser().parse_args(args)

        return PredeterminedTests(
            create_best_provider(project_file, scenario_vars, filenames),
            ModelConfig(arg_values.typer,
                        arg_values.type_interpreter,
                        arg_values.call_strategy,
                        arg_values.merge_predicate),
            tuple(filenames),
            CheckerConfig(
                lits=(
                    (() if arg_values.ignore_always_true else (True,)) +
                    (() if arg_values.ignore_always_false else (False,))
                )
            )
        )

    @classmethod
    def get_arg_parser(cls):
        parser = AbstractSemanticsChecker.get_arg_parser()
        parser.add_argument('--ignore-always-true', action='store_true',
                            help="Ignore messages for tests that are always "
                                 "true.")
        parser.add_argument('--ignore-always-false', action='store_true',
                            help="Ignore messages for tests that are always "
                                 "false.")
        return parser


checker = PredeterminedTestChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
