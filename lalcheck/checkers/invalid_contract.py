from collections import defaultdict
from xml.sax.saxutils import escape

from lalcheck.ai.constants import lits
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import ContractCheck
from lalcheck.ai.irs.basic.tools import PrettyPrinter
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AbstractSemantics
from lalcheck.checkers.support.kinds import InvalidContract
from lalcheck.checkers.support.utils import (
    collect_assumes_with_purpose, eval_expr_at
)
from lalcheck.tools import dot_printer
from lalcheck.tools.digraph import Digraph

from lalcheck.tools.scheduler import Task, Requirement


def html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def build_resulting_graph(file_name, cfg, invalids):
    def trace_id(trace):
        return str(trace)

    paths = defaultdict(list)

    for trace, derefed, precise in invalids:
        for node in trace:
            paths[node].append((trace, derefed, precise))

    new_node_map = {
        node: Digraph.Node(
            node.name,
            ___orig=node,
            **{
                trace_id(trace): (derefed, precise)
                for trace, derefed, precise in paths[node]
            }
        )
        for node in cfg.nodes
    }

    res_graph = Digraph(
        [new_node_map[n]
         for n in cfg.nodes],

        [Digraph.Edge(new_node_map[e.frm], new_node_map[e.to])
         for e in cfg.edges]
    )

    def print_orig(orig):
        if orig.data.node is not None:
            return (
                '<i>{}</i>'.format(html_render_node(orig.data.node)),
            )
        return ()

    def print_path_to_violated_contract(value):
        purpose, precise = value
        qualifier = "" if precise else "potential "
        res_str = "path to {}violated {}".format(
            qualifier,
            purpose.contract_name
        )
        return (
            '<font color="{}">{}</font>'.format('red', res_str),
        )

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(res_graph, [
            dot_printer.DataPrinter('___orig', print_orig)
        ] + [
            dot_printer.DataPrinter(
                trace_id(trace),
                print_path_to_violated_contract
            )
            for trace, _, _ in invalids
        ]))


class Results(AbstractSemanticsChecker.Results):
    """
    Contains the results of the null dereference checker.
    """
    def __init__(self, sem_analysis, invalids):
        super(Results, self).__init__(sem_analysis, invalids)

    def save_results_to_file(self, file_name):
        """
        Prints the resulting graph as a DOT file to the given file name.
        At each program point, displays where the node is part of a path
        that leads to a (potential) infeasible field access.
        """
        build_resulting_graph(
            file_name,
            self.analysis_results.cfg,
            self.diagnostics
        )

    @classmethod
    def diag_report(cls, diag):
        trace, purpose, precise = diag
        if precise:
            frmt = "{} failure"
        else:
            frmt = "(potential) {} failure"

        return (
            DiagnosticPosition.from_node(purpose.orig_call),
            frmt.format(purpose.contract_name),
            InvalidContract,
            cls.gravity(precise)
        )


def check_contracts(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_violated_contracts(analysis)


def find_violated_contracts(analysis):
    # Collect assume statements that have a ContractCheck purpose.
    contract_checks = collect_assumes_with_purpose(analysis.cfg, ContractCheck)

    # Use the semantic analysis to evaluate at those program points the
    # contract expressions.
    contract_check_values = [
        (trace, purpose, value)
        for node, check_expr, purpose in contract_checks
        for trace, value in eval_expr_at(analysis, node, check_expr)
    ]

    # Finally, keep those that might fail.
    # Store the program trace, the contract, and whether
    # the contract "might be violated" or "is always violated".
    invalids = [
        (trace, purpose, len(value) == 1)
        for trace, purpose, value in contract_check_values
        if lits.FALSE in value
    ]

    return Results(analysis, invalids)


@Requirement.as_requirement
def ViolatedContracts(provider_config, model_config, files):
    return [ContractViolationFinder(
        provider_config, model_config, files
    )]


@dataclass
class ContractViolationFinder(Task):
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
            'res': ViolatedContracts(
                self.provider_config,
                self.model_config,
                self.files
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_violated_contracts(analysis)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class ContractChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "invalid contract"

    @classmethod
    def description(cls):
        return "Finds violated pre/post-conditions and assertions"

    @classmethod
    def kinds(cls):
        return [InvalidContract]

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(ViolatedContracts)(*args, **kwargs)


checker = ContractChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
