from collections import defaultdict
from xml.sax.saxutils import escape

import lalcheck.ai.irs.basic.tree as irt
from lalcheck.ai.constants import lits
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import ExistCheck
from lalcheck.ai.irs.basic.tools import PrettyPrinter
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import AbstractSemanticsChecker
from lalcheck.checkers.support.components import AbstractSemantics
from lalcheck.tools import dot_printer
from lalcheck.tools.digraph import Digraph

from lalcheck.tools.scheduler import Task, Requirement


def html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def build_resulting_graph(file_name, cfg, infeasibles):
    def trace_id(trace):
        return str(trace)

    paths = defaultdict(list)

    for trace, derefed, precise in infeasibles:
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

    def print_path_to_infeasible_access(value):
        purpose, precise = value
        prefix = html_render_node(purpose.accessed_expr)
        qualifier = "" if precise else "potential "
        res_str = ("path to {}infeasible access {}.{} due to invalid "
                   "condition on discriminant {}.{}").format(
            qualifier,
            prefix,
            escape(purpose.field_name),
            prefix,
            escape(purpose.discr_name)
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
                print_path_to_infeasible_access
            )
            for trace, _, _ in infeasibles
        ]))


class Results(AbstractSemanticsChecker.Results):
    """
    Contains the results of the null dereference checker.
    """
    def __init__(self, sem_analysis, infeasibles):
        super(Results, self).__init__(sem_analysis, infeasibles)

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
        if ('orig_node' in purpose.accessed_expr.data
                and purpose.accessed_expr.data.orig_node is not None):

            if precise:
                frmt = "invalid field {}"
            else:
                frmt = "(potential) invalid field {}"

            return (
                purpose.accessed_expr.data.orig_node,
                frmt.format(
                    purpose.field_name,
                    purpose.discr_name
                ),
                VariantChecker.name(),
                cls.gravity(precise)
            )


def check_variants(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_invalid_accesses(analysis)


def find_invalid_accesses(analysis):
    # Retrieve nodes in the CFG that correspond to program statements.
    nodes_with_ast = (
        (node, node.data.node)
        for node in analysis.cfg.nodes
        if 'node' in node.data
    )

    # Collect those that are assume statements and that have a 'purpose' tag
    # which indicates that this assume statement was added to check
    # existence of a field.
    exist_checks = (
        (node, ast_node.expr, ast_node.data.purpose)
        for node, ast_node in nodes_with_ast
        if isinstance(ast_node, irt.AssumeStmt)
        if ExistCheck.is_purpose_of(ast_node)
    )

    # Use the semantic analysis to evaluate at those program points the
    # corresponding expression being dereferenced.
    exist_check_values = (
        (frozenset(trace) | {node}, purpose, value)
        for node, expr, purpose in exist_checks
        for anc in analysis.cfg.ancestors(node)
        for trace, value in analysis.eval_at(anc, expr).iteritems()
    )

    # Finally, keep those that might be null.
    # Store the program trace, the dereferenced expression, and whether
    # the expression "might be null" or "is always null".
    infeasibles = [
        (trace, purpose, len(value) == 1)
        for trace, purpose, value in exist_check_values
        if lits.FALSE in value
    ]

    return Results(analysis, infeasibles)


@Requirement.as_requirement
def InvalidAccesses(provider_config, model_config, files):
    return [InvalidAccessFinder(
        provider_config, model_config, files
    )]


@dataclass
class InvalidAccessFinder(Task):
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
            'res': InvalidAccesses(
                self.provider_config,
                self.model_config,
                self.files
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_invalid_accesses(analysis)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class VariantChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "invalid discriminant"

    @classmethod
    def description(cls):
        return "Finds invalid field accesses of variant records"

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(InvalidAccesses)(*args, **kwargs)


checker = VariantChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
