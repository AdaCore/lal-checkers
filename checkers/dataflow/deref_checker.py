from collections import defaultdict
from xml.sax.saxutils import escape

import lalcheck.irs.basic.tree as irt
from checker import Checker, CheckerResults
from lalcheck.constants import lits
from lalcheck.digraph import Digraph
from lalcheck.irs.basic.analyses import abstract_semantics
from lalcheck.irs.basic.purpose import DerefCheck
from lalcheck.irs.basic.tools import PrettyPrinter
from tools import dot_printer


def html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def build_resulting_graph(file_name, cfg, null_derefs):
    def trace_id(trace):
        return str(trace)

    paths = defaultdict(list)

    for trace, derefed, precise in null_derefs:
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

    def print_path_to_null_deref(value):
        derefed, precise = value
        qualifier = "" if precise else "potential "
        res_str = "path to {}null dereference of {}".format(
            qualifier, html_render_node(derefed)
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
                print_path_to_null_deref
            )
            for trace, _, _ in null_derefs
        ]))


class AnalysisResult(CheckerResults):
    """
    Contains the results of the null dereference checker.
    """
    def __init__(self, sem_analysis, null_derefs):
        super(AnalysisResult, self).__init__(sem_analysis, null_derefs)

    def save_results_to_file(self, file_name):
        """
        Prints the resulting graph as a DOT file to the given file name.
        At each program point, displays where the node is part of a path
        that leads to a (potential) null dereference.
        """
        build_resulting_graph(
            file_name,
            self.analysis_results.cfg,
            self.diagnostics
        )

    @classmethod
    def diag_message(cls, diag):
        trace, purpose, precise = diag
        if precise:
            frmt = "Null dereference of '{}'"
        else:
            frmt = "Potential null dereference of '{}'"

        return frmt.format(diag[1].data.orig_node.text)

    @classmethod
    def diag_position(cls, diag):
        return diag[1].data.orig_node.sloc_range.start


def check_derefs(prog, model, merge_pred_builder):

    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    # Retrieve nodes in the CFG that correspond to program statements.
    nodes_with_ast = (
        (node, node.data.node)
        for node in analysis.cfg.nodes
        if 'node' in node.data
    )

    # Collect those that are assume statements and that have a 'purpose' tag
    # which indicates that this assume statement was added to check
    # dereferences.
    deref_checks = (
        (node, ast_node.expr, ast_node.data.purpose.expr)
        for node, ast_node in nodes_with_ast
        if isinstance(ast_node, irt.AssumeStmt)
        if DerefCheck.is_purpose_of(ast_node)
    )

    # Use the semantic analysis to evaluate at those program points the
    # corresponding expression being dereferenced.
    derefed_values = (
        (frozenset(trace) | {node}, derefed, value)
        for node, check_expr, derefed in deref_checks
        for anc in analysis.cfg.ancestors(node)
        for trace, value in analysis.eval_at(anc, check_expr).iteritems()
    )

    # Finally, keep those that might be null.
    # Store the program trace, the dereferenced expression, and whether
    # the expression "might be null" or "is always null".
    null_derefs = [
        (trace, derefed, len(value) == 1)
        for trace, derefed, value in derefed_values
        if lits.FALSE in value
    ]

    return AnalysisResult(analysis, null_derefs)


class DerefChecker(Checker):
    def __init__(self):
        super(DerefChecker, self).__init__(
            "deref_checker",
            "Finds null dereference",
            check_derefs
        )


if __name__ == "__main__":
    DerefChecker().run()
