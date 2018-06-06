from xml.sax.saxutils import escape

from checker import Checker, CheckerResults
from lalcheck.digraph import Digraph
from lalcheck.irs.basic.analyses import abstract_semantics
from lalcheck.irs.basic.tools import PrettyPrinter
from tools import dot_printer


def html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def build_resulting_graph(file_name, cfg, dead_nodes):
    def dead_label(orig):
        return {'dead': True} if orig in dead_nodes else {}

    new_node_map = {
        node: Digraph.Node(
            node.name,
            ___orig=node,
            **dead_label(node)
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

    def print_dead_code(value):
        return (
            '<font color="{}">{}</font>'.format(
                'red', 'Dead node'
            ),
        )

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(res_graph, [
            dot_printer.DataPrinter('___orig', print_orig),
            dot_printer.DataPrinter("dead", print_dead_code)
        ]))


class Results(CheckerResults):
    """
    Contains the results of the dead code checker.
    """
    def __init__(self, sem_analysis, dead_nodes):
        super(Results, self).__init__(sem_analysis, dead_nodes)

    def save_results_to_file(self, file_name):
        """
        Prints the resulting graph as a DOT file to the given file name.
        At each program point, displays whether the node is dead.
        """
        build_resulting_graph(
            file_name,
            self.analysis_results.cfg,
            self.diagnostics
        )

    @classmethod
    def diag_message(cls, diag):
        if diag.data.node is not None:
            if ('orig_node' in diag.data.node.data
                    and diag.data.node.data.orig_node is not None):
                return "Unreachable code '{}'".format(
                    diag.data.node.data.orig_node.text
                )

    @classmethod
    def diag_position(cls, diag):
        if diag.data.node is not None:
            if ('orig_node' in diag.data.node.data
                    and diag.data.node.data.orig_node is not None):
                return diag.data.node.data.orig_node.sloc_range.start


def check_dead_code(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    dead_nodes = [
        node
        for node in analysis.cfg.nodes
        if len(analysis.semantics[node]) == 0
    ]

    return Results(analysis, dead_nodes)


class DeadCodeChecker(Checker):
    def __init__(self):
        super(DeadCodeChecker, self).__init__(
            "deadcode_checker",
            "Finds dead code",
            check_dead_code
        )


if __name__ == "__main__":
    DeadCodeChecker().run()
