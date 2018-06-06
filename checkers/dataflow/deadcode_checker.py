from xml.sax.saxutils import escape

from lalcheck.digraph import Digraph
from lalcheck.irs.basic.analyses import abstract_semantics
from lalcheck.irs.basic.tools import PrettyPrinter
from tools import dot_printer
from tools.scheduler import Task, Requirement
from checkers.support.components import AbstractSemantics
from checkers.support.checker import AbstractSemanticsChecker
from lalcheck.utils import dataclass


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


class Results(AbstractSemanticsChecker.Results):
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
                return diag.data.node.data.orig_node


def check_dead_code(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_dead_code(analysis)


def find_dead_code(analysis):
    dead_nodes = [
        node
        for node in analysis.cfg.nodes
        if len(analysis.semantics[node]) == 0
    ]

    return Results(analysis, dead_nodes)


@Requirement.as_requirement
def DeadCode(project_config, model_config, files):
    return [DeadCodeFinder(
        project_config, model_config, files
    )]


@dataclass
class DeadCodeFinder(Task):
    def __init__(self, project_config, model_config, files):
        self.project_config = project_config
        self.model_config = model_config
        self.files = files

    def requires(self):
        return {
            'sem': AbstractSemantics(
                self.project_config,
                self.model_config,
                self.files
            )
        }

    def provides(self):
        return {
            'res': DeadCode(
                self.project_config,
                self.model_config,
                self.files
            )
        }

    def run(self, sem):
        return {
            'res': [find_dead_code(analysis) for analysis in sem]
        }


class DeadCodeChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "deadcode_checker"

    @classmethod
    def description(cls):
        return "Finds dead code"

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(DeadCode)(*args, **kwargs)


checker = DeadCodeChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
