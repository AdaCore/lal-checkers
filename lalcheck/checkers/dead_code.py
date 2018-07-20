from xml.sax.saxutils import escape

from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.tools import PrettyPrinter
from lalcheck.ai.utils import dataclass
from lalcheck.checkers.support.checker import (
    AbstractSemanticsChecker, DiagnosticPosition
)
from lalcheck.checkers.support.components import AbstractSemantics
from lalcheck.tools import dot_printer
from lalcheck.tools.digraph import Digraph

from lalcheck.tools.scheduler import Task, Requirement


def html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def build_resulting_graph(file_name, cfg, dead_blocks):
    dead_nodes = frozenset(
        node
        for block in dead_blocks
        for node in block.nodes
    )

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
    def __init__(self, sem_analysis, dead_blocks):
        super(Results, self).__init__(sem_analysis, dead_blocks)

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
    def diag_report(cls, block):
        if len(block.nodes) > 0:
            start_line = block.start_node().sloc_range.start.line
            end_line = block.end_node().sloc_range.end.line
            if start_line == end_line:
                message = "unreachable code"
            else:
                message = "unreachable code (until line {})".format(end_line)
            return (
                DiagnosticPosition.from_node(block.start_node()),
                message,
                DeadCodeChecker.name(),
                cls.HIGH
            )


def check_dead_code(prog, model, merge_pred_builder):
    analysis = abstract_semantics.compute_semantics(
        prog,
        model,
        merge_pred_builder
    )

    return find_dead_code(analysis)


class Block(object):
    def __init__(self, nodes):
        """
        :param list[Digraph.Node] nodes: The nodes of the block.
        """
        self.nodes = nodes

    def start_node(self):
        return self.nodes[0].data.node.data.orig_node

    def end_node(self):
        return self.nodes[-1].data.node.data.orig_node


def _node_start_pos(x):
    node = x[0]
    orig = node.data.node.data.orig_node
    return orig.sloc_range.start.line, orig.sloc_range.start.column


def find_dead_code(analysis):
    nodes_with_origin = [
        (
            n,
            len(analysis.semantics[n]) == 0  # is node dead
        )
        for n in analysis.cfg.nodes
        if n.data.node is not None
        if 'orig_node' in n.data.node.data
        if n.data.node.data.orig_node is not None
    ]

    sorted_results = sorted(nodes_with_origin, key=_node_start_pos)

    dead_blocks = []
    current_block = Block([])

    for result in sorted_results:
        graph_node, is_dead = result
        if is_dead:
            current_block.nodes.append(graph_node)
        elif len(current_block.nodes) > 0:
            dead_blocks.append(current_block)
            current_block = Block([])

    if len(current_block.nodes) > 0:
        dead_blocks.append(current_block)

    return Results(analysis, dead_blocks)


@Requirement.as_requirement
def DeadCode(provider_config, model_config, files):
    return [DeadCodeFinder(
        provider_config, model_config, files
    )]


@dataclass
class DeadCodeFinder(Task):
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
            'res': DeadCode(
                self.provider_config,
                self.model_config,
                self.files
            )
        }

    def run(self, **sems):
        return {
            'res': [
                find_dead_code(analysis)
                for sem in sems.values()
                for analysis in sem
            ]
        }


class DeadCodeChecker(AbstractSemanticsChecker):
    @classmethod
    def name(cls):
        return "dead code"

    @classmethod
    def description(cls):
        return "Finds dead code"

    @classmethod
    def create_requirement(cls, *args, **kwargs):
        return cls.requirement_creator(DeadCode)(*args, **kwargs)


checker = DeadCodeChecker


if __name__ == "__main__":
    print("Please run this checker through the run-checkers.py script")
