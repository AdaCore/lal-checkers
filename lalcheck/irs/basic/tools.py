"""
Provides tools for using the Basic IR.
"""

from __future__ import absolute_import
from collections import defaultdict

from lalcheck.irs.basic import visitors
from lalcheck.irs.basic.tree import LabelStmt
from lalcheck.domain_ops import boolean_ops
from lalcheck.interpretations import Signature
from lalcheck.types import FunOutput
from lalcheck.utils import KeyCounter, Bunch, Transformer
from tools.digraph import Digraph


class PrettyPrinter(visitors.Visitor):
    """
    Visitor that can be used to construct a human-readable string
    representation of Basic IR nodes.
    """

    class Opts(object):
        """
        An object that holds the pretty-printing context.
        """

        def __init__(self, indent=0, print_ids=False):
            """
            :param int indent: The indentation count.
            """
            self.indent = indent
            self.print_ids = print_ids

        def indents(self, offset=0):
            """
            :param int offset: An additional indentation value.

            :return: A string filled with whitespaces, to prepend at the start
                of an indented line.

            :rtype: str
            """
            return "  " * (self.indent + offset)

        def indented(self):
            """
            :return: A new pretty printing options instance where an
                incremented "indent" field.

            :rtype: PrettyPrintOpts
            """
            return PrettyPrinter.Opts(self.indent + 1, self.print_ids)

    def __init__(self):
        self.cur_id = 0

        def next_id():
            self.cur_id += 1
            return self.cur_id

        self.id_map = defaultdict(next_id)

    @staticmethod
    def pretty_print(node, opts=None):
        """
        :param tree.Node node: The node to pretty print.
        :param PrettyPrinter.Opts | None opts: The pretty printing options.
        :return: A human-readable string representation of this node.
        :rtype: str
        """
        return node.visit(
            PrettyPrinter(),
            opts if opts is not None else PrettyPrinter.Opts()
        )

    def print_stmts(self, stmts, opts):
        indents = opts.indents(1)
        return "\n".join(map(
            lambda stmt: indents + stmt.visit(self, opts.indented()),
            stmts
        ))

    def visit_program(self, prgm, opts):
        return "Program:\n{}".format(self.print_stmts(prgm.stmts, opts))

    def visit_ident(self, ident, opts):
        return ident.var.visit(self, opts)

    def visit_var(self, var, opts):
        return "{}{}".format(
            str(var.name),
            "#{}".format(self.id_map[var]) if opts.print_ids else ""
        )

    def visit_assign(self, assign, opts):
        return "{} = {}".format(
            assign.id.visit(self, opts),
            assign.expr.visit(self, opts)
        )

    def visit_split(self, split, opts):
        indents = "\n{}".format(opts.indents())
        branches = ["split:\n{}".format(
            self.print_stmts(split.branches[0], opts)
        )] + [
            "|:\n{}".format(self.print_stmts(branch, opts))
            for branch in split.branches[1:]
        ]
        return indents.join(branches)

    def visit_loop(self, loop, opts):
        return "loop:\n{}".format(self.print_stmts(loop.stmts, opts))

    def visit_label(self, labelstmt, *args):
        return "{}:".format(labelstmt.name)

    def visit_goto(self, gotostmt, *args):
        return "goto {}".format(gotostmt.label.name)

    def visit_read(self, read, opts):
        return "read({})".format(read.id.visit(self, opts))

    def visit_use(self, use, opts):
        return "use({})".format(use.id.visit(self, opts))

    def visit_assume(self, assume, opts):
        return "assume({})".format(assume.expr.visit(self, opts))

    def visit_funcall(self, funcall, opts):
        return "{}({})".format(
            str(funcall.fun_id),
            ", ".join([arg.visit(self, opts) for arg in funcall.args])
        )

    def visit_unexpr(self, unexpr, opts):
        return "{}{}".format(
            str(unexpr.un_op),
            unexpr.expr.visit(self, opts)
        )

    def visit_lit(self, lit, opts):
        return str(lit.val)


class CFGBuilder(visitors.ImplicitVisitor):
    """
    A visitor that can be used to build the control-flow graph of the given
    program as an instance of a Digraph. Nodes of the resulting control-flow
    graph will have the following data attached to it:
    - 'widening_point': "True" iff the node can be used as a widening point.
    - 'node': the corresponding IR node which this CFG node was built from,
      or None.
    """
    def __init__(self):
        self.nodes = None
        self.edges = None
        self.jumps = None
        self.labels = None
        self.start_node = None
        self.key_counter = KeyCounter()

    def fresh(self, name):
        return "{}{}".format(name, self.key_counter.get_incr(name))

    @staticmethod
    def is_label(node):
        """
        :param tree.Node node: An IR node
        :return: Whether the node is a LabelStmt
        :rtype: bool
        """
        return isinstance(node, LabelStmt)

    def compute_reachable_nodes(self, start, reachables):
        """
        Computes the set of nodes that are reachable from the given "start"
        node using the set of edges registered so far. Reachable nodes are
        added to the given set.

        :param Digraph.Node start: The node from which to compute reachable
            nodes.

        :param set[Digraph.Node] reachables: The set of nodes that are found
            reachable so far.
        """
        def outs(node):
            """
            :return: The directly reachable nodes from the given node
            :rtype: iterable[Digraph.Node]
            """
            return (e.to for e in self.edges if e.frm == node)

        reachables.add(start)
        for node in outs(start):
            if node not in reachables:
                self.compute_reachable_nodes(node, reachables)

    def visit_program(self, prgm):
        self.nodes = []
        self.edges = []
        self.jumps = []
        self.labels = {}

        start = self.build_node("start")
        self.visit_stmts(prgm.stmts, start)

        # Generate jump edges
        for node, label in self.jumps:
            self.edges.extend([
                Digraph.Edge(node, edge.to)
                for edge in self.edges
                if edge.frm == self.labels[label]
            ])

        # Compute reachable nodes
        reachables = set()
        self.compute_reachable_nodes(start, reachables)

        # Remove all nodes and edges that are not reachable
        self.nodes = [n for n in self.nodes if n in reachables]
        self.edges = [e for e in self.edges if e.frm in reachables]

        return Digraph([start] + self.nodes, self.edges)

    def visit_split(self, splitstmt, start):
        ends = [
            self.visit_stmts(branch, start) for branch in splitstmt.branches
        ]

        join = self.build_node("split_join")
        self.register_and_link(ends, join)

        return join

    def visit_loop(self, loopstmt, start):
        loop_start = self.build_node("loop_start", is_widening_point=True)

        end = self.visit_stmts(loopstmt.stmts, loop_start)
        join = self.build_node("loop_join")

        self.register_and_link([start, end], loop_start)
        self.register_and_link([loop_start], join)
        return join

    def visit_label(self, label, start):
        self.labels[label] = start
        return start

    def visit_goto(self, goto, start):
        self.jumps.append((start, goto.label))
        return None

    def visit_assign(self, assign, start):
        n = self.build_node("assign", orig_node=assign)
        self.register_and_link([start], n)
        return n

    def visit_read(self, read, start):
        n = self.build_node("read", orig_node=read)
        self.register_and_link([start], n)
        return n

    def visit_use(self, use, start):
        n = self.build_node("use", orig_node=use)
        self.register_and_link([start], n)
        return n

    def visit_assume(self, assume, start):
        n = self.build_node("assume", orig_node=assume)
        self.register_and_link([start], n)
        return n

    def visit_stmts(self, stmts, cur):
        for stmt in stmts:
            cur = stmt.visit(self, cur)
        return cur

    def build_node(self, name, is_widening_point=False, orig_node=None):
        return Digraph.Node(
            name=self.fresh(name),
            is_widening_point=is_widening_point,
            node=orig_node
        )

    def register_and_link(self, froms, new_node):
        self.nodes.append(new_node)
        for f in froms:
            if f is not None:
                self.edges.append(Digraph.Edge(f, new_node))


class Models(visitors.Visitor):
    """
    A Models object is constructed from a typer and a type interpreter.
    With these two components, it can derive the interpretation of a type
    from the type hint provided by the frontend.

    It can then be used to build models of the given programs. A model
    can be queried for information about a node of a program. Such information
    includes the domain used to represent the value computed by that node (if
    relevant), how an operation must be interpreted (i.e. a binary addition),
    etc.
    """
    def __init__(self, typer, type_interpreter, external_def_provider=None):
        """
        :param lalcheck.types.Typer[T] typer: The typer that maps type hints
            to lalcheck types.

        :param lalcheck.types.TypeInterpreter type_interpreter: The type
            interpreter that maps lalcheck types to interpretations.

        :param function external_def_provider:
            An external def provider builder.
        """
        self.typer = typer
        self.type_interpreter = type_interpreter
        self.external_def_provider = external_def_provider

    def _type_of(self, hint):
        """
        :param T hint: The hint.
        :return: Its type.
        :rtype: lalcheck.types.Type
        """
        return self.typer.get(hint)

    def _interp_of(self, tpe):
        """
        :param lalcheck.types.Type tpe: The type.
        :return: Its interpretation.
        :rtype: lalcheck.interpretations.TypeInterpretation
        """
        return self.type_interpreter.get(tpe)

    def _typeable_to_interp(self, node):
        """
        :param tree.Node node: An expression node.
        :return: Its type interpretation.
        :rtype: lalcheck.interpretations.TypeInterpretation
        """
        return self._interp_of(self._type_of(node.data.type_hint))

    def visit_funcall(self, funcall, node_domains, defs, builders):
        dom = node_domains[funcall]

        tpe = self._type_of(funcall.data.type_hint)
        if tpe.is_a(FunOutput):
            out_indices = tpe.out_indices
            ret_tpe = tpe.get_return_type()
            ret_dom = self._interp_of(ret_tpe).domain if ret_tpe else None
        else:
            out_indices = ()
            ret_dom = dom

        input_doms = tuple(node_domains[arg] for arg in funcall.args)

        if 'additional_arg' in funcall.data:
            arg_interp = self._interp_of(
                self._type_of(
                    funcall.data.additional_arg
                )
            )
            input_doms = input_doms + (arg_interp.domain,)

        sig = Signature(
            funcall.fun_id,
            input_doms,
            ret_dom,
            out_indices
        )

        definition, inverse = defs.get(sig)

        return Bunch(
            domain=dom,
            definition=definition,
            inverse=inverse
        )

    def visit_ident(self, ident, node_domains, defs, builders):
        return ident.var.visit(self, node_domains, defs, builders)

    def visit_var(self, var, node_domains, defs, builders):
        return Bunch(domain=node_domains[var])

    def visit_lit(self, lit, node_domains, defs, builders):
        dom = node_domains[lit]
        return Bunch(domain=dom, builder=builders[dom])

    @staticmethod
    def _has_type_hint(node):
        """
        :param tree.Node node: A IR node.
        :return: True if the given node has a type hint.
        :rtype: bool
        """
        return 'type_hint' in node.data

    def _make_def_provider(self, def_provider_builders):
        @Transformer.make_memoizing
        @Transformer.from_transformer_builder
        def provider():
            aggregate_provider = Transformer.first_of(
                *(p(provider) for p in def_provider_builders)
            )
            return (aggregate_provider
                    if self.external_def_provider is None
                    else (aggregate_provider |
                          self.external_def_provider(provider)))
        return provider

    def of(self, *programs):
        """
        :param iterable[irt.Program] programs: Programs for which to build
            a model.

        :return: A dictionary that has an entry for any node in the given
            programs that has a type hint. This entry associates to the node
            valuable information, such as the domain used to represent the
            value it computes, the referenced definition if any, etc.

        :rtype: dict[tree.Node, Bunch]
        """

        model = {}
        node_domains = {}
        def_provider_builders = set()
        builders = {}

        for prog in programs:
            typeable = visitors.findall(prog, self._has_type_hint)

            for node in typeable:
                interp = self._typeable_to_interp(node)

                node_domains[node] = interp.domain
                def_provider_builders.add(interp.def_provider_builder)
                builders[interp.domain] = interp.builder

        final_provider = self._make_def_provider(def_provider_builders)

        for node in node_domains.keys():
            model[node] = node.visit(
                self,
                node_domains,
                final_provider,
                builders
            )

        return model


class ExprEvaluator(visitors.Visitor):
    """
    Can be used to evaluate expressions in the Basic IR.
    """
    def __init__(self, model):
        """
        :param dict[tree.Node, Bunch] model: A model that must have an entry
            for each node that needs be evaluated by this evaluator.
        """
        self.model = model

    def eval(self, expr, state):
        """
        :param tree.Expr expr: The expression to evaluate.

        :param tuple[object] state: The state, containing
            an entry for each Variable traversed during evaluation.

        :return: The value this expression evaluates to.

        :rtype: object
        """
        return expr.visit(self, state)

    def visit_ident(self, ident, state):
        return state[ident.var.data.index]

    def visit_funcall(self, funcall, state):
        args = [arg.visit(self, state) for arg in funcall.args]
        return self.model[funcall].definition(*args)

    def visit_lit(self, lit, state):
        return self.model[lit].builder(lit.val)


class ExprSolver(visitors.Visitor):
    """
    Can be used to solve expressions in the Basic IR.
    """
    def __init__(self, model):
        """
        :param dict[tree.Node, Bunch] model: A model that must have an entry
            for each node that needs be solved by this solver.
        """
        self.model = model
        self.eval = ExprEvaluator(model).eval

    def solve(self, expr, state):
        """
        :param tree.Expr expr: The predicate expression to solve.

        :param dict[tree.Variable, object] state: The environment, containing
            an entry for each variable traversed while solving.

        :return: A new environment for which evaluating the given expression
            returns boolean_ops.True

        :rtype: dict[tree.Variable, object]

        Note: The new environment may in fact not evaluate to True because it
        is an over-approximation of the optimal solution. However, it should
        never constructs a solution that does not contain the optimal one,
        thus making it sound for abstract interpretation.
        """

        new_state = list(state)
        res = expr.visit(self, new_state, boolean_ops.true)
        return tuple(new_state), res

    def visit_ident(self, ident, state, expected):
        var_idx = ident.var.data.index
        dom = self.model[ident].domain
        state[var_idx] = dom.meet(state[var_idx], expected)
        return True

    def visit_funcall(self, funcall, state, expected):
        args_val = [self.eval(arg, state) for arg in funcall.args]
        inv_res = self.model[funcall].inverse(
            expected, *args_val
        )

        if inv_res is None:
            return False

        if len(args_val) == 1:
            inv_res = (inv_res,)

        return all(
            arg.visit(self, state, expected_arg)
            for arg, expected_arg in zip(funcall.args, inv_res)
        )

    def visit_lit(self, lit, state, expected):
        lit_dom = self.model[lit].domain
        lit_val = self.eval(lit, state)
        return not lit_dom.is_empty(lit_dom.meet(expected, lit_val))
