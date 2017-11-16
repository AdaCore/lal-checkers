"""
Provides tools for using the Basic IR.
"""

from lalcheck.utils import KeyCounter, Bunch
from lalcheck.digraph import Digraph
from lalcheck.domain_ops import boolean_ops
import visitors

from collections import defaultdict
from funcy.calc import memoize


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
        return "{}{}".format(
            str(ident.name),
            "#{}".format(self.id_map[ident]) if opts.print_ids else ""
        )

    def visit_assign(self, assign, opts):
        return "{} = {}".format(
            assign.var.visit(self, opts),
            assign.expr.visit(self, opts)
        )

    def visit_split(self, split, opts):
        indents = opts.indents()
        return "split:\n{}\n{}|:\n{}".format(
            self.print_stmts(split.fst_stmts, opts),
            indents,
            self.print_stmts(split.snd_stmts, opts),
        )

    def visit_loop(self, loop, opts):
        return "loop:\n{}".format(self.print_stmts(loop.stmts, opts))

    def visit_read(self, read, opts):
        return "read({})".format(read.var.visit(self, opts))

    def visit_use(self, use, opts):
        return "use({})".format(use.var.visit(self, opts))

    def visit_assume(self, assume, opts):
        return "assume({})".format(assume.expr.visit(self, opts))

    def visit_binexpr(self, binexpr, opts):
        return "{} {} {}".format(
            binexpr.lhs.visit(self, opts),
            str(binexpr.bin_op),
            binexpr.rhs.visit(self, opts)
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
        self.start_node = None
        self.key_counter = KeyCounter()

    def fresh(self, name):
        return "{}{}".format(name, self.key_counter.get_incr(name))

    def visit_program(self, prgm):
        self.nodes = []
        self.edges = []

        start = self.build_node("start")
        self.visit_stmts(prgm.stmts, start)

        return Digraph(self.nodes + [start], self.edges)

    def visit_split(self, splitstmt, start):
        end_fst = self.visit_stmts(splitstmt.fst_stmts, start)
        end_snd = self.visit_stmts(splitstmt.snd_stmts, start)

        join = self.build_node("split_join")
        self.register_and_link([end_fst, end_snd], join)

        return join

    def visit_loop(self, loopstmt, start):
        loop_start = self.build_node("loop_start", is_widening_point=True)

        end = self.visit_stmts(loopstmt.stmts, loop_start)
        join = self.build_node("loop_join")

        self.register_and_link([start, end], loop_start)
        self.register_and_link([loop_start], join)
        return join

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
    def __init__(self, typer, type_interpreter):
        """
        :param lalcheck.types.Typer[T] typer: The typer that maps type hints
            to lalcheck types.

        :param lalcheck.types.TypeInterpreter type_interpreter: The type
            interpreter that maps lalcheck types to interpretations.
        """
        self.typer = typer
        self.type_interpreter = type_interpreter

    @memoize
    def _hint_to_type(self, hint):
        """
        :param T hint: The type hint.

        :return: The corresponding lalcheck type.

        :rtype: lalcheck.types.Type

        Note: requires memoization so that any two identical hints map to the
        same lalcheck type.
        """
        return self.typer.from_hint(hint)

    @memoize
    def _type_to_interp(self, tpe):
        """
        :param lalcheck.types.Type tpe: The lalcheck type.

        :return: The corresponding type interpretation.
        """
        return self.type_interpreter.from_type(tpe)

    def _typeable_to_interp(self, node):
        """
        :param tree.Node node: The expression node for which to retrieve the
            interpretation.

        :return: The associated interpretation
        """
        return self._type_to_interp(self._hint_to_type(node.data.type_hint))

    def visit_unexpr(self, unexpr, node_domains, defs, inv_defs, builders):
        dom = node_domains[unexpr]
        expr_dom = node_domains[unexpr.expr]
        tpe = (expr_dom, dom)

        return Bunch(
            domain=dom,
            definition=defs(unexpr.un_op.sym, tpe),
            inverse=inv_defs(unexpr.un_op.sym, tpe)
        )

    def visit_binexpr(self, binexpr, node_domains, defs, inv_defs, builders):
        dom = node_domains[binexpr]
        lhs_dom = node_domains[binexpr.lhs]
        rhs_dom = node_domains[binexpr.rhs]
        tpe = (lhs_dom, rhs_dom, dom)

        return Bunch(
            domain=dom,
            definition=defs(binexpr.bin_op.sym, tpe),
            inverse=inv_defs(binexpr.bin_op.sym, tpe)
        )

    def visit_ident(self, ident, node_domains, defs, inv_defs, builders):
        return Bunch(
            domain=node_domains[ident]
        )

    def visit_lit(self, lit, node_domains, defs, inv_defs, builders):
        dom = node_domains[lit]
        return Bunch(
            domain=dom,
            builder=builders[dom]
        )

    @staticmethod
    def _has_type_hint(node):
        """
        :param tree.Node node: A IR node.
        :return: True if the given node has a type hint.
        :rtype: bool
        """
        return 'type_hint' in node.data

    @staticmethod
    def _aggregate_provider(providers):
        def f(name, signature):
            for provider in providers:
                definition = provider(name, signature)
                if definition:
                    return definition
            raise LookupError("No provider for '{}' {}".format(
                name, signature
            ))

        return f

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
        def_providers = set()
        inv_def_providers = set()
        builders = {}

        for prog in programs:
            typeable = visitors.findall(prog, self._has_type_hint)

            for node in typeable:
                interp = self._typeable_to_interp(node)
                domain, domain_defs, domain_inv_defs, domain_builder = interp

                node_domains[node] = domain
                def_providers.add(domain_defs)
                inv_def_providers.add(domain_inv_defs)
                builders[domain] = domain_builder

        for node in node_domains.keys():
            model[node] = node.visit(
                self,
                node_domains,
                Models._aggregate_provider(def_providers),
                Models._aggregate_provider(inv_def_providers),
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

    def eval(self, expr, env):
        """
        :param tree.Expr expr: The expression to evaluate.

        :param dict[tree.Identifier, object] env: The environment, containing
            an entry for each identifier traversed during evaluation.

        :return: The value this expression evaluates to.

        :rtype: object
        """
        return expr.visit(self, env)

    def visit_ident(self, ident, env):
        return env[ident]

    def visit_binexpr(self, binexpr, env):
        lhs = binexpr.lhs.visit(self, env)
        rhs = binexpr.rhs.visit(self, env)
        return self.model[binexpr].definition(lhs, rhs)

    def visit_unexpr(self, unexpr, env):
        expr = unexpr.expr.visit(self, env)
        return self.model[unexpr].definition(expr)

    def visit_lit(self, lit, env):
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

    def solve(self, expr, env):
        """
        :param tree.Expr expr: The predicate expression to solve.

        :param dict[tree.Identifier, object] env: The environment, containing
            an entry for each identifier traversed while solving.

        :return: A new environment for which evaluating the given expression
            returns boolean_ops.True

        :rtype: dict[tree.Identifier, object]

        Note: The new environment may in fact not evaluate to True because it
        is an over-approximation of the optimal solution. However, it should
        never constructs a solution that does not contain the optimal one,
        thus making it sound for abstract interpretation.
        """

        new_env = env.copy()
        if not expr.visit(self, new_env, boolean_ops.true):
            return {}
        return new_env

    def visit_ident(self, ident, env, expected):
        dom = self.model[ident].domain
        env[ident] = dom.meet(env[ident], expected)
        return True

    def visit_binexpr(self, binexpr, env, expected):
        lhs_val = self.eval(binexpr.lhs, env)
        rhs_val = self.eval(binexpr.rhs, env)
        inv_res = self.model[binexpr].inverse(
            expected, lhs_val, rhs_val
        )

        if inv_res is None:
            return False

        expected_lhs, expected_rhs = inv_res
        return (binexpr.lhs.visit(self, env, expected_lhs) and
                binexpr.rhs.visit(self, env, expected_rhs))

    def visit_unexpr(self, unexpr, env, expected):
        expr_val = self.eval(unexpr.expr, env)
        expected_expr = self.model[unexpr].inverse(expected, expr_val)
        if expected_expr is None:
            return False
        return unexpr.expr.visit(self, env, expected_expr)

    def visit_lit(self, lit, env, expected):
        lit_dom = self.model[lit].domain
        lit_val = self.eval(lit, env)
        return not lit_dom.is_empty(lit_dom.meet(expected, lit_val))
