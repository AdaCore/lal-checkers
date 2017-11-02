"""
Provides tools for using the Basic IR.
"""

from lalcheck.utils import KeyCounter, Bunch
from lalcheck.digraph import Digraph
from lalcheck.domain_ops import boolean_ops
from lalcheck import domains
from tree import bin_ops, un_ops, Identifier
import visitors

from funcy.calc import memoize


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
        Creates a Models object from a typer (that maps type hints to types)
        and a type interpreter (that maps types to interpretations).
        """
        self.typer = typer
        self.type_interpreter = type_interpreter

    @memoize
    def _hint_to_type(self, hint):
        # Memoization is required to get the same type instances
        # for each identical hint
        return self.typer.from_hint(hint)

    @memoize
    def _type_to_interp(self, tpe):
        # Memoization is required to get the same interpretations
        # for each identical type
        return self.type_interpreter.from_type(tpe)

    def _typeable_to_interp(self, node):
        return self._type_to_interp(self._hint_to_type(node.data.type_hint))

    def visit_unexpr(self, unexpr, node_domains, defs, builders):
        dom = node_domains[unexpr]
        expr_dom = node_domains[unexpr.expr]
        tpe = (expr_dom, dom)

        return Bunch(
            domain=dom,
            definition=defs[unexpr.un_op.sym, tpe]
        )

    def visit_binexpr(self, binexpr, node_domains, defs, builders):
        dom = node_domains[binexpr]
        lhs_dom = node_domains[binexpr.lhs]
        rhs_dom = node_domains[binexpr.rhs]
        tpe = (lhs_dom, rhs_dom, dom)

        return Bunch(
            domain=dom,
            definition=defs[binexpr.bin_op.sym, tpe]
        )

    def visit_ident(self, ident, node_domains, defs, builders):
        return Bunch(
            domain=node_domains[ident]
        )

    def visit_lit(self, lit, node_domains, defs, builders):
        dom = node_domains[lit]
        return Bunch(
            domain=dom,
            builder=builders[dom]
        )

    @staticmethod
    def _has_type_hint(node):
        return 'type_hint' in node.data

    def of(self, *programs):
        """
        Returns a model of the given programs, that is, a dictionary that has
        an entry for any node in the given programs that has a type hint.
        This entry associates to the node valuable information, such as the
        domain used to represent the value it computes, the referenced
        definition if any, etc.
        """
        model = {}
        node_domains = {}
        defs = {}
        builders = {}

        for prog in programs:
            typeable = visitors.findall(prog, self._has_type_hint)

            for node in typeable:
                interp = self._typeable_to_interp(node)
                domain, domain_defs, domain_builder = interp

                node_domains[node] = domain
                defs.update(domain_defs)
                builders[domain] = domain_builder

        for node in node_domains.keys():
            model[node] = node.visit(self, node_domains, defs, builders)

        return model


class ExprEvaluator(visitors.Visitor):
    """
    Can be used to evaluate expressions in the Basic IR.
    """
    def __init__(self, model):
        """
        Constructs an ExprEvaluator given a model. The expression evaluator
        must only be invoked to evaluate expression which nodes have a meaning
        in the given model.
        """
        self.model = model

    def eval(self, expr, env):
        """
        Given an environment (a map from Identifier to value), evaluates
        the given expression.
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


class TrivialIntervalCS(visitors.Visitor):
    """
    A simple constraint solver that works on Interval domains.
    Solves exactly constraints of the form:
    - "expr", if expr is free of variables
    - "[not] x OP expr" or "[not] expr OP x", if expr is free of variables,
      OP is a binary operator among {<, <=, ==, !=, =>, >}, and x an
      Identifier.
    """

    AllowedBinOps = {
        bin_ops[op] for op in ('<', '<=', '==', '!=', '>=', '>')
    }
    NotOp = {
        bin_ops['<']: bin_ops['>='],
        bin_ops['<=']: bin_ops['>'],
        bin_ops['==']: bin_ops['!='],
        bin_ops['!=']: bin_ops['=='],
        bin_ops['>=']: bin_ops['<'],
        bin_ops['>']: bin_ops['<=']
    }
    OppOp = {
        bin_ops['<']: bin_ops['>'],
        bin_ops['<=']: bin_ops['>='],
        bin_ops['==']: bin_ops['=='],
        bin_ops['!=']: bin_ops['!='],
        bin_ops['>=']: bin_ops['<='],
        bin_ops['>']: bin_ops['<']
    }

    def __init__(self, model, evaluator):
        """
        Constructs a new solver.
        """
        self.model = model
        self.evaluator = evaluator

    def build_constraint(self, var, op, val, val_dom):
        if not (isinstance(self.model[var].domain, domains.Intervals) and
                isinstance(val_dom, domains.Intervals)):
            return None
        if op not in TrivialIntervalCS.AllowedBinOps:
            return None
        elif not isinstance(var, Identifier):
            return None
        elif val_dom.eq(val, val_dom.bottom):
            return None
        elif val[0] != val[1]:
            return None
        else:
            return var, op, val[0]

    def solve_constraints(self, constraints, env):
        for (var, rel_op, val) in constraints:
            domain = self.model[var].domain

            if rel_op is bin_ops['<']:
                itvs = [domain.left_unbounded(val - 1)]
            elif rel_op is bin_ops['<=']:
                itvs = [domain.left_unbounded(val)]
            elif rel_op is bin_ops['==']:
                itvs = [domain.build(val)]
            elif rel_op is bin_ops['!=']:
                itvs = []
            elif rel_op is bin_ops['>=']:
                itvs = [domain.right_unbounded(val)]
            elif rel_op is bin_ops['>']:
                itvs = [domain.right_unbounded(val + 1)]
            else:
                itvs = []

            for itv in itvs:
                env[var] = domain.meet(env[var], itv)

        return env

    def do_binexpr(self, lhs, op, rhs, env):
        lhs_dom = self.model[lhs].domain
        rhs_dom = self.model[rhs].domain
        lhs_val = self.evaluator.eval(lhs, env)
        rhs_val = self.evaluator.eval(rhs, env)

        attempts = [
            (lhs, op, rhs_val, rhs_dom),
            (rhs, TrivialIntervalCS.OppOp[op], lhs_val, lhs_dom)
        ]
        constraints = []

        for attempt in attempts:
            constraint = self.build_constraint(*attempt)
            if constraint is not None:
                constraints.append(constraint)

        return constraints

    def visit_binexpr(self, binexpr, env, inversed):
        op = (TrivialIntervalCS.NotOp[binexpr.bin_op] if inversed
              else binexpr.bin_op)
        return self.do_binexpr(binexpr.lhs, op, binexpr.rhs, env)

    def visit_unexpr(self, unexpr, env, inversed):
        if unexpr.un_op == un_ops['!']:
            return unexpr.expr.visit(self, env, not inversed)

        raise NotImplementedError("Unsupported constraint")

    def solve(self, expr, env):
        """
        Given an environment, returns a subset of this environment that
        attempts to be the closest possible to the biggest environment for
        which evaluating the given expression returns the singleton set
        containing True.
        """
        value = self.evaluator.eval(expr, env)
        if boolean_ops.Boolean.eq(value, boolean_ops.bool_true):
            return env
        elif boolean_ops.Boolean.eq(value, boolean_ops.bool_false):
            return {}
        elif boolean_ops.Boolean.eq(value, boolean_ops.bool_both):
            constraints = expr.visit(self, env, False)
            return self.solve_constraints(constraints, env)
