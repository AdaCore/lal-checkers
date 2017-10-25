"""
Provides tools for using the Basic IR.
"""

from lalcheck.utils import KeyCounter
from lalcheck.digraph import Digraph
from lalcheck import defs, domains
from ast import bin_ops, un_ops, Identifier
import visitors


class CFGBuilder(visitors.ImplicitVisitor):
    """
    A visitor that can be used to build the control-flow graph of the given
    program as an instance of a Digraph. Nodes of the resulting control-flow
    graph will have the following data attached to it:
    - 'widening_point': "True" iff the node can be used as a widening point.
    - 'node': the corresponding AST node which this CFG node was built from,
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


class Typer(object):
    """
    Once constructed, a Typer object can be used to type any amount of
    Programs. Expressions inside these programs can then be run against
    the Typer that typed them to retrieve their types.

    A Typer also stores definitions that were introduced during the typing
    of each program.
    """
    def __init__(self, type_gen):
        """
        Constructs a Typer object with the given type generator.
        A type generator is a function that, given a type hint held by a node
        and a Definitions object, will return an appropriate type for that
        node, and fill the Definitions object with the relevant definitions
        if needed.
        """
        self.type_of = {}
        self.type_map = {}
        self.defs = defs.Definitions.default()
        self.type_gen = type_gen

    def type_programs(self, programs):
        """
        Types each of the given programs.
        """
        for p in programs:
            self.type_program(p)

    def type_program(self, program):
        """
        Associates a type to any node in the given program that has a
        'type_hint' data key.
        """
        to_type = visitors.findall(
            program,
            lambda n: hasattr(n.data, 'type_hint')
        )

        for node in to_type:
            type_hint = node.data.type_hint
            if type_hint not in self.type_map:
                self.type_map[type_hint] = self.type_gen(type_hint, self.defs)
            self.type_of[node] = self.type_map[type_hint]

    def __getitem__(self, item):
        """
        Returns the type of the given node.
        Raises a KeyError if the given node was never typed by this Typer.
        """
        return self.type_of[item]


class ExprEvaluator(visitors.Visitor):
    """
    Can be used to evaluate expressions in the Basic IR.
    """
    def __init__(self, typer):
        """
        Constructs an ExprEvaluator given a Typer object. The given typer must
        have been used to type any expression that are to be evaluated by this
        ExprEvaluator.
        """
        self.typer = typer

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
        op = binexpr.bin_op.sym
        tpe = (
            self.typer[binexpr.lhs],
            self.typer[binexpr.rhs],
            self.typer[binexpr]
        )
        return self.typer.defs.lookup(op, tpe)(lhs, rhs)

    def visit_unexpr(self, unexpr, env):
        expr = unexpr.expr.visit(self, env)
        op = unexpr.un_op.sym
        tpe = (self.typer[unexpr.expr], self.typer[unexpr])
        return self.typer.defs.lookup(op, tpe)(expr)

    def visit_lit(self, lit, env):
        return self.typer[lit].build(lit.val)


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

    def __init__(self, typer, evaluator):
        """
        Constructs a new solver.
        """
        self.typer = typer
        self.evaluator = evaluator

    def build_constraint(self, var, op, val, val_dom):
        if not (isinstance(self.typer[var], domains.Intervals) and
                isinstance(val_dom, domains.Intervals)):
            return None
        if op not in TrivialIntervalCS.AllowedBinOps:
            return None
        elif not isinstance(var, Identifier):
            return None
        elif val_dom.eq(val, val_dom.bot):
            return None
        elif val[0] != val[1]:
            return None
        else:
            return var, op, val[0]

    def solve_constraints(self, constraints, env):
        for (var, rel_op, val) in constraints:
            domain = self.typer[var]

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
        lhs_dom, lhs_val = self.typer[lhs], self.evaluator.eval(lhs, env)
        rhs_dom, rhs_val = self.typer[rhs], self.evaluator.eval(rhs, env)

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
        if defs.Boolean.eq(value, defs.bool_true):
            return env
        elif defs.Boolean.eq(value, defs.bool_false):
            return {}
        elif defs.Boolean.eq(value, defs.bool_both):
            constraints = expr.visit(self, env, False)
            return self.solve_constraints(constraints, env)
