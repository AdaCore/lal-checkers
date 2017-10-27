"""
Provides tools for using the Basic IR.
"""

from lalcheck.utils import KeyCounter
from lalcheck.digraph import Digraph
from lalcheck import defs, domains
from tree import bin_ops, un_ops, Identifier
import visitors


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


class DomainCollector(object):
    """
    Once constructed, a DomainCollector object can be used to collect domains
    of any amount of Programs. Expressions inside these programs can then be
    tested against the DomainCollector that traversed them to retrieve the
    domain they were assigned.

    A DomainCollector also stores definitions that were introduced during the
    domain collection of each program inside its public 'defs' attribute.
    """
    def __init__(self, domain_gen):
        """
        Constructs a DomainCollector object with the given domain generator.
        A domain generator is a function that, given a type hint held by a node
        and a Definitions object, will return an appropriate domain for that
        node, and fill the Definitions object with the relevant definitions
        if needed.
        """
        self.domain_of = {}
        self.type_map = {}
        self.defs = defs.Definitions.default()
        self.domain_gen = domain_gen

    def collect_domains(self, *programs):
        """
        Associates a domain to any node in the given programs that has a
        'type_hint' data key.
        """
        for program in programs:
            to_assign = visitors.findall(
                program,
                lambda n: 'type_hint' in n.data
            )

            for node in to_assign:
                type_hint = node.data.type_hint
                if type_hint not in self.type_map:
                    self.type_map[type_hint] = self.domain_gen(
                        type_hint,
                        self.defs
                    )
                self.domain_of[node] = self.type_map[type_hint]

    def __getitem__(self, item):
        """
        Returns the domain of the given node.
        Raises a KeyError if the given node was never assigned a domain by
        this DomainCollector.
        """
        return self.domain_of[item]


class ExprEvaluator(visitors.Visitor):
    """
    Can be used to evaluate expressions in the Basic IR.
    """
    def __init__(self, dom_col):
        """
        Constructs an ExprEvaluator given a DomainCollector object. The given
        domain collector must have been used to assign domains to any
        expression that are to be evaluated by this ExprEvaluator.
        """
        self.dom_col = dom_col

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
            self.dom_col[binexpr.lhs],
            self.dom_col[binexpr.rhs],
            self.dom_col[binexpr]
        )
        return self.dom_col.defs.lookup(op, tpe)(lhs, rhs)

    def visit_unexpr(self, unexpr, env):
        expr = unexpr.expr.visit(self, env)
        op = unexpr.un_op.sym
        tpe = (self.dom_col[unexpr.expr], self.dom_col[unexpr])
        return self.dom_col.defs.lookup(op, tpe)(expr)

    def visit_lit(self, lit, env):
        return self.dom_col[lit].build(lit.val)


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

    def __init__(self, dom_col, evaluator):
        """
        Constructs a new solver.
        """
        self.dom_col = dom_col
        self.evaluator = evaluator

    def build_constraint(self, var, op, val, val_dom):
        if not (isinstance(self.dom_col[var], domains.Intervals) and
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
            domain = self.dom_col[var]

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
        lhs_dom, lhs_val = self.dom_col[lhs], self.evaluator.eval(lhs, env)
        rhs_dom, rhs_val = self.dom_col[rhs], self.evaluator.eval(rhs, env)

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
