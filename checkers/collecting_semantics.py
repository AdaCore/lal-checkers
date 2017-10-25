"""
An analyzer that collects semantics at each program point.
"""

from lalcheck.irs.basic.tools import (
    CFGBuilder,
    ExprEvaluator,
    TrivialIntervalCS
)

from lalcheck.irs.basic.ast import Identifier, PrettyPrintOpts
from lalcheck.irs.basic import visitors
from lalcheck.utils import KeyCounter
from lalcheck.digraph import Digraph
from lalcheck import domains
from lalcheck import dot_printer

from xml.sax.saxutils import escape
from collections import defaultdict


class VarTracker(visitors.CFGNodeVisitor):
    def __init__(self, var_set, vars_domain, vars_idx, evaluator, c_solver):
        self.vars = var_set
        self.vars_domain = vars_domain
        self.vars_idx = vars_idx
        self.evaluator = evaluator
        self.constr_solver = c_solver

    def state_to_env(self, state):
        return {v: state[self.vars_idx[v]] for v in self.vars}

    def env_to_state(self, env):
        if len(env) == 0:
            return self.vars_domain.bot
        else:
            return tuple(env[v] for v in self.vars)

    def visit_assign(self, assign, state):
        env = self.state_to_env(state)
        env[assign.var] = self.evaluator.eval(assign.expr, env)
        return self.env_to_state(env)

    def visit_assume(self, assume, state):
        return self.env_to_state(self.constr_solver.solve(
            assume.expr,
            self.state_to_env(state)
        ))

    def visit_read(self, read, state):
        return tuple(self.vars_domain.top[i] if i == self.vars_idx[read.var]
                     else x for i, x in enumerate(state))

    def visit_use(self, use, state):
        return state


class SimpleTraceLattice(domains.FiniteSubsetLattice):
    def __init__(self, *args):
        super(SimpleTraceLattice, self).__init__(*args)

    def update(self, a, b, widen=False):
        return super(SimpleTraceLattice, self).update(a, b, False)


class MergePredicateBuilder(object):
    def __init__(self, predicate):
        self.predicate = predicate

    def __or__(self, other):
        def f(trace_domain, vals_domain):
            s_pred = self.build(trace_domain, vals_domain)
            o_pred = other.build(trace_domain, vals_domain)

            return lambda a, b: s_pred(a, b) or o_pred(a, b)

        return MergePredicateBuilder(f)

    def __and__(self, other):
        def f(trace_domain, vals_domain):
            s_pred = self.build(trace_domain, vals_domain)
            o_pred = other.build(trace_domain, vals_domain)

            return lambda a, b: s_pred(a, b) and o_pred(a, b)

        return MergePredicateBuilder(f)

    def build(self, trace_domain, vals_domain):
        return self.predicate(trace_domain, vals_domain)


def _mp_always(*_):
    return lambda *_: True


def _mp_never(*_):
    return lambda *_: False


def _mp_le_traces(trace_domain, _):
    return lambda a, b: trace_domain.le(a[0], b[0])


def _mp_eq_vals(_, vals_domain):
    return lambda a, b: vals_domain.eq(a[1], b[1])


MergePredicateBuilder.Always = MergePredicateBuilder(_mp_always)
MergePredicateBuilder.Never = MergePredicateBuilder(_mp_never)
MergePredicateBuilder.Le_Traces = MergePredicateBuilder(_mp_le_traces)
MergePredicateBuilder.Eq_Vals = MergePredicateBuilder(_mp_eq_vals)


def html_render_node(node):
    return escape(node.pretty_print(PrettyPrintOpts(0)))


def save_cfg_to(file_name, cfg):
    def render_node(node):
        return (html_render_node(node),) if node is not None else ()

    def render_widening_point(is_widening_point):
        return (escape('<widening_point>'),) if is_widening_point else ()

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(cfg, [
            dot_printer.DataPrinter('node', render_node),
            dot_printer.DataPrinter('is_widening_point', render_widening_point)
        ]))


def repr_trace(trace):
    return tuple(p.name for p in trace)


def build_resulting_graph(file_name, cfg, results, trace_domain, vars_idx):
    paths = defaultdict(list)

    for node, state in results.iteritems():
        for trace, values in state:
            paths[frozenset(trace)].append(
                Digraph.Node(
                    node.name,
                    ___orig=node,
                    **{v.name: values[i] for v, i in vars_idx.iteritems()}
                )
            )

    edges = []
    for trace, nodes in paths.iteritems():
        for node in nodes:
            predecessors = [
                n
                for t, ns in paths.iteritems()
                for n in ns
                if trace_domain.le(t, trace)
                if n.data.___orig in cfg.ancestors(node.data.___orig)
            ]

            for pred in predecessors:
                edges.append(Digraph.Edge(pred, node))

    res_graph = Digraph(
        [n for _, nodes in paths.iteritems() for n in nodes],
        edges
    )

    def print_orig(orig):
        if orig.data.node is not None:
            return (
                '<i>{}</i>'.format(html_render_node(orig.data.node)),
            )
        return tuple()

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(res_graph, [
            dot_printer.DataPrinter('___orig', lambda orig: print_orig(orig))
        ] + [
            dot_printer.DataPrinter(v.name, (
                lambda name: lambda value: (name, str(value),)
            )("{} &isin;".format(v.name)))
            for v, _ in vars_idx.iteritems()
        ]))


def collect_semantics(
        prog, typer, merge_pred_builder,
        output_cfg=None, output_res=None):

    cfg = prog.visit(CFGBuilder())

    if output_cfg:
        save_cfg_to(output_cfg, cfg)

    var_set = set(visitors.findall(prog, lambda n: isinstance(n, Identifier)))
    vars_idx = {v: i for i, v in enumerate(var_set)}
    vars_domain = domains.Product(*(typer[v] for v in var_set))
    trace_domain = SimpleTraceLattice(cfg.nodes)

    evaluator = ExprEvaluator(typer)
    tcs = TrivialIntervalCS(typer, evaluator)

    widening_counter = KeyCounter()
    widening_delay = 10

    def do_widen(counter):
        # will widen when counter == widen_delay, then narrow
        return counter == widening_delay

    do_stmt = VarTracker(var_set, vars_domain, vars_idx, evaluator, tcs)

    lat = domains.Set(
        domains.Product(
            trace_domain,
            vars_domain
        ),
        merge_pred_builder.build(
            trace_domain,
            vars_domain
        )
    )

    def it(states):
        new_states = states.copy()
        print({
            k.name: {
                repr_trace(trace): values
                for trace, values in v
            }
            for k, v in new_states.iteritems()
        })

        for node in cfg.nodes:
            inputs = [new_states[anc] for anc in cfg.ancestors(node)]
            res = (lat.build([(trace_domain.bot, vars_domain.top)])
                   if len(inputs) == 0 else reduce(lat.join, inputs))

            res = lat.build([
                (
                    trace_domain.join(trace, trace_domain.build([node])),
                    updated_val
                )
                for trace, values in res
                for updated_val in [
                    node.data.node.visit(do_stmt, values)
                    if node.data.node is not None else values
                ]
                if not vars_domain.is_empty(updated_val)
            ])

            if node.data.is_widening_point:
                if do_widen(widening_counter.get_incr(node)):
                    res = lat.update(new_states[node], res, True)

            new_states[node] = res

        return new_states

    result = {n: lat.bot for n in cfg.nodes}
    last = None
    while result != last:
        last, result = result, it(result)

    if output_res:
        build_resulting_graph(
            output_res,
            cfg,
            result,
            trace_domain,
            vars_idx
        )

    return result
