"""
An analyzer that collects semantics at each program point.
"""

from lalcheck.irs.basic.tools import (
    CFGBuilder,
    ExprEvaluator,
    ExprSolver
)

from lalcheck.irs.basic.tree import Variable
from lalcheck.irs.basic.tools import PrettyPrinter
from lalcheck.irs.basic.purpose import SyntheticVariable
from lalcheck.irs.basic import visitors
from lalcheck.utils import KeyCounter
from lalcheck.digraph import Digraph
from lalcheck.interpretations import def_provider
from lalcheck import domains
from lalcheck import dot_printer

from xml.sax.saxutils import escape
from collections import defaultdict


class _VarTracker(visitors.CFGNodeVisitor):
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
            return self.vars_domain.bottom
        else:
            return tuple(env[v] for v in self.vars)

    def visit_assign(self, assign, state):
        env = self.state_to_env(state)
        env[assign.id.var] = self.evaluator.eval(assign.expr, env)
        return self.env_to_state(env)

    def visit_assume(self, assume, state):
        return self.env_to_state(self.constr_solver.solve(
            assume.expr,
            self.state_to_env(state)
        ))

    def visit_read(self, read, state):
        return tuple(self.vars_domain.top[i] if i == self.vars_idx[read.id.var]
                     else x for i, x in enumerate(state))

    def visit_use(self, use, state):
        return state


class _SimpleTraceLattice(domains.FiniteSubsetLattice):
    def __init__(self, *args):
        super(_SimpleTraceLattice, self).__init__(*args)

    def update(self, a, b, widen=False):
        return super(_SimpleTraceLattice, self).update(a, b, False)


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


class ExternalCallStrategy(object):
    def __call__(self, sig):
        """
        :param lalcheck.interpretations.Signature sig:
        :return:
        """
        raise NotImplementedError

    def as_def_provider(self):
        return def_provider(self.__call__)


class KnownTargetCallStrategy(ExternalCallStrategy):
    def __init__(self, progs):
        self.progs = progs

    def _get_provider(self, sig, prog):
        raise NotImplementedError

    def __call__(self, sig):
        for prog in self.progs:
            if sig.name == prog.data.fun_id:
                return self._get_provider(sig, prog)


class UnknownTargetCallStrategy(ExternalCallStrategy):
    def __init__(self):
        super(UnknownTargetCallStrategy, self).__init__()

    def __call__(self, sig):
        def f(*_):
            if len(sig.out_param_indices) == 0:
                return (sig.output_domain.top
                        if sig.output_domain else ())

            return tuple(
                sig.input_domains[i].top
                for i in sig.out_param_indices
            ) + ((sig.output_domain.top,) if sig.output_domain else ())

        def inv(expected, *vals):
            if len(vals) == 1:
                return vals[0]
            else:
                return vals

        return f, inv


class TopDownCallStrategy(KnownTargetCallStrategy):
    def __init__(self, progs, get_model, get_merge_pred_builder):
        super(TopDownCallStrategy, self).__init__(progs)
        self.get_model = get_model
        self.get_merge_pred_builder = get_merge_pred_builder

    def _get_provider(self, sig, prog):
        def f(*args):
            arg_values = {
                param: value
                for param, value in zip(prog.data.param_vars, args)
            }

            model = self.get_model()

            analysis = collect_semantics(
                prog,
                model,
                self.get_merge_pred_builder(),
                arg_values
            )

            envs = [
                values
                for leaf in analysis.cfg.leafs()
                for _, values in analysis.semantics[leaf].iteritems()
            ]

            param_values = [
                reduce(
                    model[var].domain.join,
                    (env[var] for env in envs)
                )
                if var in model
                else arg_values[var]
                for var in prog.data.param_vars
            ]

            result_var = prog.data.result_var
            result_value = reduce(
                model[result_var].domain.join,
                (env[result_var] for env in envs)
            ) if result_var is not None else None

            if len(sig.out_param_indices) == 0:
                return (result_value
                        if sig.output_domain else ())

            return tuple(
                param_values[i]
                for i in sig.out_param_indices
            ) + ((result_value,) if sig.output_domain else ())

        def inv(*_):
            raise NotImplementedError

        return f, inv


def _html_render_node(node):
    return escape(PrettyPrinter.pretty_print(node))


def _save_cfg_to(file_name, cfg):
    def render_node(node):
        return (_html_render_node(node),) if node is not None else ()

    def render_widening_point(is_widening_point):
        return (escape('<widening_point>'),) if is_widening_point else ()

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(cfg, [
            dot_printer.DataPrinter('node', render_node),
            dot_printer.DataPrinter('is_widening_point', render_widening_point)
        ]))


def _build_resulting_graph(file_name, cfg, results, trace_domain, model):
    paths = defaultdict(list)

    var_set = {
        v
        for node, state in results.iteritems()
        for trace, values in state.iteritems()
        for v, _ in values.iteritems()
    }

    for node, state in results.iteritems():
        for trace, values in state.iteritems():
            paths[frozenset(trace)].append(
                Digraph.Node(
                    node.name,
                    ___orig=node,
                    **{
                        v.name: value
                        for v, value in values.iteritems()
                        if not SyntheticVariable.is_purpose_of(v)
                    }
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
                '<i>{}</i>'.format(_html_render_node(orig.data.node)),
            )
        return ()

    def print_result_builder(v):
        return lambda value: (
            "{} &isin;".format(v.name),
            model[v].domain.str(value)
        )

    with open(file_name, 'w') as f:
        f.write(dot_printer.gen_dot(res_graph, [
            dot_printer.DataPrinter('___orig', print_orig)
        ] + [
            dot_printer.DataPrinter(v.name, print_result_builder(v))
            for v in var_set
        ]))


class AnalysisResults(object):
    """
    Contains the results of the collecting semantics analysis.
    """
    def __init__(self, cfg, semantics, trace_domain, vars_domain,
                 evaluator):
        self.cfg = cfg
        self.semantics = semantics
        self.trace_domain = trace_domain
        self.vars_domain = vars_domain
        self.evaluator = evaluator

    def save_cfg_to_file(self, file_name):
        """
        Prints the control-flow graph as a DOT file to the given file name.
        """
        _save_cfg_to(file_name, self.cfg)

    def save_results_to_file(self, file_name):
        """
        Prints the resulting graph as a DOT file to the given file name.
        Displays the state of each variable at each program point.
        """
        _build_resulting_graph(
            file_name,
            self.cfg,
            self.semantics,
            self.trace_domain,
            self.evaluator.model
        )

    def eval_at(self, node, expr):
        """
        Given a program point, evaluates for each program trace available at
        this program point the given expression.
        """
        return {
            trace: self.evaluator.eval(expr, env)
            for trace, env in self.semantics[node].iteritems()
        }


def collect_semantics(prog, model, merge_pred_builder, arg_values=None):

    cfg = prog.visit(CFGBuilder())

    var_set = set(visitors.findall(prog, lambda n: isinstance(n, Variable)))

    vars_idx = {v: i for i, v in enumerate(var_set)}
    vars_domain = domains.Product(*(model[v].domain for v in var_set))
    trace_domain = _SimpleTraceLattice(cfg.nodes)

    evaluator = ExprEvaluator(model)
    solver = ExprSolver(model)

    widening_counter = KeyCounter()
    widening_delay = 10

    def do_widen(counter):
        # will widen when counter == widen_delay, then narrow
        return counter == widening_delay

    do_stmt = _VarTracker(var_set, vars_domain, vars_idx, evaluator, solver)

    lat = domains.Set(
        domains.Product(
            trace_domain,
            vars_domain
        ),
        merge_pred_builder.build(
            trace_domain,
            vars_domain
        ),
        None  # We don't need a top element here.
    )

    vars_top = vars_domain.build(*(
        arg_values[var]
        if arg_values is not None and var in arg_values
        else model[var].domain.top
        for var in var_set
    ))

    def it(states):
        new_states = states.copy()

        for node in cfg.nodes:
            inputs = [new_states[anc] for anc in cfg.ancestors(node)]
            res = (lat.build([(trace_domain.bottom, vars_top)])
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

    result = {n: lat.bottom for n in cfg.nodes}
    last = None
    while result != last:
        last, result = result, it(result)

    formatted_results = {
        node: {
            trace: {
                v: values[vars_idx[v]] for v in var_set
            }
            for trace, values in state
        }
        for node, state in result.iteritems()
    }

    return AnalysisResults(
        cfg,
        formatted_results,
        trace_domain,
        vars_domain,
        evaluator
    )
