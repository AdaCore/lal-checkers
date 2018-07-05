"""
An analyzer that collects semantics at each program point.
"""

from collections import defaultdict
from xml.sax.saxutils import escape

from lalcheck.ai import domains
from lalcheck.ai.interpretations import def_provider_builder
from lalcheck.ai.irs.basic import visitors
from lalcheck.ai.irs.basic.purpose import SyntheticVariable
from lalcheck.ai.irs.basic.tree import Variable
from lalcheck.ai.utils import KeyCounter, concat_dicts
from lalcheck.tools import dot_printer
from lalcheck.tools.digraph import Digraph

from lalcheck.ai.irs.basic.tools import (
    CFGBuilder,
    ExprEvaluator,
    ExprSolver
)
from lalcheck.ai.irs.basic.tools import PrettyPrinter


def updated_state(state, var, value):
    return tuple(
        value if i == var.data.index else x
        for i, x in enumerate(state)
    )


class _VarTracker(visitors.CFGNodeVisitor):
    def __init__(self, var_set, vars_domain, evaluator, c_solver):
        self.vars = var_set
        self.evaluator = evaluator
        self.constr_solver = c_solver
        self.vars_domain = vars_domain

    def visit_assign(self, assign, state):
        return updated_state(
            state,
            assign.id.var,
            self.evaluator.eval(assign.expr, state)
        )

    def visit_assume(self, assume, state):
        new_state, success = self.constr_solver.solve(assume.expr, state)
        return new_state if success else self.vars_domain.bottom

    def visit_read(self, read, state):
        return updated_state(
            state,
            read.id.var,
            self.evaluator.model[read.id.var].domain.top
        )


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
        return def_provider_builder(self.__call__)


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

            analysis = compute_semantics(
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
                    (env[var] for env in envs),
                    model[var].domain.bottom
                )
                if var in model
                else arg_values[var]
                for var in prog.data.param_vars
            ]

            result_var = prog.data.result_var
            result_value = reduce(
                model[result_var].domain.join,
                (env[result_var] for env in envs),
                model[result_var].domain.bottom
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
    Contains the results of the abstract semantics analysis.
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

    @staticmethod
    def _to_state(env):
        last_index = max([v.data.index for v in env.keys()]) + 1
        state_list = [None] * last_index

        for var, value in env.iteritems():
            state_list[var.data.index] = value

        return tuple(state_list)

    def eval_at(self, node, expr):
        """
        Given a program point, evaluates for each program trace available at
        this program point the given expression.
        """
        return {
            trace: self.evaluator.eval(
                expr,
                self._to_state(env)
            )
            for trace, env in self.semantics[node].iteritems()
        }


_unit_domain = domains.Product()


def compute_semantics(prog, model, merge_pred_builder, arg_values=None):
    evaluator = ExprEvaluator(model)
    solver = ExprSolver(model)

    # setup widening configuration
    widening_counter = KeyCounter()
    widening_delay = 10

    def do_widen(counter):
        # will widen when counter == widen_delay, then narrow
        return counter == widening_delay

    cfg = prog.visit(CFGBuilder())
    roots = cfg.roots()
    non_roots = [n for n in cfg.nodes if n not in roots]

    # find the variables that appear in the program
    var_set = set(visitors.findall(prog, lambda n: isinstance(n, Variable)))

    # build an index
    indexed_vars = {var.data.index: var for var in var_set}
    last_index = max(indexed_vars.keys()) if len(indexed_vars) > 0 else -1

    # define the variables domain
    vars_domain = domains.Product(*(
        model[indexed_vars[i]].domain if i in indexed_vars else _unit_domain
        for i in range(last_index + 1)
    ))

    # define the trace domain
    trace_domain = _SimpleTraceLattice(cfg.nodes)

    # define the State domain that we track at each program point.
    lat = domains.Powerset(
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

    # the transfer function
    transfer_func = _VarTracker(var_set, vars_domain, evaluator, solver)

    def transfer(new_states, node, inputs):
        transferred = (
            (
                trace,
                node.data.node.visit(transfer_func, values)
                if node.data.node is not None else values
            )
            for trace, values in inputs
        )

        output = lat.build([
            (
                trace_domain.join(trace, trace_domain.build([node])),
                values
            )
            for trace, values in transferred
            if not vars_domain.is_empty(values)
        ])

        if node.data.is_widening_point:
            if do_widen(widening_counter.get_incr(node)):
                output = lat.update(new_states[node], output, True)

        return output

    def it(states):
        new_states = states.copy()

        for node in non_roots:
            new_states[node] = transfer(new_states, node, reduce(
                lat.join,
                (new_states[anc] for anc in cfg.ancestors(node))
            ))

        return new_states

    # initial state of the variables at the entry of the program
    init_vars = tuple(
        arg_values[indexed_vars[i]]
        if (i in indexed_vars and
            arg_values is not None and
            indexed_vars[i] in arg_values)
        else vars_domain.domains[i].top
        for i in range(last_index + 1)
    )

    # initial state at the the entry of the program
    init_lat = lat.build([(trace_domain.bottom, init_vars)])

    # last state of the program (all program points)
    last = concat_dicts(
        {n: transfer({}, n, init_lat) for n in roots},
        {n: lat.bottom for n in non_roots}
    )

    # current state of the program (all program points)
    result = it(last)

    # find a fix-point.
    while any(not lat.eq(x, result[i]) for i, x in last.iteritems()):
        last, result = result, it(result)

    formatted_results = {
        node: {
            trace: {
                v: values[v.data.index] for v in var_set
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
