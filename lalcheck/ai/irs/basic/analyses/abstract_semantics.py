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
        Given a signature, returns an interpretation of this function as a
        pair which contains:
        - The forward implementation of that function.
        - The backward implementation of that function.

        :param lalcheck.ai.interpretations.Signature sig: The signature for
            which to provide an interpretation.

        :rtype: (*object->object, *object->object)
        """
        raise NotImplementedError

    def as_def_provider(self):
        return def_provider_builder(self.__call__)


class KnownTargetCallStrategy(ExternalCallStrategy):
    def __init__(self, progs):
        self.progs = progs

    def _get_provider(self, sig, prog):
        """
        Given a signature and the program that it designates, returns an
        interpretation of this function as a pair which contains:
        - The forward implementation of that function.
        - The backward implementation of that function.

        :param lalcheck.ai.interpretations.Signature sig: The signature for
            which to provide an interpretation.

        :param lalcheck.ai.irs.basic.tree.Program prog: The IR of the program
            that is designated by the signature.

        :rtype: (*object->object, *object->object)
        """
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

            prog_model = self.get_model(prog)

            analysis = compute_semantics(
                prog,
                prog_model,
                self.get_merge_pred_builder(),
                arg_values
            )

            # Get all environments that can result from the analysis of the
            # function called.
            envs = [
                values
                for leaf in analysis.cfg.leafs()
                for _, values in analysis.semantics[leaf].iteritems()
            ]

            # Fetch the indices of variables that are marked out.
            out_vars = [
                prog.data.param_vars[i]
                for i in sig.out_param_indices
            ]

            # Compute their value after the call using the environments.
            param_values = tuple(
                reduce(
                    prog_model[var].domain.join,
                    (env[var] for env in envs),
                    prog_model[var].domain.bottom
                )
                for var in out_vars
            )

            # Compute the value of the return variable (if any) after the call
            # using the environments.
            result_var = prog.data.result_var
            result_value = (reduce(
                prog_model[result_var].domain.join,
                (env[result_var] for env in envs),
                prog_model[result_var].domain.bottom
            ),) if result_var is not None else ()

            if len(out_vars) == 0 and result_var is not None:
                # We must return the value directly when there is a single
                # element to return instead of a 1-tuple.
                return result_value[0]
            else:
                return param_values + result_value

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
            escape(model[v].domain.str(value))
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
                 evaluator, orig_subp):
        self.cfg = cfg
        self.semantics = semantics
        self.trace_domain = trace_domain
        self.vars_domain = vars_domain
        self.evaluator = evaluator
        self.orig_subp = orig_subp

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
        """
        Given an environment as a map from variable to value (domain element),
        generate a state vector as used by the expression evaluator. A state
        vector maps the INDEX of a variable to its value instead of the mapping
        the variable itself. For efficiency, we use a tuple instead of a dict
        to contain the mapping, although not all indices necessarily have an
        image. Those that don't have their component equal to None.

        :param dict[Variable, object] env: The environment, a map from
            variables to the domain element that they hold.

        :rtype: tuple[object]
        """

        # Because the set of indices may have holes, the maximal index is not
        # necessarily equal to the number of variables minus one. Therefore,
        # we first compute the appropriate length of the list that we will
        # use to store the state.

        max_index = (
            max(v.data.index for v in env.keys()) if len(env) > 0 else 0
        )
        state_vector = [None] * (max_index + 1)

        for var, value in env.iteritems():
            state_vector[var.data.index] = value

        return tuple(state_vector)

    def eval_at(self, node, expr):
        """
        Given a program point, evaluates for each program trace available at
        this program point the given expression.

        :param Digraph.Node node: The program point at which to evaluate the
            expression.

        :param irt.Expr expr: The expression to evaluate using the knowledge
            at that specific program point.

        :rtype: dict[frozenset[Digraph.Node], object]
        """
        return {
            trace: self.evaluator.eval(
                expr,
                self._to_state(env)
            )
            for trace, env in self.semantics[node].iteritems()
        }


_unit_domain = domains.Product()


def compute_semantics(prog, prog_model, merge_pred_builder, arg_values=None):
    evaluator = ExprEvaluator(prog_model)
    solver = ExprSolver(prog_model)

    # setup widening configuration
    visit_counter = KeyCounter()
    widening_delay = 5
    narrowing_delay = 3

    def do_widen(counter):
        # will widen when counter == widen_delay, then narrow. If it has not
        # converged after narrow_delay is reached, widening is triggered again
        # but without a follow-up narrowing.
        return (counter == widening_delay
                or counter >= narrowing_delay + widening_delay)

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
        prog_model[indexed_vars[i]].domain
        if i in indexed_vars else _unit_domain
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
            if do_widen(visit_counter.get_incr(node)):
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
        evaluator,
        prog.data.fun_id
    )
