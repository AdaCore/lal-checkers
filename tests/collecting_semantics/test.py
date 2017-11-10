import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.irs.basic.tools import Models
from lalcheck.interpretations import default_type_interpreter
from checkers.collecting_semantics import (
    collect_semantics,
    MergePredicateBuilder
)


def do(test_name, merge_predicate, idx=None):
    ctx = lal2basic.new_context()

    progs = lal2basic.extract_programs(
        ctx,
        '{}.adb'.format(test_name)
    )

    model_builder = Models(
        lal2basic.default_typer(ctx),
        default_type_interpreter
    )

    model = model_builder.of(*progs)

    collect_semantics(
        progs[0],
        model,
        merge_predicate,
        'output/{}_cfg.dot'.format(test_name),
        'output/{}{}_res.dot'.format(test_name, '' if idx is None else idx)
    )


le_trace_eq_vals = (
    MergePredicateBuilder.Le_Traces |
    MergePredicateBuilder.Eq_Vals
)

do('test_simple_if', MergePredicateBuilder.Always, '_always')
do('test_simple_if', le_trace_eq_vals, '_le_t_eq_v')

do('test_simple_if_expr', MergePredicateBuilder.Always, '_always')
do('test_simple_if_expr', le_trace_eq_vals, '_le_t_eq_v')

do('test_simple_while', le_trace_eq_vals)

do('test_simple_enum', MergePredicateBuilder.Always, '_always')

do('test_ptr', MergePredicateBuilder.Always, '_always')
do('test_ptr', le_trace_eq_vals, '_le_t_eq_v')
