import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.irs.basic.tools import Models
from lalcheck.interpretations import default_type_interpreter
from checkers.collecting_semantics import MergePredicateBuilder
from checkers.dead_code import check_dead_code


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

    analysis = check_dead_code(progs[0], model, merge_predicate)

    analysis.save_results_to_file(
        'output/{}{}_res.dot'.format(test_name, '' if idx is None else idx)
    )


le_trace_eq_vals = (
    MergePredicateBuilder.Le_Traces |
    MergePredicateBuilder.Eq_Vals
)

do('test_dead_branch_1', MergePredicateBuilder.Always, '_always')

do('test_dead_branch_2', MergePredicateBuilder.Always, '_always')

do('test_dead_branch_3', MergePredicateBuilder.Always, '_always')
do('test_dead_branch_3', le_trace_eq_vals, '_le_t_eq_v')
