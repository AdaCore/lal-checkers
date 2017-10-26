import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.irs.basic.tools import DomainCollector
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

    dom_col = DomainCollector(lal2basic.default_domain_gen)
    dom_col.collect_domains(*progs)

    collect_semantics(
        progs[0],
        dom_col,
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

do('test_simple_while', le_trace_eq_vals)
