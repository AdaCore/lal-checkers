"""
Contain common parts for test helpers that are based on the "collecting
semantics" analysis.
"""
import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.irs.basic.tools import Models
from lalcheck.interpretations import default_type_interpreter
from checkers.collecting_semantics import (
    MergePredicateBuilder
)


default_merge_predicates = {
    'le_t_eq_v': (
        MergePredicateBuilder.Le_Traces | MergePredicateBuilder.Eq_Vals
    ),
    'always': MergePredicateBuilder.Always
}


def do_analysis(checker, merge_predicates):
    ctx = lal2basic.new_context()

    progs = lal2basic.extract_programs(ctx, "test.adb")

    model_builder = Models(
        lal2basic.default_typer(ctx),
        default_type_interpreter
    )

    model = model_builder.of(*progs)

    return {
        pred_name: checker(progs[0], model, pred)
        for pred_name, pred in merge_predicates.iteritems()
    }
