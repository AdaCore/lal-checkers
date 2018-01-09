"""
Contain common parts for test helpers that are based on the "collecting
semantics" analysis.
"""
import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.interpretations import default_type_interpreter
from lalcheck.irs.basic.tools import Models
from lalcheck.irs.basic.analyses import collecting_semantics

default_merge_predicates = {
    'le_t_eq_v': (
        collecting_semantics.MergePredicateBuilder.Le_Traces |
        collecting_semantics.MergePredicateBuilder.Eq_Vals
    ),
    'always': collecting_semantics.MergePredicateBuilder.Always
}

ctx = lal2basic.ExtractionContext()


def do_analysis(checker, merge_predicates, call_strategy_name):

    progs = ctx.extract_programs("test.adb")

    call_strategies = {
        'unknown': collecting_semantics.UnknownTargetCallStrategy(),
        'topdown': collecting_semantics.TopDownCallStrategy(
            progs,
            lambda: model,
            lambda: pred
        )
    }

    model_builder = Models(
        ctx.default_typer(),
        default_type_interpreter,
        call_strategies[call_strategy_name].as_def_provider()
    )

    model = model_builder.of(*progs)

    res = {}
    for pred_name, pred in merge_predicates.iteritems():
        res[pred_name] = checker(progs[0], model, pred)

    return res, model
