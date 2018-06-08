"""
Contain common parts for test helpers that are based on the "abstract
semantics" analysis.
"""
import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.interpretations import default_type_interpreter
from lalcheck.irs.basic.tools import Models
from lalcheck.irs.basic.analyses import abstract_semantics

default_merge_predicates = {
    'le_t_eq_v': (
        abstract_semantics.MergePredicateBuilder.Le_Traces |
        abstract_semantics.MergePredicateBuilder.Eq_Vals
    ),
    'always': abstract_semantics.MergePredicateBuilder.Always
}

ctx = lal2basic.ExtractionContext()

typers = {
    'default': ctx.default_typer(),
    'unknown': lal2basic.unknown_typer
}


def do_analysis(checker,
                merge_predicates='always',
                call_strategy_name='unknown',
                typer='default'):

    progs = ctx.extract_programs_from_file("test.adb")

    call_strategies = {
        'unknown': abstract_semantics.UnknownTargetCallStrategy(),
        'topdown': abstract_semantics.TopDownCallStrategy(
            progs,
            lambda: model,
            lambda: pred
        )
    }

    model_builder = Models(
        typers[typer],
        default_type_interpreter,
        call_strategies[call_strategy_name].as_def_provider()
    )

    model = model_builder.of(*progs)

    res = {}
    for pred_name, pred in merge_predicates.iteritems():
        res[pred_name] = checker(progs[0], model, pred)

    return res, model
