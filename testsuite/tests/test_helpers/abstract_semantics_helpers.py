"""
Contain common parts for test helpers that are based on the "abstract
semantics" analysis.
"""
import lalcheck.ai.irs.basic.frontends.lal as lal2basic
from lalcheck.ai.interpretations import default_type_interpreter
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.tools import Models
from test_helper import find_test_program

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
                typer='default',
                test_subprogram_name='Test'):

    progs = ctx.extract_programs_from_file("test.adb")
    test_program = find_test_program(progs, test_subprogram_name)

    call_strategies = {
        'unknown': abstract_semantics.UnknownTargetCallStrategy(),
        'topdown': abstract_semantics.TopDownCallStrategy(
            progs,
            lambda p: model[p],
            lambda: pred
        )
    }

    model_builder = Models(
        typers[typer],
        default_type_interpreter,
        call_strategies[call_strategy_name].as_def_provider()
    )

    model = model_builder.of(*progs)
    prog_model = model[test_program]

    res = {}
    for pred_name, pred in merge_predicates.iteritems():
        res[pred_name] = checker(test_program, prog_model, pred)

    return res, prog_model
