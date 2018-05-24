import os
import re
import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.interpretations import default_type_interpreter
from lalcheck.irs.basic.tools import Models
from lalcheck.irs.basic.analyses import abstract_semantics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs=1)


default_merge_predicates = {
    'le_t_eq_v': (
        abstract_semantics.MergePredicateBuilder.Le_Traces |
        abstract_semantics.MergePredicateBuilder.Eq_Vals
    ),
    'always': abstract_semantics.MergePredicateBuilder.Always
}


gnatcoverage_dir = os.path.join(os.environ['EXT_SRC'], 'gnatcoverage')

ctx = lal2basic.ExtractionContext(
    os.path.join(gnatcoverage_dir, 'gnatcov.gpr'),
    {'BINUTILS_SRC_DIR': '/doesnotexists',
     'BINUTILS_BUILD_DIR': '/doesnotexists'}
)


def do_analysis(checker, merge_predicates, call_strategy_name):
    args = parser.parse_args()
    if args.file is not None:
        input_files = [os.path.join(gnatcoverage_dir, args.file[0])]
    else:
        input_files = [
            "ali_files.adb",
            "ali_files.ads",
            "annotations-dynamic_html.adb",
            "annotations-dynamic_html.ads",
            "annotations-html.adb",
            "annotations-html.ads",
            "annotations-report.adb",
            "annotations-report.ads",
            "annotations-xcov.adb",
            "annotations-xcov.ads",
            "annotations-xml.adb",
            "annotations-xml.ads",
            "annotations.adb",
            "annotations.ads",
            "arch__32.ads",
            "arch__64.ads",
            "argparse.adb",
            "argparse.ads",
            "binary_files.adb"
        ]
        input_files = [os.path.join(gnatcoverage_dir, f) for f in input_files]

    progs = []
    for i, f in enumerate(input_files):
        print("{:02d}%: {}".format(
            int(((i + 1.0) / len(input_files)) * 100),
            f
        ))
        progs.extend(ctx.extract_programs_from_file(f))

    call_strategies = {
        'unknown': abstract_semantics.UnknownTargetCallStrategy(),
        'topdown': abstract_semantics.TopDownCallStrategy(
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


def run():
    results, model = do_analysis(
        abstract_semantics.compute_semantics,
        default_merge_predicates, 'unknown'
    )

    for pred_name, analysis in results.iteritems():
        print(pred_name)
        print(analysis)
        print("-------------------")


run()
