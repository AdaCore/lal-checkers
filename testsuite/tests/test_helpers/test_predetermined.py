"""
Output the JSON-formatted results of the "test always true" checker.
"""

import json
import os

import abstract_semantics_helpers
import test_helper
from lalcheck.checkers.predetermined_test import check_predetermined_tests


def format_analysis_results(results):
    return json.dumps({
        pred_name: [
            {
                'traces': sorted([
                    sorted([n.name for n in trace])
                    for trace in traces
                ], key=lambda t: "".join(t)),
                'kind': always_true
            }
            for traces, _, always_true in analysis.diagnostics
        ]
        for pred_name, analysis in results.iteritems()
    }, sort_keys=True, indent=2)


@test_helper.run
def run(args):
    results, _ = abstract_semantics_helpers.do_analysis(
        check_predetermined_tests,
        abstract_semantics_helpers.default_merge_predicates,
        args.call_strategy,
        args.typer,
        args.test_subprogram
    )

    if args.output_dir is not None:
        test_helper.ensure_dir(args.output_dir)

    for pred_name, analysis in results.iteritems():
        if args.output_dir is not None:
            analysis.analysis_results.save_cfg_to_file(os.path.join(
                args.output_dir,
                'cfg.dot'
            ))
            analysis.analysis_results.save_results_to_file(os.path.join(
                args.output_dir,
                'sem_{}.dot'.format(pred_name)
            ))

    print(str(format_analysis_results(results)))
