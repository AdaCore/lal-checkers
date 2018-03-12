"""
Output the JSON-formatted results of the "abstract semantics" analysis.
"""

from checkers.contract_checker import check_contracts
import abstract_semantics_helpers
import test_helper

import json
import os


def format_analysis_results(results):
    return json.dumps({
        pred_name: [
            {
                'trace': sorted([n.name for n in trace]),
                'precise': precise
            }
            for trace, _, precise in analysis.diagnostics
        ]
        for pred_name, analysis in results.iteritems()
    }, sort_keys=True, indent=2)


@test_helper.run
def run(args):
    results, _ = abstract_semantics_helpers.do_analysis(
        check_contracts,
        abstract_semantics_helpers.default_merge_predicates,
        args.call_strategy
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
            analysis.save_results_to_file(os.path.join(
                args.output_dir,
                'violated_contracts_{}.dot'.format(pred_name)
            ))

    print(str(format_analysis_results(results)))
