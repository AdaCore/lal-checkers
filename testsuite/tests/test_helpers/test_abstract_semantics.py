"""
Output the JSON-formatted results of the "abstract semantics" analysis.
"""

import json
import os

import abstract_semantics_helpers
import test_helper
from lalcheck.ai.irs.basic.analyses import abstract_semantics
from lalcheck.ai.irs.basic.purpose import SyntheticVariable


def trace_id(trace):
    return "".join(sorted([n.name for n in trace]))


def format_analysis_results(results, model):
    return json.dumps({
        pred_name: {
            node.name: [
                {
                    'trace:': sorted([n.name for n in trace]),
                    'values': {
                        var.name: model[var].domain.str(value)
                        for var, value in values.iteritems()
                        if not SyntheticVariable.is_purpose_of(var)
                    }
                }
                for trace, values in sorted(
                    state.iteritems(), key=lambda x: trace_id(x[0])
                )
            ]
            for node, state in sorted(
                analysis.semantics.iteritems(), key=lambda x: x[0].name
            )
        }
        for pred_name, analysis in results.iteritems()
    }, sort_keys=True, indent=2)


@test_helper.run
def run(args):
    results, model = abstract_semantics_helpers.do_analysis(
        abstract_semantics.compute_semantics,
        abstract_semantics_helpers.default_merge_predicates,
        args.call_strategy,
        args.typer,
        args.test_subprogram
    )

    if args.output_dir is not None:
        test_helper.ensure_dir(args.output_dir)

        cfg_file = os.path.join(args.output_dir, 'cfg.dot')
        sem_pattern = os.path.join(args.output_dir, 'sem_{}.dot')

    for pred_name, analysis in results.iteritems():
        if args.output_dir is not None:
            analysis.save_cfg_to_file(cfg_file)
            analysis.save_results_to_file(sem_pattern.format(pred_name))

    print(str(format_analysis_results(results, model)))
