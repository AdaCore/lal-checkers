import argparse

import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.interpretations import default_type_interpreter
from lalcheck.irs.basic.tools import Models
from lalcheck.irs.basic.analyses import collecting_semantics
import time


def lal_subprogram_info(subp):
    return (
        subp.f_subp_spec.f_subp_name.text,
        subp.f_subp_spec.f_subp_name.sloc_range.start,
    )


class CheckerResults(object):
    def __init__(self, analysis_results, diagnostics):
        self.analysis_results = analysis_results
        self.diagnostics = diagnostics


class Checker(object):
    def __init__(self, checker_name, checker_descr, checker_fun):
        self.checker_name = checker_name
        self.checker_descr = checker_descr
        self.checker_fun = checker_fun

        self.parser = argparse.ArgumentParser(description=self.checker_descr)
        self.parser.add_argument('--output-format', default="codepeer")
        self.parser.add_argument('--path-sensitive', action='store_true')
        self.parser.add_argument('--call-strategy', default="unknown")
        self.parser.add_argument('--project', default=None)
        self.parser.add_argument('--model', default=None)
        self.parser.add_argument('--timings', action='store_true')
        self.parser.add_argument('--print-analysis', action='store_true')
        self.parser.add_argument('file')
        self.args = None

    @staticmethod
    def _default_emit(*_):
        print("warning")

    def _emit_codepeer_message(self, file, line, column, proc_name,
                               proc_file, proc_line, proc_column, msg):
        print("{}:{}:{} warning: {}:{}:{}:{}: {} {}".format(
            file, line, column,
            proc_name, proc_file, proc_line, proc_column,
            msg,
            "[{}]".format(self.checker_name)
        ))

    def report(self, diag):
        raise NotImplementedError

    def position(self, diag):
        raise NotImplementedError

    def run(self):
        start_time = time.clock()

        args = self.args = self.parser.parse_args()

        ctx = lal2basic.ExtractionContext(args.project)

        frontend_start_time = time.clock()

        if args.project is None:
            progs = ctx.extract_programs_from_file(args.file)
        else:
            if args.model is not None:
                ctx.use_model(args.model)

            progs = ctx.extract_programs_from_provider(args.file, 'body')

        call_strategy_unknown = (
            collecting_semantics.UnknownTargetCallStrategy().as_def_provider()
        )

        call_strategy_topdown = collecting_semantics.TopDownCallStrategy(
                progs,
                lambda: model,
                lambda: merge_predicate
            ).as_def_provider()

        call_strategies = {
            'unknown': call_strategy_unknown,
            'topdown': call_strategy_topdown | call_strategy_unknown
        }

        model_gen_start_time = time.clock()

        model_builder = Models(
            ctx.default_typer(lal2basic.unknown_typer),
            default_type_interpreter,
            call_strategies[args.call_strategy]
        )

        model = model_builder.of(*progs)

        if args.path_sensitive:
            merge_predicate = (
                collecting_semantics.MergePredicateBuilder.Le_Traces |
                collecting_semantics.MergePredicateBuilder.Eq_Vals
            )
        else:
            merge_predicate = collecting_semantics.MergePredicateBuilder.Always

        analysis_start_time = time.clock()

        analyses = {
            prog: self.checker_fun(prog, model, merge_predicate)
            for prog in progs
        }

        end_time = time.clock()

        if args.timings:
            print("IR Generation: {} seconds.".format(
                model_gen_start_time - frontend_start_time
            ))
            print("Model Generation: {} seconds.".format(
                analysis_start_time - model_gen_start_time
            ))
            print("Analysis: {} seconds.".format(
                end_time - analysis_start_time
            ))
            print("Total: {} seconds.".format(
                end_time - start_time
            ))

        if args.output_format == 'codepeer':
            emit_message = self._emit_codepeer_message
        else:
            emit_message = self._default_emit

        for prog, analysis in analyses.iteritems():
            prog_info = lal_subprogram_info(prog.data.orig_node)
            if args.print_analysis:
                analysis.analysis_results.save_results_to_file(
                    prog_info[0] + ".dot"
                )

            for diag in analysis.diagnostics:
                pos = self.position(diag)
                msg = self.report(diag)

                if msg is not None and pos is not None:
                    emit_message(
                        args.file, pos.line, pos.column,
                        prog_info[0],
                        args.file, prog_info[1].line, prog_info[1].column,
                        msg
                    )
