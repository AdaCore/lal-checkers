import argparse

import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.interpretations import default_type_interpreter
from lalcheck.irs.basic.tools import Models
from lalcheck.irs.basic.analyses import collecting_semantics


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
        args = self.args = self.parser.parse_args()
        ctx = lal2basic.ExtractionContext()

        progs = ctx.extract_programs(args.file)

        call_strategies = {
            'unknown': collecting_semantics.UnknownTargetCallStrategy(),
            'topdown': collecting_semantics.TopDownCallStrategy(
                progs,
                lambda: model,
                lambda: merge_predicate
            )
        }

        model_builder = Models(
            ctx.default_typer(),
            default_type_interpreter,
            call_strategies[args.call_strategy].as_def_provider()
        )

        model = model_builder.of(*progs)

        if args.path_sensitive:
            merge_predicate = (
                collecting_semantics.MergePredicateBuilder.Le_Traces |
                collecting_semantics.MergePredicateBuilder.Eq_Vals
            )
        else:
            merge_predicate = collecting_semantics.MergePredicateBuilder.Always

        analyses = {
            prog: self.checker_fun(prog, model, merge_predicate)
            for prog in progs
        }

        if args.output_format == 'codepeer':
            emit_message = self._emit_codepeer_message
        else:
            emit_message = self._default_emit

        for prog, analysis in analyses.iteritems():
            prog_info = lal_subprogram_info(prog.data.orig_node)
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
