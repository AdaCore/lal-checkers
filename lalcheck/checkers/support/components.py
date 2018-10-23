from collections import namedtuple

import lalcheck.ai.interpretations as interps
import lalcheck.ai.irs.basic.analyses.abstract_semantics as abstract_analysis
import lalcheck.ai.irs.basic.frontends.lal as lal2basic
import lalcheck.ai.irs.basic.tools as irtools
from lalcheck.ai.irs.basic.visitors import count as ircount
from lalcheck.ai.utils import dataclass

from lalcheck.checkers.support.utils import token_count

from lalcheck.tools.scheduler import Task, Requirement
from lalcheck.tools.logger import log, log_stdout
from lalcheck.tools.parallel_tools import keepalive

import sys
import traceback
import time

ProjectProvider = namedtuple(
    'ProjectProvider', ['project_file', 'scenario_vars', 'target']
)
AutoProvider = namedtuple(
    'AutoProvider', ['files']
)
ModelConfig = namedtuple(
    'ModelConfig', ['typer', 'type_interpreter', 'call_strategy',
                    'merge_predicate_builder']
)


@Requirement.as_requirement
def AnalysisContext(provider_config):
    return [AnalysisContextCreator(provider_config)]


@Requirement.as_requirement
def ExtractionContext(provider_config):
    return [ExtractionContextCreator(provider_config)]


@Requirement.as_requirement
def AnalysisUnit(provider_config, filename):
    return [LALAnalyser(provider_config, filename)]


@Requirement.as_requirement
def IRTrees(provider_config, filename):
    return [IRGenerator(provider_config, filename)]


@Requirement.as_requirement
def IRModel(provider_config, model_config, filenames):
    return [ModelGenerator(provider_config, model_config, filenames)]


@Requirement.as_requirement
def AbstractSemantics(
        provider_config,
        model_config,
        filenames,
        analysis_file):
    return [
        AbstractAnalyser(
            provider_config,
            model_config,
            filenames,
            analysis_file
        )
    ]


@dataclass
class AnalysisContextCreator(Task):
    def __init__(self, provider_config):
        self.provider_config = provider_config

    def requires(self):
        return {}

    def provides(self):
        return {'ctx': AnalysisContext(self.provider_config)}

    def run(self):
        if isinstance(self.provider_config, AutoProvider):
            provider = lal2basic.lal.UnitProvider.auto(
                self.provider_config.files
            )
        else:
            provider = lal2basic.lal.UnitProvider.for_project(
                project_file=self.provider_config.project_file,
                scenario_vars=dict(self.provider_config.scenario_vars),
                target=self.provider_config.target
            )
        return {'ctx': lal2basic.lal.AnalysisContext(unit_provider=provider)}


@dataclass
class ExtractionContextCreator(Task):
    def __init__(self, provider_config):
        self.provider_config = provider_config

    def requires(self):
        return {'ctx': AnalysisContext(self.provider_config)}

    def provides(self):
        return {'res': ExtractionContext(self.provider_config)}

    def run(self, ctx):
        return {'res': lal2basic.ExtractionContext(ctx)}


@dataclass
class LALAnalyser(Task):
    def __init__(self, provider_config, filename):
        self.provider_config = provider_config
        self.filename = filename

    def requires(self):
        return {'ctx': AnalysisContext(self.provider_config)}

    def provides(self):
        return {'res': AnalysisUnit(self.provider_config, self.filename)}

    def _keepalive(self):
        # Libadalang is pretty fast at parsing and will probably never get
        # stuck at this stage, so we let it do its thing for at least 10
        # seconds.
        keepalive(10, self.filename)

    def run(self, ctx):
        try:
            unit = ctx.get_from_file(self.filename)
            if unit.root is not None:
                return {'res': unit}

            log('error', '\n'.join(str(diag) for diag in unit.diagnostics))
        except Exception as e:
            with log_stdout('info'):
                print('error: libadalang failed to analyze {}: {}.'.format(
                    self.filename,
                    e
                ))
                traceback.print_exc(file=sys.stdout)
        return {'res': None}


@dataclass
class IRGenerator(Task):
    def __init__(self, provider_config, filename):
        self.provider_config = provider_config
        self.filename = filename

    def requires(self):
        return {
            'ctx': ExtractionContext(self.provider_config),
            'unit': AnalysisUnit(self.provider_config, self.filename)
        }

    def provides(self):
        return {'res': IRTrees(self.provider_config, self.filename)}

    def _keepalive(self, root):
        # Based on empirical testing. For ~1500 tokens, IR generation takes
        # ~1sec on a good cpu. We add 10 seconds for libadalang to perform
        # name resolution.
        keepalive(token_count(root) / 1500.0 + 10, self.filename)

    def run(self, ctx, unit):
        irtree = None
        if unit is not None:
            self._keepalive(unit.root)

            log('info', 'Transforming {}'.format(self.filename))
            try:
                start_t = time.clock()
                irtree = ctx.extract_programs_from_unit(unit)
                end_t = time.clock()

                log('timings', "Transformation of {} took {}s.".format(
                    self.filename, end_t - start_t
                ))
            except Exception as e:
                with log_stdout('info'):
                    print('error: could not generate IR for file {}: {}.'
                          .format(self.filename, e))
                    traceback.print_exc(file=sys.stdout)
        return {'res': irtree}


@dataclass
class ModelGenerator(Task):
    def __init__(self, provider_config, model_config, filenames):
        self.provider_config = provider_config
        self.model_config = model_config
        self.filenames = filenames

    def requires(self):
        req = {
            'ir_{}'.format(i): IRTrees(self.provider_config, filename)
            for i, filename in enumerate(self.filenames)
        }
        req.update({'ctx': ExtractionContext(self.provider_config)})
        return req

    def provides(self):
        return {
            'model': IRModel(
                self.provider_config,
                self.model_config,
                self.filenames
            )
        }

    @staticmethod
    def get_typer_for(ctx, name):
        if name == 'default':
            return ctx.default_typer()
        elif name == 'default_robust':
            return ctx.default_typer(lal2basic.unknown_typer)
        elif name == 'unknown':
            return lal2basic.unknown_typer
        else:
            raise LookupError('Unknown typer {}'.format(name))

    @staticmethod
    def get_type_interpreter_for(name):
        if name == 'default':
            return interps.default_type_interpreter
        else:
            raise LookupError('Uknown type interpreter {}'.format(name))

    @staticmethod
    def get_call_strategies_for(name, progs, model_getter, mpb_getter):
        unknown_call_strat = (abstract_analysis.UnknownTargetCallStrategy()
                              .as_def_provider())
        if name == 'topdown':
            return abstract_analysis.TopDownCallStrategy(
                progs, model_getter, mpb_getter
            ).as_def_provider(), unknown_call_strat
        elif name == 'unknown':
            return unknown_call_strat,
        else:
            raise LookupError('Unknown call strategy {}'.format(name))

    @staticmethod
    def get_merge_pred_builder_for(name):
        if name == 'le_t_eq_v':
            return (
                abstract_analysis.MergePredicateBuilder.Le_Traces |
                abstract_analysis.MergePredicateBuilder.Eq_Vals
            )
        elif name == 'always':
            return abstract_analysis.MergePredicateBuilder.Always
        else:
            raise LookupError("Unknown merge predicate builder {}".format(
                name
            ))

    def _keepalive(self):
        # Model generation is generally pretty fast and should not get stuck.
        # Therefore, we let it run for a safe amount of time, depending on the
        # number of files.
        keepalive(len(self.filenames))

    def run(self, **kwargs):
        ctx = kwargs['ctx']
        res = None
        self._keepalive()
        try:
            progs = [
                prog
                for key, ir in kwargs.iteritems()
                if key.startswith('ir')
                if ir is not None
                for prog in ir
            ]
            modeler = irtools.Models(
                self.get_typer_for(ctx, self.model_config.typer),
                self.get_type_interpreter_for(
                    self.model_config.type_interpreter
                ),
                *self.get_call_strategies_for(
                    self.model_config.call_strategy,
                    progs,
                    lambda p: models[p],
                    lambda: merge_pred_builder
                )
            )
            models = modeler.of(*progs)
            merge_pred_builder = self.get_merge_pred_builder_for(
                self.model_config.merge_predicate_builder
            )
            res = (models, merge_pred_builder)
        except Exception as e:
            with log_stdout('info'):
                print('error: could not create model: {}.'.format(e))
                traceback.print_exc(file=sys.stdout)
        return {'model': res}


@dataclass
class AbstractAnalyser(Task):
    def __init__(self,
                 provider_config,
                 model_config,
                 filenames,
                 analysis_file):
        self.provider_config = provider_config
        self.model_config = model_config
        self.filenames = filenames
        self.analysis_file = analysis_file

    def requires(self):
        return {
            'ir': IRTrees(self.provider_config, self.analysis_file),
            'model_and_merge_pred': IRModel(
                self.provider_config,
                self.model_config,
                self.filenames
            )
        }

    def provides(self):
        return {
            'res': AbstractSemantics(
                self.provider_config,
                self.model_config,
                self.filenames,
                self.analysis_file
            )
        }

    def _keepalive(self, prog):
        # Based on empirical testing. In most cases, for ~2000 IR nodes,
        # analysis takes ~1sec on a good cpu. However, depending on the
        # cyclomatic complexity of the subprogram, it can take in some cases
        # a lot more than that. Therefore, we say that it can take a maximum of
        # 40 seconds for ~2000 IR nodes.
        # todo: design some heuristics to evaluate complexity of an IR program.
        keepalive(ircount(prog) / 50.0, self.analysis_file)

    def run(self, ir, model_and_merge_pred):
        res = []
        if ir is not None and model_and_merge_pred is not None:
            log('info', 'Analyzing file {}'.format(self.analysis_file))

            model, merge_pred_builder = model_and_merge_pred

            start_t = time.clock()

            for prog in ir:
                self._keepalive(prog)
                fun = prog.data.fun_id
                subp_start_t = time.clock()

                try:
                    res.append(abstract_analysis.compute_semantics(
                        prog, model[prog], merge_pred_builder
                    ))
                except Exception as e:
                    with log_stdout('info'):
                        print('error: analysis of subprocedure {}({}) failed: '
                              '{}.'.format(fun.f_subp_spec.f_subp_name.text,
                                           fun.sloc_range, e))
                        traceback.print_exc(file=sys.stdout)

                subp_end_t = time.clock()

                log(
                    'timings',
                    " - Analysis of subprocedure {} took {}s.".format(
                        fun.f_subp_spec.f_subp_name.text,
                        subp_end_t - subp_start_t
                    )
                )

            end_t = time.clock()
            log('timings', "Analysis of {} took {}s.".format(
                self.analysis_file, end_t - start_t
            ))
            log('progress', 'analyzed {}'.format(self.analysis_file))

        return {'res': res}
