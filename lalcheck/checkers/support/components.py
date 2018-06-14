from collections import namedtuple

import lalcheck.ai.interpretations as interps
import lalcheck.ai.irs.basic.analyses.abstract_semantics as abstract_analysis
import lalcheck.ai.irs.basic.frontends.lal as lal2basic
import lalcheck.ai.irs.basic.tools as irtools
from lalcheck.ai.utils import dataclass

from lalcheck.tools.scheduler import Task, Requirement

ProjectProvider = namedtuple(
    'ProjectProvider', ['project_file', 'scenario_vars']
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
                self.provider_config.project_file,
                dict(self.provider_config.scenario_vars)
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

    def run(self, ctx):
        return {'res': ctx.get_from_file(self.filename)}


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

    def run(self, ctx, unit):
        return {'res': ctx.extract_programs_from_unit(unit)}


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
    def get_call_strategy_for(name, progs, model_getter, mpb_getter):
        if name == 'topdown':
            return abstract_analysis.TopDownCallStrategy(
                progs, model_getter, mpb_getter
            )
        elif name == 'unknown':
            return abstract_analysis.UnknownTargetCallStrategy()
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

    def run(self, **kwargs):
        ctx = kwargs['ctx']
        progs = [
            prog
            for key, ir in kwargs.iteritems()
            if key.startswith('ir')
            if ir is not None
            for prog in ir
        ]
        modeler = irtools.Models(
            self.get_typer_for(ctx, self.model_config.typer),
            self.get_type_interpreter_for(self.model_config.type_interpreter),
            self.get_call_strategy_for(
                self.model_config.call_strategy,
                progs,
                lambda: models,
                lambda: merge_pred_builder
            ).as_def_provider()
        )
        models = modeler.of(*progs)
        merge_pred_builder = self.get_merge_pred_builder_for(
            self.model_config.merge_predicate_builder
        )
        return {'model': (models, merge_pred_builder)}


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

    def run(self, ir, model_and_merge_pred):
        model, merge_pred_builder = model_and_merge_pred
        if ir is None:
            return {'res': []}

        return {
            'res': [
                abstract_analysis.compute_semantics(
                    prog, model, merge_pred_builder
                )
                for prog in ir
            ]
        }
