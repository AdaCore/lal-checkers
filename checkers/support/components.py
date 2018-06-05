from collections import namedtuple

import lalcheck.interpretations as interps
import lalcheck.irs.basic.analyses.abstract_semantics as abstract_analysis
import lalcheck.irs.basic.frontends.lal as lal2basic
import lalcheck.irs.basic.tools as irtools
from lalcheck.utils import dataclass
from tools.scheduler import Task, Requirement

ProjectConfig = namedtuple(
    'ProjectConfig', ['project_file', 'scenario_vars']
)
ModelConfig = namedtuple(
    'ModelConfig', ['typer', 'type_interpreter', 'call_strategy',
                    'merge_predicate_builder']
)


@Requirement.as_requirement
def AnalysisContext(project_config):
    return [AnalysisContextCreator(project_config)]


@Requirement.as_requirement
def ExtractionContext(project_config):
    return [ExtractionContextCreator(project_config)]


@Requirement.as_requirement
def AnalysisUnit(project_config, filename):
    return [LALAnalyser(project_config, filename)]


@Requirement.as_requirement
def IRTrees(project_config, filename):
    return [IRGenerator(project_config, filename)]


@Requirement.as_requirement
def IRModel(project_config, model_config, filenames):
    return [ModelGenerator(project_config, model_config, filenames)]


@Requirement.as_requirement
def AbstractSemantics(
        project_config,
        model_config,
        filenames):
    return [
        AbstractAnalyser(
            project_config,
            model_config,
            filenames
        )
    ]


@dataclass
class AnalysisContextCreator(Task):
    def __init__(self, project_config):
        self.project_config = project_config

    def requires(self):
        return {}

    def provides(self):
        return {'ctx': AnalysisContext(self.project_config)}

    def run(self):
        return {'ctx': lal2basic.lal.AnalysisContext(
            unit_provider=lal2basic.lal.UnitProvider.for_project(
                self.project_config.project_file,
                dict(self.project_config.scenario_vars)
            )
        )}


@dataclass
class ExtractionContextCreator(Task):
    def __init__(self, project_config):
        self.project_config = project_config

    def requires(self):
        return {'ctx': AnalysisContext(self.project_config)}

    def provides(self):
        return {'res': ExtractionContext(self.project_config)}

    def run(self, ctx):
        return {'res': lal2basic.ExtractionContext(ctx)}


@dataclass
class LALAnalyser(Task):
    def __init__(self, project_config, filename):
        self.project_config = project_config
        self.filename = filename

    def requires(self):
        return {'ctx': AnalysisContext(self.project_config)}

    def provides(self):
        return {'res': AnalysisUnit(self.project_config, self.filename)}

    def run(self, ctx):
        return {'res': ctx.get_from_file(self.filename)}


@dataclass
class IRGenerator(Task):
    def __init__(self, project_config, filename):
        self.project_config = project_config
        self.filename = filename

    def requires(self):
        return {
            'ctx': ExtractionContext(self.project_config),
            'unit': AnalysisUnit(self.project_config, self.filename)
        }

    def provides(self):
        return {'res': IRTrees(self.project_config, self.filename)}

    def run(self, ctx, unit):
        return {'res': ctx.extract_programs_from_unit(unit)}


@dataclass
class ModelGenerator(Task):
    def __init__(self, project_config, model_config, filenames):
        self.project_config = project_config
        self.model_config = model_config
        self.filenames = filenames

    def requires(self):
        req = {
            'ir_{}'.format(i): IRTrees(self.project_config, filename)
            for i, filename in enumerate(self.filenames)
        }
        req.update({'ctx': ExtractionContext(self.project_config)})
        return req

    def provides(self):
        return {
            'model': IRModel(
                self.project_config,
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
                 project_config,
                 model_config,
                 filenames):
        self.project_config = project_config
        self.model_config = model_config
        self.filenames = filenames

    def requires(self):
        req = {
            'ir_{}'.format(i): IRTrees(self.project_config, filename)
            for i, filename in enumerate(self.filenames)
        }
        req.update({'model': IRModel(
            self.project_config,
            self.model_config,
            self.filenames
        )})
        return req

    def provides(self):
        return {
            'res': AbstractSemantics(
                self.project_config,
                self.model_config,
                self.analyis_config,
                self.filenames
            )
        }

    def run(self, **kwargs):
        model, merge_pred_builder = kwargs['model']
        progs = [
            prog
            for key, ir in kwargs.iteritems()
            if key.startswith('ir')
            if ir is not None
            for prog in ir
        ]

        return {
            'res': [
                abstract_analysis.compute_semantics(
                    prog, model, merge_pred_builder
                )
                for prog in progs
            ]
        }
