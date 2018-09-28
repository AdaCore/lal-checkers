"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal
from lalcheck.ai.utils import profile
from lalcheck.tools.logger import log_stdout, log

import analysis
import typers
import utils
import time
from codegen import ConvertUniversalTypes, gen_ir


unknown_typer = typers.unknown_typer


class ExtractionContext(object):
    """
    The libadalang-based frontend interface. Provides method for extracting
    IR programs from Ada source files (see extract_programs), as well as
    a default typer for those programs (see default_typer).

    Note: programs extracted using different ExtractionContext are not
    compatible. Also, this extraction context must be kept alive as long
    as the programs parsed with it are intended to be used.
    """
    def __init__(self, lal_ctx=None):
        self.lal_ctx = lal.AnalysisContext() if lal_ctx is None else lal_ctx

        # Get a dummy node, needed to call static properties of libadalang.
        dummy = self.lal_ctx.get_from_buffer(
            "<dummy>", 'package Dummy is end;'
        ).root

        # Find the Character TypeDecl.
        char_type = dummy.p_standard_unit.root.find(
            lambda x: x.is_a(lal.TypeDecl) and x.f_name.text == "Character"
        )

        float_type = dummy.p_standard_unit.root.find(
            lambda x: x.is_a(lal.TypeDecl) and x.f_name.text == "Float"
        )

        self.evaluator = utils.ConstExprEvaluator(
            dummy.p_bool_type,
            dummy.p_int_type,
            float_type,
            char_type,
            dummy.p_universal_int_type,
            dummy.p_universal_real_type
        )

        self.type_models = {}
        self.fun_models = {}
        self.subpdata = {}
        self._internal_typer = self.default_typer()

    @staticmethod
    def for_project(project_file, scenario_vars={}):
        return ExtractionContext(lal.AnalysisContext(
            unit_provider=lal.UnitProvider.for_project(
                project_file,
                scenario_vars
            )
        ))

    @staticmethod
    def empty():
        return ExtractionContext()

    def extract_programs_from_file(self, ada_file):
        """
        :param str ada_file: A path to the Ada source file from which to
            extract programs.

        :return: a Basic IR Program for each subprogram body that exists in the
            given source code.

        :rtype: iterable[irt.Program]
        """
        return self._extract_from_unit(self.lal_ctx.get_from_file(ada_file))

    def extract_programs_from_provider(self, name, kind):
        return self._extract_from_unit(self.lal_ctx.get_from_provider(
            name, kind
        ))

    def extract_programs_from_unit(self, unit):
        """
        :param lal.AnalysisUnit unit: The already parsed compilation unit.
        """
        return self._extract_from_unit(unit)

    @profile()
    def use_model(self, name):
        model_unit = self.lal_ctx.get_from_provider(name, "specification")
        with log_stdout('error'):
            for diag in model_unit.diagnostics:
                print('   {}'.format(diag))
                return

        model_ofs = model_unit.root.findall(
            lambda x: (x.is_a(lal.AspectAssoc)
                       and x.f_id.text.lower() == "model_of")
        )

        type_models = {
            aspect.parent.parent.parent: aspect.f_expr.p_referenced_decl
            for aspect in model_ofs
            if aspect.parent.parent.parent.is_a(lal.TypeDecl, lal.SubtypeDecl)
        }

        fun_models = {
            aspect.parent.parent.parent: aspect.f_expr.p_referenced_decl
            for aspect in model_ofs
            if aspect.parent.parent.parent.is_a(lal.SubpDecl)
        }

        if len(type_models) + len(fun_models) != len(model_ofs):
            with log_stdout('info'):
                print("warning: detected usage of Model_Of in an unknown "
                      "context.")

        for tdecl, ref in type_models.iteritems():
            if ref is None:
                with log_stdout('info'):
                    print("warning: Model_Of used on '{}' refers to an unknown"
                          " type.".format(tdecl.f_name.text))
            else:
                self.type_models[ref] = tdecl

        for fdecl, ref in fun_models.iteritems():
            if ref is None:
                with log_stdout('info'):
                    print(
                        "warning: Model_Of used on '{}' refers to an unknown "
                        "procedure/function.".format(
                            fdecl.f_subp_spec.f_subp_name.text
                        )
                    )
            else:
                self.fun_models[ref] = fdecl

    @profile()
    def _extract_from_unit(self, unit):
        if unit.root is None:
            with log_stdout('error'):
                print('Could not parse {}:'.format(unit.filename))
                for diag in unit.diagnostics:
                    print('   {}'.format(diag))
                    return

        unit.populate_lexical_env()

        subpdata = analysis.traverse_unit(unit.root)
        self.subpdata.update(subpdata)

        progs = []
        for subp, subpuserdata in subpdata.iteritems():
            if subp.is_a(lal.BaseSubpBody):
                start_t = time.clock()
                progs.append(
                    gen_ir(self, subp, self._internal_typer, subpuserdata)
                )
                end_t = time.clock()
                log(
                    'timings',
                    " - Transformation of subprocedure {} took {}s".format(
                        subp.f_subp_spec.f_subp_name.text,
                        end_t - start_t
                    )
                )

        converter = ConvertUniversalTypes(self.evaluator, self._internal_typer)

        for prog in progs:
            prog.visit(converter)

        return progs

    def standard_typer(self):
        """
        :return: A Typer for Ada standard types of programs parsed using
            this extraction context.

        :rtype: types.Typer[lal.AdaNode]
        """
        return typers.standard_typer(self)

    def model_typer(self, inner_typer):
        return typers.model_typer(self, inner_typer)

    def default_typer(self, fallback_typer=None):
        """
        :return: The default Typer for Ada programs parsed using this
            extraction context.

        :rtype: types.Typer[lal.AdaNode]
        """
        return typers.default_typer(self, fallback_typer)
