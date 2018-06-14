import argparse
import os

import lalcheck.ai.irs.basic.frontends.lal as lal2basic
from lalcheck.ai.interpretations import default_type_interpreter
from lalcheck.ai.irs.basic.analyses import abstract_semantics

from lalcheck.ai.irs.basic.tools import Models

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

ctx = lal2basic.ExtractionContext.for_project(
    os.path.join(gnatcoverage_dir, 'gnatcov.gpr'),
    {'BINUTILS_SRC_DIR': '/doesnotexist',
     'BINUTILS_BUILD_DIR': '/doesnotexist'}
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
            "argparse.adb",
            "argparse.ads",
            "binary_files.adb",
            "binary_files.ads",
            "cfg_dump.adb",
            "cfg_dump.ads",
            "check_scos.adb",
            "check_scos.ads",
            "checkpoints.adb",
            "checkpoints.ads",
            "coff.ads",
            "command_line.adb",
            "command_line.ads",
            "commands.adb",
            "commands.ads",
            "convert.adb",
            "convert.ads",
            "coverage-object.adb",
            "coverage-object.ads",
            "coverage-source.adb",
            "coverage-source.ads",
            "coverage-tags.adb",
            "coverage-tags.ads",
            "coverage.adb",
            "coverage.ads",
            "decision_map.adb",
            "decision_map.ads",
            "diagnostics.adb",
            "diagnostics.ads",
            "disa_arm.adb",
            "disa_arm.ads",
            "disa_common.adb",
            "disa_common.ads",
            "disa_lmp.adb",
            "disa_lmp.ads",
            "disa_ppc.adb",
            "disa_ppc.ads",
            "disa_sparc.adb",
            "disa_sparc.ads",
            "disa_symbolize.adb",
            "disa_symbolize.ads",
            "disa_thumb.adb",
            "disa_thumb.ads",
            "disa_x86.adb",
            "disa_x86.ads",
            "disassemble_insn_properties.adb",
            "disassemble_insn_properties.ads",
            "disassemblers.adb",
            "disassemblers.ads",
            "display.adb",
            "display.ads",
            "dwarf.ads",
            "dwarf_handling.adb",
            "dwarf_handling.ads",
            "elf32.adb",
            "elf32.ads",
            "elf64.adb",
            "elf64.ads",
            "elf_common.adb",
            "elf_common.ads",
            "elf_disassemblers.adb",
            "elf_disassemblers.ads",
            "elf_files.adb",
            "elf_files.ads",
            "execs_dbase.adb",
            "execs_dbase.ads",
            "factory_registry.adb",
            "factory_registry.ads",
            "files_table.adb",
            "files_table.ads",
            "gnatcov.adb",
            "hex_images.adb",
            "hex_images.ads",
            "highlighting.adb",
            "highlighting.ads",
            "inputs.adb",
            "inputs.ads",
            "libopcodes_bind/dis_opcodes.ads",
            "mc_dc.adb",
            "mc_dc.ads",
            "object_locations.adb",
            "object_locations.ads",
            "outputs.adb",
            "outputs.ads",
            "pecoff_files.adb",
            "pecoff_files.ads",
            "perf_counters.adb",
            "perf_counters.ads",
            "ppc_descs.adb",
            "ppc_descs.ads",
            "project.adb",
            "project.ads",
            "qemu_traces.ads",
            "qemu_traces_entries__32.ads",
            "rundrv-config.adb",
            "rundrv-config.ads",
            "rundrv.adb",
            "rundrv.ads",
            "sc_obligations.adb",
            "sc_obligations.ads",
            "slocs.adb",
            "slocs.ads",
            "sparc_descs.ads",
            "strings.adb",
            "strings.ads",
            "swaps.adb",
            "swaps.ads",
            "switches.adb",
            "switches.ads",
            "symbols.adb",
            "symbols.ads",
            "traces.adb",
            "traces.ads",
            "traces_dbase.adb",
            "traces_dbase.ads",
            "traces_disa.adb",
            "traces_disa.ads",
            "traces_dump.adb",
            "traces_dump.ads",
            "traces_elf.adb",
            "traces_elf.ads",
            "traces_files.adb",
            "traces_files.ads",
            "traces_files_list.adb",
            "traces_files_list.ads",
            "traces_lines.adb",
            "traces_lines.ads",
            "traces_names.adb",
            "traces_names.ads",
            "traces_stats.adb",
            "traces_stats.ads",
            "version.ads"
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
        ctx.default_typer(lal2basic.unknown_typer),
        default_type_interpreter,
        call_strategies[call_strategy_name].as_def_provider()
    )

    model = model_builder.of(*progs)

    for prog in progs:
        for pred_name, pred in merge_predicates.iteritems():
            ada_prog = prog.data.fun_id
            dir_path = "output/{}/{}".format(pred_name, ada_prog.unit.filename)
            ensure_dir(dir_path)
            analysis = checker(prog, model, pred)
            path = "{}/{}.dot".format(
                dir_path,
                ada_prog.f_subp_spec.f_subp_name.text
            )
            analysis.save_results_to_file(path)


def ensure_dir(path):
    """
    Ensure that the directory at the given path exists. (Creates it if it
    doesn't).

    :param str path: The desired path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def run():
    do_analysis(
        abstract_semantics.compute_semantics,
        default_merge_predicates, 'unknown'
    )


run()
