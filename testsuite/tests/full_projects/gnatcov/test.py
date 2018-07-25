import argparse
import os
from lalcheck import checker_runner

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, nargs=1)

gnatcoverage_dir = os.path.join(os.environ['EXT_SRC'], 'gnatcoverage')
project_file = os.path.join(gnatcoverage_dir, 'gnatcov.gpr')
scenario_vars = {'BINUTILS_SRC_DIR': '/doesnotexist',
                 'BINUTILS_BUILD_DIR': '/doesnotexist'}

checkers = [
    'lalcheck.checkers.invalid_contract',
    'lalcheck.checkers.dead_code',
    'lalcheck.checkers.null_dereference',
    'lalcheck.checkers.invalid_discriminant',
    'lalcheck.checkers.bad_unequal',
    'lalcheck.checkers.same_logic',
    'lalcheck.checkers.same_operands',
    'lalcheck.checkers.same_test',
    'lalcheck.checkers.same_then_else'
]


def run():
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

    args = (['-P', project_file] +
            ['-X{}={}'.format(*entry) for entry in scenario_vars.iteritems()] +
            ['--checkers', '{}'.format(';'.join([c for c in checkers]))] +
            ['--files', '{}'.format(';'.join(input_files))] +
            ['--log', 'error'])

    diags = checker_runner.run(args, diagnostic_action='return')
    diags = [
        "{}:{}: {}".format(
            pos.start[0],
            pos.start[1],
            msg
        )
        for pos, msg, _, _ in diags
    ]
    diags.sort()

    for diag in diags:
        print(diag)


run()
