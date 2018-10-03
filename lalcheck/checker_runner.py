import argparse
import importlib
from multiprocessing import cpu_count
from itertools import izip_longest
from functools import partial
from collections import defaultdict
import traceback
import sys

from lalcheck.checkers.support.checker import (
    Checker, CheckerResults, ProviderConfig
)
from lalcheck.tools.digraph import Digraph
from lalcheck.tools.dot_printer import gen_dot, DataPrinter
from lalcheck.tools.scheduler import Scheduler
from lalcheck.tools.parallel_tools import parallel_map, keepalive
from lalcheck.tools import logger

parser = argparse.ArgumentParser(description='lal-checker runner.')

provider_group = parser.add_mutually_exclusive_group(required=False)
provider_group.add_argument('-P', default=None, metavar='FILE_PATH', type=str,
                            help="The path to the project file to use.")
provider_group.add_argument('--provider-files-from', default=None,
                            metavar='FILE_PATH', type=str,
                            help="The path to a file containing a list of "
                                 "files that are dependencies of the files to "
                                 "analyze.")
provider_group.add_argument('--provider-files', default=None,
                            metavar='FILE_PATHS', type=str,
                            help="A list of files that are dependencies of the"
                                 " files to analyze.")

parser.add_argument('-X', action='append', metavar='VAR=STR', type=str,
                    default=[])
parser.add_argument('--target', default=None, metavar="TARGET", type=str,
                    help="The target to use. Overrides the one given in the "
                         "project file, if any.")

files_group = parser.add_mutually_exclusive_group(required=True)
files_group.add_argument('--files-from', metavar='FILE_PATH', type=str)
files_group.add_argument('--files', metavar='FILE_PATHS', action='append',
                         type=str, help='File paths separated by semicolons.')
files_group.add_argument('--list-categories', action='store_true',
                         help='Prints a list of message kinds together with: '
                              'the subset of checkers that can actually '
                              'output this message kind; a description of '
                              'this message kind')
files_group.add_argument('--checkers-help', action='store_true',
                         help='Prints a help message for each checker '
                              'passed through --checkers of --checkers-from.')

checkers_group = parser.add_mutually_exclusive_group(required=True)
checkers_group.add_argument('--checkers-from', metavar='FILE_PATH', type=str)
checkers_group.add_argument('--checkers', metavar='CHECKERS', action='append',
                            type=str, help='Checker commands separated by'
                                           'semicolons.')

parser.add_argument('--log', metavar="CATEGORIES", type=str,
                    default="progress;error;internal-error;diag-{}".format(
                        CheckerResults.HIGH
                    ),
                    help='Categories separated by semicolons.')
parser.add_argument('--log-to-file', metavar=("FILE", "CATEGORIES"), nargs=2,
                    type=str, default=[], action='append',
                    help='The first argument is the file name in which to log.'
                         ' The second argument is a list of categories '
                         'separated by semicolons.')

parser.add_argument('--codepeer-output', action='store_true')
parser.add_argument('--export-schedule', type=str)
parser.add_argument('--partition-size', default=10, type=int,
                    help='The amount of files that will be batched in a'
                         'partition. A higher number means less computing'
                         'time, but more memory consumption.')
parser.add_argument('-j', default=1, type=int,
                    help='The number of process to spawn in parallel, each'
                         'of which deals with a single partition at a time.')
parser.add_argument('--timeout-factor', default=10.0, type=float,
                    help="This allows processes to live longer than the "
                         "expected timeout. You may increase this number on "
                         "slower machines to ensure that processes are not "
                         "killed too early.")


BUILT_IN_CHECKERS_FORMAT = 'lalcheck.checkers.{}'


def lines_from_file(filename):
    """
    Returns the list of lines of the file.
    If the file cannot be opened, an error is logged and an empty list is
    returned.

    :param str filename: The path to the file to read.
    :rtype: list[str]
    """
    try:
        with open(filename) as f:
            return [l.strip() for l in f.readlines()]
    except EnvironmentError:
        logger.log('error', 'error: cannot read file {}'.format(filename))
        return []


def get_line_count(filename):
    """
    Returns the number of lines ('\n') in the given file.
    """
    try:
        with open(filename, 'r') as f:
            return sum(1 for _ in f)
    except EnvironmentError:
        return 0


def sort_files_by_line_count(filenames):
    """
    Sorts the given list of files in descending order according to the number
    of lines in each file (returns a new list).

    :type filenames: list[str] | None
    """
    if filenames is None:
        return []

    file_lines = [(f, get_line_count(f)) for f in filenames]
    file_lines.sort(key=lambda x: x[1], reverse=True)
    return [f for f, l in file_lines]


def clear_file(fname):
    """
    Erases all of the content of the given file.

    :param str fname: The path to the file to clear.
    """
    with open(fname, 'w+'):
        pass


def set_logger(args):
    filters = {f: sys.stdout for f in args.log.split(';')}

    for filenames_categories in args.log_to_file:
        # filenames_categories is the flattened list
        # [file_1, cats_1, file_2, cats_2, ...]

        unflattened = [
            filenames_categories[i:i+2]
            for i in range(0, len(filenames_categories), 2)
        ]
        # unflattened is the list:
        # [(file_1, cats_1), (file_2, cats_2), ...]

        for fname, categories in unflattened:
            clear_file(fname)
            log_file = open(fname, 'a')  # file will be closed upon destruction
            for category in categories.split(';'):
                filters[category] = log_file

    logger.set_logger(logger.Logger(filters))


def import_checker(module_path):
    """
    Tries to import the checker defined in the given module path. If the
    exact path does not resolve to a python module, a second attempt is done
    assuming that the given module path is actually the name of one of the
    built-in checkers.

    :param str module_path: path to the python module containing the checker.
    :raise ImportError: when the module could not be imported.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return importlib.import_module(
            BUILT_IN_CHECKERS_FORMAT.format(module_path)
        )


def commands_from_file_or_list(file_path, commands):
    """
    Returns a list of commands, by taking them either from a file (if
    file_path is not None) or from a list of strings.

    If both are None, returns None.

    :param str | None file_path: The path to a file that contains a list of
        commands on each line.
    :param list[str] | None commands: The list of command strings, where each
        string holds multiple commands separated by semicolons.
    :rtype: list[str] | None
    """
    if file_path is not None:
        return lines_from_file(file_path)
    elif commands is not None:
        return [
            c
            for cs in commands
            for c in cs.split(';')
        ]
    else:
        return None


def get_working_checkers(args):
    """
    Retrieves the list of checkers to run from the information passed as
    command-line arguments using the --checkers or --checkers-from options.

    The return value is a list of pairs of (Checker, list[str]) corresponding
    to the actual checker objects to run together with a list of arguments
    specific to each checker run. Additionnally, a boolean is returned to
    indicate whether all checkers were successfully loaded or not.

    Errors may be reported if the specified checkers are not found or do
    not export the checker interface.

    :param argparse.Namespace args: The command-line arguments.
    :rtype: list[(Checker, list[str])], bool
    """

    checker_commands = commands_from_file_or_list(
        args.checkers_from, args.checkers
    )
    split_commands = [command.split() for command in checker_commands]

    checkers = []

    for checker_args in split_commands:
        # checker_args[0] is the python module to the checker. The rest are
        # additional arguments that must be passed to that checker.
        try:
            checker_module = import_checker(checker_args[0])
        except ImportError:
            logger.log('error', 'Failed to import checker module {}.'.format(
                checker_args[0]
            ))
        else:
            if not hasattr(checker_module, 'checker'):
                logger.log('error', 'Checker {} does not export a "checker" '
                                    'object.'.format(checker_module))
            elif not issubclass(checker_module.checker, Checker):
                logger.log('error',
                           'Checker {} does not inherit the '
                           '"lalcheck.checkers.support.checker.Checker" '
                           'interface.'.format(checker_module))
            else:
                checkers.append((checker_module.checker, checker_args[1:]))

    return checkers, len(checkers) == len(checker_commands)


def create_provider_config(args, analysis_files):
    """
    Creates a ProviderConfig object using the switches have been passed
    as argument, among "-P", "-X", "--target", "--provider-files[-from]".

    Note that if the "--provider-files[-from]" switches have not been
    specified, it uses the set of files to analyze as provider files.

    :param argparse.Namespace args: The command-line arguments.
    :param list[str] analysis_files: The files to analyze.
    :rtype: ProviderConfig
    """
    project_file = args.P
    scenario_vars = dict([eq.split('=') for eq in args.X])
    provider_files = set(commands_from_file_or_list(
        args.provider_files_from, args.provider_files
    ) or [])
    target = args.target

    # Make sure that files to analyze are part of the auto provider.
    provider_files.update(analysis_files)

    if project_file is None and len(scenario_vars) > 0:
        logger.log('error', "warning: use of scenario vars without a "
                            "project file.")

    if project_file is None and target is not None:
        logger.log('error', "warning: specifying a target without a "
                            "project file.")

    return ProviderConfig(
        project_file=project_file,
        scenario_vars=tuple(scenario_vars.iteritems()),
        provider_files=tuple(provider_files),
        target=target
    )


def get_requirements(provider_config, checkers, files_to_check):
    """
    Returns a list of requirements corresponding to the execution of the
    specified checkers on the specified list of files.

    For example, the checker null_dereference creates a requirement NullDerefs,
    which in itself contains all the information necessary to compute the list
    of null dereferences in the given list of files: NullDerefs says that it
    can be fulfilled with the NullDerefFinder task, which itself depends on
    other requirements, etc. The scheduler then takes care of arranging the
    tasks in order to fulfill all the necessary requirements.

    :param ProviderConfig provider_config: The provider configuration.
    :param list[(Checker, list[str])] checkers: The checkers to run.
    :param list[str] files_to_check: The files to analyze.
    :rtype: list[lalcheck.tools.scheduler.Requirement]
    """
    requirements = []

    for checker, checker_args in checkers:
        requirements.append(checker.create_requirement(
            provider_config=provider_config,
            analysis_files=tuple(files_to_check),
            args=checker_args
        ))

    return requirements


def get_schedules(requirements):
    """
    Given a list of requirements, creates a list of schedule such that every
    requirement is fulfilled by each schedule.

    :param list[lalcheck.tools.scheduler.Requirement] requirements: The
        requirements to fulfill.
    :rtype: list[lalcheck.tools.scheduler.Schedule]
    """
    scheduler = Scheduler()
    return scheduler.schedule({
        'res_{}'.format(i): req
        for i, req in enumerate(requirements)
    })


def export_schedule(schedule, export_path):
    """
    Exports the given schedule as a dot graph. Each horizontally aligned
    elements can be ran in parallel.

    :param lalcheck.tools.scheduler.Schedule schedule: The schedule to export.
    :param str export_path: The path to the dot file to write. The .dot
        extension is appended here.
    """
    nice_colors = [
        '#093145', '#107896', '#829356', '#3C6478', '#43ABC9', '#B5C689',
        '#BCA136', '#C2571A', '#9A2617', '#F26D21', '#C02F1D', '#F58B4C',
        '#CD594A'
    ]

    def var_printer(var):
        name = str(var)
        color = nice_colors[hash(name) % len(nice_colors)]

        def colored(x):
            return '<font color="{}">{}</font>'.format(color, x)

        def printer(x):
            return (
                '<i>{}</i>'.format(colored(name)),
                colored(x)
            )

        return printer

    tasks = frozenset(
        task
        for batch in schedule.batches
        for task in batch
    )

    varset = frozenset(
        var
        for task in tasks
        for var in vars(task).keys()
    )

    task_to_node = {
        task: Digraph.Node(
            task.__class__.__name__,
            **vars(task)
        )
        for task in tasks
    }

    edges = [
        Digraph.Edge(task_to_node[a], task_to_node[b])
        for a in tasks
        for b in tasks
        if len(frozenset(a.provides().values()) &
               frozenset(b.requires().values())) > 0
    ]

    digraph = Digraph(task_to_node.values(), edges)
    dot = gen_dot(digraph, [
        DataPrinter(
            var,
            var_printer(var)
        )
        for var in varset
    ])
    with open("{}.dot".format(export_path), 'w') as export_file:
        export_file.write(dot)


def report_diag(args, report):
    """
    Given a diagnostic report, creates the final string to output to the user.

    :param argparse.Namespace args: The command-line arguments.
    :param (DiagnosticPosition, str, MessageKind, str) report: The report to
        output.
    :rtype: str
    """
    pos, msg, kind, _ = report

    if args.codepeer_output:
        return "{}:{}:{}: warning: {}:{}:{}:{}: {} [{}]".format(
            pos.filename, pos.start[0], pos.start[1],
            pos.proc_name or "unknown", pos.proc_filename or "unknown",
            pos.proc_start[0] if pos.proc_start is not None else "unknown",
            pos.proc_start[1] if pos.proc_start is not None else "unknown",
            msg, kind.name()
        )
    else:
        return "{}:{}:{} {}".format(
            pos.filename, pos.start[0], pos.start[1], msg
        )


def list_categories(checkers):
    """
    :param list[(Checker, list[str])] checkers: The checkers for which
        to output information about the kind of messages they can output.
    """
    kinds_map = defaultdict(list)

    for checker, _ in checkers:
        for kind in checker.kinds():
            kinds_map[kind].append(checker)

    for kind, checkers in kinds_map.iteritems():
        print("{} ({}) - {}".format(
            kind.name(),
            ", ".join(c.name() for c in checkers),
            kind.description()
        ))


def print_checkers_help(checkers):
    """
    Prints the usage of the given list of checkers.
    :param list[Checker] checkers: The checkers for which to print usage.
    """
    for i, (checker, _) in enumerate(checkers):
        checker_parser = checker.get_arg_parser()
        checker_parser.description = checker.description()
        checker_parser.prog = checker.name()
        checker_parser.print_help()
        if i != len(checkers) - 1:
            print('\n')
            print('_' * 80)
            print('\n')


def on_timeout(file_cause):
    """
    Function that will be called if one of the partition times out.

    :param string file_cause: The filename of the file that caused the time
        out. We know that the cause will be a file in this case because that
        is how checkers are using the timeout mechanism.
    """
    if file_cause is not None:
        error_msg = "process timed out due to file {}.".format(file_cause)
    else:
        error_msg = "process timed out."

    logger.log('error', error_msg)


def do_partition(args, provider_config, checkers, partition):
    """
    Runs the checkers on a single partition of the whole set of files.
    Returns a list of diagnostics.

    :param argparse.Namespace args: The command-line arguments.
    :param ProviderConfig provider_config: The provider configuration.
    :param list[(Checker, list[str])] checkers: The list of checkers to run
        together with their specific arguments.
    :param (int, list[str]) partition: The index of that partition and the list
        of files that make up that partition.
    :rtype: list[(DiagnosticPosition, str, MessageKind, str)]
    """
    set_logger(args)
    keepalive(2)

    diags = []
    index, files = partition

    logger.log(
        'debug',
        "started partition {} with files {}".format(index, files)
    )

    try:
        reqs = get_requirements(provider_config, checkers, files)
        schedule = get_schedules(reqs)[0]

        if args.export_schedule is not None:
            export_schedule(
                schedule, "{}{}".format(args.export_schedule, index)
            )

        for checker_result in schedule.run().values():
            for program_result in checker_result:
                for diag in program_result.diagnostics:
                    report = program_result.diag_report(diag)
                    if report is not None:
                        diags.append(report)
    except Exception:
        with logger.log_stdout('internal-error'):
            traceback.print_exc(file=sys.stdout)
    finally:
        logger.log('debug', "completed partition {}".format(index))
        return diags


def do_all(args, diagnostic_action):
    """
    Main routine of the driver. Uses the user-provided arguments to run
    checkers on a list of files. Takes care of parallelizing the process,
    distributing the tasks, etc.

    :param argparse.Namespace args: The command-line arguments.
    :param str diagnostic_action: Action to perform with the diagnostics that
        are produced by the checkers. Can be one of:
         - 'return': Return them as a list.
         - 'log': Output them in the logger.
    """
    args.j = cpu_count() if args.j <= 0 else args.j
    working_files = sort_files_by_line_count(
        commands_from_file_or_list(args.files_from, args.files)
    )
    provider_config = create_provider_config(args, working_files)
    ps = args.partition_size
    ps = len(working_files) / args.j if ps == 0 else ps
    ps = max(ps, 1)

    def compute_partitions():
        # Input: list of files sorted by line count: [file_1, ..., file_n]

        # Step 1: Distribute them among the j cores: [
        #     [file_1, file_{1+j}, file_{1+2j}, ...],
        #     [file_2, file_{2+j}, file_{2+2j}, ...],
        #     ...,
        #     [file_j, file_{2j}, file_{3j}, ...]
        # ]
        process_parts = [working_files[i::args.j] for i in range(args.j)]

        # Step 2: Split each in lists of maximum length ps: [
        #     [[file_1, ..., file_{1+(ps-1)*j}], [file_{1+ps*j}, ...], ...}]],
        #     [[file_2, ..., file_{2+(ps-1)*j}], [file_{2+ps*j}, ...], ...],
        #     ...,
        #     [[file_j, ..., file_{ps*j}], [file_{(ps+1)*j}, ...], ...]
        # ]
        split_parts = [
            [part[i:i+ps] for i in range(0, len(part), ps)]
            for part in process_parts
        ]

        # Final step: Transpose and flatten the above matrix to make sure each
        # core gets a partition of size ps: [
        #     [file_1, ..., file_{1+(ps-1)*j}],
        #     [file_2, ..., file_{2+(ps-1)*j}],
        #     ...,
        #     [file_j, ..., file_{ps*j}],
        #
        #     ...,
        #
        #     [file_{1+ps*j}, ...],
        #     [file_{2+ps*j}, ...],
        #     ...,
        #     [file_{(ps+1)*j}, ...],
        #
        #     ...
        # ]
        return [
            x
            for p in izip_longest(*split_parts, fillvalue=[])
            for x in p
        ]

    partitions = compute_partitions()
    checkers, checker_loading_success = get_working_checkers(args)

    if not checker_loading_success:
        logger.log('error', 'Some checkers could not be loaded, exiting.')
        sys.exit(1)

    if args.list_categories:
        list_categories(checkers)
    if args.checkers_help:
        print_checkers_help(checkers)

    logger.log('info', 'Created {} partitions of {} files.'.format(
        len(partitions), ps
    ))
    logger.log(
        'info',
        'Parallel analysis done in batches of {} partitions'.format(
            args.j
        )
    )

    all_diags = parallel_map(
        process_count=args.j,
        target=partial(do_partition, args, provider_config, checkers),
        elements=enumerate(partitions),
        timeout_factor=args.timeout_factor,
        timeout_callback=on_timeout
    )

    reports = []

    for diags in all_diags:
        for diag in diags:
            if diagnostic_action == 'log':
                logger.log('diag-{}'.format(diag[3]), report_diag(args, diag))
            elif diagnostic_action == 'return':
                reports.append(diag)

    return reports


def run(argv, diagnostic_action='log'):
    """
    Runs the driver with the given arguments. The diagnostic_action parameter
    allows choosing between logging the diagnostics found by the checkers
    returning them as a list of strings.

    :param list[str] argv: Driver arguments.
    :param str diagnostic_action: Either 'log' or 'return'.
    :rtype: list[str] | None
    """
    args = parser.parse_args(argv)
    set_logger(args)
    return do_all(args, diagnostic_action)


if __name__ == "__main__":
    run(sys.argv[1:])
