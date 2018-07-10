import argparse
import importlib
from multiprocessing import Pool, cpu_count
from itertools import izip_longest
import signal
import traceback

from lalcheck.checkers.support.checker import Checker, CheckerResults
from lalcheck.tools.digraph import Digraph
from lalcheck.tools.dot_printer import gen_dot, DataPrinter
from lalcheck.tools.scheduler import Scheduler
from lalcheck.tools import logger

parser = argparse.ArgumentParser(description='lal-checker runner.')

parser.add_argument('-P', default=None, metavar='PROJECT_FILE', type=str)

parser.add_argument('-X', action='append', metavar='VAR=STR', type=str,
                    default=[])

files_group = parser.add_mutually_exclusive_group(required=True)
files_group.add_argument('--files-from', metavar='FILE_PATH', type=str)
files_group.add_argument('--files', metavar='FILE_PATHS', action='append',
                         type=str, help='File paths separated by semicolons.')

checkers_group = parser.add_mutually_exclusive_group(required=True)
checkers_group.add_argument('--checkers-from', metavar='FILE_PATH', type=str)
checkers_group.add_argument('--checkers', metavar='CHECKERS', action='append',
                            type=str, help='Checker commands separated by'
                                           'semicolons.')

parser.add_argument('--log', metavar="CATEGORIES", type=str,
                    default="error;internal-error;diag-{}".format(
                        CheckerResults.HIGH
                    ),
                    help='Categories separated by semicolons.')

parser.add_argument('--codepeer-output', action='store_true')
parser.add_argument('--export-schedule', type=str)
parser.add_argument('--partition-size', default=0, type=int,
                    help='The amount of files that will be batched in a'
                         'partition. A higher number means less computing'
                         'time, but more memory consumption.')
parser.add_argument('-j', default=0, type=int,
                    help='The number of process to spawn in parallel, each'
                         'of which deals with a single partition at a time.')


args = None


def lines_from_file(filename):
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
    """
    file_lines = [(f, get_line_count(f)) for f in filenames]
    file_lines.sort(key=lambda x: x[1], reverse=True)
    return [f for f, l in file_lines]


def set_logger():
    logger.set_logger(logger.Logger.with_std_output(args.log.split(';')))


def get_working_files():
    if args.files_from:
        return lines_from_file(args.files_from)
    else:
        return [
            f
            for fs in args.files
            for f in fs.split(';')
        ]


def get_requirements(files_to_check):
    project_file = args.P
    scenario_vars = dict([eq.split('=') for eq in args.X])

    if project_file is None and len(scenario_vars) > 0:
        logger.log('info', "warning: use of scenario vars without a "
                           "project file.")

    if args.checkers_from:
        checker_commands = lines_from_file(args.checkers_from)
    else:
        checker_commands = [
            c
            for cs in args.checkers
            for c in cs.split(';')
        ]

    checker_args = [command.split() for command in checker_commands]

    requirements = []
    for checker_arg in checker_args:
        try:
            checker_module = importlib.import_module(checker_arg[0])
        except ImportError:
            logger.log('error', 'Failed to import checker module {}.'.format(
                checker_arg[0]
            ))
        else:
            if not hasattr(checker_module, 'checker'):
                logger.log('error', 'Checker {} does not export a "checker" '
                                    'object.'.format(checker_module))
            elif not issubclass(checker_module.checker, Checker):
                logger.log('error', 'checker {} does not inherit the '
                                    '"checkers.support.checker.Checker" '
                                    'interface.'.format(checker_module))

            requirements.append(checker_module.checker.create_requirement(
                project_file=project_file,
                scenario_vars=scenario_vars,
                filenames=files_to_check,
                args=checker_arg[1:]
            ))

    return requirements


def get_schedules(requirements):
    scheduler = Scheduler()
    return scheduler.schedule({
        'res_{}'.format(i): req
        for i, req in enumerate(requirements)
    })


def export_schedule(schedule, export_path):
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
    with open(export_path, 'w') as export_file:
        export_file.write(dot)


def report_diag(report):
    """
    Given a diagnostic report, creates the final string to output to the user.
    :param (DiagnosticPosition, str, str, str) report: The report to output.
    :rtype: str
    """
    pos, msg, flag, _ = report

    if args.codepeer_output:
        return "{}:{}:{}: warning: {}:{}:{}:{}: {} [{}]".format(
            pos.filename, pos.start[0], pos.start[1],
            pos.proc_name or "unknown", pos.proc_filename or "unknown",
            pos.proc_start[0] or "unknown", pos.proc_start[1] or "unknown",
            msg, flag
        )
    else:
        return "{}:{}:{} {}".format(
            pos.filename, pos.start[0], pos.start[1], msg
        )


def do_partition(partition):
    diags = []

    try:
        reqs = get_requirements(partition)
        schedule = get_schedules(reqs)[0]

        if args.export_schedule is not None:
            export_schedule(schedule, args.export_schedule)

        for checker_result in schedule.run().values():
            for program_result in checker_result:
                for diag in program_result.diagnostics:
                    report = program_result.memoized_diag_report(diag)
                    if report is not None:
                        diags.append(report)
    except Exception:
        with logger.log_stdout('internal-error'):
            traceback.print_exc(file=sys.stdout)
    finally:
        return diags


def do_all(diagnostic_action):
    args.j = cpu_count() if args.j <= 0 else args.j
    working_files = sort_files_by_line_count(get_working_files())
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

    logger.log('info', 'Created {} partitions of {} files.'.format(
        len(partitions), ps
    ))
    logger.log(
        'info',
        'Parallel analysis done in batches of {} partitions'.format(
            args.j
        )
    )

    # To handle Keyboard interrupt, the child processes must inherit the
    # SIG_IGN (ignore signal) handler from the parent process. (see
    # https://stackoverflow.com/a/35134329)
    orig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = Pool(args.j, maxtasksperchild=1)

    # Restore the original handler of the parent process.
    signal.signal(signal.SIGINT, orig_handler)

    all_diags = p.map_async(do_partition, partitions, chunksize=1)

    # Keyboard interrupts are ignored if we use wait() or get() without any
    # timeout argument. By using this loop, we mimic the behavior of an
    # infinite timeout but allow keyboard interrupts to go through.
    while not all_diags.ready():
        all_diags.wait(1)

    p.close()
    p.join()

    reports = []

    for diags in all_diags.get():
        for diag in diags:
            if diagnostic_action == 'log':
                logger.log('diag-{}'.format(diag[3]), report_diag(diag))
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
    global args
    args = parser.parse_args(argv)

    set_logger()

    return do_all(diagnostic_action)


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
