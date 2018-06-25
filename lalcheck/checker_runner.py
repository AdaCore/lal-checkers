import argparse
import importlib

import libadalang as lal
from lalcheck.checkers.support.checker import Checker
from lalcheck.tools.digraph import Digraph
from lalcheck.tools.dot_printer import gen_dot, DataPrinter
from lalcheck.tools.scheduler import Scheduler

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

parser.add_argument('--codepeer-output', action='store_true')
parser.add_argument('--export-schedule', type=str)


args = parser.parse_args()


def lines_from_file(filename):
    try:
        with open(filename) as f:
            return [l.strip() for l in f.readlines()]
    except IOError:
        print('error: cannot read file {}'.format(filename))


def get_requirements():
    project_file = args.P
    scenario_vars = dict([eq.split('=') for eq in args.X])

    if project_file is None and len(scenario_vars) > 0:
        print ("warning: use of scenario vars without a project file.")

    if args.files_from:
        files_to_check = lines_from_file(args.files_from)
    else:
        files_to_check = [
            f
            for fs in args.files
            for f in fs.split(';')
        ]

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
        checker_module = importlib.import_module(checker_arg[0])
        if not hasattr(checker_module, 'checker'):
            print('error: checker {} does not export a '
                  '"checker" object.'.format(checker_module))
        elif not issubclass(checker_module.checker, Checker):
            print('error: checker {} does not inherit the '
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


def closest_enclosing(node, *tpes):
    """
    Given a libadalang node n, returns its closest enclosing libadalang node of
    one of the given types which directly or indirectly contains n.

    :param lal.AdaNode node: The node from which to start the search.
    :param *type tpes: The kind of node to look out for.
    :rtype: lal.AdaNode|None
    """
    while node.parent is not None:
        node = node.parent
        if node.is_a(*tpes):
            return node
    return None


def report_diag(report):
    node, msg, flag = report
    filename = node.unit.filename
    pos = node.sloc_range.start

    if args.codepeer_output:
        subp = closest_enclosing(node, lal.SubpBody)

        if subp is None:
            print('No enclosing subprogram')
            return

        spec = subp.f_subp_spec
        proc_name = spec.f_subp_name.text
        proc_filename = filename
        proc_pos = spec.sloc_range.start

        print("{}:{}:{}: warning: {}:{}:{}:{}: {} [{}]".format(
            filename, pos.line, pos.column,
            proc_name, proc_filename, proc_pos.line, proc_pos.column,
            msg, flag
        ))
    else:
        print("{}:{}:{} {}".format(filename, pos.line, pos.column, msg))


def main():
    reqs = get_requirements()
    schedule = get_schedules(reqs)[0]

    if args.export_schedule is not None:
        export_schedule(schedule, args.export_schedule)

    for checker_result in schedule.run().values():
        for program_result in checker_result:
            for diag in program_result.diagnostics:
                report = program_result.diag_report(diag)
                if report is not None:
                    report_diag(report)


if __name__ == "__main__":
    main()
