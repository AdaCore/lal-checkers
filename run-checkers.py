import argparse
import importlib
from tools.scheduler import Scheduler


parser = argparse.ArgumentParser(description='lal-checker runner.')
parser.add_argument('-P', default=None, metavar='PROJECT_FILE', type=str)

parser.add_argument('-X', action='append', metavar='VAR=STR', type=str,
                    default=[])

parser.add_argument('--files-from', metavar='PATH', type=str, required=True)

parser.add_argument('--checkers-from', metavar='PATH', type=str, required=True)


def lines_from_file(filename):
    try:
        with open(filename) as f:
            return [l.strip() for l in f.readlines()]
    except IOError:
        print('error: cannot read file {}'.format(filename))


def get_requirements():
    args = parser.parse_args()

    project_file = args.P
    scenario_vars = dict([eq.split('=') for eq in args.X])

    if project_file is None and len(scenario_vars) > 0:
        print ("warning: use of scenario vars without a project file.")

    files_to_check = lines_from_file(args.files_from)
    checker_commands = lines_from_file(args.checkers_from)
    checker_args = [command.split() for command in checker_commands]

    requirements = []
    for checker_arg in checker_args:
        checker_module = importlib.import_module(checker_arg[0])
        if not hasattr(checker_module, 'create_requirement'):
            print('error: checker {} does not export a '
                  '"create_requirement" function.'.format(checker_module))

        requirements.append(checker_module.create_requirement(
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


if __name__ == "__main__":
    reqs = get_requirements()
    schedule = get_schedules(reqs)[0]
    for checker_result in schedule.run().values():
        for program_result in checker_result:
            for diag in program_result.diagnostics:
                pos = program_result.diag_position(diag)
                msg = program_result.diag_message(diag)
                if pos is not None and msg is not None:
                    print("{}: {}".format(pos, msg))
