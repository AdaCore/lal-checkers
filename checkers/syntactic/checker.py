import libadalang as lal
import utils
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--project', default=None, help='The project file',
                    type=str)
parser.add_argument('--codepeer-output', action='store_true')
parser.add_argument('files', help='The files to analyze',
                    type=str, nargs='+', metavar='F')


class Checker(object):
    """
    Base class for lightweight checkers. Simply override the "name" and "run"
    class-methods to give a name to your checker and implement its logic.
    """
    def __init__(self, provider):
        """
        :param lal.UnitProvider provider: The unit provider to use for this
            checker.
        """
        self.ctx = lal.AnalysisContext(unit_provider=provider)
        self.output_format = 'default'

    def set_output_format(self, output_format):
        """
        Sets the output format to use.

        :param output_format: The formatting to use when report messages.
            Either 'default' or 'codepeer'.
        """
        self.output_format = output_format

    @classmethod
    def name(cls):
        """
        Should return the name of the checker.
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def for_project(cls, project_file, scenario_vars=None):
        """
        Instantiates this checker using a project file and scenario vars.

        :param str project_file: The path to the project file.
        :param dict[str, str] scenario_vars: The scenario variables.
        :rtype: Checker
        """
        provider = lal.UnitProvider.for_project(project_file, scenario_vars)
        return cls(provider)

    @classmethod
    def auto(cls, input_files):
        """
        Instantiates this checker using a set of input files.

        :param iterable[str] input_files: An iterable of file paths.
        :rtype: Checker
        """
        provider = lal.UnitProvider.auto(input_files)
        return cls(provider)

    @classmethod
    def build_and_run(cls):
        """
        "Main" procedure which parses command-line arguments, instantiates a
        checker and runs it on the provided set of files.
        """
        args = parser.parse_args()
        if args.project is not None:
            checker = cls.for_project(args.project)
        else:
            checker = cls.auto(args.files)

        if args.codepeer_output:
            checker.set_output_format('codepeer')

        checker.check_files(*args.files)

    def run(self, unit):
        """
        Override this method to implement the actual logic of the checker.

        :param lal.CompilationUnit unit: The compilation unit to run the
            checker on.
        """
        raise NotImplementedError

    def check_files(self, *files):
        """
        Runs the checker on a given set of files.

        :param *str files: The set of file paths to check.
        """
        for f in files:
            unit = self.ctx.get_from_file(f)
            if unit.root is None:
                print('Could not parse {}:'.format(f))
                for diag in unit.diagnostics:
                    self.fatal('   {}'.format(diag))
                return

            self.run(unit)

    @staticmethod
    def fatal(msg):
        """
        Outputs an error message corresponding to a failure of the checker
        itself.

        :param object msg:
        """
        print(msg)

    def report(self, node, msg, flag=None):
        """
        Outputs an error/warning message in a specific format.

        :param lal.AdaNode node: The node at the origin of the error.
        :param str msg: The error message to report
        :param str|None flag: The category of the error.
        """
        if flag is None:
            flag = self.name()

        filename = node.unit.filename
        pos = node.sloc_range.start
        subp = utils.closest_enclosing(node, lal.SubpBody)

        if subp is None:
            self.fatal('No enclosing subprogram')
            return

        spec = subp.f_subp_spec
        proc_name = spec.f_subp_name.text
        proc_filename = filename
        proc_pos = spec.sloc_range.start

        if self.output_format == 'codepeer':
            print("{}:{}:{} warning: {}:{}:{}:{}: {} {}".format(
                filename, pos.line, pos.column,
                proc_name, proc_filename, proc_pos.line, proc_pos.column,
                msg,
                "[{}]".format(flag)
            ))
        else:
            print("{}:{}:{} {}".format(filename, pos.line, pos.column, msg))
