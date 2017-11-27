import os
import os.path
import sys


from gnatpython.ex import PIPE, Run, STDOUT
from gnatpython import fileutils
from gnatpython.testsuite.driver import TestDriver


class PythonDriver(TestDriver):
    """
    If the 'helper' option is used in the YAML file, the corresponding test
    helper (from the 'tests/test_helpers' directory) is run with a set of
    standard arguments ('output_dir', etc.). Otherwise, the default 'test.py'
    file is run.
    """

    #
    # Driver entry points
    #

    py_file = 'test.py'
    """
    Name of the file for the Python script to run.
    """

    out_file = 'actual.out'
    """
    Name of the file for output redirection.
    """

    expected_file = 'test.out'
    """
    Name of the file that contains the expected output.
    """

    timeout = 300
    """
    Timeout (in seconds) to run the Python script.
    """

    @property
    def python_interpreter(self):
        """
        Return the path to the Python interpreter to use to run tests.

        :rtype: str
        """
        return self.global_env['options'].with_python or sys.executable

    @property
    def test_helpers_dir(self):
        """
        Return the path to the "test_helpers" directory.

        :rtype: str
        """
        return os.path.join(self.global_env['test_dir'], 'test_helpers')

    @property
    def python_interpreter_args(self):
        """
        Return the arguments to pass to the python interpreter.
        :rtype: list[str]
        """
        if 'helper' in self.test_env:
            py_file = os.path.join(
                self.test_helpers_dir,
                self.test_env['helper']
            )
            return [py_file, '--output_dir=output']
        else:
            return [self.py_file]

    def test_working_dir(self, *args):
        """
        Build a path under the temporary directory created for this testcase.

        :param list[str] args: Path components.
        :rtype: str
        """
        return os.path.join(self.global_env['working_dir'],
                            self.test_env['test_name'],
                            *args)

    def tear_up(self):
        super(PythonDriver, self).tear_up()
        fileutils.sync_tree(self.test_env['test_dir'], self.test_working_dir())

    def run(self):
        # Run the Python script and redirect its output to `self.out_file`.
        argv = [self.python_interpreter] + self.python_interpreter_args

        p = Run(argv, timeout=self.timeout, output=PIPE, error=STDOUT,
                cwd=self.test_working_dir())

        with open(self.test_working_dir(self.out_file), 'a') as f:
            f.write(p.out)

        if p.status != 0:
            self.result.actual_output += '{} returned status code {}\n'.format(
                ' '.join(argv), p.status
            )
            self.result.actual_output += p.out
            self.result.set_status('FAILED', 'error status code')

    def analyze(self):
        diff = fileutils.diff(self.test_working_dir(self.expected_file),
                              self.test_working_dir(self.out_file))
        if diff:
            self.result.actual_output += diff
            self.result.set_status('FAILED', 'diff in output')
        else:
            self.result.set_status('PASSED')
