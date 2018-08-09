"""
Provide the test_runner.run decorator to use on functions to test.
"""

import argparse
import os


parser = argparse.ArgumentParser(description="Generic Test Helper")
parser.add_argument('--output_dir', required=False)
parser.add_argument('--call_strategy', required=False, default='unknown')
parser.add_argument('--typer', required=False, default='default')
parser.add_argument('--test_subprogram', required=False, default='Test')


def run(fun):
    """
    Decorator to use on testing functions. Invokes the given function with a
    dict containing standard helper arguments, such as 'output_dir'.

    :param (argparse.Namespace)->None fun: The function to test.
    :rtype: None
    """
    args = parser.parse_args()
    fun(args)


def ensure_dir(path):
    """
    Ensure that the directory at the given path exists. (Creates it if it
    doesn't).

    :param str path: The desired path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_test_program(progs, name):
    return next(
        prog
        for prog in progs
        if prog.data.fun_id.f_subp_spec.f_subp_name.text == name
    )
