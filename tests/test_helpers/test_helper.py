"""
Provide the test_runner.run decorator to use on functions to test.
"""

import argparse


parser = argparse.ArgumentParser(description="Generic Test Helper")
parser.add_argument('--output_dir', required=False)


def run(fun):
    """
    Decorator to use on testing functions. Invokes the given function with a
    dict containing standard helper arguments, such as 'output_dir'.

    :param (argparse.Namespace)->None fun: The function to test.
    :rtype: None
    """
    args = parser.parse_args()
    fun(args)
