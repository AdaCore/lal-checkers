import runpy
import sys

if len(sys.argv) <= 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("Usage: python {}: <checker-module> <args>".format(sys.argv[0]))
    ex = "Example: python {} checkers.syntactic.check_same_logic test.adb"
    print(ex.format(sys.argv[0]))
else:
    del sys.argv[0]
    runpy.run_module(sys.argv[0], run_name='__main__')
