import lalcheck.irs.basic.frontends.lal as lal2basic
from lalcheck.irs.basic.tools import PrettyPrinter


def do(test_name, expected_output):
    ctx = lal2basic.new_context()

    progs = lal2basic.extract_programs(
        ctx,
        '{}.adb'.format(test_name)
    )

    output = PrettyPrinter.pretty_print(
        progs[0],
        PrettyPrinter.Opts(print_ids=True)
    )
    print(output)

    assert output.strip() == expected_output.strip()


do('test_constexpr_1', """
Program:
  read(x#1)
  split:
    assume(False)
    x#1 = 1
  |:
    assume(True)
""")

do('test_constant_1', """
Program:
  read(x#1)
  split:
    assume(True)
    x#1 = 1
  |:
    assume(False)
""")

do('test_static_range', """
Program:
  read(x#1)
  x#1 = 40
""")
