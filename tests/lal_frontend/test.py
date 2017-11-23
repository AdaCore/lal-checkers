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

do('test_if_expr_1', """
Program:
  read(x#1)
  read(C#2)
  read(A#3)
  read(B#4)
  split:
    assume(C#2)
    tmp0#5 = A#3
  |:
    assume(!C#2)
    tmp0#5 = B#4
  x#1 = tmp0#5
""")

do('test_if_expr_2', """
Program:
  read(x#1)
  read(C1#2)
  read(C2#3)
  read(A1#4)
  read(A2#5)
  read(A3#6)
  split:
    assume(C1#2)
    split:
      assume(C2#3)
      tmp1#7 = A1#4
    |:
      assume(!C2#3)
      tmp1#7 = A2#5
    tmp0#8 = tmp1#7
  |:
    assume(!C1#2)
    tmp0#8 = A3#6
  x#1 = tmp0#8
""")

do('test_ptr_1', """
Program:
  read(x#1)
  y#2 = 0
  read(b#3)
  split:
    assume(b#3)
    x#1 = &y#2
  |:
    assume(!b#3)
    x#1 = null
  split:
    assume(x#1 == null)
    y#2 = 2
  |:
    assume(!x#1 == null)
    assume(x#1 != null)
    y#2 = *x#1
""")

do('test_if_expr_3', """
Program:
  read(x#1)
  read(C1#2)
  read(C2#3)
  read(A1#4)
  read(A2#5)
  split:
    assume(C1#2)
    tmp0#6 = C1#2
  |:
    assume(!C1#2)
    tmp0#6 = C2#3
  split:
    assume(tmp0#6)
    tmp1#7 = A1#4
  |:
    assume(!tmp0#6)
    tmp1#7 = A2#5
  x#1 = tmp1#7
""")
