"""
Defines the TypeInterpreter interface, as well as a few common
TypeInterpreters.
"""

from lalcheck.domain_ops import (
    boolean_ops,
    interval_ops,
    finite_lattice_ops
)
from lalcheck import types
from lalcheck import domains


class TypeInterpreter(object):
    """
    A TypeInterpreter is used to determine how a type is interpreted in the
    backend. It must provide the domain used, a set of definitions that are
    available on this new domain, as well as a builder function to build
    elements of this domain from literal values.
    """
    def from_type(self, tpe):
        """
        Given a type, returns the domain that this TypeInterpreter would use to
        represent the type, a dictionary of definitions, as well as a builder
        for this domain. Returns None if this interpreter does not provide any
        interpretation for the given type.
        """
        raise NotImplementedError

    def __or__(self, other):
        """
        Creates a new interpreter by combining two interpreters. This new
        TypeInterpreter will try to interpret the given type using the first
        TypeInterpreter. It it failed (returned None), it will interpret it
        using the second TypeInterpreter.
        """
        @type_interpreter
        def f(hint):
            x = self.from_type(hint)
            return x if x is not None else other.from_type(hint)

        return f


def type_interpreter(fun):
    """
    A useful decorator to use on a function to turn it into an TypeInterpreter.
    The decorated function must receive a type as parameter and return
    an TypeInterpreter instance.
    """
    class AnonymousInterpreter(TypeInterpreter):
        def from_type(self, tpe):
            return fun(tpe)

    return AnonymousInterpreter()


@type_interpreter
def default_boolean_interpreter(tpe):
    if tpe.is_a(types.Boolean):
        bool_dom = boolean_ops.Boolean
        un_fun_dom = (bool_dom, bool_dom)
        bin_fun_dom = (bool_dom, bool_dom, bool_dom)

        defs = {
            ('!', un_fun_dom): boolean_ops.not_,
            ('&&', bin_fun_dom): boolean_ops.and_,
            ('||', bin_fun_dom): boolean_ops.or_,

            ('==', bin_fun_dom): finite_lattice_ops.eq(bool_dom),
            ('!=', bin_fun_dom): finite_lattice_ops.neq(bool_dom)
        }

        inv_defs = {
            ('!', un_fun_dom): boolean_ops.inv_not,
            ('&&', bin_fun_dom): boolean_ops.inv_and,
            ('||', bin_fun_dom): boolean_ops.inv_or,

            ('==', bin_fun_dom): finite_lattice_ops.inv_eq(bool_dom),
            ('!=', bin_fun_dom): finite_lattice_ops.inv_neq(bool_dom)
        }

        builder = boolean_ops.lit

        return bool_dom, defs, inv_defs, builder


@type_interpreter
def default_int_range_interpreter(tpe):
    if tpe.is_a(types.IntRange):
        int_dom = domains.Intervals(tpe.frm, tpe.to)
        bool_dom = boolean_ops.Boolean
        unary_fun_dom = (int_dom, int_dom)
        binary_fun_dom = (int_dom, int_dom, int_dom)
        binary_rel_dom = (int_dom, int_dom, bool_dom)

        defs = {
            ('+', binary_fun_dom): interval_ops.add_no_wraparound(int_dom),
            ('-', binary_fun_dom): interval_ops.sub_no_wraparound(int_dom),

            ('<', binary_rel_dom): interval_ops.lt(int_dom),
            ('<=', binary_rel_dom): interval_ops.le(int_dom),
            ('==', binary_rel_dom): interval_ops.eq(int_dom),
            ('!=', binary_rel_dom): interval_ops.neq(int_dom),
            ('>=', binary_rel_dom): interval_ops.ge(int_dom),
            ('>', binary_rel_dom): interval_ops.gt(int_dom),
            ('-', unary_fun_dom): interval_ops.inverse(int_dom),
        }

        inv_defs = {
            ('+', binary_fun_dom): interval_ops.inv_add_no_wraparound(int_dom),
            ('-', binary_fun_dom): interval_ops.inv_sub_no_wraparound(int_dom),

            ('<', binary_rel_dom): interval_ops.inv_lt(int_dom),
            ('<=', binary_rel_dom): interval_ops.inv_le(int_dom),
            ('==', binary_rel_dom): interval_ops.inv_eq(int_dom),
            ('!=', binary_rel_dom): interval_ops.inv_neq(int_dom),
            ('>=', binary_rel_dom): interval_ops.inv_ge(int_dom),
            ('>', binary_rel_dom): interval_ops.inv_gt(int_dom),
            ('-', unary_fun_dom): interval_ops.inv_inverse(int_dom),
        }

        builder = interval_ops.lit(int_dom)

        return int_dom, defs, inv_defs, builder


@type_interpreter
def default_enum_interpreter(tpe):
    if tpe.is_a(types.Enum):
        enum_dom = domains.FiniteLattice.of_subsets(set(tpe.lits))
        bool_dom = boolean_ops.Boolean
        bin_rel_dom = (enum_dom, enum_dom, bool_dom)

        defs = {
            ('==', bin_rel_dom): finite_lattice_ops.eq(enum_dom),
            ('!=', bin_rel_dom): finite_lattice_ops.neq(enum_dom)
        }

        inv_defs = {
            ('==', bin_rel_dom): finite_lattice_ops.inv_eq(enum_dom),
            ('!=', bin_rel_dom): finite_lattice_ops.inv_neq(enum_dom)
        }

        builder = finite_lattice_ops.lit(enum_dom)

        return enum_dom, defs, inv_defs, builder


@type_interpreter
def simple_access_interpreter(tpe):
    if tpe.is_a(types.Pointer):
        ptr_dom = domains.FiniteLattice.of_subsets({'null', 'not_null'})
        bool_dom = boolean_ops.Boolean
        bin_rel_dom = (ptr_dom, ptr_dom, bool_dom)

        defs = {
            ('==', bin_rel_dom): finite_lattice_ops.eq(ptr_dom),
            ('!=', bin_rel_dom): finite_lattice_ops.neq(ptr_dom)
        }

        inv_defs = {
            ('==', bin_rel_dom): finite_lattice_ops.inv_eq(ptr_dom),
            ('!=', bin_rel_dom): finite_lattice_ops.inv_neq(ptr_dom)
        }

        builder = finite_lattice_ops.lit(ptr_dom)

        return ptr_dom, defs, inv_defs, builder


default_type_interpreter = (
    default_boolean_interpreter |
    default_int_range_interpreter |
    default_enum_interpreter |
    simple_access_interpreter
)
