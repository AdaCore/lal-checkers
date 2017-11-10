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
    a type interpretation.
    """
    class AnonymousInterpreter(TypeInterpreter):
        def from_type(self, tpe):
            return fun(tpe)

    return AnonymousInterpreter()


def dict_to_provider(def_dict):
    """
    Converts a dictionnary of definitions indexed by their names and domain
    signatures to a provider function.
    """
    def provider(name, signature):
        if (name, signature) in def_dict:
            return def_dict[name, signature]

    return provider


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

        return (
            bool_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            builder
        )


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

        return (
            int_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            builder
        )


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

        return (
            enum_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            builder
        )


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
        null = builder('null')
        notnull = builder('not_null')

        def def_provider(name, sig):
            if (name, sig) in defs:
                return defs[name, sig]
            elif name == '*' and len(sig) == 2 and sig[0] == ptr_dom:
                elem_dom = sig[1]

                def deref(ptr):
                    return elem_dom.bottom if ptr == null else elem_dom.top

                return deref
            elif name == '&' and len(sig) == 2 and sig[1] == ptr_dom:
                elem_dom = sig[0]

                def address(elem):
                    return notnull

                return address

        def inv_def_provider(name, sig):
            if (name, sig) in defs:
                return inv_defs[name, sig]
            elif name == '*' and len(sig) == 2 and sig[0] == ptr_dom:
                elem_dom = sig[1]

                def inv_deref(elem, e_constr):
                    if ptr_dom.is_empty(e_constr) or elem_dom.is_empty(elem):
                        return None

                    if ptr_dom.le(notnull, e_constr):
                        return notnull

                    return None

                return inv_deref
            elif name == '&' and len(sig) == 2 and sig[1] == ptr_dom:
                elem_dom = sig[0]

                def inv_address(ptr, e_constr):
                    if ptr_dom.is_empty(ptr) or elem_dom.is_empty(e_constr):
                        return None

                    return e_constr

                return inv_address

        return (
            ptr_dom,
            def_provider,
            inv_def_provider,
            builder
        )


default_type_interpreter = (
    default_boolean_interpreter |
    default_int_range_interpreter |
    default_enum_interpreter |
    simple_access_interpreter
)
