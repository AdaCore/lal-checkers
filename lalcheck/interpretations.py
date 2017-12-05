"""
Defines the TypeInterpreter interface, as well as a few common
TypeInterpreters.
"""

from lalcheck.domain_ops import (
    boolean_ops,
    interval_ops,
    finite_lattice_ops
)
from lalcheck.constants import ops, lits
from lalcheck.utils import Transformer
from lalcheck import types
from lalcheck import domains


class TypeInterpretation(object):
    """
    A type interpretation is how the middle-end type is represented when
    doing abstract interpretation. It provides the abstract domain used to
    represent elements of the type, the implementation of the different
    operations available for that type, etc.
    """
    def __init__(self, domain, def_provider, inv_def_provider, builder):
        """
        :param domains.AbstractDomain domain: The abstract domain used to
            represent the type.

        :param (str, tuple[domains.AbstractDomain])->function def_provider:
            A function which can be called with the name and signature
            (domains of all its operands) of the desired definition to
            retrieve it.

        :param (str, tuple[domains.AbstractDomain])->function inv_def_provider:
            A function which can be called with the name and signature
            (domains of all its operands) of the desired definition to
            retrieve its inverse definition.

        :param function builder: A function used to build elements of the
            domain from literal values.
        """
        self.domain = domain
        self.def_provider = def_provider
        self.inv_def_provider = inv_def_provider
        self.builder = builder


type_interpreter = Transformer.as_transformer


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
            (ops.NOT, un_fun_dom): boolean_ops.not_,
            (ops.AND, bin_fun_dom): boolean_ops.and_,
            (ops.OR, bin_fun_dom): boolean_ops.or_,

            (ops.EQ, bin_fun_dom): finite_lattice_ops.eq(bool_dom),
            (ops.NEQ, bin_fun_dom): finite_lattice_ops.neq(bool_dom)
        }

        inv_defs = {
            (ops.NOT, un_fun_dom): boolean_ops.inv_not,
            (ops.AND, bin_fun_dom): boolean_ops.inv_and,
            (ops.OR, bin_fun_dom): boolean_ops.inv_or,

            (ops.EQ, bin_fun_dom): finite_lattice_ops.inv_eq(bool_dom),
            (ops.NEQ, bin_fun_dom): finite_lattice_ops.inv_neq(bool_dom)
        }

        builder = boolean_ops.lit

        return TypeInterpretation(
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
            (ops.PLUS, binary_fun_dom):
                interval_ops.add_no_wraparound(int_dom),

            (ops.MINUS, binary_fun_dom):
                interval_ops.sub_no_wraparound(int_dom),

            (ops.LT, binary_rel_dom): interval_ops.lt(int_dom),
            (ops.LE, binary_rel_dom): interval_ops.le(int_dom),
            (ops.EQ, binary_rel_dom): interval_ops.eq(int_dom),
            (ops.NEQ, binary_rel_dom): interval_ops.neq(int_dom),
            (ops.GE, binary_rel_dom): interval_ops.ge(int_dom),
            (ops.GT, binary_rel_dom): interval_ops.gt(int_dom),
            (ops.NEG, unary_fun_dom): interval_ops.negate(int_dom),
        }

        inv_defs = {
            (ops.PLUS, binary_fun_dom):
                interval_ops.inv_add_no_wraparound(int_dom),

            (ops.MINUS, binary_fun_dom):
                interval_ops.inv_sub_no_wraparound(int_dom),

            (ops.LT, binary_rel_dom): interval_ops.inv_lt(int_dom),
            (ops.LE, binary_rel_dom): interval_ops.inv_le(int_dom),
            (ops.EQ, binary_rel_dom): interval_ops.inv_eq(int_dom),
            (ops.NEQ, binary_rel_dom): interval_ops.inv_neq(int_dom),
            (ops.GE, binary_rel_dom): interval_ops.inv_ge(int_dom),
            (ops.GT, binary_rel_dom): interval_ops.inv_gt(int_dom),
            (ops.NEG, unary_fun_dom): interval_ops.inv_inverse(int_dom)
        }

        builder = interval_ops.lit(int_dom)

        return TypeInterpretation(
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
            (ops.EQ, bin_rel_dom): finite_lattice_ops.eq(enum_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.neq(enum_dom)
        }

        inv_defs = {
            (ops.EQ, bin_rel_dom): finite_lattice_ops.inv_eq(enum_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.inv_neq(enum_dom)
        }

        builder = finite_lattice_ops.lit(enum_dom)

        return TypeInterpretation(
            enum_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            builder
        )


@type_interpreter
def simple_access_interpreter(tpe):
    if tpe.is_a(types.Pointer):
        ptr_dom = domains.FiniteLattice.of_subsets({lits.NULL, lits.NOT_NULL})
        bool_dom = boolean_ops.Boolean
        bin_rel_dom = (ptr_dom, ptr_dom, bool_dom)

        defs = {
            (ops.EQ, bin_rel_dom): finite_lattice_ops.eq(ptr_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.neq(ptr_dom)
        }

        inv_defs = {
            (ops.EQ, bin_rel_dom): finite_lattice_ops.inv_eq(ptr_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.inv_neq(ptr_dom)
        }

        builder = finite_lattice_ops.lit(ptr_dom)
        null = builder(lits.NULL)
        notnull = builder(lits.NOT_NULL)

        def def_provider(name, sig):
            if (name, sig) in defs:
                return defs[name, sig]

            elif (name == ops.DEREF and
                    len(sig) == 2 and
                    sig[0] == ptr_dom):
                elem_dom = sig[1]

                def deref(ptr):
                    return elem_dom.bottom if ptr == null else elem_dom.top

                return deref

            elif (name == ops.ADDRESS and
                    len(sig) == 2 and
                    sig[1] == ptr_dom):
                elem_dom = sig[0]

                def address(elem):
                    return notnull

                return address

        def inv_def_provider(name, sig):
            if (name, sig) in defs:
                return inv_defs[name, sig]

            elif (name == ops.DEREF and
                    len(sig) == 2 and
                    sig[0] == ptr_dom):
                elem_dom = sig[1]

                def inv_deref(elem, e_constr):
                    if ptr_dom.is_empty(e_constr) or elem_dom.is_empty(elem):
                        return None

                    if ptr_dom.le(notnull, e_constr):
                        return notnull

                    return None

                return inv_deref

            elif (name == ops.ADDRESS and
                    len(sig) == 2 and
                    sig[1] == ptr_dom):
                elem_dom = sig[0]

                def inv_address(ptr, e_constr):
                    if ptr_dom.is_empty(ptr) or elem_dom.is_empty(e_constr):
                        return None

                    return e_constr

                return inv_address

        return TypeInterpretation(
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
