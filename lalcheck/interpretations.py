"""
Defines the TypeInterpreter interface, as well as a few common
TypeInterpreters.
"""

from lalcheck.domain_ops import (
    boolean_ops,
    interval_ops,
    finite_lattice_ops,
    product_ops
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


TypeInterpreter = Transformer
"""
TypeInterpreter[T] is equivalent to Transformer[T, TypeInterpretation]
"""

type_interpreter = Transformer.as_transformer
delegating_type_interpreter = Transformer.from_transformer_builder
memoizing_type_interpreter = Transformer.make_memoizing


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


def default_simple_pointer_interpreter(inner_interpreter):
    """
    Builds a simple type interpreter for pointers, representing them as either
    null or nonnull. Provides comparison ops and deref/address.

    :param TypeInterpreter inner_interpreter: interpreter for pointer elements.
    :rtype: TypeInterpreter
    """
    @Transformer.as_transformer
    def get_pointer_element(tpe):
        """
        :param types.Type tpe: The type.
        :return: The type of the element of the pointer, if relevant.
        :rtype: type.Type
        """
        if tpe.is_a(types.Pointer):
            return tpe.elem_type

    @Transformer.as_transformer
    def pointer_interpreter(elem_interpretation):
        """
        :param TypeInterpretation elem_interpretation: The interpretation of
            the pointer element.

        :return: A type interpreter for pointers of such elements.
        :rtype: TypeInterpreter
        """
        ptr_dom = domains.FiniteLattice.of_subsets({lits.NULL, lits.NOT_NULL})
        elem_dom = elem_interpretation.domain
        bool_dom = boolean_ops.Boolean
        bin_rel_dom = (ptr_dom, ptr_dom, bool_dom)
        deref_dom = (ptr_dom, elem_dom)
        address_dom = (elem_dom, ptr_dom)

        builder = finite_lattice_ops.lit(ptr_dom)
        null = builder(lits.NULL)
        notnull = builder(lits.NOT_NULL)

        def deref(ptr):
            return elem_dom.bottom if ptr == null else elem_dom.top

        def inv_deref(elem, e_constr):
            if ptr_dom.is_empty(e_constr) or elem_dom.is_empty(elem):
                return None

            if ptr_dom.le(notnull, e_constr):
                return notnull

            return None

        def address(elem):
            return notnull

        def inv_address(ptr, e_constr):
            if ptr_dom.is_empty(ptr) or elem_dom.is_empty(e_constr):
                return None

            return e_constr

        defs = {
            (ops.EQ, bin_rel_dom): finite_lattice_ops.eq(ptr_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.neq(ptr_dom),
            (ops.DEREF, deref_dom): deref,
            (ops.ADDRESS, address_dom): address

        }

        inv_defs = {
            (ops.EQ, bin_rel_dom): finite_lattice_ops.inv_eq(ptr_dom),
            (ops.NEQ, bin_rel_dom): finite_lattice_ops.inv_neq(ptr_dom),
            (ops.DEREF, deref_dom): inv_deref,
            (ops.ADDRESS, address_dom): inv_address
        }

        return TypeInterpretation(
            ptr_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            builder
        )

    return get_pointer_element >> inner_interpreter >> pointer_interpreter


def default_product_interpreter(elem_interpreter):
    """
    Builds a type interpreter for product types, using the product domain of
    the domains of each of its components.

    :param TypeInterpreter elem_interpreter: interpreter for elements of
        the product.

    :rtype: TypeInterpreter
    """
    @Transformer.as_transformer
    def get_elements(tpe):
        """
        :param types.Type tpe: The type.
        :return: The type of the elements of the record, if relevant.
        :rtype: list[type.Type]
        """
        if tpe.is_a(types.Product):
            return tpe.elem_types

    @Transformer.as_transformer
    def product_interpreter(elem_interpretations):
        """
        :param list[TypeInterpretation] elem_interpretations:
        :return:
        """
        elem_doms = [interp.domain for interp in elem_interpretations]
        prod_dom = domains.Product(*elem_doms)
        constructor_dom = tuple(elem_doms) + (prod_dom,)
        bool_dom = boolean_ops.Boolean
        bin_rel_dom = (prod_dom, prod_dom, bool_dom)

        elem_eq_defs = [
            interp.def_provider(
                ops.EQ, (
                    interp.domain,
                    interp.domain,
                    bool_dom
                )
            )
            for interp in elem_interpretations
        ]

        elem_inv_eq_defs, elem_inv_neq_defs = (
            [
                interp.inv_def_provider(
                    op, (
                        interp.domain,
                        interp.domain,
                        bool_dom
                    )
                )
                for interp in elem_interpretations
            ]
            for op in [ops.EQ, ops.NEQ]
        )

        defs = {
            (ops.NEW, constructor_dom): product_ops.construct(prod_dom),
            (ops.EQ, bin_rel_dom): product_ops.eq(elem_eq_defs),
            (ops.NEQ, bin_rel_dom): product_ops.neq(elem_eq_defs)
        }

        inv_defs = {
            (ops.NEW, constructor_dom): product_ops.inv_construct(prod_dom),
            (ops.EQ, bin_rel_dom): product_ops.inv_eq(
                prod_dom, elem_inv_eq_defs, elem_eq_defs
            ),
            (ops.NEQ, bin_rel_dom): product_ops.inv_neq(
                prod_dom, elem_inv_eq_defs, elem_eq_defs
            )
        }

        return TypeInterpretation(
            prod_dom,
            dict_to_provider(defs),
            dict_to_provider(inv_defs),
            product_ops.lit
        )

    return get_elements >> elem_interpreter.lifted() >> product_interpreter


@memoizing_type_interpreter
@delegating_type_interpreter
def default_type_interpreter():
    return (
        default_boolean_interpreter |
        default_int_range_interpreter |
        default_enum_interpreter |
        default_simple_pointer_interpreter(default_type_interpreter) |
        default_product_interpreter(default_type_interpreter)
    )
