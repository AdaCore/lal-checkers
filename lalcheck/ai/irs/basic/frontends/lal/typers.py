import libadalang as lal
from lalcheck.ai import types
from lalcheck.ai.utils import Transformer

from utils import (
    StackType, PointerType, ExtendedCallReturnType,
    record_fields, is_array_type_decl, eval_as_real
)


@Transformer.as_transformer
def _eval_as_int(x):
    """
    Given an arbitrary Ada node, tries to evaluate it to an integer.

    :param lal.AdaNode x: The node to evaluate
    :rtype: int | None
    """
    try:
        return x.p_eval_as_int
    except (lal.PropertyError, lal.NativeException, OSError):
        return None


_eval_as_real = (
    Transformer.as_transformer(eval_as_real).catch(NotImplementedError)
)


@types.delegating_typer
def int_range_typer():
    """
    :return: A typer for int ranges.
    :rtype: types.Typer[lal.AdaNode]
    """

    @Transformer.as_transformer
    def get_operands(hint):
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.SignedIntTypeDef):
                rng = hint.f_type_def.f_range.f_range
                return rng.f_left, rng.f_right

    @types.Typer
    def to_int_range(xs):
        return types.IntRange(*xs)

    return get_operands >> (_eval_as_int & _eval_as_int) >> to_int_range


@types.delegating_typer
def real_range_typer():
    """
    Returns a typer for real type declarations.

    Note that this typer is very basic and ignores the delta (in fixed point
    type declarations) and digits.

    Also, it may conservatively fall back to the range -inf .. inf if it cannot
    parse the actual range.

    :rtype: types.Typer[lal.AdaNode]
    """

    @Transformer.as_transformer
    def get_real_type_def(hint):
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.FloatingPointDef,
                                    lal.DecimalFixedPointDef):
                return hint.f_type_def

    @Transformer.as_transformer
    def get_operands(real_def):
        if real_def.f_range is not None:
            rng = real_def.f_range.f_range
            return rng.f_left, rng.f_right

    @Transformer.as_transformer
    def infinite_range(_):
        return float('-inf'), float('inf')

    @types.Typer
    def to_real_range(xs):
        return types.RealRange(*xs)

    get_specified_range = (
        get_operands >>
        ((_eval_as_real & _eval_as_real) | infinite_range)
    )

    return (
        get_real_type_def >>
        (get_specified_range | infinite_range) >>
        to_real_range
    )


@types.delegating_typer
def int_mod_typer():
    """
    :return: A typer for mod int types.
    :rtype: types.Typer[lal.AdaNode]
    """

    @Transformer.as_transformer
    def get_modulus(hint):
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.ModIntTypeDef):
                return hint.f_type_def.f_expr

    @types.Typer
    def to_int_range(modulus):
        return types.IntRange(0, modulus - 1)

    return get_modulus >> _eval_as_int >> to_int_range


@types.typer
def enum_typer(hint):
    """
    :param lal.AdaNode hint: the lal type.
    :return: The corresponding lalcheck type.
    :rtype: types.Enum
    """
    if hint.is_a(lal.TypeDecl):
        if hint.f_type_def.is_a(lal.EnumTypeDef):
            literals = hint.f_type_def.findall(lal.EnumLiteralDecl)
            return types.Enum([lit.f_name.text for lit in literals])


_pointer_type = types.Pointer()


@types.Typer
def access_typer(hint):
    """
    :param lal.AdaNode hint: the lal type.
    :return: The lal type being accessed.
    :rtype: lal.AdaNode
    """
    if hint.is_a(lal.TypeDecl):
        try:
            if hint.p_is_access_type():
                return _pointer_type
        except lal.PropertyError:
            pass
    elif hint.is_a(PointerType):
        return _pointer_type


def record_component_typer(inner_typer):
    """
    :param types.Typer[lal.AdaNode] inner_typer: A typer for the types of the
        components.

    :return: A typer for record components.

    :rtype: types.Typer[lal.ComponentDecl | lal.DiscriminantSpec]
    """
    @Transformer.as_transformer
    def get_component_type(component):
        if component.is_a(lal.ComponentDecl):
            return component.f_component_def.f_type_expr
        elif component.is_a(lal.DiscriminantSpec):
            component.f_type_expr.f_name.p_resolve_names
            return component.f_type_expr

    return get_component_type >> inner_typer


def record_typer(comp_typer):
    """
    :param types.Typer[lal.ComponentDecl | lal.DiscriminantSpec] comp_typer:
        A typer for components of products.

    :return: A typer for record types.

    :rtype: types.Typer[lal.AdaNode]
    """
    @Transformer.as_transformer
    def get_elements(hint):
        """
        :param lal.AdaNode hint: the lal type.
        :return: The components of the record type, if relevant.
        :rtype: list[lal.AdaNode]
        """
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.RecordTypeDef):
                return [field.decl for field in record_fields(hint)]

    to_product = Transformer.as_transformer(types.Product)

    # Get the elements -> type them all -> generate the product type.
    return get_elements >> comp_typer.lifted() >> to_product


def name_typer(inner_typer):
    """
    :param types.Typer[lal.AdaNode] inner_typer: A typer for elements
        being referred by identifiers.

    :return: A typer for names.

    :rtype: types.Typer[lal.AdaNode]
    """

    # Create a simple placeholder object for when there are no constraint.
    # (object() is called because None would mean that we failed to fetch the
    # constraint).
    no_constraint = object()

    @Transformer.as_transformer
    def resolved_name_and_constraint(hint):
        """
        :param lal.AdaNode hint: the lal type expression.
        :return: The type declaration associated to the name, if relevant,
            as well as the optional constraint.
        :rtype: (lal.BaseTypeDecl, lal.AdaNode)
        """
        if hint.is_a(lal.SubtypeIndication):
            try:
                return (
                    hint.p_designated_type_decl,
                    hint.f_constraint if hint.f_constraint is not None
                    else no_constraint
                )
            except lal.PropertyError:
                pass

    def identity_if_is_a(tpe):
        """
        Given a type tpe, returns a transformer which fails if its element
        to transform is not of type tpe, or else leaves it untouched.
        """
        @Transformer.as_transformer
        def f(x):
            if x.is_a(tpe):
                return x
        return f

    @Transformer.as_transformer
    def get_range(constraint):
        """
        If the given constraint is a range constraint, returns the expressions
        for its left and right hand side in a pair.
        """
        if constraint is not no_constraint:
            if constraint.is_a(lal.RangeConstraint):
                rng = constraint.f_range.f_range
                if rng.is_a(lal.BinOp) and rng.f_op.is_a(lal.OpDoubleDot):
                    return rng.f_left, rng.f_right

    @Transformer.as_transformer
    def refined_int_range(args):
        """
        From the given pair containing an IntRange type on the left and an
        evaluated range constraint on the right (a pair of integers), return
        a new IntRange instance with the constraint applied.
        """
        tpe, (c_left, c_right) = args
        assert (tpe.frm <= c_left and tpe.to >= c_right)
        return type(tpe)(c_left, c_right)

    # Transforms a lal.RangeConstraint into a pair of int ('First and 'Last)
    get_range_constraint = get_range >> (_eval_as_int & _eval_as_int)

    # Transforms a pair (types.IntRange, lal.RangeConstraint) into a new,
    # refined types.IntRange instance.
    when_constrained_range = (
        (identity_if_is_a(types.IntRange) & get_range_constraint) >>
        refined_int_range
    )

    # 1. Resolve the name reference and the optional constraint into a pair.
    # 2. Type the left (0th) element of the tuple (which contains the referred
    #    declaration).
    # 3. If the computed type is an IntRange and the constraint is a range
    #    constraint, refine the IntRange. Else ignore the constraint and simply
    #    return the previously computed type which is in the left (0th) element
    #    of the tuple.
    return (resolved_name_and_constraint >>
            inner_typer.for_index(0) >>
            (when_constrained_range | Transformer.project(0)))


def anonymous_typer(inner_typer):
    """
    :param types.Typer[lal.AdaNode] inner_typer: A typer for the types declared
        by the anonymous typer.

    :return: A typer for anonymous types.

    :rtype: types.Typer[lal.AdaNode]
    """
    @Transformer.as_transformer
    def get_type_decl(hint):
        """
        :param lal.AdaNode hint: the lal type expression.
        :return: The type declaration associated to the anonymous type, if
            relevant.
        :rtype: lal.BaseTypeDecl
        """
        if hint.is_a(lal.AnonymousType):
            return hint.f_type_decl

    return get_type_decl >> inner_typer


def derived_typer(inner_typer):
    @Transformer.as_transformer
    def get_derived_type(hint):
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.DerivedTypeDef):
                return hint.f_type_def.f_subtype_indication

    return get_derived_type >> inner_typer


def array_typer(indices_typer, component_typer):
    """
    :param types.Typer[lal.AdaNode] indices_typer: Typer for array indices.

    :param types.Typer[lal.AdaNode] component_typer: Typer for array
        components.

    :return: A Typer for array types.

    :rtype: types.Typer[lal.AdaNode]
    """
    @Transformer.as_transformer
    def get_array_attributes(hint):
        if is_array_type_decl(hint):
            return (
                hint.f_type_def.f_indices.f_list,
                hint.f_type_def.f_component_type.f_type_expr
            )

    @Transformer.as_transformer
    def to_array(attribute_types):
        index_types, component_type = attribute_types
        return types.Array(index_types, component_type)

    return (
        get_array_attributes >>
        (indices_typer.lifted() & component_typer) >>
        to_array
    )


def subp_ret_typer(inner):
    """
    :param types.Typer[lal.AdaNode] inner: Typer for return type components.
    :rtype: types.Typer[lal.AdaNode]
    """
    @Transformer.as_transformer
    def get_components(hint):
        if hint.is_a(ExtendedCallReturnType):
            return (
                hint.out_indices,
                hint.out_types if hint.ret_type is None
                else hint.out_types + (hint.ret_type,)
            )

    @Transformer.as_transformer
    def to_output(x):
        out_indices, out_types = x
        return types.FunOutput(tuple(out_indices), out_types)

    return (
        get_components >>
        (Transformer.identity() & inner.lifted()) >>
        to_output
    )


def subtyper(inner):
    """
    :param types.Typer[lal.AdaNode] inner: Typer for base types.
    :rtype: types.Typer[lal.AdaNode]
    """
    @types.typer
    def get_subtyped(hint):
        if hint.is_a(lal.SubtypeDecl):
            return hint.f_subtype

    return get_subtyped >> inner


@types.typer
def ram_typer(hint):
    if hint.is_a(StackType):
        return types.DataStorage()


_unknown_type = types.Unknown()


@types.memoizing_typer
@types.typer
def unknown_typer(_):
    return _unknown_type


@types.memoizing_typer
@types.typer
def none_typer(hint):
    if hint is None:
        return _unknown_type


def standard_typer(ctx):
    """
    :return: A Typer for Ada standard types of programs parsed using
        this extraction context.

    :rtype: types.Typer[lal.AdaNode]
    """
    bool_type = ctx.evaluator.bool
    int_type = ctx.evaluator.int
    char_type = ctx.evaluator.char

    @types.typer
    def typer(hint):
        """
        :param lal.AdaNode hint: the lal type.
        :return: The corresponding lalcheck type.
        :rtype: types.Boolean | types.IntRange
        """
        if hint == bool_type:
            return types.Boolean()
        elif hint == int_type:
            return types.IntRange(-2 ** 31, 2 ** 31 - 1)
        elif hint == char_type:
            return types.ASCIICharacter()

    return typer


def model_typer(ctx, inner_typer):
    @Transformer.as_transformer
    def find_modeled_type(hint):
        model = ctx.type_models.get(hint, None)
        if model is not None:
            return hint, model

    @Transformer.as_transformer
    def to_model_type(tpes):
        return types.ModeledType(*tpes)

    return find_modeled_type >> inner_typer.lifted() >> to_model_type


def default_typer(ctx, fallback_typer):
    """
    :return: The default Typer for Ada programs parsed using this
        extraction context.

    :rtype: types.Typer[lal.AdaNode]
    """

    std_typer = standard_typer(ctx)

    @types.memoizing_typer
    @types.delegating_typer
    def typer():
        return model_typer(ctx, typer_without_model) | typer_without_model

    @types.memoizing_typer
    @types.delegating_typer
    def typer_without_model():
        """
        :rtype: types.Typer[lal.AdaNode]
        """
        typr = (none_typer |
                std_typer |
                int_range_typer |
                real_range_typer |
                int_mod_typer |
                enum_typer |
                access_typer |
                record_typer(record_component_typer(typer)) |
                name_typer(typer) |
                anonymous_typer(typer) |
                derived_typer(typer) |
                array_typer(typer, typer) |
                subp_ret_typer(typer) |
                subtyper(typer) |
                ram_typer)

        return typr if fallback_typer is None else typr | fallback_typer

    return typer
