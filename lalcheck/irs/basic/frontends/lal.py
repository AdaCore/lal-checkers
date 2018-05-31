"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt, purpose
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.constants import ops, lits, access_paths
from lalcheck.utils import KeyCounter, Transformer, profile
from lalcheck import types

from funcy.calc import memoize


_lal_op_type_to_symbol = {
    (lal.OpLt, 2): ops.LT,
    (lal.OpLte, 2): ops.LE,
    (lal.OpEq, 2): ops.EQ,
    (lal.OpNeq, 2): ops.NEQ,
    (lal.OpGte, 2): ops.GE,
    (lal.OpGt, 2): ops.GT,
    (lal.OpAnd, 2): ops.AND,
    (lal.OpOr, 2): ops.OR,
    (lal.OpPlus, 2): ops.PLUS,
    (lal.OpMinus, 2): ops.MINUS,
    (lal.OpDoubleDot, 2): ops.DOT_DOT,

    (lal.OpMinus, 1): ops.NEG,
    (lal.OpNot, 1): ops.NOT,
}

_attr_to_unop = {
    'first': ops.GET_FIRST,
    'last': ops.GET_LAST,
    'model': ops.GET_MODEL
}


class Mode(object):
    """
    Represent the mode of an Ada variable.
    """
    Local = 0
    Global = 1
    Out = 2

    @staticmethod
    def from_lal_mode(mode):
        """
        Returns the equivalent mode.
        :param lal.Mode mode: The lal mode.
        :rtype: int
        """
        if mode.is_a(lal.ModeIn, lal.ModeDefault):
            return Mode.Local
        elif mode.is_a(lal.ModeOut, lal.ModeInOut):
            return Mode.Out


class _RecordField(object):
    """
    Data structure that holds information about the field of a record.
    """
    def __init__(self, decl, record_decl, name, index, conditions=[]):
        """
        :param lal.ComponentDecl | lal.DiscriminantSpec decl: The node where
            the field is declared.

        :param lal.TypeDecl record_decl: The record declaration in which this
            field is defined.

        :param lal.Identifier name: The identifier of the field.

        :param int index: The index of the field inside the record.

        :param list[(lal.Identifier, lal.AlternativesList)] conditions:
            The conditions that must hold in order for this field to exist,
            where a condition is a tuple with (the selector discriminant,
            the value(s) that it must have for the condition to hold).
        """
        self.decl = decl
        self.record_decl = record_decl
        self.name = name
        self.index = index
        self.conds = conditions

    def is_referred_by(self, name):
        """
        Returns True if the given identifier refers to this record field.

        :param lal.Identifier name: The identifier to test.
        :rtype: bool
        """
        return (name.p_referenced_decl == self.decl and
                name.text == self.name.text)

    def field_type_expr(self):
        """
        Returns the type of this field, be it a component decl or a
        discriminant spec.

        :rtype: lal.BaseTypeDecl
        """
        if self.decl.is_a(lal.ComponentDecl):
            return self.decl.f_component_def.f_type_expr
        else:
            return self.decl.f_type_expr


class _ValueHolder(object):
    """
    Holds a value.
    """
    def __init__(self, init=None):
        """
        :param T init: The value to initialize the held value with.
        """
        self.value = init


class _ExtendedCallReturnType(object):
    def __init__(self, out_indices, out_types, ret_type=None):
        self.out_indices = out_indices
        self.out_types = out_types
        self.ret_type = ret_type

    def __eq__(self, other):
        return (isinstance(other, _ExtendedCallReturnType) and
                self.out_indices == other.out_indices and
                self.out_types == other.out_types and
                self.ret_type == other.ret_type)

    def __hash__(self):
        return hash((self.out_indices, self.out_types, self.ret_type))

    def is_a(self, tpe):
        return isinstance(self, tpe)


class _PointerType(object):
    def __init__(self, elem_type):
        self.elem_type = elem_type

    def __eq__(self, other):
        return (isinstance(other, _PointerType) and
                self.elem_type == other.elem_type)

    def __hash__(self):
        return hash(self.elem_type)

    def is_a(self, tpe):
        return isinstance(self, tpe)


class _StackType(object):
    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, _StackType)

    def __hash__(self):
        return hash(self.__class__)

    def is_a(self, tpe):
        return isinstance(self, tpe)


def _closest(node, *tpe):
    """
    Returns the closest parent of this node which is of the given type.
    :param lal.AdaNode node: The node from which to start
    :param type* tpe: The types to look for.
    :rtype: lal.AnaNode | None
    """

    if node.is_a(*tpe):
        return node
    elif node.parent is not None:
        return _closest(node.parent, *tpe)
    else:
        return None


@memoize
def _record_fields(record_decl):
    """
    Returns an iterable of the fields of the given record, where a field is
    identified by the pair (its declaration, its name).

    :param lal.TypeDecl record_decl: The record whose fields to list.
    :rtype: list[_RecordField]
    """
    res = []
    i = _ValueHolder(0)

    def add_component_fields(component_list, conds=[]):
        """
        :param lal.ComponentList component_list:
        :param list[(lal.Identifier, lal.AlternativesList)] conds:
        :return:
        """
        for component in component_list.f_components:
            if not component.is_a(lal.NullComponentDecl):
                for name in component.f_ids:
                    res.append(_RecordField(
                        component, record_decl, name, i.value, conds)
                    )
                    i.value += 1

        variant_part = component_list.f_variant_part
        if variant_part is not None:
            for variant in variant_part.f_variant:
                add_component_fields(variant.f_components, conds + [(
                    variant_part.f_discr_name,
                    variant.f_choices
                )])

    if record_decl.f_discriminants is not None:
        for discr in record_decl.f_discriminants.f_discr_specs:
            for name in discr.f_ids:
                res.append(_RecordField(discr, record_decl, name, i.value))
                i.value += 1

    tp_def = record_decl.f_type_def

    if tp_def.is_a(lal.DerivedTypeDef):
        record_def = tp_def.f_record_extension
    else:
        record_def = tp_def.f_record_def

    add_component_fields(record_def.f_components)
    return res


@memoize
def _proc_parameters(proc):
    """
    Returns the parameters from the given subprogram that have the "Out" mode.

    :param lal.SubpBody | lal.SubpDecl proc: The procedure for which to
        retrieve the parameters.

    :rtype: iterable[(int, lal.Identifier, lal.ParamSpec)]
    """
    spec = proc.f_subp_spec

    def gen():
        i = 0

        if spec.f_subp_params is not None:
            for param in spec.f_subp_params.f_params:
                for name in param.f_ids:
                    yield (i, name, param)
                    i += 1

    return list(gen())


@memoize
def _subprogram_param_types(subp):
    """
    Returns a tuple containing the types of each parameter of the given
    subprogram.

    :param lal.SubpDecl subp: The subprogram to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    params = _proc_parameters(subp)
    return tuple(p.f_type_expr for _, _, p in params)


def _array_access_param_types(array_type_decl):
    """
    Returns a tuple containing the types of each parameter of the function
    that accesses an element of an array of the given array type.

    :param lal.TypeDecl array_type_decl: The array type to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    array_def = array_type_decl.f_type_def
    index_types = tuple(array_def.f_indices.f_list)
    return (array_type_decl,) + index_types


def _array_assign_param_types(array_type_decl):
    """
    Returns a tuple containing the types of each parameter of the function
    that updates an element of an array of the given array type.

    :param lal.TypeDecl array_type_decl: The array type to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    array_def = array_type_decl.f_type_def
    component_type = array_def.f_component_type.f_type_expr
    index_types = tuple(array_def.f_indices.f_list)
    return (array_type_decl, component_type) + index_types


def _field_access_param_types(field_info):
    """
    Returns a tuple containing the types of each parameter of the function
    that accesses the given field.

    :param _RecordField field_info: The field to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    return field_info.record_decl,


def _field_assign_param_types(field_info):
    """
    Returns a tuple containing the types of each parameter of the function
    that updates the given field.

    :param _RecordField field_info: The field to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    return field_info.record_decl, field_info.field_type_expr()


def _stack_assign_param_types(var):
    """
    Returns a tuple containing the types of each parameter of the function
    that updates the given variable that is stored in the stack.

    :param irt.Variable var: The spilled variable to consider.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    return _StackType(), var.data.type_hint


def _deref_assign_param_types(ptr_type, val_type):
    """
    Returns a tuple containing the types of each parameter of the function
    that assigns a value of the given val_type to a pointer of type ptr_type.

    :param lal.BaseTypeDecl ptr_type: The type of the access.
    :param lal.BaseTypeDecl val_type: The type of the value.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    return _StackType(), ptr_type, val_type


def _find_cousin_conditions(record_fields, cond_prefix):
    """
    Returns the set of conditions that come after the given prefix list of
    conditions.

    :param list[_RecordField] record_fields: The fields of the record.

    :param list[(lal.Identifier, lal.AlternativesList)] cond_prefix:
        The prefix list of conditions.

    :rtype: set[(lal.Identifier, lal.AlternativesList)]
    """
    return set(
        field.conds[len(cond_prefix)]
        for field in record_fields
        if len(field.conds) == len(cond_prefix) + 1
        if field.conds[:len(cond_prefix)] == cond_prefix
    )


def _field_info(field_id):
    """
    Computes the index of the given record's field. Example:
    type Foo is record
      x, y : Integer;
      z : Integer;
    end record;

    => "x" has index 0, "y" has index 1, "z" has index 2

    :param lal.Identifier field_id: An identifier referring to a record's
        field.
    :rtype: _RecordField
    """
    ref = field_id.p_referenced_decl
    record_decl = _closest(ref, lal.TypeDecl)

    return next(
        field
        for field in _record_fields(record_decl)
        if field.is_referred_by(field_id)
    )


def _is_record_field(ident):
    """
    Returns true if the given name refers to the field of a record.
    :param lal.Identifier ident: The identifier to check.
    """
    ref = ident.p_referenced_decl
    return (ref is not None and
            ref.is_a(lal.ComponentDecl, lal.DiscriminantSpec))


def _is_array_type_decl(tpe):
    """
    Returns True iff the given type is an array type decl.
    :param lal.AdaNode tpe: The type to check.
    """
    if tpe is not None and tpe.is_a(lal.TypeDecl):
        return tpe.f_type_def.is_a(lal.ArrayTypeDef)

    return False


_IGNORES_GLOBAL_STATE = 0
_READS_GLOBAL_STATE = 1
_WRITES_GLOBAL_STATE = 2


@memoize
def _contains_access_type(typer, type_hint):
    """
    Returns True if the given type is an access type or contains one, None if
    unknown.

    :param types.Typer[lal.AdaNode] typer:
    :param lal.AdaNode type_hint:
    :rtype: bool | None.
    """
    def has_pointer(tpe):
        return tpe.is_a(types.Pointer) or any(
            has_pointer(child) for child in tpe.children()
        )

    try:
        return has_pointer(typer.get(type_hint))
    except ValueError:
        return None


@memoize
def _find_global_access(typer, proc):
    """
    Analyzes the side effects of the given subprogram. returns the following:
    - _IGNORES_GLOBAL_STATE if the function doesn't access the global state.
    - _READS_GLOBAL_STATE if the function may read from the global state.
    - _WRITES_GLOBAL_STATE if the function may write to the global state.

    :param types.Typer[lal.AdaNode] typer:
    :param lal.SubpBody | lal.SubpDecl proc:
    :return:
    """
    for i, id, param in _proc_parameters(proc):
        if _contains_access_type(typer, param.f_type_expr):
            return _WRITES_GLOBAL_STATE

    return _IGNORES_GLOBAL_STATE


def _index_of(elem, iterable):
    i = 0
    for x in iterable:
        if x == elem:
            return i
        i += 1
    return -1


@profile()
@memoize
def _find_vars_to_spill(ctx, node):
    """
    :param lal.AdaNode node:
    :return:
    """
    if node is None:
        return set()

    def accessed_var(expr):
        if expr.is_a(lal.Identifier):
            return expr.p_xref
        elif expr.is_a(lal.DottedName):
            return accessed_var(expr.f_prefix)
        elif expr.is_a(lal.AttributeRef) and expr.f_attribute.text == 'Model':
            return accessed_var(expr.f_prefix)
        else:
            return None

    def is_accessed(node):
        if node.parent.is_a(lal.AttributeRef):
            if node.parent.f_attribute.text == 'Access':
                if node.parent.f_prefix == node:
                    return True
        elif node.parent.is_a(lal.ParamAssoc):
            try:
                call_expr = node.parent.parent.parent
                if call_expr.is_a(lal.CallExpr):
                    ref = call_expr.f_name.p_referenced_decl
                    if (ref is not None and
                            ref.is_a(lal.SubpBody, lal.SubpDecl)):
                        procs = [ref]
                        model = ctx.fun_models.get(ref, None)
                        if model is not None:
                            procs.append(model)

                        for proc in procs:
                            params_to_spill = _find_vars_to_spill(
                                ctx,
                                proc.f_aspects
                            )

                            if len(params_to_spill) > 0:
                                param_indexes_to_spill = {
                                    _index_of(param, param.parent)
                                    for param in params_to_spill
                                }
                                arg_index = _index_of(
                                    node.parent, node.parent.parent
                                )
                                return arg_index in param_indexes_to_spill

            except lal.PropertyError:
                pass
        return False

    return {accessed_var(n) for n in node.findall(is_accessed)}


@memoize
def retrieve_function_contracts(ctx, proc):
    """
    :param lal.SubpDecl | lal.SubpBody proc:
    :return:
    """
    pres, posts = [], []

    def collect_pre_posts(proc):
        if proc.f_aspects is not None:
            for aspect in proc.f_aspects.f_aspect_assocs:
                if aspect.f_id.text == 'Pre':
                    pres.append((aspect.f_expr, True))
                elif aspect.f_id.text == 'Model_Pre':
                    pres.append((aspect.f_expr, True))
                elif aspect.f_id.text == 'Post':
                    posts.append((aspect.f_expr, True))
                elif aspect.f_id.text == 'Model_Post':
                    posts.append((aspect.f_expr, False))

    collect_pre_posts(proc)

    model = ctx.fun_models.get(proc, None)
    if model is not None:
        collect_pre_posts(model)

    return pres, posts


@profile()
def _gen_ir(ctx, subp, typer):
    """
    Generates Basic intermediate representation from a lal subprogram body.

    :param ExtractionContext ctx: The program extraction context.

    :param lal.SubpBody subp: The subprogram body from which to generate IR.

    :param types.Typer[lal.AdaNode] typer: A typer for any lal AdaNode.

    :return: a Basic Program.

    :rtype: irt.Program
    """

    var_decls = {}
    param_vars = []
    tmp_vars = KeyCounter()

    # Used to assign a unique index to each variable.
    var_idx = _ValueHolder(0)

    def next_var_idx():
        var_idx.value += 1
        return var_idx.value - 1

    # A synthetic variable used to store the result of a function (what is
    # returned). Use a ValueHolder so it can be rebound by closures (e.g.
    # transform_spec).
    result_var = _ValueHolder()

    # Pre-transform every label, as a label might not have been seen yet when
    # transforming a goto statement.
    labels = {
        label_decl: irt.LabelStmt(
            label_decl.f_name.text,
            orig_node=label_decl
        )
        for label_decl in subp.findall(lal.LabelDecl)
    }

    # Make a label for the end of the function, so that return statement
    # can jump to it.
    func_end_label = irt.LabelStmt("end")

    # Store the loops which we are currently in while traversing the syntax
    # tree. The tuple (loop_statement, exit_label) is stored.
    loop_stack = []

    # Store a mapping of lal.AdaNode to irt.Node. Whenever a
    # lal.Identifier refers to such a key present in the mapping, it is
    # substituted by the corresponding irt node.
    substitutions = {}

    # Contains variables that are spilled.
    to_spill = _find_vars_to_spill(ctx, subp)

    stack = irt.Identifier(
        irt.Variable(
            "$stack",
            type_hint=_StackType(),
            mode=Mode.Local,
            index=next_var_idx()
        ),
        type_hint=_StackType()
    )

    def fresh_name(name):
        """
        :param str name: The base name of the variable.
        :return: A fresh name that shouldn't collide with any other.
        :rtype: str
        """
        return "{}{}".format(name, tmp_vars.get_incr(name))

    def transform_operator(lal_op, arity):
        """
        :param lal.Op lal_op: The lal operator to convert.
        :param int arity: The arity of the operator
        :return: The corresponding Basic IR operator.
        :rtype: str
        """
        return _lal_op_type_to_symbol[type(lal_op), arity]

    def unimplemented(node):
        """
        To be called when the implementation cannot transform some node.

        :param lal.AdaNode node: The node that cannot be transformed.
        :raise NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Cannot transform "{}" ({})'.format(node.text, type(node))
        )

    def unimplemented_expr(expr):
        """
        To be called when the implementation cannot transform some expression.

        This will return a compatible expression (type-wise) that is sound
        w.r.t to the original expression, but less precise.

        :param lal.Expr expr: The expr that this routine fails to transform.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        return [], new_expression_replacing_var("unimpl", expr)

    def unimplemented_dest(dest):
        """
        To be called when the implementation cannot transform a destination
        (see gen_actual_dest).

        :param lal.Expr dest: The destination expression
        :rtype: list[irt.Stmt], (irt.Identifier, irt.Expr)
        """
        unimplemented(dest)

    def new_expression_replacing_var(name, replaced_expr):
        """
        Some lal expressions such as if expressions and case expressions
        cannot be described in the Basic IR as an expression and must therefore
        be transformed as several statements which modify a temporary variable
        that holds the result of the original expression.

        This function creates such a variable.

        :param str name: The base name of the synthetic variable to generate.

        :param lal.Expr replaced_expr: The expression being replaced by this
            variable.

        :return: A new identifier of a new variable.

        :rtype: irt.Identifier
        """
        return irt.Identifier(
            irt.Variable(
                fresh_name(name),
                purpose=purpose.SyntheticVariable(),
                type_hint=replaced_expr.p_expression_type,
                orig_node=replaced_expr,
                mode=Mode.Local,
                index=next_var_idx()
            ),
            type_hint=replaced_expr.p_expression_type,
            orig_node=replaced_expr
        )

    def get_var(expr, ref):
        """
        :param lal.Expr expr:
        :param lal.ObjectDecl ref:
        :return:
        """
        var = var_decls.get(ref)
        if var is None:
            return new_expression_replacing_var("unknown", expr)

        if ref in to_spill:
            return irt.FunCall(
                ops.GetName(var.data.index),
                [stack],
                type_hint=var.data.type_hint,
                orig_node=expr
            )
        else:
            return irt.Identifier(
                var,
                type_hint=var.data.type_hint,
                orig_node=expr
            )

    def update_var(expr, ref):
        var = var_decls[ref]
        if ref in to_spill:
            return irt.FunCall(
                ops.GetName(var.data.index),
                [stack],
                type_hint=var.data.type_hint,
                orig_node=expr
            )
        else:
            return irt.Identifier(
                var,
                type_hint=var.data.type_hint,
                orig_node=expr
            )

    def read_var(expr, ref):
        """
        :param lal.Expr expr:
        :param lal.ObjectDecl ref:
        :return:
        """
        var = var_decls[ref]
        if ref in to_spill:
            return irt.FunCall(
                ops.GetName(var.data.index),
                [stack],
                type_hint=var.data.type_hint,
                orig_node=expr
            )
        else:
            return [irt.ReadStmt(
                get_var(expr, ref),
                orig_node=expr
            )]

    def gen_split_stmt(cond, then_stmts, else_stmts, **data):
        """
        :param lal.Expr cond: The condition of the if statement.

        :param iterable[irt.Stmt] then_stmts: The already transformed then
            statements.

        :param iterable[irt.Stmt] else_stmts: The already transformed else
            statements.

        :param **object data: user data on the generated split statement.

        :return: The corresponding split-assume statements.

        :rtype: list[irt.Stmt]
        """
        cond_pre_stmts, cond = transform_expr(cond)
        not_cond = irt.FunCall(
            ops.NOT,
            [cond],
            type_hint=cond.data.type_hint
        )

        assume_cond, assume_not_cond = (
            irt.AssumeStmt(x) for x in [cond, not_cond]
        )

        return cond_pre_stmts + [
            irt.SplitStmt(
                [
                    [assume_cond] + then_stmts,
                    [assume_not_cond] + else_stmts,
                ],
                **data
            )
        ]

    def binexpr_builder(op, type_hint):
        """
        :param str op: The binary operator.

        :param lal.AdaNode type_hint: The type hint to attach to the
            binary expressions.

        :return: A function taking an lhs and an rhs and returning a binary
            expression using this builder's operator.

        :rtype: (irt.Expr, irt.Expr)->irt.Expr
        """
        def build(lhs, rhs):
            return irt.FunCall(
                op, [lhs, rhs],
                type_hint=type_hint
            )
        return build

    def gen_case_condition(expr, values):
        """
        Example:

        `gen_case_condition(X, [1, 2, 10 .. 20])`

        Will generate the following condition:

        `X == 1 || X == 2 || X >= 10 || X <= 20`

        :param irt.Expr expr: The case's selector expression.

        :param list[int | ConstExprEvaluator.Range] values:
            The different possible literal values.

        :return: An expression corresponding to the condition check for
            entering an alternative of an Ada case statement.

        :rtype: irt.Expr
        """
        def gen_lit(value):
            if isinstance(value, int):
                return irt.Lit(
                    value,
                    type_hint=ctx.evaluator.universal_int
                )
            raise NotImplementedError("Cannot transform literal")

        def gen_single(value):
            if isinstance(value, int):
                return irt.FunCall(
                    ops.EQ,
                    [expr, gen_lit(value)],
                    type_hint=ctx.evaluator.bool
                )
            elif isinstance(value, ConstExprEvaluator.Range):
                if (isinstance(value.first, int) and
                        isinstance(value.last, int)):
                    return irt.FunCall(
                        ops.AND,
                        [
                            irt.FunCall(
                                ops.GE,
                                [expr, gen_lit(value.first)],
                                type_hint=ctx.evaluator.bool
                            ),
                            irt.FunCall(
                                ops.LE,
                                [expr, gen_lit(value.last)],
                                type_hint=ctx.evaluator.bool
                            )
                        ],
                        type_hint=ctx.evaluator.bool
                    )

            raise NotImplementedError("Cannot transform when condition")

        conditions = [gen_single(value) for value in values]

        if len(conditions) > 1:
            return reduce(
                binexpr_builder(ops.OR, ctx.evaluator.bool),
                conditions
            )
        else:
            return conditions[0]

    def transform_short_circuit_ops(bin_expr):
        """
        :param lal.BinOp bin_expr: A binary expression that involves a short-
            circuit operation (and then / or else).

        :return: The transformation of the given expression.

        :rtype:  (list[irt.Stmt], irt.Expr)
        """
        res = new_expression_replacing_var("tmp", bin_expr)
        res_eq_true, res_eq_false = (irt.AssignStmt(
            res,
            irt.Lit(
                literal,
                type_hint=bin_expr.p_expression_type
            )
        ) for literal in [lits.TRUE, lits.FALSE])

        if bin_expr.f_op.is_a(lal.OpAndThen):
            # And then is transformed as such:
            #
            # Ada:
            # ------------
            # x := C1 and then C2;
            #
            # Basic IR:
            # -------------
            # split:
            #   assume(C1)
            #   split:
            #     assume(C2)
            #     res = True
            #   |:
            #     assume(!C2)
            #     res = False
            # |:
            #   assume(!C1)
            #   res = False
            # x = res

            res_stmts = gen_split_stmt(
                bin_expr.f_left,
                gen_split_stmt(
                    bin_expr.f_right,
                    [res_eq_true],
                    [res_eq_false]
                ),
                [res_eq_false]
            )
        else:
            # Or else is transformed as such:
            #
            # Ada:
            # ------------
            # x := C1 or else C2;
            #
            # Basic IR:
            # -------------
            # split:
            #   assume(C1)
            #   res = True
            # |:
            #   assume(!C1)
            #   split:
            #     assume(C2)
            #     res = True
            #   |:
            #     assume(!C2)
            #     res = False
            # x = res
            res_stmts = gen_split_stmt(
                bin_expr.f_left,
                [res_eq_true],
                gen_split_stmt(
                    bin_expr.f_right,
                    [res_eq_true],
                    [res_eq_false]
                )
            )

        return res_stmts, res

    def if_expr_alt_transformer_of(var):
        """
        :param irt.Identifier var: The synthetic variable used in the
            transformation of an if expression.

        :return: A transformer for if expression's alternatives.

        :rtype: (lal.Expr)->list[irt.Stmt]
        """
        def transformer(expr):
            """
            :param lal.Expr expr: The if-expression's alternative's expression.
            :return: Its transformation.
            :rtype: list[irt.Stmt]
            """
            pre_stmts, tr_expr = transform_expr(expr)
            return pre_stmts + [irt.AssignStmt(var, tr_expr)]

        return transformer

    def case_expr_alt_transformer_of(var):
        """
        :param irt.Identifier var: The synthetic variable used in the
            transformation of a case expression.

        :return: A transformer for case expression's alternatives.

        :rtype: (lal.CaseExprAlternative)->list[irt.Stmt]
        """

        def transformer(alt):
            """
            :param lal.CaseExprAlternative alt: The case-expression's
                alternative.

            :return: Its transformation.

            :rtype: list[irt.Stmt]
            """
            pre_stmts, tr_expr = transform_expr(alt.f_expr)
            return pre_stmts + [irt.AssignStmt(var, tr_expr)]

        return transformer

    def case_stmt_alt_transformer(alt):
        """
        :param lal.CaseStmtAlternative alt: The case-statement's alternative.
        :return: Its transformation.
        :rtype: list[irt.Stmt]
        """
        return transform_stmts(alt.f_stmts)

    def gen_if_base(alternatives, transformer):
        """
        Transforms a chain of if-elsifs.

        :param iterable[(lal.Expr | None, lal.AbstractNode)] alternatives:
            Each alternative of the chain, represented by a pair holding:
                - The condition of the alternative (None for the "else" one).
                - The "then" part of the alternative.

        :param (lal.AbstractNode)->list[irt.Stmt] transformer: The function
            which transforms the "then" part of an alternative.

        :return: The transformation of the if-elsif chain as chain of nested
            split statements.

        :rtype: list[irt.Stmt]
        """
        cond, lal_node = alternatives[0]
        stmts = transformer(lal_node)

        return stmts if cond is None else gen_split_stmt(
            cond,
            stmts,
            gen_if_base(alternatives[1:], transformer),
            orig_node=lal_node
        )

    def gen_case_base(selector_expr, alternatives, transformer, orig_node):
        """
        Transforms a case construct.

        :param lal.Expr selector_expr: The selector of the case.

        :param iterable[object] alternatives: The alternatives of the case
            construct.

        :param object->list[irt.Stmt] transformer: The function which
            transforms the "then" part of an alternative.

        :param lal.AbstractNode orig_node: The lal node of the case construct.

        :return: The transformation of the case construct as a multi-branch
            split statement.

        :rtype: list[irt.Stmt]
        """

        # Transform the selector expression
        case_pre_stmts, case_expr = transform_expr(selector_expr)

        # Evaluate the choices of each alternative that is not the "others"
        # one. Choices are statically known values, meaning the evaluator
        # should never fail.
        # Also store the transformed statements of each alternative.
        case_alts = [
            (
                [
                    ctx.evaluator.eval(transform_expr(choice)[1])
                    for choice in alt.f_choices
                ],
                transformer(alt)
            )
            for alt in alternatives
            if not any(
                choice.is_a(lal.OthersDesignator)
                for choice in alt.f_choices
            )
        ]

        # Store the transformed statements of the "others" alternative.
        others_potential_stmts = [
            transformer(alt)
            for alt in alternatives
            if any(
                choice.is_a(lal.OthersDesignator)
                for choice in alt.f_choices
            )
        ]

        # Build the conditions that correspond to matching the choices,
        # for each alternative that is not the "others".
        # See `gen_case_condition`.
        alts_conditions = [
            gen_case_condition(case_expr, choices)
            for choices, _ in case_alts
        ]

        # Build the condition for the "others" alternative, which is the
        # negation of the disjunction of all the previous conditions.
        others_condition = irt.FunCall(
            ops.NOT,
            [
                reduce(
                    binexpr_builder(ops.OR, ctx.evaluator.bool),
                    alts_conditions
                )
            ],
            type_hint=ctx.evaluator.bool
        )

        # Generate the branches of the split statement.
        branches = [
            [irt.AssumeStmt(cond)] + stmts
            for cond, (choices, stmts) in
            zip(alts_conditions, case_alts)
        ] + [
            [irt.AssumeStmt(others_condition)] + others_stmts
            for others_stmts in others_potential_stmts
        ]

        return case_pre_stmts + [irt.SplitStmt(
            branches,
            orig_node=orig_node
        )]

    def gen_actual_dest(dest, expr):
        """
        Examples:
        - gen_actual_dest(`x`, "3") is called when transforming `x := 3`. In
          this case, ("x", "3") is returned.

        - gen_actual_dest(`r.p.x`, "12") is called when transforming
          `r.p.x := 12`. It will produce in this case (
            "r",
            "Updated_I(r, Updated_J(Get_I(r), 12))"
          ). Where I is the index of the "p" field in "r", and J of the "x"
          field in "r.p".

        :param lal.Expr dest: The destination of the assignment (lhs).
        :param irt.Expr expr: The expression to assign (rhs).
        :rtype: list[irt.Stmt], (irt.Identifier, irt.Expr)
        """
        if dest.is_a(lal.Identifier, lal.DefiningName):
            ref = dest.p_xref if dest.is_a(lal.Identifier) else dest

            var = var_decls[ref]

            if ref in to_spill:
                updated_index = var.data.index
                return [], (
                    stack,
                    irt.FunCall(
                        ops.UpdatedName(updated_index),
                        [stack, expr],
                        type_hint=stack.data.type_hint,
                        orig_node=dest,
                        param_types=_stack_assign_param_types(var)
                    )
                )
            else:
                return [], (irt.Identifier(
                    var,
                    type_hint=var.data.type_hint,
                    orig_node=dest
                ), expr)

        elif dest.is_a(lal.DottedName) and _is_record_field(dest.f_suffix):
            field_info = _field_info(dest.f_suffix)
            updated_index = field_info.index
            prefix_pre_stmts, prefix_expr = transform_expr(dest.f_prefix)

            exist_stmts = gen_field_existence_condition(
                prefix_expr,
                dest.f_suffix
            )

            pre_stmts, ret = gen_actual_dest(dest.f_prefix, irt.FunCall(
                ops.UpdatedName(updated_index),
                [prefix_expr, expr],
                type_hint=dest.f_prefix.p_expression_type,
                orig_node=dest.f_prefix,
                param_types=_field_assign_param_types(field_info)
            ))
            return prefix_pre_stmts + exist_stmts + pre_stmts, ret

        elif dest.is_a(lal.CallExpr):
            if dest.f_name.p_referenced_decl.is_a(lal.TypeDecl):
                # type conversion
                return unimplemented_dest(dest)

            prefix_pre_stmts, prefix_expr = transform_expr(dest.f_name)

            if dest.f_suffix.is_a(lal.BinOp, lal.AttributeRef):
                return unimplemented_dest(dest)

            try:
                name_tpe = dest.f_name.p_expression_type
                if not typer.get(name_tpe).is_a(types.Array):
                    return unimplemented_dest(dest)
            except ValueError:
                return unimplemented_dest(dest)

            suffixes = [
                transform_expr(suffix.f_r_expr)
                for suffix in dest.f_suffix
            ]
            suffix_pre_stmts = [
                suffix_stmt
                for suffix in suffixes
                for suffix_stmt in suffix[0]
            ]
            suffix_exprs = [suffix[1] for suffix in suffixes]
            pre_stmts, ret = gen_actual_dest(dest.f_name, irt.FunCall(
                ops.UPDATED,
                [prefix_expr, expr] + suffix_exprs,
                type_hint=dest.f_name.p_expression_type,
                orig_node=dest,
                param_types=_array_assign_param_types(
                    dest.f_name.p_expression_type
                )
            ))
            return (
                prefix_pre_stmts + pre_stmts + suffix_pre_stmts,
                ret
            )

        elif dest.is_a(lal.ExplicitDeref):
            prefix_pre_stmts, prefix_expr = transform_expr(dest.f_prefix)

            return prefix_pre_stmts, (stack, irt.FunCall(
                ops.UPDATED,
                [stack, prefix_expr, expr],
                type_hint=stack.data.type_hint,
                orig_node=dest,
                param_types=_deref_assign_param_types(
                    dest.f_prefix.p_expression_type,
                    dest.p_expression_type
                )
            ))

        return unimplemented_dest(dest)

    def gen_field_existence_condition(prefix, field):
        """
        Returns a list of assume statements describing the conditions that
        must hold for the given field to exist for the given object.

        :param irt.Expr prefix: The object whose field is being accessed.
        :param lal.Identifier field: The field being accessed
        :rtype: list[irt.AssumeStmt]
        """
        info = _field_info(field)
        all_fields = _record_fields(
            _closest(field.p_referenced_decl, lal.TypeDecl)
        )

        res = []
        for i, (discr, alternatives) in enumerate(info.conds):
            discr_getter = irt.FunCall(
                ops.GetName(_field_info(discr).index), [prefix],
                type_hint=discr.p_expression_type
            )

            if not any(x.is_a(lal.OthersDesignator) for x in alternatives):
                choices = [
                    ctx.evaluator.eval(transform_expr(choice)[1])
                    for choice in alternatives
                ]

                condition = gen_case_condition(discr_getter, choices)
            else:
                # Find all the other conditions
                cousin_conditions = _find_cousin_conditions(
                    all_fields, info.conds[:i]
                )

                other_alts = [
                    alts
                    for _, alts in cousin_conditions
                    if not any(
                        x.is_a(lal.OthersDesignator) for x in alts
                    )
                ]

                alts_conditions = [
                    gen_case_condition(discr_getter, [
                        ctx.evaluator.eval(transform_expr(choice)[1])
                        for choice in alt
                    ])
                    for alt in other_alts
                ]

                condition = irt.FunCall(
                    ops.NOT,
                    [
                        reduce(
                            binexpr_builder(ops.OR, ctx.evaluator.bool),
                            alts_conditions
                        )
                    ],
                    type_hint=ctx.evaluator.bool
                )

            res.append(irt.AssumeStmt(
                condition,
                purpose=purpose.ExistCheck(
                    prefix,
                    field.text,
                    discr.text
                )
            ))

        return res

    def gen_access_path(expr):
        if expr.is_a(lal.Identifier):
            ref = expr.p_xref

            if ref in substitutions:
                expr = substitutions[ref][1].data.orig_node
                ref = expr.p_xref

            var = var_decls[ref]

            return irt.FunCall(
                access_paths.Var(var.data.index),
                [stack],
                type_hint=_PointerType(expr.p_expression_type),
                additional_arg=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.DottedName) and _is_record_field(expr.f_suffix):
            info = _field_info(expr.f_suffix)
            if info is None:
                unimplemented(expr)

            updated_index = info.index

            return irt.FunCall(
                access_paths.Field(updated_index),
                [gen_access_path(expr.f_prefix)],
                type_hint=_PointerType(expr.p_expression_type),
                additional_arg=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.AttributeRef) and expr.f_attribute.text == "Model":
            updated_index = 1

            return irt.FunCall(
                access_paths.Field(updated_index),
                [gen_access_path(expr.f_prefix)],
                type_hint=_PointerType(expr.p_expression_type),
                additional_arg=expr.p_expression_type,
                orig_node=expr
            )

        unimplemented(expr)

    def gen_assignment(assign_dest, expr_pre_stmts, expr, orig_node=None):
        """
        Returns the statements equivalent to the assignment of the given
        expression to the given destination.

        :param lal.Expr assign_dest: The destination of the assignment.

        :param irt.Expr assign_expr: The expression assigned, already
            transformed.

        :param lal.AdaNode orig_node: The original node.

        :rtype: list[irt.Stmt]
        """
        dest_pre_stmts, (dest, updated_expr) = gen_actual_dest(
            assign_dest, expr
        )
        return dest_pre_stmts + expr_pre_stmts + [
            irt.AssignStmt(
                dest,
                updated_expr,
                orig_node=orig_node
            )
        ]

    @profile()
    def gen_contract_conditions(proc, pres, posts,
                                args_in, args_out, ret, orig_call):
        """
        Generates assume statements for checking pre/post conditions of the
        given procedure.

        :param lal.SubpBody | lal.SubpDecl proc: The procedure for which to
            generate contract checking conditions.

        :param list[lal.Expr] pres: The list of precondition expressions.

        :param list[lal.Expr] posts: The list of postcondition expressions.

        :param list[irt.Expr] args_in: The expressions used to retrieve
            the input arguments of the function, indexed by their position.

        :param dict[int, irt.Expr] args_out: The expressions used to retrieve
            the output arguments of the function, indexed by their position.

        :param irt.Expr | None ret: The expression used to retrieve the result
            of the function, if any.

        :param lal.AdaNode orig_call: The original call node.
        """

        procs = [proc]
        model = ctx.fun_models.get(proc, None)
        if model is not None:
            procs.append(model)

        for proc in procs:
            for i, name, param in _proc_parameters(proc):
                substitutions[name] = ([], args_in[i])

        pre_stmts = [
            stmt
            for pre, must_check in pres
            for stmts, expr in [transform_expr(pre)]
            for stmt in stmts + [irt.AssumeStmt(
                expr,
                purpose=purpose.ContractCheck("precondition", orig_call),
                orig_node=pre
            )]
        ]

        for proc in procs:
            substitutions[proc, 'result'] = ([], ret)

        for proc in procs:
            for i, name, param in _proc_parameters(proc):
                if i in args_out:
                    substitutions[name, 'old'] = ([], args_in[i])
                    substitutions[name] = ([], args_out[i])

        post_stmts = [
            stmt
            for post, must_check in posts
            for stmts, expr in [transform_expr(post)]
            for stmt in stmts + [irt.AssumeStmt(
                expr,
                purpose=(
                    purpose.ContractCheck("postcondition", orig_call)
                    if must_check else None
                ),
                orig_node=post
            )]
        ]

        return pre_stmts, post_stmts

    @profile()
    def gen_call_expr(prefix, args, type_hint, orig_node):
        """
        Call expressions are transformed the following way:

        Ada:
        ----------------
        r := F(x_1, ..., x_n);

        Basic IR:
        ----------------

        1. `F` does not have any out parameters:

        r := F(x_1, ..., x_n);

        2. `F` has out parameters (say, x_1 and x_2):

        tmp := F(x_1, ..., x_n);
        x_1 := Get_0(tmp)
        x_2 := Get_1(tmp)
        r := Get_2(tmp)

        Additionally, contract checks are added around the call.

        :param lal.Expr prefix: The object being called.
        :param lal.BinOp|lal.AssocList args: The arguments passed.
        :param lal.AdaNode type_hint: The type of the call expression.
        :param lal.AdaNode orig_node: The original call node.
        :rtype: (list[irt.Stmt], irt.Expr)
        """

        if isinstance(args, lal.BinOp):
            # array slices
            return unimplemented_expr(orig_node)

        if any(x.f_designator is not None for x in args):
            return unimplemented_expr(orig_node)

        arg_exprs = [suffix.f_r_expr for suffix in args]

        suffixes = [transform_expr(e) for e in arg_exprs]
        suffix_pre_stmts = [
            suffix_stmt
            for suffix in suffixes
            for suffix_stmt in suffix[0]
        ]
        suffix_exprs = [suffix[1] for suffix in suffixes]

        def gen_out_arg_assignment(i):
            def do(out_expr):
                return gen_assignment(arg_exprs[i], [], out_expr)
            return do

        if prefix.is_a(lal.Identifier, lal.DottedName):
            ref = prefix.p_referenced_decl
            if ref is not None and ref.is_a(lal.SubpBody, lal.SubpDecl):
                # The call target is statically known.

                if any(p.f_default_expr is not None
                       for _, _, p in _proc_parameters(ref)):
                    return unimplemented_expr(orig_node)

                if ref.metadata.f_dottable_subp:
                    # handle dot calls
                    prefix_expr = prefix.f_prefix
                    arg_exprs.insert(0, prefix_expr)
                    prefix_expr_tr = transform_expr(prefix_expr)
                    suffixes.insert(0, prefix_expr_tr)
                    suffix_pre_stmts = prefix_expr_tr[0] + suffix_pre_stmts
                    suffix_exprs.insert(0, prefix_expr_tr[1])

                out_params = [
                    (i, param.f_type_expr, gen_out_arg_assignment(i))
                    for i, _, param in _proc_parameters(ref)
                    if param.f_mode.is_a(lal.ModeOut, lal.ModeInOut)
                ]

                # Pass stack too

                global_access = _find_global_access(typer, ref)

                if (global_access == _READS_GLOBAL_STATE or
                        global_access == _WRITES_GLOBAL_STATE):
                    offset = var_idx.value

                    suffix_exprs.append(irt.FunCall(
                        ops.OffsetName(offset),
                        [stack],
                        type_hint=stack.data.type_hint
                    ))

                    if global_access == _WRITES_GLOBAL_STATE:
                        out_params.append((
                            len(_proc_parameters(ref)),
                            stack.data.type_hint,
                            lambda expr: [irt.AssignStmt(
                                stack,
                                irt.FunCall(
                                    ops.COPY_OFFSET,
                                    [stack, expr],
                                    type_hint=stack.data.type_hint
                                )
                            )]
                        ))

                pres, posts = retrieve_function_contracts(ctx, ref)

                if len(out_params) == len(pres) == len(posts) == 0:
                    return suffix_pre_stmts, irt.FunCall(
                        ref,
                        suffix_exprs,
                        orig_node=orig_node,
                        type_hint=type_hint,
                        param_types=_subprogram_param_types(ref)
                    )
                else:
                    ret_tpe = _ExtendedCallReturnType(
                        tuple(index for index, _, _ in out_params),
                        tuple(param_type for _, param_type, _ in out_params),
                        type_hint
                    ) if len(out_params) != 0 else type_hint

                    ret_var = irt.Identifier(
                        irt.Variable(
                            fresh_name("ret"),
                            purpose=purpose.SyntheticVariable(),
                            type_hint=ret_tpe,
                            orig_node=orig_node,
                            mode=Mode.Local,
                            index=next_var_idx()
                        ),
                        type_hint=ret_tpe,
                        orig_node=orig_node
                    )

                    call = [irt.AssignStmt(
                        ret_var,
                        irt.FunCall(
                            ref,
                            suffix_exprs,
                            orig_node=orig_node,
                            type_hint=ret_tpe,
                            param_types=_subprogram_param_types(ref)
                        )
                    )]

                    out_arg_exprs = {
                        j: irt.FunCall(
                            ops.GetName(i),
                            [ret_var],
                            type_hint=tpe
                        )
                        for i, (j, tpe, _) in enumerate(out_params)
                    }

                    assignments = [
                        stmt
                        for i, _, assign_out in out_params
                        for stmt in assign_out(out_arg_exprs[i])
                    ]

                    if type_hint is None:
                        res = None
                    elif len(out_params) == 0:
                        res = ret_var
                    else:
                        res = irt.FunCall(
                            ops.GetName(len(out_params)),
                            [ret_var],
                            type_hint=type_hint
                        )

                    pre_conds, post_conds = gen_contract_conditions(
                        ref,
                        pres,
                        posts,
                        suffix_exprs,
                        out_arg_exprs,
                        res,
                        orig_node
                    )

                    return (
                        (suffix_pre_stmts +
                         pre_conds +
                         call +
                         post_conds +
                         assignments),
                        res
                    )

        if _is_array_type_decl(prefix.p_expression_type):
            prefix_pre_stmts, prefix_expr_tr = transform_expr(prefix)

            return prefix_pre_stmts + suffix_pre_stmts, irt.FunCall(
                ops.CALL,
                [prefix_expr_tr] + suffix_exprs,
                type_hint=type_hint,
                orig_node=orig_node,
                param_types=_array_access_param_types(prefix.p_expression_type)
            )

        return unimplemented_expr(orig_node)

    def transform_dereference(derefed_expr, deref_type, deref_orig):
        """
        Generate the IR code that dereferences the given expression, as such:
        Ada:
        ----------------
        x := F(y.all);

        Basic IR:
        ----------------
        assume(y != null)
        x := F(y.all)

        :param lal.Expr derefed_expr: The expression being dereferenced.
        :param lal.AdaNode deref_type: The type of the dereference expression.
        :param lal.Expr deref_orig: The original dereference node.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        # Transform the expression being dereferenced and build the
        # assume expression stating that the expr is not null.
        expr_pre_stmts, expr = transform_expr(derefed_expr)
        assumed_expr = irt.FunCall(
            ops.NEQ,
            [
                expr,
                irt.Lit(
                    lits.NULL,
                    type_hint=derefed_expr.p_expression_type
                )
            ],
            type_hint=derefed_expr.p_bool_type
        )

        # Build the assume statement as mark it as a deref check, so as
        # to inform deref checkers that this assume statement was
        # introduced for that purpose.
        return expr_pre_stmts + [irt.AssumeStmt(
            assumed_expr,
            purpose=purpose.DerefCheck(expr)
        )], irt.FunCall(
            ops.DEREF,
            [expr, stack],
            type_hint=deref_type,
            orig_node=deref_orig
        )

    def transform_record_aggregate(expr):
        """
        :param lal.Aggregate expr: The aggregate expression.
        :return: Its IR transformation.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        record_decl = expr.p_expression_type
        all_fields = list(_record_fields(record_decl))
        field_init = [None] * len(all_fields)
        others_expr_idx = None

        r_exprs_pre_stmts, r_exprs = zip(*[
            transform_expr(assoc.f_r_expr)
            if not assoc.f_r_expr.is_a(lal.BoxExpr)
            else ([], None)  # todo: replace None by default expr
            for assoc in expr.f_assocs
        ])

        for i, assoc in enumerate(expr.f_assocs):
            if len(assoc.f_designators) == 0:
                indexes = [i]
            elif (len(assoc.f_designators) == 1 and
                  assoc.f_designators[0].is_a(lal.OthersDesignator)):
                others_expr_idx = i
                continue
            else:
                indexes = [
                    _field_info(designator).index
                    for designator in assoc.f_designators
                ]

            for idx in indexes:
                field_init[idx] = r_exprs[i]

        if others_expr_idx is not None:
            for i in range(len(field_init)):
                if field_init[i] is None:
                    field_init[i] = r_exprs[others_expr_idx]

        def build_record(record_expr, i=0):
            if i >= len(all_fields):
                return record_expr
            elif field_init[i] is None:
                return build_record(record_expr, i + 1)
            else:
                return build_record(
                    irt.FunCall(
                        ops.UpdatedName(i),
                        [record_expr, field_init[i]],
                        type_hint=record_expr.data.type_hint,
                        orig_node=expr,
                        param_types=_field_assign_param_types(all_fields[i])
                    ), i + 1
                )

        record_var = new_expression_replacing_var("tmp", expr)
        building_stmt = irt.AssignStmt(record_var, build_record(record_var))

        return sum(r_exprs_pre_stmts, []) + [building_stmt], record_var

    def transform_array_aggregate(expr):
        """
        :param lal.Aggregate expr: The aggregate expression.
        :return: its IR transformation.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        # array_def = expr.p_expression_type.f_type_def

        return unimplemented_expr(expr)

    def transform_decl_ref(expr):
        """
        Transforms an expression that refers to a declaration.

        :param lal.Expr expr: The expression to transform.
        :rtype: irt.Expr
        """
        decl = expr.p_referenced_decl

        if decl.is_a(lal.SubpBody, lal.SubpDecl):
            return gen_call_expr(expr, [], expr.p_expression_type, expr)
        elif decl.is_a(lal.EnumLiteralDecl):
            return [], irt.Lit(
                expr.text,
                type_hint=decl.parent.parent.parent,
                orig_node=expr
            )
        elif decl.is_a(lal.NumberDecl):
            return transform_expr(decl.f_expr)
        elif decl.is_a(lal.TypeDecl):
            if decl.f_type_def.is_a(lal.SignedIntTypeDef):
                return transform_expr(decl.f_type_def.f_range.f_range)
        elif decl.is_a(lal.SubtypeDecl):
            constr = decl.f_subtype.f_constraint
            if constr.is_a(lal.RangeConstraint):
                return transform_expr(constr.f_range.f_range)

        return unimplemented_expr(expr)

    @profile()
    def transform_expr(expr):
        """
        :param lal.Expr expr: The expression to transform.

        :return: A list of statements that must directly precede the statement
            that uses the expression being transformed, as well as the
            transformed expression.

        :rtype: (list[irt.Stmt], irt.Expr)
        """

        if expr.is_a(lal.ParenExpr):
            return transform_expr(expr.f_expr)

        elif expr.is_a(lal.BinOp):

            if expr.f_op.is_a(lal.OpAndThen, lal.OpOrElse):
                return transform_short_circuit_ops(expr)
            else:
                lhs_pre_stmts, lhs = transform_expr(expr.f_left)
                rhs_pre_stmts, rhs = transform_expr(expr.f_right)

                return lhs_pre_stmts + rhs_pre_stmts, irt.FunCall(
                    transform_operator(expr.f_op, 2),
                    [lhs, rhs],
                    type_hint=expr.p_expression_type,
                    orig_node=expr
                )

        elif expr.is_a(lal.UnOp):
            inner_pre_stmts, inner_expr = transform_expr(expr.f_expr)
            return inner_pre_stmts, irt.FunCall(
                transform_operator(expr.f_op, 1),
                [inner_expr],
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.CallExpr):
            return gen_call_expr(
                expr.f_name,
                expr.f_suffix,
                expr.p_expression_type,
                expr
            )

        elif expr.is_a(lal.IfExpr):
            # If expressions are transformed as such:
            #
            # Ada:
            # ---------------
            # x := (if C1 then A elsif C2 then B else C);
            #
            #
            # Basic IR:
            # ---------------
            # split:
            #   assume(C1)
            #   tmp := A
            # |:
            #   assume(!C1)
            #   split:
            #     assume(C2)
            #     tmp := B
            #  |:
            #     assume(!C2)
            #     tmp := C
            # x := tmp

            # Generate the temporary variable, make sure it is marked as
            # synthetic so as to inform checkers not to emit irrelevant
            # messages.
            tmp = new_expression_replacing_var("tmp", expr)

            return gen_if_base([
                (expr.f_cond_expr, expr.f_then_expr)
            ] + [
                (part.f_cond_expr, part.f_then_expr)
                for part in expr.f_alternatives
            ] + [
                (None, expr.f_else_expr)
            ], if_expr_alt_transformer_of(tmp)), tmp

        elif expr.is_a(lal.CaseExpr):
            # Case expressions are transformed as such:
            #
            # Ada:
            # ---------------
            # y := case x is
            #      when CST1 => E1,
            #      when CST2 | CST3 => E2,
            #      when RANGE => E3,
            #      when SUBTYPE => E4,
            #      when others => E5;
            #
            #
            # Basic IR:
            # ---------------
            # split:
            #   assume(x == CST1)
            #   tmp = E1
            # |:
            #   assume(x == CST2 || x == CST3)
            #   tmp = E2
            # |:
            #   assume(x >= GetFirst(Range) && x <= GetLast(Range))
            #   tmp = E3
            # |:
            #   assume(x >= GetFirst(Subtype) && x <= GetLast(Subtype))
            #   tmp = E4
            # |:
            #   assume(!(x == CST1 || (x == CST2 || x == CST3) ||
            #          x >= GetFirst(Range) && x <= GetLast(Range) ||
            #          x >= GetFirst(Subtype) && x <= GetLast(Subtype)))
            #   tmp = E5
            #  y := tmp
            #
            # Note: In Ada, case expressions must be complete and *disjoint*.
            # This allows us to transform the case in a split of N branches
            # instead of in a chain of if-elsifs.

            # Generate the temporary variable, make sure it is marked as
            # synthetic so as to inform checkers not to emit irrelevant
            # messages.
            tmp = new_expression_replacing_var("tmp", expr)

            return gen_case_base(
                expr.f_expr,
                expr.f_cases,
                case_expr_alt_transformer_of(tmp),
                expr
            ), tmp

        elif expr.is_a(lal.Identifier):
            # Transform the identifier according what it refers to.
            ref = expr.p_xref
            if ref is None:
                return unimplemented_expr(expr)
            elif ref in substitutions:
                return substitutions[ref]
            elif ref.p_basic_decl.is_a(lal.ObjectDecl, lal.ParamSpec):
                return [], get_var(expr, ref)
            else:
                return transform_decl_ref(expr)

        elif expr.is_a(lal.DottedName):
            # Field access is transformed as such:
            # Ada:
            # ---------------
            # r := x.f
            #
            # Basic IR:
            # ---------------
            # assume([f exists])
            # r = Get_N(x)
            #
            # Where N is the index of the field "f" in the record x (see
            # _compute_field_index), and the condition [f exists] is an
            # expression that must be true for x to have the field f (only
            # relevant for variant records).
            # Additionally, if an implicit dereference takes place, the
            # relevant assume statements are also inserted.

            if expr.f_prefix.p_expression_type is None:
                return unimplemented_expr(expr)
            elif _is_record_field(expr.f_suffix):
                if expr.f_prefix.p_expression_type.p_is_access_type:
                    accessed_type = (expr.f_prefix.p_expression_type
                                     .f_type_def.f_subtype_indication
                                     .p_designated_type_decl_from(expr))
                    prefix_pre_stmts, prefix = transform_dereference(
                        expr.f_prefix, accessed_type, expr.f_prefix
                    )
                else:
                    prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)

                exists_stmts = gen_field_existence_condition(
                    prefix,
                    expr.f_suffix
                )

                field_info = _field_info(expr.f_suffix)
                return prefix_pre_stmts + exists_stmts, irt.FunCall(
                    ops.GetName(field_info.index),
                    [prefix],
                    type_hint=expr.p_expression_type,
                    orig_node=expr,
                    param_types=_field_access_param_types(field_info)
                )
            else:
                return transform_decl_ref(expr)

        elif expr.is_a(lal.IntLiteral):
            return [], irt.Lit(
                expr.p_eval_as_int,
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.NullLiteral):
            return [], irt.Lit(
                access_paths.Null(),
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.CharLiteral):
            return [], irt.Lit(expr.text[1:-1], type_hint=ctx.evaluator.char)

        elif expr.is_a(lal.StringLiteral):
            lit = new_expression_replacing_var("tmp", expr)
            text = expr.text[1:-1]

            def build_lit(i):
                if i == len(text):
                    return lit
                else:
                    return irt.FunCall(
                        ops.UPDATED,
                        [
                            build_lit(i + 1),
                            irt.Lit(text[i], type_hint=ctx.evaluator.char),
                            irt.Lit(
                                i + 1,
                                type_hint=ctx.evaluator.universal_int
                            ),
                        ],
                        type_hint=expr.p_expression_type,
                        orig_node=expr,
                        param_types=_array_assign_param_types(
                            expr.p_expression_type
                        )
                    )

            return [irt.AssignStmt(lit, build_lit(0))], lit

        elif expr.is_a(lal.Aggregate):
            type_decl = expr.p_expression_type
            if type_decl.is_a(lal.TypeDecl):
                type_def = type_decl.f_type_def
                if type_def.is_a(lal.RecordTypeDef):
                    return transform_record_aggregate(expr)
                elif type_def.is_a(lal.ArrayTypeDef):
                    return transform_array_aggregate(expr)

        elif expr.is_a(lal.ExplicitDeref):
            return transform_dereference(
                expr.f_prefix,
                expr.p_expression_type,
                expr
            )

        elif expr.is_a(lal.AttributeRef):
            # AttributeRefs are transformed using an unary operator.
            attribute_text = expr.f_attribute.text.lower()

            if (attribute_text in _attr_to_unop and
                    expr.f_prefix.p_expression_type is not None):
                prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)
                return prefix_pre_stmts, irt.FunCall(
                    _attr_to_unop[attribute_text],
                    [prefix],
                    type_hint=expr.p_expression_type,
                    orig_node=expr
                )
            elif attribute_text == 'access':
                return [], gen_access_path(expr.f_prefix)
            elif attribute_text == 'result':
                return substitutions[expr.f_prefix.p_referenced_decl, 'result']
            elif attribute_text == 'old':
                return substitutions[expr.f_prefix.p_referenced_decl, 'old']
            elif attribute_text == 'image':
                if expr.f_prefix.p_referenced_decl == expr.p_int_type:
                    arg_pre_stmts, arg_expr = transform_expr(
                        expr.f_args[0].f_r_expr
                    )
                    return arg_pre_stmts, irt.FunCall(
                        ops.IMAGE,
                        [arg_expr],
                        type_hint=expr.p_expression_type,
                        orig_node=expr
                    )
            elif attribute_text == 'length':
                return unimplemented_expr(expr)
            elif attribute_text in ('first', 'last'):
                try:
                    return [], irt.Lit(
                        expr.p_eval_as_int,
                        type_hint=expr.p_expression_type,
                        orig_node=expr
                    )
                except lal.PropertyError:
                    pass

        return unimplemented_expr(expr)

    def transform_spec(spec):
        """
        :param lal.SubpSpec spec: The subprogram's specification
        :return:
        """
        if spec.f_subp_params is not None:
            params = spec.f_subp_params.f_params
        else:
            params = []

        for param in params:
            mode = Mode.from_lal_mode(param.f_mode)
            for var_id in param.f_ids:
                param_var = irt.Variable(
                    var_id.text,
                    type_hint=param.f_type_expr,
                    orig_node=var_id,
                    mode=mode,
                    index=next_var_idx()
                )
                var_decls[var_id] = param_var
                param_vars.append(param_var)

        if spec.f_subp_returns is not None:
            result_var.value = irt.Identifier(
                irt.Variable(
                    fresh_name("result"),
                    type_hint=spec.f_subp_returns,
                    index=next_var_idx()
                ),
                type_hint=spec.f_subp_returns
            )

        if (_find_global_access(typer, subp) ==
                _WRITES_GLOBAL_STATE):
            param_vars.append(stack.var)

        return []

    def transform_decl(decl):
        """
        :param lal.BasicDecl decl: The lal decl to transform.

        :return: A (potentially empty) list of statements that emulate the
            Ada semantics of the declaration.

        :rtype: list[irt.Stmt]
        """
        if decl.is_a(lal.TypeDecl, lal.SubtypeDecl,
                     lal.NumberDecl, lal.PackageDecl, lal.PackageBody):
            return []

        elif decl.is_a(lal.ObjectDecl):
            tdecl = decl.f_type_expr

            for var_id in decl.f_ids:
                var_decls[var_id] = irt.Variable(
                    var_id.text,
                    type_hint=tdecl,
                    orig_node=var_id,
                    mode=Mode.Out,
                    index=next_var_idx()
                )

            if decl.f_default_expr is None:
                return [
                    irt.ReadStmt(
                        irt.Identifier(
                            var_decls[var_id],
                            type_hint=tdecl,
                            orig_node=var_id
                        ),
                        orig_node=decl
                    )
                    for var_id in decl.f_ids
                ]
            else:
                dval_pre_stmts, dval_expr = transform_expr(decl.f_default_expr)

                actuals_dests = [
                    gen_actual_dest(var_id, dval_expr)
                    for var_id in decl.f_ids
                ]

                return dval_pre_stmts + [
                    stmt
                    for dest_pre_stmts, (dest, updated) in actuals_dests
                    for stmt in dest_pre_stmts + [
                        irt.AssignStmt(dest, updated, orig_node=decl)
                    ]
                ]

        elif decl.is_a(lal.SubpBody, lal.SubpDecl):
            return []

        elif decl.is_a(lal.IncompleteTypeDecl):
            return []

        unimplemented(decl)

    @profile()
    def transform_stmt(stmt):
        """
        :param lal.Stmt stmt: The lal statement to transform.

        :return: A list of statement that emulate the Ada semantics of the
            statement being transformed.

        :rtype: list[irt.Stmt]
        """

        if stmt.is_a(lal.AssignStmt):
            expr_pre_stmts, expr = transform_expr(stmt.f_expr)
            return gen_assignment(stmt.f_dest, expr_pre_stmts, expr, stmt)

        elif stmt.is_a(lal.CallStmt):
            call_expr = stmt.f_call
            if call_expr.is_a(lal.Identifier, lal.DottedName):
                return gen_call_expr(
                    call_expr, [], call_expr.p_expression_type, stmt
                )[0]
            elif call_expr.is_a(lal.AttributeRef):
                unimplemented(stmt)
            else:
                return gen_call_expr(
                    call_expr.f_name,
                    call_expr.f_suffix,
                    call_expr.p_expression_type,
                    stmt
                )[0]

        elif stmt.is_a(lal.DeclBlock):
            decls = transform_decls(stmt.f_decls.f_decls)
            stmts = transform_stmts(stmt.f_stmts.f_stmts)
            return decls + stmts

        elif stmt.is_a(lal.BeginBlock):
            return transform_stmts(stmt.f_stmts.f_stmts)

        elif stmt.is_a(lal.IfStmt):
            # If statements are transformed as such:
            #
            # Ada:
            # ---------------
            # if C1 then
            #   S1;
            # elsif C2 then
            #   S2;
            # else
            #   S3;
            # end if;
            #
            #
            # Basic IR:
            # ---------------
            # split:
            #   assume(C1)
            #   S1
            # |:
            #   assume(!C1)
            #   split:
            #     assume(C2)
            #     S2
            #  |:
            #     assume(!C2)
            #     S3

            return gen_if_base([
                (stmt.f_cond_expr, stmt.f_then_stmts)
            ] + [
                (part.f_cond_expr, part.f_stmts)
                for part in stmt.f_alternatives
            ] + [
                (None, stmt.f_else_stmts)
            ], transform_stmts)

        elif stmt.is_a(lal.CaseStmt):
            # Case statements are transformed as such:
            #
            # Ada:
            # ---------------
            # case x is
            #   when CST1 =>
            #     S1;
            #   when CST2 | CST3 =>
            #     S2;
            #   when RANGE =>
            #     S3;
            #   when SUBTYPE =>
            #     S4;
            #   when others =>
            #     S5;
            # end case;
            #
            #
            # Basic IR:
            # ---------------
            # split:
            #   assume(x == CST1)
            #   S1
            # |:
            #   assume(x == CST2 || x == CST3)
            #   S2
            # |:
            #   assume(x >= GetFirst(Range) && x <= GetLast(Range))
            #   S3
            # |:
            #   assume(x >= GetFirst(Subtype) && x <= GetLast(Subtype))
            #   S4
            # |:
            #   assume(!(x == CST1 || (x == CST2 || x == CST3) ||
            #          x >= GetFirst(Range) && x <= GetLast(Range) ||
            #          x >= GetFirst(Subtype) && x <= GetLast(Subtype)))
            #   S5
            #
            # Note: In Ada, case statements must be complete and *disjoint*.
            # This allows us to transform the case in a split of N branches
            # instead of in a chain of if-elsifs.

            return gen_case_base(
                stmt.f_expr,
                stmt.f_alternatives,
                case_stmt_alt_transformer,
                stmt
            )

        elif stmt.is_a(lal.LoopStmt):
            exit_label = irt.LabelStmt(fresh_name('exit_loop'))

            loop_stack.append((stmt, exit_label))
            loop_stmts = transform_stmts(stmt.f_stmts)
            loop_stack.pop()

            return [irt.LoopStmt(loop_stmts, orig_node=stmt), exit_label]

        elif stmt.is_a(lal.WhileLoopStmt):
            # While loops are transformed as such:
            #
            # Ada:
            # ----------------
            # while C loop
            #   S;
            # end loop;
            #
            # Basic IR:
            # ----------------
            # loop:
            #   assume(C)
            #   S;
            # assume(!C)

            # Transform the condition of the while loop
            cond_pre_stmts, cond = transform_expr(stmt.f_spec.f_expr)

            # Build its inverse. It is appended at the end of the loop. We know
            # that the inverse condition is true once the control goes out of
            # the loop as long as there are not exit statements.
            not_cond = irt.FunCall(
                ops.NOT,
                [cond],
                type_hint=cond.data.type_hint
            )

            exit_label = irt.LabelStmt(fresh_name('exit_while_loop'))

            loop_stack.append((stmt, exit_label))
            loop_stmts = transform_stmts(stmt.f_stmts)
            loop_stack.pop()

            return [irt.LoopStmt(
                cond_pre_stmts +
                [irt.AssumeStmt(cond)] +
                loop_stmts,
                orig_node=stmt
            ), irt.AssumeStmt(not_cond), exit_label]

        elif stmt.is_a(lal.ForLoopStmt):
            # todo
            return []

        elif stmt.is_a(lal.Label):
            # Use the pre-transformed label.
            return [labels[stmt.f_decl]]

        elif stmt.is_a(lal.GotoStmt):
            label = labels[stmt.f_label_name.p_referenced_decl]
            return [irt.GotoStmt(label, orig_node=stmt)]

        elif stmt.is_a(lal.NamedStmt):
            return transform_stmt(stmt.f_stmt)

        elif stmt.is_a(lal.ExitStmt):
            # Exit statements are transformed as such:
            #
            # Ada:
            # ----------------
            # loop
            #   exit when C
            # end loop;
            #
            # Basic IR:
            # ----------------
            # loop:
            #   split:
            #     assume(C)
            #     goto [AFTER_LOOP]
            #   |:
            #     assume(!C)
            # [AFTER_LOOP]

            if stmt.f_loop_name is None:
                # If not loop name is specified, take the one on top of the
                # loop stack.
                exited_loop = loop_stack[-1]
            else:
                named_loop_decl = stmt.f_loop_name.p_referenced_decl
                ref_loop = named_loop_decl.parent.f_stmt
                # Find the exit label corresponding to the exited loop.
                exited_loop = next(
                    loop for loop in loop_stack
                    if loop[0] == ref_loop
                )

            # The label to jump to is stored in the second component of the
            # loop tuple.
            loop_exit_label = exited_loop[1]
            exit_goto = irt.GotoStmt(loop_exit_label)

            if stmt.f_cond_expr is None:
                # If there is no "when" part, only generate a goto statement.
                return [exit_goto]
            else:
                # Else emulate the behavior with split-assume statements.
                return gen_split_stmt(
                    stmt.f_cond_expr,
                    [exit_goto],
                    [],
                    orig_node=stmt
                )

        elif stmt.is_a(lal.ReturnStmt):
            stmts = []

            if stmt.f_return_expr is not None:
                ret_pre_stmts, ret_expr = transform_expr(stmt.f_return_expr)
                stmts.extend(ret_pre_stmts)
                stmts.append(irt.AssignStmt(
                    result_var.value,
                    ret_expr,
                    orig_node=stmt
                ))

            stmts.append(irt.GotoStmt(
                func_end_label,
                orig_node=stmt
            ))

            return stmts

        elif stmt.is_a(lal.NullStmt):
            return []

        elif stmt.is_a(lal.ExceptionHandler):
            # todo ?
            return []

        unimplemented(stmt)

    def print_warning(subject, exception):
        print("warning: ignored '{}'".format(subject))
        message = getattr(exception, 'message', None)
        if message is not None:
            print("\treason: {}".format(message))

    def transform_decls(decls):
        """
        :param iterable[lal.BasicDecl] decls: An iterable of decls
        :return: The transformed list of statements.
        :rtype: list[irt.Stmt]
        """
        res = []
        for decl in decls:
            try:
                res.extend(transform_decl(decl))
            except (lal.PropertyError, NotImplementedError,
                    KeyError, NotConstExprError) as e:
                print_warning(decl.text, e)
        return res

    def transform_stmts(stmts):
        """
        :param iterable[lal.Stmt] stmts: An iterable of stmts
        :return: The transformed list of statements.
        :rtype: list[irt.Stmt]
        """
        res = []
        for stmt in stmts:
            try:
                res.extend(transform_stmt(stmt))
            except (lal.PropertyError, NotImplementedError,
                    KeyError, NotConstExprError) as e:
                print_warning(stmt.text, e)
        return res

    return irt.Program(
        transform_spec(subp.f_subp_spec) +
        transform_decls(subp.f_decls.f_decls) +
        transform_stmts(subp.f_stmts.f_stmts) +
        [func_end_label],
        fun_id=subp,
        orig_node=subp,
        result_var=result_var.value.var if result_var.value else None,
        param_vars=param_vars
    )


class ConvertUniversalTypes(IRImplicitVisitor):
    """
    Visitor that mutates the given IR tree so as to remove references to
    universal types from in node data's type hints.
    """

    def __init__(self, evaluator, typer):
        """
        :param ConstExprEvaluator evaluator: A const expr evaluator.
        :param types.Typer[lal.AdaNode]: A typer
        """
        super(ConvertUniversalTypes, self).__init__()

        self.evaluator = evaluator
        self.typer = typer

    def has_universal_type(self, expr):
        """
        :param irt.Expr expr: A Basic IR expression.

        :return: True if the expression is either of universal int type, or
            universal real type.

        :rtype: bool
        """
        return expr.data.type_hint in [
            self.evaluator.universal_int,
            self.evaluator.universal_real
        ]

    def is_integer_type(self, tpe):
        """
        Returns True if the given type denotes an integer type declaration.
        :param lal.BaseTypeDecl tpe:
        :rtype: bool
        """
        try:
            return self.typer.get(tpe).is_a(types.IntRange)
        except ValueError:
            return False

    def is_compatible(self, tpe, universal_tpe):
        """
        Returns True if the given type is compatible with the given universal
        type. For example, Integer types are compatible with universal ints,
        etc.

        :param lal.BaseTypeDecl tpe: The type to check compatibility with.
        :param lal.BaseTypeDecl universal_tpe: The universal type.
        :rtype: bool
        """
        if universal_tpe == self.evaluator.universal_int:
            return self.is_integer_type(tpe)
        else:
            raise NotImplementedError

    def compatible_type(self, universal_tpe):
        """
        Returns a "concrete" type that is compatible with the given universal
        type. For example, Integer can be returned for universal ints.

        :param lal.BaseTypeDecl universal_tpe: The universal type.
        :rtype: lal.BaseTypeDecl
        """
        if universal_tpe == self.evaluator.universal_int:
            return self.evaluator.int
        else:
            raise NotImplementedError

    def try_convert_expr(self, expr, expected_type):
        """
        :param irt.Expr expr: A Basic IR expression.

        :param lal.AdaNode expected_type: The expected type hint of the
            expression.

        :return: An equivalent expression which does not have an universal
            type.

        :rtype: irt.Expr
        """
        try:
            return irt.Lit(
                self.evaluator.eval(expr),
                type_hint=expected_type
            )
        except NotConstExprError:
            if self.has_universal_type(expr):
                expr.data = expr.data.copy(type_hint=expected_type)
            expr.visit(self, expr.data.type_hint)
            return expr

    def visit_assign(self, assign):
        assign.expr = self.try_convert_expr(
            assign.expr,
            assign.id.data.type_hint if not self.has_universal_type(assign.id)
            else self.compatible_type(assign.id.data.type_hint)
        )

    def visit_assume(self, assume):
        assume.expr = self.try_convert_expr(assume.expr, self.evaluator.bool)

    def visit_funcall(self, funcall, expected_type):
        if any(self.has_universal_type(arg) for arg in funcall.args):
            if 'param_types' in funcall.data:
                funcall.args = [
                    self.try_convert_expr(arg, param_type)
                    for param_type, arg in zip(
                        funcall.data.param_types,
                        funcall.args
                    )
                ]
            else:
                # Otherwise, assume that functions that accept one argument
                # as universal int/real need all their arguments to be of the
                # same type, which is true for arithmetic ops, comparison ops,
                # etc.
                not_universals = [
                    arg.data.type_hint
                    for arg in funcall.args
                    if not self.has_universal_type(arg)
                ]
                for i in range(len(funcall.args)):
                    arg = funcall.args[i]

                    if not self.has_universal_type(arg):
                        tpe = arg.data.type_hint
                    elif (len(not_universals) > 0 and
                            self.is_compatible(not_universals[0],
                                               arg.data.type_hint)):
                        tpe = not_universals[0]
                    elif self.is_compatible(expected_type, arg.data.type_hint):
                        tpe = expected_type
                    else:
                        tpe = self.compatible_type(arg.data.type_hint)

                    funcall.args[i] = self.try_convert_expr(arg, tpe)
        else:
            funcall.args = [
                self.try_convert_expr(arg, arg.data.type_hint)
                for arg in funcall.args
            ]


class NotConstExprError(ValueError):
    def __init__(self):
        super(NotConstExprError, self).__init__()


ADA_TRUE = 'True'
ADA_FALSE = 'False'


class ConstExprEvaluator(IRImplicitVisitor):
    """
    Used to evaluate expressions statically.
    See eval.
    """

    class Range(object):
        def __init__(self, first, last):
            self.first = first
            self.last = last

    Ops = {
        (ops.AND, 2): lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) and ConstExprEvaluator.to_bool(y)
        ),
        (ops.OR, 2): lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) or ConstExprEvaluator.to_bool(y)
        ),

        (ops.NEQ, 2): lambda x, y: ConstExprEvaluator.from_bool(x != y),
        (ops.EQ, 2): lambda x, y: ConstExprEvaluator.from_bool(x == y),
        (ops.LT, 2): lambda x, y: ConstExprEvaluator.from_bool(x < y),
        (ops.LE, 2): lambda x, y: ConstExprEvaluator.from_bool(x <= y),
        (ops.GE, 2): lambda x, y: ConstExprEvaluator.from_bool(x >= y),
        (ops.GT, 2): lambda x, y: ConstExprEvaluator.from_bool(x > y),
        (ops.DOT_DOT, 2): lambda x, y: ConstExprEvaluator.Range(x, y),

        (ops.PLUS, 2): lambda x, y: x + y,
        (ops.MINUS, 2): lambda x, y: x - y,

        (ops.NOT, 1): lambda x: ConstExprEvaluator.from_bool(
            not ConstExprEvaluator.to_bool(x)
        ),
        (ops.NEG, 1): lambda x: -x,
        (ops.GET_FIRST, 1): lambda x: x.first,
        (ops.GET_LAST, 1): lambda x: x.last
    }

    def __init__(self, bool_type, int_type, char_type, u_int_type,
                 u_real_type):
        """
        :param lal.AdaNode bool_type: The standard boolean type.
        :param lal.AdaNode int_type: The standard int type.
        :param lal.AdaNode char_type: The standard char type.
        :param lal.AdaNode u_int_type: The standard universal int type.
        :param lal.AdaNode u_real_type: The standard universal real type.
        """
        super(ConstExprEvaluator, self).__init__()
        self.bool = bool_type
        self.int = int_type
        self.char = char_type
        self.universal_int = u_int_type
        self.universal_real = u_real_type

    @staticmethod
    def to_bool(x):
        """
        :param str x: The boolean to convert.
        :return: The representation of the corresponding boolean literal.
        :rtype: bool
        """
        return x == ADA_TRUE

    @staticmethod
    def from_bool(x):
        """
        :param bool x: The representation of a boolean literal to convert.
        :return: The corresponding boolean.
        :rtype: str
        """
        return ADA_TRUE if x else ADA_FALSE

    def eval(self, expr):
        """
        Evaluates an expression, returning the value it evaluates to.

        :param irt.Expr expr: A Basic IR expression to evaluate.
        :rtype: int | str | ConstExprEvaluator.Range
        :raise NotConstExprError: if the expression is not a constant.
        :raise NotImplementedError: if implementation is incomplete.
        """
        return self.visit(expr)

    @memoize
    def visit(self, expr):
        """
        To use instead of node.visit(self). Performs memoization, so as to
        avoid evaluating expression referred to by constant symbols multiple
        times.

        :param irt.Expr expr: The IR Basic expression to evaluate

        :return: The value of this expression.

        :rtype: int | str | ConstExprEvaluator.Range
        """
        return expr.visit(self)

    def visit_ident(self, ident):
        raise NotConstExprError

    def visit_funcall(self, funcall):
        try:
            op = ConstExprEvaluator.Ops[funcall.fun_id, len(funcall.args)]
            return op(*(
                self.visit(arg) for arg in funcall.args
            ))
        except KeyError:
            raise NotConstExprError

    def visit_lit(self, lit):
        return lit.val


@types.delegating_typer
def int_range_typer():
    """
    :return: A typer for int ranges.
    :rtype: types.Typer[lal.AdaNodetypes]
    """

    @Transformer.as_transformer
    def get_operands(hint):
        if hint.is_a(lal.TypeDecl):
            if hint.f_type_def.is_a(lal.SignedIntTypeDef):
                rng = hint.f_type_def.f_range.f_range
                return rng.f_left, rng.f_right

    @Transformer.as_transformer
    def eval_as_int(x):
        try:
            return x.p_eval_as_int
        except (lal.PropertyError, lal.NativeException):
            return None

    @types.Typer
    def to_int_range(xs):
        return types.IntRange(*xs)

    return get_operands >> (eval_as_int & eval_as_int) >> to_int_range


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
            if hint.p_is_access_type:
                return _pointer_type
        except lal.PropertyError:
            pass
    elif hint.is_a(_PointerType):
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
                return [field.decl for field in _record_fields(hint)]

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
    @Transformer.as_transformer
    def resolved_name(hint):
        """
        :param lal.AdaNode hint: the lal type expression.
        :return: The type declaration associated to the name, if relevant.
        :rtype: lal.BaseTypeDecl
        """
        if hint.is_a(lal.SubtypeIndication):
            # todo: Take constraint into account
            try:
                return hint.p_designated_type_decl
            except lal.PropertyError:
                pass

    return resolved_name >> inner_typer


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
        if _is_array_type_decl(hint):
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
        if hint.is_a(_ExtendedCallReturnType):
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
    def get_subtype(hint):
        if hint.is_a(lal.SubtypeDecl):
            return hint.f_subtype

    return get_subtype >> inner


@types.typer
def ram_typer(hint):
    if hint.is_a(_StackType):
        return types.DataStorage()


@types.memoizing_typer
@types.typer
def unknown_typer(_):
    return types.Unknown()


@types.memoizing_typer
@types.typer
def none_typer(hint):
    if hint is None:
        return types.Product([])


class ExtractionContext(object):
    """
    The libadalang-based frontend interface. Provides method for extracting
    IR programs from Ada source files (see extract_programs), as well as
    a default typer for those programs (see default_typer).

    Note: programs extracted using different ExtractionContext are not
    compatible. Also, this extraction context must be kept alive as long
    as the programs parsed with it are intended to be used.
    """
    def __init__(self, project_file=None, scenario_vars=None):
        if project_file is None:
            self.lal_ctx = lal.AnalysisContext()
        else:
            self.lal_ctx = lal.AnalysisContext(
                unit_provider=lal.UnitProvider.for_project(
                    project_file,
                    scenario_vars
                )
            )

        # Get a dummy node, needed to call static properties of libadalang.
        dummy = self.lal_ctx.get_from_buffer(
            "<dummy>", 'package Dummy is end;'
        ).root

        # Find the Character TypeDecl.
        char_type = dummy.p_standard_unit.root.find(
            lambda x: x.is_a(lal.TypeDecl) and x.f_name.text == "Character"
        )

        self.evaluator = ConstExprEvaluator(
            dummy.p_bool_type,
            dummy.p_int_type,
            char_type,
            dummy.p_universal_int_type,
            dummy.p_universal_real_type
        )

        self.type_models = {}
        self.fun_models = {}
        self._internal_typer = self.default_typer()

    def extract_programs_from_file(self, ada_file):
        """
        :param str ada_file: A path to the Ada source file from which to
            extract programs.

        :return: a Basic IR Program for each subprogram body that exists in the
            given source code.

        :rtype: iterable[irt.Program]
        """
        return self._extract_from_unit(self.lal_ctx.get_from_file(ada_file))

    def extract_programs_from_provider(self, name, kind):
        return self._extract_from_unit(self.lal_ctx.get_from_provider(
            name, kind
        ))

    @profile()
    def use_model(self, name):
        model_unit = self.lal_ctx.get_from_provider(name, "specification")
        for diag in model_unit.diagnostics:
            print('   {}'.format(diag))
            return

        model_ofs = model_unit.root.findall(
            lambda x: x.is_a(lal.AspectAssoc) and x.f_id.text == "Model_Of"
        )

        type_models = {
            aspect.parent.parent.parent: aspect.f_expr.p_referenced_decl
            for aspect in model_ofs
            if aspect.parent.parent.parent.is_a(lal.TypeDecl, lal.SubtypeDecl)
        }

        fun_models = {
            aspect.parent.parent.parent: aspect.f_expr.p_referenced_decl
            for aspect in model_ofs
            if aspect.parent.parent.parent.is_a(lal.SubpDecl)
        }

        if len(type_models) + len(fun_models) != len(model_ofs):
            print("warning: detected usage of Model_Of in an unknown context.")

        for tdecl, ref in type_models.iteritems():
            if ref is None:
                print("warning: Model_Of used on '{}' refers to an unknown "
                      "type.".format(tdecl.f_name.text))
            else:
                self.type_models[ref] = tdecl

        for fdecl, ref in fun_models.iteritems():
            if ref is None:
                print(
                    "warning: Model_Of used on '{}' refers to an unknown "
                    "procedure/function.".format(
                        fdecl.f_subp_spec.f_subp_name.text
                    )
                )
            else:
                self.fun_models[ref] = fdecl

    @profile()
    def _extract_from_unit(self, unit):
        if unit.root is None:
            print('Could not parse {}:'.format(unit.filename))
            for diag in unit.diagnostics:
                print('   {}'.format(diag))
                return

        unit.populate_lexical_env()

        progs = [
            _gen_ir(self, subp, self._internal_typer)
            for subp in unit.root.findall(lal.SubpBody)
        ]

        converter = ConvertUniversalTypes(self.evaluator, self._internal_typer)

        for prog in progs:
            prog.visit(converter)

        return progs

    def standard_typer(self):
        """
        :return: A Typer for Ada standard types of programs parsed using
            this extraction context.

        :rtype: types.Typer[lal.AdaNode]
        """
        bool_type = self.evaluator.bool
        int_type = self.evaluator.int
        char_type = self.evaluator.char

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

    def model_typer(self, inner_typer):
        @Transformer.as_transformer
        def find_modeled_type(hint):
            model = self.type_models.get(hint, None)
            if model is not None:
                return hint, model

        @Transformer.as_transformer
        def to_model_type(tpes):
            return types.ModeledType(*tpes)

        return find_modeled_type >> inner_typer.lifted() >> to_model_type

    def default_typer(self, fallback_typer=None):
        """
        :return: The default Typer for Ada programs parsed using this
            extraction context.

        :rtype: types.Typer[lal.AdaNode]
        """

        standard_typer = self.standard_typer()

        @types.memoizing_typer
        @types.delegating_typer
        def typer():
            return self.model_typer(typer_without_model) | typer_without_model

        @types.memoizing_typer
        @types.delegating_typer
        def typer_without_model():
            """
            :rtype: types.Typer[lal.AdaNode]
            """
            typr = (none_typer |
                    standard_typer |
                    int_range_typer |
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
