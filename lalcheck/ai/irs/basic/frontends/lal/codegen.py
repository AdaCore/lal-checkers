import libadalang as lal
from lalcheck.ai import types
from lalcheck.ai.constants import lits, ops, access_paths
from lalcheck.ai.irs.basic import tree as irt, purpose
from lalcheck.ai.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.ai.utils import KeyCounter, Transformer, profile, unzip
from lalcheck.tools.logger import log_stdout
from funcy.calc import memoize

from utils import (
    Mode,
    NotConstExprError,
    StackType, PointerType, ExtendedCallReturnType,
    ValueHolder, record_fields, proc_parameters, get_field_info,
    is_array_type_decl, is_record_field, is_access_to_subprogram,
    closest, get_subp_identity
)

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


class AccessingUnspilledVar(ValueError):
    """
    Is raised when accessing a variable that was not spilled
    """
    def __init__(self, *args):
        super(AccessingUnspilledVar, self).__init__(*args)


def _subprogram_param_types(subp, globs, with_stack):
    """
    Returns a tuple containing the types of each parameter of the given
    subprogram.

    :param lal.SubpDecl subp: The subprogram to consider.
    :param list[lal.DefiningName] globs: The global variables it accesses.
    :param bool with_stack: Whether the subprogram takes the optional stack
        as an additional argument.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    params = proc_parameters(subp)
    return tuple(
        p.f_type_expr for _, _, p in params
    ) + tuple(
        g.p_basic_decl.f_type_expr for g in globs
    ) + (
        (StackType(),) if with_stack else ()
    )


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
    return StackType(), var.data.type_hint


def _deref_assign_param_types(ptr_type, val_type):
    """
    Returns a tuple containing the types of each parameter of the function
    that assigns a value of the given val_type to a pointer of type ptr_type.

    :param lal.BaseTypeDecl ptr_type: The type of the access.
    :param lal.BaseTypeDecl val_type: The type of the value.
    :rtype: tuple[lal.BaseTypeDecl]
    """
    return StackType(), ptr_type, val_type


def _string_literal_param_types(array_type_decl, text):
    """
    Returns a tuple containing the types of each parameter of the function
    that creates the array containing the given string literal.

    :param lal.BaseTypeDecl array_type_decl: The array type to consider.
    :param str text: The content of the string literal.
    :rtype: tuple[lal.BaseTypeDecl]
    """

    assignment_param_types = _array_assign_param_types(array_type_decl)

    assert(len(assignment_param_types) == 3)

    component_type = assignment_param_types[1]
    index_type = assignment_param_types[2]

    return (array_type_decl,) + (index_type, component_type) * len(text)


def get_subp_interface(ctx, subp):
    """
    Returns a pair containing the parameter types (plus the return value if
    any) and the call interface of the given subprogram.
    See access_paths.Subprogram.CallInterface.

    :param ExtractionContext ctx: The current extraction context.
    :param lal.SubpDecl | lal.SubpBody subp: The subprogram for which to
        retrieve this interface.
    :rtype: (list[lal.TypeDecl], access_paths.Subprogram.CallInterface)
    """
    params = [
        param for _, _, param in proc_parameters(subp)
    ]

    param_types = [param.f_type_expr for param in params]
    out_indices = [
        i
        for i, param in enumerate(params)
        if param.f_mode.is_a(lal.ModeOut, lal.ModeInOut)
    ]

    subp_data = ctx.subpdata.get(subp)
    if subp_data is not None:
        for global_id in subp_data.all_global_vars:
            param_types.append(
                PointerType(global_id.p_basic_decl.f_type_expr)
            )

    global_access = _find_global_access(ctx, subp)
    if global_access != _IGNORES_GLOBAL_STATE:
        param_types.append(StackType())
        if global_access == _WRITES_GLOBAL_STATE:
            out_indices.append(len(param_types) - 1)

    ret_type = subp.f_subp_spec.f_subp_returns
    if ret_type is not None:
        param_types.append(ret_type)

    return param_types, access_paths.Subprogram.CallInterface(
        out_indices, global_access, ret_type is not None
    )


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
    except Transformer.TransformationFailure:
        return None


@memoize
def _find_global_access(ctx, proc):
    """
    Analyzes the side effects of the given subprogram. returns the following:
    - _IGNORES_GLOBAL_STATE if the function doesn't access the global state.
    - _READS_GLOBAL_STATE if the function may read from the global state.
    - _WRITES_GLOBAL_STATE if the function may write to the global state.

    :param ExtractionContext ctx: The extraction context.
    :param lal.SubpBody | lal.SubpDecl proc: The subprogram to analyze.
    :rtype: int
    """
    procdata = ctx.subpdata.get(get_subp_identity(proc))
    if procdata is not None and len(procdata.all_global_vars) > 0:
        # The subprogram has access to global variables.
        return _WRITES_GLOBAL_STATE

    for _, _, param in proc_parameters(proc):
        if _contains_access_type(ctx._internal_typer, param.f_type_expr):
            # The subprogram takes pointers as parameters, and may therefore
            # produce global side effects.
            return _WRITES_GLOBAL_STATE

    return _IGNORES_GLOBAL_STATE


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
                aspect_name = aspect.f_id.text.lower()
                if aspect_name == 'pre':
                    pres.append((aspect.f_expr, True))
                elif aspect_name == 'model_pre':
                    pres.append((aspect.f_expr, True))
                elif aspect_name == 'post':
                    posts.append((aspect.f_expr, True))
                elif aspect_name == 'model_post':
                    posts.append((aspect.f_expr, False))

    collect_pre_posts(proc)

    model = ctx.fun_models.get(proc, None)
    if model is not None:
        collect_pre_posts(model)

    return pres, posts


@profile()
def gen_ir(ctx, subp, typer, subpdata):
    """
    Generates Basic intermediate representation from a lal subprogram body.

    :param ExtractionContext ctx: The program extraction context.

    :param lal.SubpBody subp: The subprogram body from which to generate IR.

    :param types.Typer[lal.AdaNode] typer: A typer for any lal AdaNode.

    :param analysis.SubpAnalysisData subpdata: The information computed by
        the analysis about the given subprogram.

    :return: a Basic Program.

    :rtype: irt.Program
    """

    param_vars = []
    tmp_vars = KeyCounter()

    # A synthetic variable used to store the result of a function (what is
    # returned). Use a ValueHolder so it can be rebound by closures (e.g.
    # transform_spec).
    result_var = ValueHolder()

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
    to_spill = subpdata.vars_to_spill

    class StorageManager(object):
        """
        Singleton object (see variable "stores") that is used to track all
        variables considered in this program. It can be used to register new
        variables and to access them using their libadalang defining name.

        This object manages variable handles (see VariableHandle) and provides
        a common API for manipulating different kind of variables without
        having to specialize your code for each possible underlying
        representation of a variable (irt.Variable, index on the stack, etc.).
        """
        class VariableHandle(object):
            """
            The base class for variable handles, stores the libadalang
            DefiningName associated to the variable.
            """
            def __init__(self, defining_name):
                """
                :param lal.DefiningName|None defining_name: The defining name
                    of the Ada variable that this handle wraps.
                """
                self.defining_name = defining_name

            def havoc(self, **data):
                """
                Returns a list of statements which, when executed, destroys
                the previous knowledge that we had on the variable.

                :param **object data: User data to attach on the returned
                    statements.
                :rtype: list[irt.Stmt]
                """
                raise NotImplementedError

            def get(self, **data):
                """
                Returns an expression which holds the value of the variable.
                :param **object data: The data to attach to the expression.

                :rtype: irt.Expr | irt.Identifier
                """
                raise NotImplementedError

            def gen_dest(self, expr, **data):
                """
                Generates a pair (LHS, RHS) for an assignment of expr into the
                variable. RHS originates from expr but might be refined.

                :param irt.Expr expr: The expression to assign
                :param **object data: User data to attach to the generated
                    object.
                :rtype: (irt.Expr, irt.Expr)
                """
                raise NotImplementedError

            def access(self, **data):
                """
                Returns an expression representing an access (in the sense of
                pointer) to this variable.

                :param **object data: User data to attach to the generated
                    expression.
                :rtype: irt.Expr
                """
                raise NotImplementedError

        class NormalVariableHandle(VariableHandle):
            """
            Represents local variables which cannot be aliased, and which are
            always read and written to directly (wraps an irt.Variable object).
            """
            def __init__(self, defining_name, var):
                """
                :param lal.DefiningName | None defining_name: The defining name
                    of the Ada variable that this handle wraps.
                :param irt.Variable var: The intermediate language node
                    representing the variable.
                """
                StorageManager.VariableHandle.__init__(self, defining_name)
                self.var = var

            def havoc(self, read_orig_node, **id_data):
                # todo: remove read statements all together.
                return [
                    irt.ReadStmt(self.get(**id_data), orig_node=read_orig_node)
                ]

            def get(self, **data):
                return irt.Identifier(
                    self.var, type_hint=self.var.data.type_hint, **data
                )

            def gen_dest(self, expr, **data):
                return self.get(**data), expr

            def access(self, **data):
                raise AccessingUnspilledVar

        class SpilledRelativeVariableHandle(VariableHandle):
            """
            Represents a local variable that lives on the stack.
            """
            def __init__(self, defining_name, var_index, type_hint):
                """
                :param lal.DefiningName | None defining_name: The defining name
                    of the Ada variable that this handle wraps.
                :param int var_index: The index of this variable, to use as
                    its location on the stack.
                :param lal.TypeDecl type_hint: The type of the variable.
                """
                StorageManager.VariableHandle.__init__(self, defining_name)
                self.var_index = var_index
                self.type_hint = type_hint

            def havoc(self, **data):
                raise NotImplementedError

            def get(self, **data):
                return irt.FunCall(
                    ops.GetName(self.var_index),
                    [stack],
                    type_hint=self.type_hint,
                    param_types=[stack.data.type_hint],
                    **data
                )

            def gen_dest(self, expr, **data):
                return stack, irt.FunCall(
                    ops.UpdatedName(self.var_index),
                    [stack, expr],
                    type_hint=stack.data.type_hint,
                    param_types=[stack.data.type_hint, self.type_hint],
                    **data
                )

            def access(self, **data):
                return irt.FunCall(
                    access_paths.Var(self.var_index),
                    [stack],
                    type_hint=PointerType(self.type_hint),
                    additional_args=[self.type_hint],
                    **data
                )

        class SpilledAbsoluteVariableHandle(VariableHandle):
            """
            Represents a variable that lives in the stack, but which is not
            necessarily local. It is accessed using another variable handle,
            representing its address.
            """
            def __init__(self, defining_name, address_var, type_hint):
                """
                :param lal.DefiningName | None defining_name: The defining name
                    of the Ada variable that this handle wraps.
                :param VariableHandle address_var: The handle representing
                    the address of this variable.
                :param lal.TypeDecl type_hint: The type of this variable.
                """
                StorageManager.VariableHandle.__init__(self, defining_name)
                self.address_var = address_var
                self.type_hint = type_hint

            def havoc(self, **data):
                raise NotImplementedError

            def get(self, **data):
                address_expr = self.address_var.get()
                return irt.FunCall(
                    ops.DEREF,
                    [address_expr, stack],
                    type_hint=self.type_hint,
                    param_types=[address_expr.data.type_hint,
                                 stack.data.type_hint],
                    **data
                )

            def gen_dest(self, expr, **data):
                address_expr = self.address_var.get()
                return stack, irt.FunCall(
                    ops.UPDATED,
                    [stack, address_expr, expr],
                    type_hint=stack.data.type_hint,
                    param_types=_deref_assign_param_types(
                        address_expr.data.type_hint,
                        expr.data.type_hint
                    ),
                    **data
                )

            def access(self, **data):
                return self.address_var.get(**data)

        def __init__(self):
            self.vars = {}
            self.var_index = 0

        def register_normal(self, defining_name=None, name=None, **data):
            """
            Creates a new "normal" variable handle. At least one of (name,
            defining_name) must be given.

            :param lal.DefiningName | None defining_name: The defining name of
                the Ada variable that this handle wraps.
            :param str|None name: The name to associate to the variable.
            :param **object data: User data to attach to this variable.
            :rtype: NormalVariableHandle
            """
            if name is None:
                assert defining_name is not None
                name = defining_name.text

            var = self.NormalVariableHandle(defining_name, irt.Variable(
                name,
                index=self.var_index,
                **data
            ))

            self.var_index += 1

            if defining_name is not None:
                self.vars[defining_name] = var

            return var

        def register_spilled_relative(self, type_hint, defining_name=None):
            """
            Creates a new "spilled relative" variable handle.

            :param lal.TypeDecl type_hint: The type of this variable.
            :param lal.DefiningName | None defining_name: The defining name of
                the Ada variable that this handle wraps.
            :rtype: SpilledRelativeVariableHandle
            """
            var = self.SpilledRelativeVariableHandle(
                defining_name, self.var_index, type_hint
            )
            self.var_index += 1

            if defining_name is not None:
                self.vars[defining_name] = var

            return var

        def register_spilled_absolute(self, address_handle, type_hint,
                                      defining_name=None):
            """
            Creates a new "spilled absolute" variable handle.

            :param VariableHandle address_handle: The variable handle of the
                variable holding the address of this variable.
            :param lal.TypeDecl type_hint: The type of this variable.
            :param lal.DefiningName | None defining_name: The defining name of
                the Ada variable that this handle wraps.
            :rtype: SpilledAbsoluteVariableHandle
            """
            var = self.SpilledAbsoluteVariableHandle(
                defining_name, address_handle, type_hint
            )

            if defining_name is not None:
                self.vars[defining_name] = var

            return var

        def get(self, defining_name, default=None):
            """
            Finds the variable of the given defining name.
            :param lal.DefiningName defining_name: The defining name to look
                for.
            :param object | None default: The value to return if the lookup
                fails.
            :rtype: VariableHandle | object
            """
            return self.vars.get(defining_name, default)

        def current_index(self):
            """
            Returns the current variable index.
            :rtype: int
            """
            return self.var_index

    stores = StorageManager()

    stack = stores.register_normal(
        name="$stack", type_hint=StackType(), mode=Mode.Local
    ).get()

    class CallExprBuilder(object):
        """
        Builder object which can be used to simplify the creation of call
        expressions.
        """
        def __init__(self, orig_call_expr, called_name, res_type):
            """
            :param lal.AdaNode orig_call_expr: The original call expression
                in the source code.
            :param lal.SubpBody | lal.SubpDecl called_name: The subprogram
                being called.
            :param lal.AdaNode res_type: The type of the returned value.
            """
            self.orig_call_expr = orig_call_expr
            self.called_name = called_name
            self.res_type = res_type
            self.args = []
            self.preconditions = []
            self.postconditions = []

        def add_argument(self, param_type, arg_stmts_expr, assigner=None):
            """
            Append a new argument to the call expression.
            :param lal.AdaNode param_type: The expected type of the argument.
                (The type of the parameter).
            :param (list[irt.Stmt], irt.Expr) arg_stmts_expr: The transformed
                expression of the argument as returned by transform_expr.
            :param (irt.Expr -> irt.AssignStmt) | None assigner: If not None,
                this argument is "out" and the assigner function will be
                called to generate the assignment of the out parameter after
                the call. The expression passed to that function describes the
                value of that argument as modified by the function called.
                See gen_call_expr for an example of usage.
            """
            self.args.append((param_type, arg_stmts_expr, assigner))

        @staticmethod
        def _stack_assigner(expr):
            return [irt.AssignStmt(
                stack, irt.FunCall(
                    ops.COPY_OFFSET,
                    [stack, expr],
                    type_hint=stack.data.type_hint
                )
            )]

        def add_stack_argument(self, out=False):
            """
            Adds the stack as argument to this call expression.
            :param bool out: Indicates if the stack should be updated after
                the call.
            """
            self.add_argument(
                stack.data.type_hint,
                ([], irt.FunCall(
                    ops.OffsetName(stores.current_index()),
                    [stack],
                    type_hint=stack.data.type_hint
                )),
                self._stack_assigner if out else None
            )

        def add_contracts(self, preconditions, postconditions):
            """
            Adds new pre- and postconditions to that call.
            :param list[lal.Expr] preconditions: Preconditions to add.
            :param list[lal.Expr] postconditions: Postconditions to add.
            """
            self.preconditions.extend(preconditions)
            self.postconditions.extend(postconditions)

        def build(self):
            """
            Generates the call expression.
            :rtype: list[irt.Stmt], irt.Expr
            """
            param_types, arg_stmts_expr, args_out = unzip(self.args, 3)

            arg_pre_stmts, arg_exprs = unzip(arg_stmts_expr, 2)
            arg_pre_stmts = [
                stmt
                for pre_stmts in arg_pre_stmts
                for stmt in pre_stmts
            ]

            out_indices = tuple(
                i
                for i, out in enumerate(args_out)
                if out is not None
            )

            if len(out_indices) > 0:
                ret_type = ExtendedCallReturnType(
                    out_indices,
                    tuple(param_types[i] for i in out_indices),
                    self.res_type
                )
            else:
                ret_type = self.res_type

            call_expr = irt.FunCall(
                self.called_name,
                arg_exprs,
                param_types=param_types,
                type_hint=ret_type,
                orig_node=self.orig_call_expr
            )

            call_assignment = []
            out_exprs = {}
            assignments = []

            if len(out_indices) > 0 or len(self.postconditions) > 0:
                # If we have out arguments or postconditions, we have no choice
                # but to store the result of the call in a temporary variable
                # "ret" so that we can:
                #  - retrieve each out argument.
                #  - apply postconditions on it.
                ret_var = stores.register_normal(
                    name=fresh_name("ret"),
                    purpose=purpose.SyntheticVariable(),
                    mode=Mode.Local,
                    type_hint=ret_type
                ).get()

                call_assignment = [irt.AssignStmt(
                    ret_var, call_expr, orig_node=self.orig_call_expr
                )]

                if len(out_indices) > 0:
                    out_exprs = {
                        out_index: irt.FunCall(
                            ops.GetName(i), [ret_var],
                            type_hint=param_types[out_index]
                        )
                        for i, out_index in enumerate(out_indices)
                    }
                    assignments = [
                        stmt
                        for out_index, out_expr in out_exprs.iteritems()
                        for stmt in args_out[out_index](out_expr)
                    ]

                    result_expr = irt.FunCall(
                        ops.GetName(len(out_indices)),
                        [ret_var],
                        type_hint=self.res_type
                    )
                else:
                    result_expr = ret_var
            else:
                result_expr = call_expr

            preconds_stmts, postconds_stmts = gen_contract_conditions(
                self.called_name,
                self.preconditions,
                self.postconditions,
                arg_exprs,
                out_exprs,
                result_expr,
                self.orig_call_expr
            )

            return (
                arg_pre_stmts +
                preconds_stmts +
                call_assignment +
                postconds_stmts +
                assignments
            ), result_expr

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
        return _lal_op_type_to_symbol.get((type(lal_op), arity))

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
        tpe = replaced_expr.p_expression_type
        return stores.register_normal(
            name=fresh_name(name),
            purpose=purpose.SyntheticVariable(),
            type_hint=tpe,
            orig_node=replaced_expr,
            mode=Mode.Local
        ).get(orig_node=replaced_expr)

    def get_var(expr, ref):
        """
        :param lal.Expr expr:
        :param lal.ObjectDecl ref:
        :return:
        """
        var = stores.get(ref)
        if var is None:
            return new_expression_replacing_var("unknown", expr)
        else:
            return var.get(orig_node=expr)

    def gen_split_stmt(orig_cond, then_stmts, else_stmts, **data):
        """
        :param lal.Expr orig_cond: The condition of the if statement.

        :param iterable[irt.Stmt] then_stmts: The already transformed then
            statements.

        :param iterable[irt.Stmt] else_stmts: The already transformed else
            statements.

        :param **object data: user data on the generated split statement.

        :return: The corresponding split-assume statements.

        :rtype: list[irt.Stmt]
        """
        cond_pre_stmts, cond = transform_expr(orig_cond)
        not_cond = irt.FunCall(
            ops.NOT,
            [cond],
            type_hint=cond.data.type_hint
        )

        assume_cond, assume_not_cond = (
            irt.AssumeStmt(cond,
                           purpose=purpose.PredeterminedCheck(orig_cond)),
            irt.AssumeStmt(not_cond)
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

    def build_relational(op, lhs, rhs):
        """
        Generate a function call to the given binary relational operator
        between the two given expressions.

        :param str op: The operator to call.
        :param irt.Expr lhs: The left-hand side.
        :param irt.Expr rhs: The right-hand side.
        :rtype: irt.Expr
        """
        return irt.FunCall(op, [lhs, rhs], type_hint=ctx.evaluator.bool)

    def gen_for_condition(iter_var, iter_expr):
        """
        Generate the condition (I in E) where I is the iteration variable and
        E the expression on which to iterate over. E.g:

        - I in 0 .. 10: I >= 0 && I <= 10
        - I in Positive: I >= 1 && I <= 2**32 - 1

        :param irt.Expr iter_var: The iteration variable.
        :param lal.Expr iter_expr: The LAL node representing the expression to
            iterate over.
        :rtype: irt.Expr
        """
        def build_range_condition(low, high):
            """
            Build the IR expression for the call "I >= <low> && I <= <high>".

            :param irt.Expr low: The expression for the lower bound.
            :param irt.Expr high: The expression for the higher bound.
            :rtype: irt.Expr
            """
            return build_relational(
                ops.AND,
                build_relational(ops.GE, iter_var, low),
                build_relational(ops.LE, iter_var, high)
            )

        def gen_when_range_lit(lhs_val, rhs_val):
            """
            Generate the expression for when the two range bounds are known
            literals.

            :param object lhs_val: The literal value for the lower bound.
            :param object rhs_val: The literal value the higher bound.
            :rtype: (list[irt.Stmt], irt.Expr)
            """
            return [], build_range_condition(
                irt.Lit(lhs_val, type_hint=iter_var.data.type_hint),
                irt.Lit(rhs_val, type_hint=iter_var.data.type_hint)
            )

        def gen_when_range_exprs(lhs_expr, rhs_expr):
            """
            Generate the expression for when the two range bounds are
            arbitrarily complex expressions.

            :param lal.Expr lhs_expr: The expression for the lower bound.
            :param lal.Expr rhs_expr: The expression for the higher bound.
            :rtype: (list[irt.Stmt], irt.Expr)
            """
            lhs_pre_stmts, lhs_ir_expr = transform_expr(lhs_expr)
            rhs_pre_stmts, rhs_ir_expr = transform_expr(rhs_expr)
            return lhs_pre_stmts + rhs_pre_stmts, build_range_condition(
                lhs_ir_expr, rhs_ir_expr
            )

        def gen_when_type_reference(ref):
            """
            Generate the expression for when the expression to iterate over
            is a reference to a type, in which case it means we are going to
            iterate over its discrete range.

            :param lal.TypeDecl ref: The type declaration to iterate over.
            :rtype: (list[irt.Stmt], irt.Expr)
            """
            try:
                actual_type = typer.get(ref)
                if actual_type.is_a(types.IntRange):
                    return gen_when_range_lit(
                        actual_type.frm, actual_type.to
                    )
                elif actual_type.is_a(types.Enum):
                    return gen_when_range_lit(
                        actual_type.lits[0], actual_type.lits[-1]
                    )
            except Transformer.TransformationFailure:
                pass
            raise NotImplementedError("Cannot transform type reference")

        if (iter_expr.is_a(lal.AttributeRef)
                and iter_expr.f_attribute.text.lower() == 'range'):
            iter_expr = iter_expr.f_prefix

        if (iter_expr.is_a(lal.Identifier)
                and iter_expr.p_referenced_decl.is_a(lal.BaseTypeDecl)):
            return gen_when_type_reference(iter_expr.p_referenced_decl)
        elif iter_expr.is_a(lal.DiscreteSubtypeIndication):
            return gen_when_type_reference(iter_expr)
        elif (iter_expr.is_a(lal.BinOp)
                and iter_expr.f_op.is_a(lal.OpDoubleDot)):
            return gen_when_range_exprs(iter_expr.f_left, iter_expr.f_right)

        unimplemented(iter_expr)

    def gen_case_condition(expr, choices):
        """
        Example:

        `gen_case_condition(X, [1, 2, 10 .. 20])`

        Will generate the following condition:

        `X == 1 || X == 2 || X >= 10 || X <= 20`

        :param irt.Expr expr: The case's selector expression.

        :param list[lal.AdaNode] choices: The list of alternatives as lal
            nodes.

        :return: An expression corresponding to the condition check for
            entering an alternative of an Ada case statement.

        :rtype: irt.Expr
        """
        try:
            tpe = typer.get(expr.data.type_hint)
        except Transformer.TransformationFailure:
            raise NotImplementedError("Selected expression cannot be typed")

        def gen_lit(lit_value, lit_expr):
            if tpe.is_a(types.ASCIICharacter):
                return irt.Lit(
                    chr(lit_value),
                    type_hint=expr.data.type_hint,
                    orig_node=lit_expr
                )
            elif tpe.is_a(types.IntRange):
                return irt.Lit(
                    lit_value,
                    type_hint=expr.data.type_hint,
                    orig_node=lit_expr
                )
            elif tpe.is_a(types.Enum):
                return irt.Lit(
                    tpe.lits[lit_value],
                    type_hint=expr.data.type_hint,
                    orig_node=lit_expr
                )
            else:
                return unimplemented_expr(lit_expr)[1]

        def gen_single(choice):
            def gen_range(first_val, last_val, first_expr, last_expr):
                first_lit = gen_lit(first_val, first_expr)
                last_lit = gen_lit(last_val, last_expr)
                return build_relational(
                    ops.AND,
                    build_relational(ops.GE, expr, first_lit),
                    build_relational(ops.LE, expr, last_lit)
                )

            if (choice.is_a(lal.Identifier)
                    and choice.p_referenced_decl.is_a(lal.SubtypeDecl)):
                try:
                    actual_type = typer.get(choice.p_referenced_decl)
                    if actual_type.is_a(types.IntRange):
                        return gen_range(
                            actual_type.frm, actual_type.to,
                            None, None
                        )
                except Transformer.TransformationFailure:
                    pass
                raise NotImplementedError("Cannot transform subtype reference")
            elif choice.is_a(lal.BinOp) and choice.f_op.is_a(lal.OpDoubleDot):
                return gen_range(
                    choice.f_left.p_eval_as_int, choice.f_right.p_eval_as_int,
                    choice.f_left, choice.f_right
                )
            else:
                return irt.FunCall(
                    ops.EQ,
                    [expr, gen_lit(choice.p_eval_as_int, choice)],
                    type_hint=ctx.evaluator.bool
                )

        conditions = [gen_single(choice) for choice in choices]

        return reduce(
            binexpr_builder(ops.OR, ctx.evaluator.bool),
            conditions
        )

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
            (alt.f_choices, transformer(alt))
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

    def gen_dereference_dest(prefix, prefix_type, dest, dest_type, expr):
        """
        Generates the actual destination for a dereference expression.
        Example:

        E.g. x.all := 2 becomes $stack = Updated($stack, x, 2)

        :param lal.Expr prefix: The expression to dereference ("x" in the
            example).
        :param lal.TypeDecl prefix_type: The type of the expression to
            dereference (an access type).
        :param lal.Expr dest: The dereference expression ("x.all" in the
            example).
        :param lal.TypeDecl dest_type: The type of the dereferencing expression
            (the type of the accessed elements).
        :param irt.Expr expr: The expression on the RHS.
        :rtype: list[irt.Stmt], (irt.Identifier, irt.Expr)
        """
        prefix_pre_stmts, prefix_expr = transform_expr(prefix)

        return prefix_pre_stmts, (stack, irt.FunCall(
            ops.UPDATED,
            [stack, prefix_expr, expr],
            type_hint=stack.data.type_hint,
            orig_node=dest,
            param_types=_deref_assign_param_types(
                prefix_type,
                dest_type
            )
        ))

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

            var = stores.get(ref)
            if var is not None:
                return [], var.gen_dest(expr, orig_node=dest)

        elif dest.is_a(lal.DottedName) and is_record_field(dest.f_suffix):
            implicit_deref = dest.f_prefix.p_expression_type.p_is_access_type()
            if implicit_deref:
                prefix_type = (dest.f_prefix.p_expression_type
                               .f_type_def.f_subtype_indication
                               .p_designated_type_decl_from(dest))
                prefix_pre_stmts, prefix_expr = transform_dereference(
                    dest.f_prefix, prefix_type, dest.f_prefix
                )
            else:
                prefix_type = dest.f_prefix.p_expression_type
                prefix_pre_stmts, prefix_expr = transform_expr(dest.f_prefix)

            exist_stmts = gen_field_existence_condition(
                prefix_expr,
                dest.f_suffix
            )

            field_info = get_field_info(dest.f_suffix)
            updated_index = field_info.index

            updated_expr = irt.FunCall(
                ops.UpdatedName(updated_index),
                [prefix_expr, expr],
                type_hint=prefix_type,
                orig_node=dest.f_prefix,
                param_types=_field_assign_param_types(field_info)
            )

            if implicit_deref:
                pre_stmts, ret = gen_dereference_dest(
                    dest.f_prefix, dest.f_prefix.p_expression_type,
                    dest.f_prefix, prefix_type, updated_expr
                )
            else:
                pre_stmts, ret = gen_actual_dest(dest.f_prefix, updated_expr)

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
            except Transformer.TransformationFailure:
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
            return gen_dereference_dest(
                dest.f_prefix, dest.f_prefix.p_expression_type,
                dest, dest.p_expression_type, expr
            )

        return unimplemented_dest(dest)

    def gen_field_existence_condition(prefix, field):
        """
        Returns a list of assume statements describing the conditions that
        must hold for the given field to exist for the given object.

        :param irt.Expr prefix: The object whose field is being accessed.
        :param lal.Identifier field: The field being accessed
        :rtype: list[irt.AssumeStmt]
        """
        info = get_field_info(field)
        all_fields = record_fields(
            closest(field.p_referenced_decl, lal.TypeDecl)
        )

        res = []
        for i, (discr, alternatives) in enumerate(info.conds):
            discr_getter = irt.FunCall(
                ops.GetName(get_field_info(discr).index), [prefix],
                type_hint=discr.p_expression_type
            )

            if not any(x.is_a(lal.OthersDesignator) for x in alternatives):
                condition = gen_case_condition(discr_getter, alternatives)
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
                    gen_case_condition(discr_getter, alt)
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
        """
        Generates the IR expression corresponding to an access on the given
        expression.

        :param lal.Expr expr: The expression on which an access is taken.
        :rtype: irt.Expr
        """

        if expr.is_a(lal.Identifier):
            decl = expr.p_referenced_decl
            if decl.is_a(lal.BaseSubpBody, lal.BasicSubpDecl):
                # Access on subprogram. Generate a function call which name
                # carries information on the subprogram that is accessed,
                # including the types of its parameters, etc.
                # The arguments of that function call are the addresses of
                # its captures.

                accessed_subp = get_subp_identity(decl)
                subp_data = ctx.subpdata.get(accessed_subp)

                if subp_data is not None:
                    captures = [
                        stores.get(global_id).access()
                        for global_id in subp_data.all_global_vars
                    ]
                else:
                    captures = []

                param_types, call_interface = get_subp_interface(
                    ctx, accessed_subp
                )

                return irt.FunCall(
                    access_paths.Subprogram(accessed_subp, call_interface),
                    captures,
                    type_hint=PointerType(expr.p_expression_type),
                    additional_args=param_types,
                    orig_node=expr,
                )
            else:
                # Access on variable
                ref = expr.p_xref

                if ref in substitutions:
                    expr = substitutions[ref][1].data.orig_node
                    ref = expr.p_xref

                var = stores.get(ref)
                if var is not None:
                    return var.access(orig_node=expr)

        elif expr.is_a(lal.DottedName) and is_record_field(expr.f_suffix):
            info = get_field_info(expr.f_suffix)
            if info is None:
                unimplemented(expr)

            updated_index = info.index

            return irt.FunCall(
                access_paths.Field(updated_index),
                [gen_access_path(expr.f_prefix)],
                type_hint=PointerType(expr.p_expression_type),
                additional_args=[expr.p_expression_type],
                orig_node=expr
            )

        elif (expr.is_a(lal.AttributeRef)
                and expr.f_attribute.text.lower() == "model"):
            updated_index = 1

            return irt.FunCall(
                access_paths.Field(updated_index),
                [gen_access_path(expr.f_prefix)],
                type_hint=PointerType(expr.p_expression_type),
                additional_args=[expr.p_expression_type],
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

        if len(pres) == len(posts) == 0:
            return [], []

        procs = [proc]
        model = ctx.fun_models.get(proc, None)
        if model is not None:
            procs.append(model)

        for proc in procs:
            for i, name, param in proc_parameters(proc):
                substitutions[name] = ([], args_in[i])

        pre_stmts = [
            stmt
            for pre, must_check in pres
            for stmts, expr in [transform_expr(pre)]
            for stmt in stmts + [irt.AssumeStmt(
                expr,
                purpose=purpose.ContractCheck("precondition", orig_call),
                orig_node=orig_call
            )]
        ]

        for proc in procs:
            substitutions[proc, 'result'] = ([], ret)

        for proc in procs:
            for i, name, param in proc_parameters(proc):
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
                orig_node=orig_call
            )]
        ]

        return pre_stmts, post_stmts

    def zip_with_params(params, prefix, args, is_dot_call):
        """
        Zips the given parameters with the given arguments.

        :param list[(int, lal.Identifier, lal.ParamSpec)] params: The
            parameters to zip with.
        :param lal.Name prefix: The name being called.
        :param lal.AssocList args: The arguments of the call.
        :param bool is_dot_call: Whether the original call is a dot call.
        :rtype: list[(lal.ParamSpec, lal.Expr)]
        """
        actual_args = [(arg.f_designator, arg.f_r_expr) for arg in args]
        if is_dot_call:
            actual_args.insert(0, (None, prefix.f_prefix))

        param_index = {
            name: i
            for i, name, _ in params
        }

        ordered_args = [None] * len(param_index)
        for i, (designator, expr) in enumerate(actual_args):
            if designator is None:
                arg_index = i
            else:
                arg_index = param_index[designator.p_xref]

            ordered_args[arg_index] = expr

        return [
            (param, arg if arg is not None else param.f_default_expr)
            for (_, _, param), arg in zip(params, ordered_args)
        ]

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

        3. `F` is an access on function (without out parameters):

        r := Call(F, x_1, ..., x_n, $stack)

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

        if prefix.is_a(lal.Identifier, lal.DottedName):
            ref = prefix.p_referenced_decl
            if ref is not None and ref.is_a(lal.SubpBody, lal.SubpDecl):
                # The call target is statically known.

                called_name = ref
                if ref.is_a(lal.BasicSubpDecl):
                    try:
                        body = ref.p_body_part
                        if body is not None:
                            called_name = body
                    except lal.PropertyError:
                        pass
                builder = CallExprBuilder(orig_node, called_name, type_hint)

                param_args = zip_with_params(
                    proc_parameters(ref),
                    prefix,
                    args,
                    prefix.p_is_dot_call
                )

                for param, arg_expr in param_args:
                    param_type = param.f_type_expr

                    assigner = None
                    if param.f_mode.is_a(lal.ModeOut, lal.ModeInOut):
                        def assigner(out_expr, arg_expr=arg_expr):
                            return gen_assignment(arg_expr, [], out_expr)

                    builder.add_argument(
                        param_type, transform_expr(arg_expr), assigner
                    )

                # Pass global variables by reference

                called_subpdata = ctx.subpdata.get(get_subp_identity(ref))
                if called_subpdata is not None:
                    for global_var_id in called_subpdata.all_global_vars:
                        local_id = stores.get(global_var_id).access()
                        builder.add_argument(
                            local_id.data.type_hint, ([], local_id)
                        )

                # Pass stack too

                global_access = _find_global_access(ctx, ref)

                if global_access != _IGNORES_GLOBAL_STATE:
                    builder.add_stack_argument(
                        global_access == _WRITES_GLOBAL_STATE
                    )

                builder.add_contracts(*retrieve_function_contracts(ctx, ref))

                return builder.build()

        if is_access_to_subprogram(prefix.p_expression_type):
            subp_type = prefix.p_expression_type.f_type_def

            # Transform the expression being dereferenced and build the
            # assume expression stating that the prefix is not null.
            prefix_pre_stmts, prefix_expr = transform_expr(prefix)
            assumed_expr = irt.FunCall(
                ops.NEQ,
                [
                    prefix_expr,
                    irt.Lit(lits.NULL, type_hint=prefix.p_expression_type)
                ],
                type_hint=ctx.evaluator.bool
            )
            prefix_pre_stmts.append(irt.AssumeStmt(
                assumed_expr,
                purpose=purpose.DerefCheck(prefix_expr)
            ))

            builder = CallExprBuilder(orig_node, ops.CALL, type_hint)
            builder.add_argument(subp_type, (prefix_pre_stmts, prefix_expr))

            param_args = zip_with_params(
                proc_parameters(subp_type),
                prefix,
                args,
                False
            )

            for param, arg_expr in param_args:
                param_type = param.f_type_expr

                assigner = None
                if param.f_mode.is_a(lal.ModeOut, lal.ModeInOut):
                    def assigner(out_expr, arg_expr=arg_expr):
                        return gen_assignment(arg_expr, [], out_expr)

                builder.add_argument(
                    param_type, transform_expr(arg_expr), assigner
                )

            builder.add_stack_argument(out=True)
            return builder.build()

        if is_array_type_decl(prefix.p_expression_type):
            arr_type = prefix.p_expression_type.f_type_def
            builder = CallExprBuilder(orig_node, ops.CALL, type_hint)

            builder.add_argument(arr_type, transform_expr(prefix))

            for idx_type, arg in zip(arr_type.f_indices.f_list, args):
                builder.add_argument(idx_type, transform_expr(arg.f_r_expr))

            return builder.build()

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

        Note that dereference can also be a function call. Check out
        gen_call_expr to see how calls through subprogram pointers are
        transformed.

        :param lal.Expr derefed_expr: The expression being dereferenced.
        :param lal.AdaNode deref_type: The type of the dereference expression.
        :param lal.Expr deref_orig: The original dereference node.
        :rtype: (list[irt.Stmt], irt.Expr)
        """

        if is_access_to_subprogram(derefed_expr.p_expression_type):
            return gen_call_expr(derefed_expr, [], deref_type, deref_orig)

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
        all_fields = list(record_fields(record_decl))
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
                    get_field_info(designator).index
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
                iop = transform_operator(expr.f_op, 2)
                if iop is not None:
                    lhs_pre_stmts, lhs = transform_expr(expr.f_left)
                    rhs_pre_stmts, rhs = transform_expr(expr.f_right)

                    return lhs_pre_stmts + rhs_pre_stmts, irt.FunCall(
                        transform_operator(expr.f_op, 2),
                        [lhs, rhs],
                        type_hint=expr.p_expression_type,
                        orig_node=expr
                    )

        elif expr.is_a(lal.UnOp):
            iop = transform_operator(expr.f_op, 1)
            if iop is not None:
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
            elif ref.p_basic_decl.is_a(lal.ObjectDecl, lal.ParamSpec,
                                       lal.ForLoopVarDecl):
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
            elif is_record_field(expr.f_suffix):
                if expr.f_prefix.p_expression_type.p_is_access_type():
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

                field_info = get_field_info(expr.f_suffix)
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
            return [], irt.Lit(
                expr.p_denoted_value,
                type_hint=ctx.evaluator.char
            )

        elif expr.is_a(lal.StringLiteral):
            lit = new_expression_replacing_var("tmp", expr)
            text = expr.p_denoted_value

            def build_lit():
                # Transform the string literal into a call to the "String"
                # method as such:
                #
                # Ada:
                # ---------------------------------
                # "abc"
                #
                # Basic IR:
                # ---------------------------------
                # String(tmp, 0, a, 1, b, 2, c)

                args = [lit]  # type: list[irt.Expr]
                for i in range(len(text)):
                    args.append(irt.Lit(
                        i + 1, type_hint=ctx.evaluator.universal_int
                    ))
                    args.append(irt.Lit(
                        text[i], type_hint=ctx.evaluator.char
                    ))
                return irt.FunCall(
                    ops.STRING,
                    args,
                    type_hint=expr.p_expression_type,
                    orig_node=expr,
                    param_types=_string_literal_param_types(
                        expr.p_expression_type, text
                    )
                )

            return [irt.AssignStmt(lit, build_lit())], lit

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
                try:
                    return [], gen_access_path(expr.f_prefix)
                except AccessingUnspilledVar:
                    # The best thing we can do in this case is to assume that
                    # the pointer is not null (we know that because it should
                    # have been an access on a local variable).
                    ret = new_expression_replacing_var("ptr", expr)
                    tpe = expr.p_expression_type
                    return [irt.AssumeStmt(irt.FunCall(
                        ops.NEQ,
                        [ret, irt.Lit(lits.NULL, type_hint=tpe)],
                        param_types=[tpe, tpe],
                        type_hint=ctx.evaluator.bool
                    ))], ret
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
                param_vars.append(stores.register_normal(
                    defining_name=var_id,
                    type_hint=param.f_type_expr,
                    orig_node=var_id,
                    mode=mode
                ).var)

        if spec.f_subp_returns is not None:
            result_var.value = stores.register_normal(
                name=fresh_name("result"),
                type_hint=spec.f_subp_returns
            ).get()

        for glob_id in subpdata.all_global_vars:
            glob_access_param = stores.register_normal(
                name=glob_id.text + "_ptr",
                type_hint=PointerType(glob_id.p_basic_decl.f_type_expr),
                orig_node=glob_id,
                mode=Mode.Local
            )
            param_vars.append(glob_access_param.var)
            stores.register_spilled_absolute(
                defining_name=glob_id,
                address_handle=glob_access_param,
                type_hint=glob_id.p_basic_decl.f_type_expr
            )

        if _find_global_access(ctx, subp) != _IGNORES_GLOBAL_STATE:
            param_vars.append(stack.var)

        return []

    def transform_decl(decl):
        """
        :param lal.BasicDecl decl: The lal decl to transform.

        :return: A (potentially empty) list of statements that emulate the
            Ada semantics of the declaration.

        :rtype: list[irt.Stmt]
        """
        if decl.is_a(lal.TypeDecl, lal.SubtypeDecl, lal.IncompleteTypeDecl,
                     lal.NumberDecl, lal.PackageDecl, lal.PackageBody,
                     lal.SubpDecl, lal.SubpBody,
                     lal.UsePackageClause, lal.UseTypeClause,
                     lal.GenericSubpInstantiation,
                     lal.GenericPackageInstantiation):
            return []

        elif decl.is_a(lal.ObjectDecl):
            tdecl = decl.f_type_expr

            for var_id in decl.f_ids:
                if var_id in to_spill:
                    stores.register_spilled_relative(
                        defining_name=var_id,
                        type_hint=tdecl
                    )
                else:
                    stores.register_normal(
                        defining_name=var_id,
                        type_hint=tdecl,
                        orig_node=var_id,
                        mode=Mode.Out
                    )

            if decl.f_default_expr is None:
                return [
                    stmt
                    for var_id in decl.f_ids
                    for stmt in stores.get(var_id).havoc(
                        read_orig_node=decl, orig_node=var_id
                    )
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
            elif call_expr.is_a(lal.ExplicitDeref):
                return gen_call_expr(
                    call_expr.f_prefix, [], call_expr.p_expression_type, stmt
                )[0]
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
                [irt.AssumeStmt(
                    cond,
                    purpose=purpose.PredeterminedCheck(stmt.f_spec.f_expr)
                )] +
                loop_stmts,
                orig_node=stmt
            ), irt.AssumeStmt(not_cond), exit_label]

        elif stmt.is_a(lal.ForLoopStmt):
            # For loops are transformed as such:
            #
            # Ada:
            # ----------------
            # for I in E loop
            #   S;
            # end loop;
            #
            # Basic:
            # ----------------
            # assume(I in E)
            # loop:
            #   S;
            #
            # Where "I in E" is transformed into the adequate expression. For
            # example, when E is a range indication, "I in L .. R" would be
            # "I >= L && I <= R".
            #
            # For now, only a few subsets of for loops are handled.
            # todo: handle "for x in X'Range" where X is an array.
            # todo: handle "for x of ..." pattern.

            spec = stmt.f_spec
            if spec.f_loop_type.is_a(lal.IterTypeIn):
                loop_var = spec.f_var_decl.f_id
                var_expr = stores.register_normal(
                    defining_name=loop_var,
                    type_hint=loop_var.p_expression_type,
                    orig_node=loop_var,
                    mode=Mode.Local
                ).get()

                pre_stmts, cond = gen_for_condition(var_expr, spec.f_iter_expr)

                exit_label = irt.LabelStmt(fresh_name('exit_for_loop'))

                loop_stack.append((stmt, exit_label))
                loop_stmts = transform_stmts(stmt.f_stmts)
                loop_stack.pop()

                return pre_stmts + [irt.AssumeStmt(cond), irt.LoopStmt(
                    loop_stmts
                ), exit_label]

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

        elif stmt.is_a(lal.ExtendedReturnStmt):
            stmts = []
            var_decl = stmt.f_decl.f_ids[0]
            stmts.extend(transform_decl(stmt.f_decl))
            stmts.extend(transform_stmts(stmt.f_stmts.f_stmts))

            var = stores.get(var_decl)
            if var is None:
                unimplemented(stmt)

            stmts.append(irt.AssignStmt(
                result_var.value,
                var.get(orig_node=var.var.data.orig_node),
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

        elif stmt.is_a(lal.PragmaNode):
            if stmt.f_id.text.lower() == "assert":
                expr_pre_stmts, expr = transform_expr(stmt.f_args[0].f_expr)
                return expr_pre_stmts + [irt.AssumeStmt(
                    expr,
                    purpose=purpose.ContractCheck('assertion', stmt),
                    orig_node=stmt
                )]

        unimplemented(stmt)

    def print_warning(subject, exception):
        with log_stdout('info'):
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

    def transform_expr_function_body(expr):
        """
        Generate the body of this subprogram when it is an expression
        function.

        :param lal.Expr expr: The expression body of the expression function.
        :rtype: list[irt.Stmt]
        """
        res = []
        try:
            ret_expr_stmts, ret_expr = transform_expr(expr)
            res.extend(ret_expr_stmts)
            res.append(irt.AssignStmt(result_var.value, ret_expr))
        except (lal.PropertyError, NotImplementedError,
                KeyError, NotConstExprError) as e:
            print_warning(expr.text, e)
        return res

    def transform_subp_body(subp):
        """
        Generate the body of this subprogram when it is an actual subprogram
        body.

        :param lal.SubpBody subp: The subprogram which body to transform.
        :rtype: list[irt.Stmt]
        """
        return (transform_decls(subp.f_decls.f_decls) +
                transform_stmts(subp.f_stmts.f_stmts))

    # Generate function body statements. Subp may be an expression function or
    # a normal subprogram.

    fun_body_stmts = transform_spec(subp.f_subp_spec)
    if subp.is_a(lal.ExprFunction):
        fun_body_stmts.extend(transform_expr_function_body(subp.f_expr))
    else:
        fun_body_stmts.extend(transform_subp_body(subp))
    fun_body_stmts.append(func_end_label)

    return irt.Program(
        fun_body_stmts,
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
        except Transformer.TransformationFailure:
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
            data = {}
            if 'orig_node' in expr.data:
                data['orig_node'] = expr.data.orig_node

            return irt.Lit(
                self.evaluator.eval(expr),
                type_hint=expected_type,
                **data
            )
        except NotConstExprError:
            if self.has_universal_type(expr):
                expr.data = expr.data.copy(type_hint=expected_type)
            expr.visit(self, expr.data.type_hint)
            return expr

    def visit_assign(self, assign):
        if self.has_universal_type(assign.id):
            # When the variable assigned has an universal type, we must find
            # a compatible "concrete" type to replace it with.
            compatible_type = self.compatible_type(assign.id.data.type_hint)
            assign.id.visit(self, compatible_type)

        assign.expr = self.try_convert_expr(
            assign.expr, assign.id.data.type_hint
        )

    def visit_assume(self, assume):
        assume.expr = self.try_convert_expr(assume.expr, self.evaluator.bool)

    def visit_funcall(self, funcall, expected_type):
        if any(self.has_universal_type(arg) for arg in funcall.args):
            if 'param_types' in funcall.data:
                assert(len(funcall.data.param_types) == len(funcall.args))
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

    def visit_ident(self, ident, *expected_type):
        # An identifier doesn't always act as an expression (see ReadStmt).
        # However when it does, it must have an "expected_type" argument,
        # coming from one of the above visits.
        if len(expected_type) == 1:
            if self.has_universal_type(ident.var):
                ident.var.data = ident.var.data.copy(
                    type_hint=expected_type[0]
                )
