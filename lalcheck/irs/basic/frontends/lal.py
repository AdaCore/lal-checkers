"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt, purpose
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.constants import ops, lits
from lalcheck.utils import KeyCounter, Transformer
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
    'Access': ops.ADDRESS,
    'First': ops.GET_FIRST,
    'Last': ops.GET_LAST,
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


def _record_fields(record_def):
    """
    Returns an iterable of the fields of the given record, where a field is
    identified by the pair (its ComponentDecl, its Identifier).

    :param lal.RecordDef record_def: The record whose fields to list.
    """
    for component in record_def.f_components.f_components:
        for name in component.f_ids:
            yield (component, name.text)


def _compute_field_index(field_id):
    """
    Computes the index of the given record's field. Example:
    type Foo is record
      x, y : Integer;
      z : Integer;
    end record;

    => "x" has index 0, "y" has index 1, "z" has index 2

    :param lal.Identifier field_id: An identifier referring to a record's
        field.
    :rtype: int
    """
    record_def = field_id.p_referenced_decl.parent.parent.parent
    return next(
        i
        for i, field in enumerate(_record_fields(record_def))
        if field == (field_id.p_referenced_decl, field_id.text)
    )


def _is_array_type_decl(tpe):
    """
    Returns True iff the given type is an array type decl.
    :param lal.AdaNode tpe: The type to check.
    """
    if tpe.is_a(lal.TypeDecl):
        return tpe.f_type_def.is_a(lal.ArrayTypeDef)
    return False


def _gen_ir(ctx, subp):
    """
    Generates Basic intermediate representation from a lal subprogram body.

    :param ExtractionContext ctx: The program extraction context.

    :param lal.SubpBody subp: The subprogram body from which to generate IR.

    :return: a Basic Program.

    :rtype: irt.Program
    """

    var_decls = {}
    tmp_vars = KeyCounter()

    # Pre-transform every label, as a label might not have been seen yet when
    # transforming a goto statement.
    labels = {
        label_decl: irt.LabelStmt(
            label_decl.f_name.text,
            orig_node=label_decl
        )
        for label_decl in subp.findall(lal.LabelDecl)
    }

    # Store the loops which we are currently in while traversing the syntax
    # tree. The tuple (loop_statement, exit_label) is stored.
    loop_stack = []

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
        :param lal.AdaNode node: The node that cannot be transformed.
        :raise NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Cannot transform "{}" ({})'.format(node.text, type(node))
        )

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
                mode=Mode.Local
            ),
            type_hint=replaced_expr.p_expression_type,
            orig_node=replaced_expr
        )

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
                    type_hint=ctx.evaluator.int
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
        :rtype: list[irt.Stmt], irt.Identifier, irt.Expr
        """
        if dest.is_a(lal.Identifier):
            return [], (irt.Identifier(
                var_decls[
                    dest.p_referenced_decl,
                    dest.text
                ],
                type_hint=dest.p_expression_type,
                orig_node=dest
            ), expr)

        elif dest.is_a(lal.DottedName):
            updated_index = _compute_field_index(dest.f_suffix)
            prefix_pre_stmts, prefix_expr = transform_expr(dest.f_prefix)

            pre_stmts, ret = gen_actual_dest(dest.f_prefix, irt.FunCall(
                ops.updated(updated_index),
                [prefix_expr, expr],
                type_hint=dest.f_prefix.p_expression_type,
                orig_node=dest.f_prefix,
                purpose=purpose.FieldAssignment(
                    updated_index,
                    dest.f_suffix.p_expression_type
                )
            ))
            return prefix_pre_stmts + pre_stmts, ret

        elif dest.is_a(lal.CallExpr):
            prefix_pre_stmts, prefix_expr = transform_expr(dest.f_name)
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
                purpose=purpose.CallAssignment(dest.f_name.p_expression_type)
            ))
            return (
                prefix_pre_stmts + pre_stmts + suffix_pre_stmts,
                ret
            )

        unimplemented(dest)

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
            [expr],
            type_hint=deref_type,
            orig_node=deref_orig
        )

    def transform_record_aggregate(expr):
        """
        :param lal.Aggregate expr: The aggregate expression.
        :return: Its IR transformation.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        record_def = expr.p_expression_type.f_type_def.f_record_def
        all_fields = list(_record_fields(record_def))
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
                    _compute_field_index(designator)
                    for designator in assoc.f_designators
                ]

            for idx in indexes:
                field_init[idx] = r_exprs[i]

        for i in range(len(field_init)):
            if field_init[i] is None:
                field_init[i] = r_exprs[others_expr_idx]

        return sum(r_exprs_pre_stmts, []), irt.FunCall(
            ops.NEW,
            field_init,
            type_hint=expr.p_expression_type,
            orig_node=expr
        )

    def transform_array_aggregate(expr):
        """
        :param lal.Aggregate expr: The aggregate expression.
        :return: its IR transformation.
        :rtype: (list[irt.Stmt], irt.Expr)
        """
        array_def = expr.p_expression_type.f_type_def
        print(array_def.dump())
        unimplemented(expr)

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
            prefix_pre_stmts, prefix_expr = transform_expr(expr.f_name)
            suffixes = [
                transform_expr(suffix.f_r_expr)
                for suffix in expr.f_suffix
            ]
            suffix_pre_stmts = [
                suffix_stmt for suffix in suffixes
                for suffix_stmt in suffix[0]
            ]
            suffix_exprs = [suffix[1] for suffix in suffixes]

            return prefix_pre_stmts + suffix_pre_stmts, irt.FunCall(
                ops.CALL,
                [prefix_expr] + suffix_exprs,
                type_hint=expr.p_expression_type,
                orig_node=expr,
                callee_type=expr.f_name.p_expression_type
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
            #      when others => E4;
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
            #   assume(!(x == CST1 || (x == CST2 || x == CST3) ||
            #          x >= GetFirst(Range) && x <= GetLast(Range)))
            #   tmp = E4
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
            ref = expr.p_referenced_decl
            if ref.is_a(lal.ObjectDecl, lal.ParamSpec):
                return [], irt.Identifier(
                    var_decls[ref, expr.text],
                    type_hint=expr.p_expression_type,
                    orig_node=expr
                )
            elif ref.is_a(lal.EnumLiteralDecl):
                return [], irt.Lit(
                    expr.text,
                    type_hint=ref.parent.parent.parent,
                    orig_node=expr
                )
            elif ref.is_a(lal.NumberDecl):
                return transform_expr(ref.f_expr)
            elif ref.is_a(lal.TypeDecl):
                if ref.f_type_def.is_a(lal.SignedIntTypeDef):
                    return transform_expr(ref.f_type_def.f_range.f_range)

        elif expr.is_a(lal.DottedName):
            # Field access is transformed as such:
            # Ada:
            # ---------------
            # r := x.f
            #
            # Basic IR:
            # ---------------
            # r = Get_N(x)
            #
            # Where N is the index of the field "f" in the record x
            # (see _compute_field_index).
            if expr.f_prefix.p_expression_type.p_is_access_type:
                access_type_def = expr.f_prefix.p_expression_type.f_type_def
                accessed_type = access_type_def.f_subtype_indication
                prefix_pre_stmts, prefix = transform_dereference(
                    expr.f_prefix, accessed_type, expr.f_prefix
                )
            else:
                prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)

            return prefix_pre_stmts, irt.FunCall(
                ops.get(_compute_field_index(expr.f_suffix)),
                [prefix],
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.IntLiteral):
            return [], irt.Lit(
                int(expr.f_tok.text),
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.NullLiteral):
            return [], irt.Lit(
                lits.NULL,
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.Aggregate):
            type_def = expr.p_expression_type.f_type_def
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

            prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)
            return prefix_pre_stmts, irt.FunCall(
                _attr_to_unop[expr.f_attribute.text],
                [prefix],
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        unimplemented(expr)

    def transform_spec(spec):
        """
        :param lal.SubpSpec spec: The subprogram's specification
        :return:
        """
        params = spec.f_subp_params.f_params

        for param in params:
            mode = Mode.from_lal_mode(param.f_mode)
            for var_id in param.f_ids:
                var_decls[param, var_id.text] = irt.Variable(
                    var_id.text,
                    type_hint=param.f_type_expr,
                    orig_node=var_id,
                    mode=mode
                )

        return []

    def transform_decl(decl):
        """
        :param lal.BasicDecl decl: The lal decl to transform.

        :return: A (potentially empty) list of statements that emulate the
            Ada semantics of the declaration.

        :rtype: list[irt.Stmt]
        """
        if decl.is_a(lal.TypeDecl, lal.NumberDecl):
            return []
        elif decl.is_a(lal.ObjectDecl):
            tdecl = decl.f_type_expr.p_designated_type_decl

            for var_id in decl.f_ids:
                var_decls[decl, var_id.text] = irt.Variable(
                    var_id.text,
                    type_hint=tdecl,
                    orig_node=var_id,
                    mode=Mode.Out
                )

            if decl.f_default_expr is None:
                return [
                    irt.ReadStmt(
                        irt.Identifier(
                            var_decls[decl, var_id.text],
                            type_hint=tdecl,
                            orig_node=var_id
                        ),
                        orig_node=decl
                    )
                    for var_id in decl.f_ids
                ]
            else:
                dval_pre_stmts, dval_expr = transform_expr(decl.f_default_expr)
                return dval_pre_stmts + [
                    irt.AssignStmt(
                        irt.Identifier(
                            var_decls[decl, var_id.text],
                            type_hint=tdecl,
                            orig_node=var_id
                        ),
                        dval_expr,
                        orig_node=decl
                    )
                    for var_id in decl.f_ids
                ]

        unimplemented(decl)

    def transform_stmt(stmt):
        """
        :param lal.Stmt stmt: The lal statement to transform.

        :return: A list of statement that emulate the Ada semantics of the
            statement being transformed.

        :rtype: list[irt.Stmt]
        """

        if stmt.is_a(lal.AssignStmt):
            expr_pre_stmts, expr = transform_expr(stmt.f_expr)
            dest_pre_stmts, (dest, updated_expr) = gen_actual_dest(
                stmt.f_dest, expr
            )
            return dest_pre_stmts + expr_pre_stmts + [
                irt.AssignStmt(
                    dest,
                    updated_expr,
                    orig_node=stmt
                )
            ]

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
            #   when others =>
            #     S4;
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
            #   assume(!(x == CST1 || (x == CST2 || x == CST3) ||
            #          x >= GetFirst(Range) && x <= GetLast(Range)))
            #   S4
            #
            # Note: In Ada, case statements must be complete and *disjoint*.
            # This allows us to transform the case in a split of N branches
            # instead of in a chain of if-elsifs.

            return gen_case_base(
                stmt.f_case_expr,
                stmt.f_case_alts,
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

            if stmt.f_condition is None:
                # If there is no "when" part, only generate a goto statement.
                return [exit_goto]
            else:
                # Else emulate the behavior with split-assume statements.
                return gen_split_stmt(
                    stmt.f_condition,
                    [exit_goto],
                    [],
                    orig_node=stmt
                )

        elif stmt.is_a(lal.ExceptionHandler):
            # todo ?
            return []

        unimplemented(stmt)

    def transform_decls(decls):
        """
        :param iterable[lal.BasicDecl] decls: An iterable of decls
        :return: The transformed list of statements.
        :rtype: list[irt.Stmt]
        """
        res = []
        for decl in decls:
            res.extend(transform_decl(decl))
        return res

    def transform_stmts(stmts):
        """
        :param iterable[lal.Stmt] stmts: An iterable of stmts
        :return: The transformed list of statements.
        :rtype: list[irt.Stmt]
        """
        res = []
        for stmt in stmts:
            res.extend(transform_stmt(stmt))
        return res

    return irt.Program(
        transform_spec(subp.f_subp_spec) +
        transform_decls(subp.f_decls.f_decls) +
        transform_stmts(subp.f_stmts.f_stmts),
        orig_node=subp
    )


class ConvertUniversalTypes(IRImplicitVisitor):
    """
    Visitor that mutates the given IR tree so as to remove references to
    universal types from in node data's type hints.
    """

    def __init__(self, evaluator):
        """
        :param ConstExprEvaluator evaluator: A const expr evaluator.
        """
        super(ConvertUniversalTypes, self).__init__()

        self.evaluator = evaluator

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
            expr.visit(self)
            return expr

    def visit_assign(self, assign):
        assign.expr = self.try_convert_expr(
            assign.expr,
            assign.id.data.type_hint
        )

    def visit_assume(self, assume):
        assume.expr = self.try_convert_expr(assume.expr, self.evaluator.bool)

    def visit_funcall(self, funcall):
        if any(self.has_universal_type(arg) for arg in funcall.args):
            if 'callee_type' in funcall.data:
                # Case where we have information about the callee type
                tpe = funcall.data.callee_type
                if _is_array_type_decl(tpe):
                    # if the "callee" is an array
                    indices = tpe.f_type_def.f_indices.f_list
                    funcall.args[1:] = [
                        self.try_convert_expr(arg, indices[i])
                        for i, arg in enumerate(funcall.args[1:])
                    ]
            elif purpose.FieldAssignment.is_purpose_of(funcall):
                # Case where the function is an update call replacing a
                # field assignment. (p.x = y)
                funcall.args[1] = self.try_convert_expr(
                    funcall.args[1],
                    funcall.data.purpose.field_type_hint
                )
            elif purpose.CallAssignment.is_purpose_of(funcall):
                # Case where the function is an udpate call replacing a
                # call assignment. (a(i) = b)
                tpe = funcall.data.purpose.callee_type
                if _is_array_type_decl(tpe):
                    # if the "callee" is an array
                    tdef = tpe.f_type_def
                    indices = tdef.f_indices.f_list
                    funcall.args[1] = self.try_convert_expr(
                        funcall.args[1],
                        tdef.f_component_type.f_type_expr
                    )
                    funcall.args[2:] = [
                        self.try_convert_expr(arg, indices[i])
                        for i, arg in enumerate(funcall.args[2:])
                    ]
            else:
                # Otherwise, assume that functions that accept one argument
                # as universal int/real need all their arguments to be of the
                # same type, which is true for arithmetic ops, comparison ops,
                # etc.
                expected_type = next(
                    arg.data.type_hint
                    for arg in funcall.args
                    if not self.has_universal_type(arg)
                )

                funcall.args = [
                    self.try_convert_expr(arg, expected_type)
                    for arg in funcall.args
                ]

        super(ConvertUniversalTypes, self).visit_funcall(funcall)


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

    def __init__(self, bool_type, int_type, u_int_type, u_real_type):
        """
        :param lal.AdaNode bool_type: The standard boolean type.
        :param lal.AdaNode int_type: The standard int type.
        :param lal.AdaNode u_int_type: The standard universal int type.
        :param lal.AdaNode u_real_type: The standard universal real type.
        """
        super(ConstExprEvaluator, self).__init__()
        self.bool = bool_type
        self.int = int_type
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


@types.typer
def int_range_typer(hint):
    """
    :param lal.AdaNode hint: the lal type.
    :return: The corresponding lalcheck type.
    :rtype: types.IntRange
    """
    if hint.is_a(lal.TypeDecl):
        if hint.f_type_def.is_a(lal.SignedIntTypeDef):
            rng = hint.f_type_def.f_range.f_range
            frm = int(rng.f_left.text)
            to = int(rng.f_right.text)
            return types.IntRange(frm, to)


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
            return types.Enum([lit.f_enum_identifier.text for lit in literals])


def access_typer(inner_typer):
    """
    :param types.Typer[lal.AdaNode] inner_typer: A typer for elements
        being accessed.

    :return: A Typer for Ada's access types.

    :rtype: types.Typer[lal.AdaNode]
    """

    @Transformer.as_transformer
    def accessed_type(hint):
        """
        :param lal.AdaNode hint: the lal type.
        :return: The lal type being accessed.
        :rtype: lal.AdaNode
        """
        if hint.is_a(lal.TypeDecl):
            if hint.p_is_access_type:
                return hint.f_type_def.f_subtype_indication

    to_pointer = Transformer.as_transformer(types.Pointer)
    return accessed_type >> inner_typer >> to_pointer


def record_typer(elem_typer):
    """
    :param types.Typer[lal.AdaNode] elem_typer: A typer for elements of
        products.

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
                r_def = hint.f_type_def.f_record_def
                return [
                    decl.f_component_def.f_type_expr
                    for decl, _ in _record_fields(r_def)
                ]

    to_product = Transformer.as_transformer(types.Product)

    # Get the elements -> type them all -> generate the product type.
    return get_elements >> elem_typer.lifted() >> to_product


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
            return hint.f_name.p_referenced_decl

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


class ExtractionContext(object):
    """
    The libadalang-based frontend interface. Provides method for extracting
    IR programs from Ada source files (see extract_programs), as well as
    a default typer for those programs (see default_typer).

    Note: programs extracted using different ExtractionContext are not
    compatible. Also, this extraction context must be kept alive as long
    as the programs parsed with it are intended to be used.
    """
    def __init__(self):
        self.lal_ctx = lal.AnalysisContext()

        # Get a dummy node, needed to call static properties of libadalang.
        dummy = self.lal_ctx.get_from_buffer(
            "<dummy>", 'package Dummy is end;'
        ).root

        self.evaluator = ConstExprEvaluator(
            dummy.p_bool_type,
            dummy.p_int_type,
            dummy.p_universal_int_type,
            dummy.p_universal_real_type
        )

    def _parse_file(self, ada_file):
        """
        Parses the given file.

        :param str ada_file: The path to the file to parse.
        :rtype: lal.AnalysisUnit
        """
        return self.lal_ctx.get_from_file(ada_file)

    def extract_programs(self, ada_file):
        """
        :param str ada_file: A path to the Ada source file from which to
            extract programs.

        :return: a Basic IR Program for each subprogram body that exists in the
            given source code.

        :rtype: iterable[irt.Program]
        """
        unit = self._parse_file(ada_file)

        if unit.root is None:
            print('Could not parse {}:'.format(ada_file))
            for diag in unit.diagnostics:
                print('   {}'.format(diag))
                return

        unit.populate_lexical_env()

        progs = [
            _gen_ir(self, subp)
            for subp in unit.root.findall((
                lal.SubpBody,
                lal.ExprFunction
            ))
        ]

        converter = ConvertUniversalTypes(self.evaluator)

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

        return typer

    def default_typer(self):
        """
        :return: The default Typer for Ada programs parsed using this
            extraction context.

        :rtype: types.Typer[lal.AdaNode]
        """

        standard_typer = self.standard_typer()

        @types.memoizing_typer
        @types.delegating_typer
        def typer():
            """
            :rtype: types.Typer[lal.AdaNode]
            """
            return (standard_typer |
                    int_range_typer |
                    enum_typer |
                    access_typer(typer) |
                    record_typer(typer) |
                    name_typer(typer) |
                    anonymous_typer(typer) |
                    array_typer(typer, typer))

        return typer
