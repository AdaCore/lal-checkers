"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt, purpose
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.constants import ops, lits
from lalcheck.utils import KeyCounter
from lalcheck import types

from funcy.calc import memoize


_lal_op_type_to_symbol = {
    (lal.OpLt, 2): irt.bin_ops[ops.LT],
    (lal.OpLte, 2): irt.bin_ops[ops.LE],
    (lal.OpEq, 2): irt.bin_ops[ops.EQ],
    (lal.OpNeq, 2): irt.bin_ops[ops.NEQ],
    (lal.OpGte, 2): irt.bin_ops[ops.GE],
    (lal.OpGt, 2): irt.bin_ops[ops.GT],
    (lal.OpAnd, 2): irt.bin_ops[ops.AND],
    (lal.OpOr, 2): irt.bin_ops[ops.OR],
    (lal.OpPlus, 2): irt.bin_ops[ops.PLUS],
    (lal.OpMinus, 2): irt.bin_ops[ops.MINUS],
    (lal.OpDoubleDot, 2): irt.bin_ops[ops.DOT_DOT],

    (lal.OpMinus, 1): irt.un_ops[ops.NEG],
    (lal.OpNot, 1): irt.un_ops[ops.NOT],
}

_attr_to_unop = {
    'Access': irt.un_ops[ops.ADDRESS],
    'First': irt.un_ops[ops.GET_FIRST],
    'Last': irt.un_ops[ops.GET_LAST],
}


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
        :rtype: irt.Operator
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
                orig_node=replaced_expr
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
        not_cond = irt.UnExpr(
            irt.un_ops[ops.NOT],
            cond,
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
        :param irt.Operator op: The binary operator.

        :return: A function taking an lhs and an rhs and returning a binary
            expression using this builder's operator.

        :rtype: (irt.Expr, irt.Expr)->irt.Expr
        """
        def build(lhs, rhs):
            return irt.BinExpr(
                lhs, op, rhs,
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
                return irt.BinExpr(
                    expr,
                    irt.bin_ops[ops.EQ],
                    gen_lit(value),
                    type_hint=ctx.evaluator.bool
                )
            elif isinstance(value, ConstExprEvaluator.Range):
                if (isinstance(value.first, int) and
                        isinstance(value.last, int)):
                    return irt.BinExpr(
                        irt.BinExpr(
                            expr,
                            irt.bin_ops[ops.GE],
                            gen_lit(value.first),
                            type_hint=ctx.evaluator.bool
                        ),
                        irt.bin_ops[ops.AND],
                        irt.BinExpr(
                            expr,
                            irt.bin_ops[ops.LE],
                            gen_lit(value.last),
                            type_hint=ctx.evaluator.bool
                        ),
                        type_hint=ctx.evaluator.bool
                    )

            raise NotImplementedError("Cannot transform when condition")

        conditions = [gen_single(value) for value in values]

        if len(conditions) > 1:
            return reduce(
                binexpr_builder(irt.bin_ops[ops.OR], ctx.evaluator.bool),
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
        others_condition = irt.UnExpr(
            irt.un_ops[ops.NOT],
            reduce(
                binexpr_builder(irt.bin_ops[ops.OR], ctx.evaluator.bool),
                alts_conditions
            ),
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

        if expr.is_a(lal.BinOp):

            if expr.f_op.is_a(lal.OpAndThen, lal.OpOrElse):
                return transform_short_circuit_ops(expr)
            else:
                lhs_pre_stmts, lhs = transform_expr(expr.f_left)
                rhs_pre_stmts, rhs = transform_expr(expr.f_right)

                return lhs_pre_stmts + rhs_pre_stmts, irt.BinExpr(
                    lhs,
                    transform_operator(expr.f_op, 2),
                    rhs,
                    type_hint=expr.p_expression_type,
                    orig_node=expr
                )

        elif expr.is_a(lal.UnOp):
            inner_pre_stmts, inner_expr = transform_expr(expr.f_expr)
            return inner_pre_stmts, irt.UnExpr(
                transform_operator(expr.f_op, 1),
                inner_expr,
                type_hint=expr.p_expression_type,
                orig_node=expr
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
            if ref.is_a(lal.ObjectDecl):
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

        elif expr.is_a(lal.ExplicitDeref):
            # Explicit dereferences are transformed as such:
            #
            # Ada:
            # ----------------
            # x := F(y.all);
            #
            # Basic IR:
            # ----------------
            # assume(y != null)
            # x := F(y.all)

            # Transform the expression being dereferenced and build the
            # assume expression stating that the prefix is not null.
            prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)
            assumed_expr = irt.BinExpr(
                prefix,
                irt.bin_ops[ops.NEQ],
                irt.Lit(
                    lits.NULL,
                    type_hint=expr.f_prefix.p_expression_type
                ),
                type_hint=expr.p_bool_type
            )

            # Build the assume statement as mark it as a deref check, so as
            # to inform deref checkers that this assume statement was
            # introduced for that purpose.
            return prefix_pre_stmts + [irt.AssumeStmt(
                assumed_expr,
                purpose=purpose.DerefCheck(prefix)
            )], irt.UnExpr(
                irt.un_ops[ops.DEREF],
                prefix,
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        elif expr.is_a(lal.AttributeRef):
            # AttributeRefs are transformed using an unary operator.

            prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)
            return prefix_pre_stmts, irt.UnExpr(
                _attr_to_unop[expr.f_attribute.text],
                prefix,
                type_hint=expr.p_expression_type,
                orig_node=expr
            )

        unimplemented(expr)

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
                    orig_node=var_id
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
            return expr_pre_stmts + [
                irt.AssignStmt(
                    irt.Identifier(
                        var_decls[
                            stmt.f_dest.p_referenced_decl,
                            stmt.f_dest.text
                        ],
                        type_hint=stmt.f_dest.p_expression_type,
                        orig_node=stmt.f_dest
                    ),
                    expr,
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
            not_cond = irt.UnExpr(
                irt.un_ops[ops.NOT],
                cond,
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

        :param lal.BaseTypeDecl expected_type: The expected type hint of the
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

    def visit_binexpr(self, binexpr):
        expected_type = (binexpr.lhs.data.type_hint
                         if self.has_universal_type(binexpr.rhs)
                         else binexpr.rhs.data.type_hint)

        binexpr.lhs = self.try_convert_expr(binexpr.lhs, expected_type)
        binexpr.rhs = self.try_convert_expr(binexpr.rhs, expected_type)

    def visit_unexpr(self, unexpr):
        unexpr.expr.visit(self)


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

    BinOps = {
        ops.AND: lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) and ConstExprEvaluator.to_bool(y)
        ),
        ops.OR: lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) or ConstExprEvaluator.to_bool(y)
        ),

        ops.NEQ: lambda x, y: ConstExprEvaluator.from_bool(x != y),
        ops.EQ: lambda x, y: ConstExprEvaluator.from_bool(x == y),
        ops.LT: lambda x, y: ConstExprEvaluator.from_bool(x < y),
        ops.LE: lambda x, y: ConstExprEvaluator.from_bool(x <= y),
        ops.GE: lambda x, y: ConstExprEvaluator.from_bool(x >= y),
        ops.GT: lambda x, y: ConstExprEvaluator.from_bool(x > y),
        ops.DOT_DOT: lambda x, y: ConstExprEvaluator.Range(x, y),

        ops.PLUS: lambda x, y: x + y,
        ops.MINUS: lambda x, y: x - y
    }

    UnOps = {
        ops.NOT: lambda x: ConstExprEvaluator.from_bool(
            not ConstExprEvaluator.to_bool(x)
        ),
        ops.NEG: lambda x: -x,
        ops.GET_FIRST: lambda x: x.first,
        ops.GET_LAST: lambda x: x.last
    }

    def __init__(self, bool_type, int_type, u_int_type, u_real_type):
        """
        :param lal.BaseTypeDecl bool_type: The standard boolean type.
        :param lal.BaseTypeDecl int_type: The standard int type.
        :param lal.BaseTypeDecl u_int_type: The standard universal int type.
        :param lal.BaseTypeDecl u_real_type: The standard universal real type.
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

    def visit_binexpr(self, binexpr):
        try:
            op = ConstExprEvaluator.BinOps[binexpr.bin_op.sym]
            return op(
                self.visit(binexpr.lhs),
                self.visit(binexpr.rhs)
            )
        except KeyError:
            raise NotConstExprError

    def visit_unexpr(self, unexpr):
        try:
            op = ConstExprEvaluator.UnOps[unexpr.un_op.sym]
            return op(self.visit(unexpr.expr))
        except KeyError:
            raise NotConstExprError

    def visit_lit(self, lit):
        return lit.val


@types.typer
def int_range_typer(hint):
    """
    :param lal.BaseTypeDecl hint: the lal type.
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
    :param lal.BaseTypeDecl hint: the lal type.
    :return: The corresponding lalcheck type.
    :rtype: types.Enum
    """
    if hint.is_a(lal.TypeDecl):
        if hint.f_type_def.is_a(lal.EnumTypeDef):
            literals = hint.f_type_def.findall(lal.EnumLiteralDecl)
            return types.Enum([lit.f_enum_identifier.text for lit in literals])


def access_typer(inner_typer):
    """
    :param types.Typer[lal.BaseTypeDecl] inner_typer: A typer for elements
        being accessed.

    :return: A Typer for Ada's access types.

    :rtype: types.Typer[lal.BaseTypeDecl]
    """

    @types.typer
    def typer(hint):
        """
        :param lal.BaseTypeDecl hint: the lal type.
        :return: The corresponding lalcheck type.
        :rtype: types.Pointer
        """
        if hint.p_is_access_type:
            accessed_type = hint.f_type_def.f_subtype_indication.f_name
            tpe = inner_typer.from_hint(accessed_type.p_referenced_decl)
            if tpe:
                return types.Pointer(tpe)

    return typer


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

        :rtype: types.Typer[lal.BaseTypeDecl]
        """
        bool_type = self.evaluator.bool
        int_type = self.evaluator.int

        @types.typer
        def typer(hint):
            """
            :param lal.BaseTypeDecl hint: the lal type.
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

        :rtype: types.Typer[lal.BaseTypeDecl]
        """

        @types.delegating_typer
        def typer():
            """
            :rtype: types.Typer[lal.BaseTypeDecl]
            """
            return (self.standard_typer() |
                    int_range_typer |
                    enum_typer |
                    access_typer(typer))

        return typer
