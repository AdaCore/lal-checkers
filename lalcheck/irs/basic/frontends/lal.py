"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt, purpose
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.irs.basic.tools import PrettyPrinter
from lalcheck.constants import ops, lits
from lalcheck.utils import KeyCounter
from lalcheck import types

from funcy.calc import memoize


_lal_op_type_2_symbol = {
    (lal.OpLt, 2): irt.bin_ops[ops.LT],
    (lal.OpLte, 2): irt.bin_ops[ops.LE],
    (lal.OpEq, 2): irt.bin_ops[ops.EQ],
    (lal.OpNeq, 2): irt.bin_ops[ops.NEQ],
    (lal.OpGte, 2): irt.bin_ops[ops.GE],
    (lal.OpGt, 2): irt.bin_ops[ops.GT],
    (lal.OpPlus, 2): irt.bin_ops[ops.PLUS],
    (lal.OpMinus, 2): irt.bin_ops[ops.MINUS],
    (lal.OpDoubleDot, 2): irt.bin_ops[ops.DOT_DOT],

    (lal.OpMinus, 1): irt.un_ops[ops.NEG],
    (lal.OpNot, 1): irt.un_ops[ops.NOT],
}

_attr_2_unop = {
    'Access': irt.un_ops[ops.ADDRESS],
    'First': irt.un_ops[ops.GET_FIRST],
    'Last': irt.un_ops[ops.GET_LAST],
}


def _gen_ir(subp):
    """
    Generates Basic intermediate representation from a lal subprogram body.

    :param lal.SubpBody subp: The subprogram body from which to generate IR.

    :return: a Basic Program.

    :rtype: irt.Program
    """

    var_decls = {}
    tmp_vars = KeyCounter()

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
        return _lal_op_type_2_symbol[type(lal_op), arity]

    def unimplemented(node):
        """
        :param lal.AdaNode node: The node that cannot be transformed.
        :raise NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Cannot transform "{}" ({})'.format(node.text, type(node))
        )

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
            lhs_pre_stmts, lhs = transform_expr(expr.f_left)
            rhs_pre_stmts, rhs = transform_expr(expr.f_right)

            return lhs_pre_stmts + rhs_pre_stmts, irt.BinExpr(
                lhs,
                transform_operator(expr.f_op, 2),
                rhs,
                type_hint=expr.p_expression_type
            )

        elif expr.is_a(lal.UnOp):
            inner_pre_stmts, inner_expr = transform_expr(expr.f_expr)
            return inner_pre_stmts, irt.UnExpr(
                transform_operator(expr.f_op, 1),
                inner_expr,
                type_hint=expr.p_expression_type
            )

        elif expr.is_a(lal.IfExpr):
            # If expressions are transformed as such:
            #
            # Ada:
            # ---------------
            # x := (if C then A else B);
            #
            #
            # Basic IR:
            # ---------------
            # split:
            #   assume(cond)
            #   tmp := A
            # |:
            #   assume(!cond)
            #   tmp := B
            # x := tmp

            # Transform the condition and build its inverse.
            cond_pre_stmts, cond = transform_expr(expr.f_cond_expr)
            not_cond = irt.UnExpr(
                irt.un_ops[ops.NOT],
                cond,
                type_hint=cond.data.type_hint
            )

            # Generate the temporary variable, make sure it is marked as
            # synthetic so as to inform checkers not to emit irrelevant
            # messages.
            tmp = irt.Identifier(
                fresh_name("tmp"),
                type_hint=expr.p_expression_type,
                purpose=purpose.SyntheticVariable()
            )

            # Transform the "then" and "else" subexpressions.
            then_stmts, then_expr = transform_expr(expr.f_then_expr)
            else_stmts, else_expr = transform_expr(expr.f_else_expr)

            # Generate the "then" statements of our split statement.
            fst_stmts = (
                [irt.AssumeStmt(cond)] +
                then_stmts +
                [irt.AssignStmt(tmp, then_expr)]
            )

            # Generate the "else" statements of our split statement.
            snd_stmts = (
                [irt.AssumeStmt(not_cond)] +
                else_stmts +
                [irt.AssignStmt(tmp, else_expr)]
            )

            return cond_pre_stmts + [irt.SplitStmt(fst_stmts, snd_stmts)], tmp

        elif expr.is_a(lal.Identifier):
            # Transform the identifier according what it refers to.

            ref = expr.p_referenced_decl
            if ref.is_a(lal.ObjectDecl):
                return [], var_decls[ref, expr.text]
            elif ref.is_a(lal.EnumLiteralDecl):
                return [], irt.Lit(
                    expr.text,
                    type_hint=ref.parent.parent
                )
            elif ref.is_a(lal.NumberDecl):
                return transform_expr(ref.f_expr)
            elif ref.is_a(lal.TypeDecl):
                if ref.f_type_def.is_a(lal.SignedIntTypeDef):
                    return transform_expr(ref.f_type_def.f_range.f_range)

        elif expr.is_a(lal.IntLiteral):
            return [], irt.Lit(
                int(expr.f_tok.text),
                type_hint=expr.p_expression_type
            )

        elif expr.is_a(lal.NullLiteral):
            return [], irt.Lit(
                lits.NULL,
                type_hint=expr.p_expression_type
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
                type_hint=expr.p_expression_type
            )

        elif expr.is_a(lal.AttributeRef):
            # AttributeRefs are transformed using an unary operator.

            prefix_pre_stmts, prefix = transform_expr(expr.f_prefix)
            return prefix_pre_stmts, irt.UnExpr(
                _attr_2_unop[expr.f_attribute.text],
                prefix,
                type_hint=expr.p_expression_type
            )

        unimplemented(expr)

    def transform_decl(decl):
        """
        :param lal.BasicDecl decl: The lal decl to transform.

        :return: A (potentially empty) list of statements that emulate the
            Ada semantics of the declaration.

        :rtype: list[irt.Stmt]
        """
        if decl.is_a(lal.TypeDecl, lal.EnumTypeDecl, lal.NumberDecl):
            return []
        elif decl.is_a(lal.ObjectDecl):
            tdecl = decl.f_type_expr.p_designated_type_decl

            for var_id in decl.f_ids:
                var_decls[decl, var_id.text] = irt.Identifier(
                    var_id.text,
                    type_hint=tdecl
                )

            if decl.f_default_expr is None:
                return [irt.ReadStmt(var_decls[decl, var_id.text])
                        for var_id in decl.f_ids]
            else:
                dval_pre_stmts, dval_expr = transform_expr(decl.f_default_expr)
                return dval_pre_stmts + [
                    irt.AssignStmt(
                        var_decls[decl, var_id.text],
                        dval_expr
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
                    var_decls[stmt.f_dest.p_referenced_decl, stmt.f_dest.text],
                    expr
                )
            ]

        elif stmt.is_a(lal.IfStmt):
            cond_pre_stmts, cond = transform_expr(stmt.f_cond_expr)
            not_cond = irt.UnExpr(
                irt.un_ops[ops.NOT],
                cond,
                type_hint=cond.data.type_hint
            )

            then_stmts = [irt.AssumeStmt(cond)]
            else_stmts = [irt.AssumeStmt(not_cond)]

            then_stmts.extend(transform_stmts(stmt.f_then_stmts))
            else_stmts.extend(transform_stmts(stmt.f_else_stmts))

            return cond_pre_stmts + [irt.SplitStmt(then_stmts, else_stmts)]

        elif stmt.is_a(lal.CaseStmt):
            # todo
            return []

        elif stmt.is_a(lal.LoopStmt):
            return [irt.LoopStmt(transform_stmts(stmt.f_stmts))]

        elif stmt.is_a(lal.WhileLoopStmt):
            cond_pre_stmts, cond = transform_expr(stmt.f_spec.f_expr)
            return [irt.LoopStmt(
                cond_pre_stmts +
                [irt.AssumeStmt(cond)] +
                transform_stmts(stmt.f_stmts)
            )]

        elif stmt.is_a(lal.ForLoopStmt):
            # todo
            return []

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
        transform_stmts(subp.f_stmts.f_stmts)
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
            assign.var.data.type_hint
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
        :param irt.Expr expr: A Basic IR expression to evaluate.
        :return: The value which this expression evalutes to.
        :rtype: int | str
        :raise NotConstExprError: if the expression is not a constant.
        :raise NotImplementedError: if implementation is incomplete.
        """
        res = self.visit(expr)
        if res is not None:
            return res
        else:
            raise NotImplementedError("Cannot evaluate `{}`".format(
                PrettyPrinter.pretty_print(expr)
            ))

    @memoize
    def visit(self, expr):
        """
        To use instead of node.visit(self). Performs memoization, so as to
        avoid evaluating expression referred to by constant symbols multiple
        times.

        :param irt.Expr expr: The IR Basic expression to evaluate

        :return: The value of this expression.

        :rtype: int | str
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
    if hint.is_a(lal.EnumTypeDecl):
        literals = hint.findall(lal.EnumLiteralDecl)
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


def standard_typer_of(ctx):
    """
    :param ExtractionContext ctx: The program extraction context.
    :return: A Typer for Ada's standard types.
    :rtype: types.Typer[lal.BaseTypeDecl]
    """
    bool_type = ctx.evaluator.bool
    int_type = ctx.evaluator.int

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


class ExtractionContext(object):
    def __init__(self, lal_ctx):
        """
        :param lal.AnalysisContext lal_ctx: The libadalang analysis context.
        """
        self.lal_ctx = lal_ctx

        dummy = lal_ctx.get_from_buffer(
            "<dummy>", 'package Dummy is end;'
        ).root

        self.evaluator = ConstExprEvaluator(
            dummy.p_bool_type,
            dummy.p_int_type,
            dummy.p_universal_int_type,
            dummy.p_universal_real_type
        )

    def parse_file(self, ada_file):
        return self.lal_ctx.get_from_file(ada_file)


def new_context():
    """
    Creates a new program extraction context.

    Programs extracted with the same context have compatible standard types.

    Note that the context must be kept alive as long as long as the
    programs that were extracted with this context are intended to be used.

    :return: A new libadalang analysis context.

    :rtype: ExtractionContext
    """
    return ExtractionContext(lal.AnalysisContext())


def extract_programs(ctx, ada_file):
    """
    :param ExtractionContext ctx: The extraction context.

    :param str ada_file: A path to the Ada source file from which to extract
        programs.

    :return: a Basic IR Program for each subprogram body that exists in the
        given source code.

    :rtype: iterable[irt.Program]
    """
    unit = ctx.parse_file(ada_file)

    if unit.root is None:
        print('Could not parse {}:'.format(ada_file))
        for diag in unit.diagnostics:
            print('   {}'.format(diag))
            return

    unit.populate_lexical_env()

    progs = [
        _gen_ir(subp)
        for subp in unit.root.findall((
            lal.SubpBody,
            lal.ExprFunction
        ))
    ]

    converter = ConvertUniversalTypes(ctx.evaluator)

    for prog in progs:
        prog.visit(converter)

    return progs


def default_typer(ctx):
    """
    :param ExtractionContext ctx: The program extraction context.
    :return: The default Typer for the Ada language.
    :rtype: types.Typer[lal.BaseTypeDecl]
    """

    standard_typer = standard_typer_of(ctx)

    @types.delegating_typer
    def typer():
        """
        :rtype: types.Typer[lal.BaseTypeDecl]
        """
        return (standard_typer |
                int_range_typer |
                enum_typer |
                access_typer(typer))

    return typer
