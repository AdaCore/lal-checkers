"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.irs.basic.tools import PrettyPrinter
from lalcheck.constants import ops, lits
from lalcheck import types

from funcy.calc import memoize


_lal_op_type_2_symbol = {
    (lal.OpLt, 2): irt.bin_ops[ops.Lt],
    (lal.OpLte, 2): irt.bin_ops[ops.Le],
    (lal.OpEq, 2): irt.bin_ops[ops.Eq],
    (lal.OpNeq, 2): irt.bin_ops[ops.Neq],
    (lal.OpGte, 2): irt.bin_ops[ops.Ge],
    (lal.OpGt, 2): irt.bin_ops[ops.Gt],
    (lal.OpPlus, 2): irt.bin_ops[ops.Plus],
    (lal.OpMinus, 2): irt.bin_ops[ops.Minus],
    (lal.OpDoubleDot, 2): irt.bin_ops[ops.DotDot],

    (lal.OpMinus, 1): irt.un_ops[ops.Neg],
    (lal.OpNot, 1): irt.un_ops[ops.Not],
}

_attr_2_unop = {
    'Access': irt.un_ops[ops.Address],
    'First': irt.un_ops[ops.GetFirst],
    'Last': irt.un_ops[ops.GetLast],
}


def _gen_ir(subp):
    """
    Generates Basic intermediate representation from a lal subprogram body.

    :param lal.SubpBody subp: The subprogram body from which to generate IR.

    :return: a Basic Program.

    :rtype: irt.Program
    """

    var_decls = {}

    def transform_operator(lal_op, arity):
        return _lal_op_type_2_symbol[type(lal_op), arity]

    def unimplemented(node):
        raise NotImplementedError(
            'Cannot transform "{}" ({})'.format(node.text, type(node))
        )

    def transform_expr(expr, ctx):
        if expr.is_a(lal.ParenExpr):
            return transform_expr(expr.f_expr, ctx)

        if expr.is_a(lal.BinOp):
            def lhs_ctx(lhs):
                def rhs_ctx(rhs):
                    return ctx(irt.BinExpr(
                        lhs,
                        transform_operator(expr.f_op, 2),
                        rhs,
                        type_hint=expr.p_expression_type
                    ))

                return transform_expr(expr.f_right, rhs_ctx)

            return transform_expr(expr.f_left, lhs_ctx)

        elif expr.is_a(lal.UnOp):
            return transform_expr(expr.f_expr, lambda operand: ctx(
                irt.UnExpr(
                    transform_operator(expr.f_op, 1),
                    operand,
                    type_hint=expr.p_expression_type
                )
            ))

        elif expr.is_a(lal.IfExpr):
            def cond_ctx(cond):
                def then_ctx(thn):
                    def else_ctx(els):
                        not_cond = irt.UnExpr(
                            irt.un_ops[ops.Not],
                            cond,
                            type_hint=cond.data.type_hint
                        )

                        then_stmts = [irt.AssumeStmt(cond)]
                        else_stmts = [irt.AssumeStmt(not_cond)]

                        then_stmts.extend(ctx(thn))
                        else_stmts.extend(ctx(els))

                        return [irt.SplitStmt(then_stmts, else_stmts)]

                    return transform_expr(expr.f_else_expr, else_ctx)

                return transform_expr(expr.f_then_expr, then_ctx)

            return transform_expr(expr.f_cond_expr, cond_ctx)

        elif expr.is_a(lal.Identifier):
            ref = expr.p_referenced_decl
            if ref.is_a(lal.ObjectDecl):
                return ctx(var_decls[ref, expr.text])
            elif ref.is_a(lal.EnumLiteralDecl):
                return ctx(irt.Lit(
                    expr.text,
                    type_hint=ref.parent.parent
                ))
            elif ref.is_a(lal.NumberDecl):
                return transform_expr(ref.f_expr, ctx)
            elif ref.is_a(lal.TypeDecl):
                if ref.f_type_def.is_a(lal.SignedIntTypeDef):
                    return transform_expr(ref.f_type_def.f_range.f_range, ctx)

        elif expr.is_a(lal.IntLiteral):
            return ctx(irt.Lit(
                int(expr.f_tok.text),
                type_hint=expr.p_expression_type
            ))

        elif expr.is_a(lal.NullLiteral):
            return ctx(irt.Lit(
                lits.Null,
                type_hint=expr.p_expression_type
            ))

        elif expr.is_a(lal.ExplicitDeref):
            def prefix_ctx(prefix):
                assumed_expr = irt.BinExpr(
                    prefix,
                    irt.bin_ops[ops.Neq],
                    irt.Lit(
                        lits.Null,
                        type_hint=expr.f_prefix.p_expression_type
                    ),
                    type_hint=expr.p_bool_type
                )
                assume_stmt = irt.AssumeStmt(
                    assumed_expr,
                    purpose={
                        'kind': 'deref_check',
                        'obj': prefix
                    }
                )
                return [assume_stmt] + ctx(
                    irt.UnExpr(
                        irt.un_ops[ops.Deref],
                        prefix,
                        type_hint=expr.p_expression_type
                    )
                )

            return transform_expr(expr.f_prefix, prefix_ctx)

        elif expr.is_a(lal.AttributeRef):
            return transform_expr(expr.f_prefix, lambda prefix: ctx(
                irt.UnExpr(
                    _attr_2_unop[expr.f_attribute.text],
                    prefix,
                    type_hint=expr.p_expression_type
                )
            ))

        unimplemented(expr)

    def transform_decl(decl):
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
                return transform_expr(decl.f_default_expr, lambda def_val: [
                    irt.AssignStmt(var_decls[decl, var_id.text], def_val)
                    for var_id in decl.f_ids
                ])

        unimplemented(decl)

    def transform_stmt(stmt):
        if stmt.is_a(lal.AssignStmt):
            return transform_expr(stmt.f_expr, lambda expr: [
                irt.AssignStmt(
                    var_decls[stmt.f_dest.p_referenced_decl, stmt.f_dest.text],
                    expr
                )
            ])

        elif stmt.is_a(lal.IfStmt):
            def cond_ctx(cond):
                not_cond = irt.UnExpr(
                    irt.un_ops[ops.Not],
                    cond,
                    type_hint=cond.data.type_hint
                )

                then_stmts = [irt.AssumeStmt(cond)]
                else_stmts = [irt.AssumeStmt(not_cond)]

                then_stmts.extend(transform_stmts(stmt.f_then_stmts))
                else_stmts.extend(transform_stmts(stmt.f_else_stmts))

                # todo
                # for sub in stmt.f_alternatives:
                #    traverse_branch(sub, nulls, neg_cond=stmt.f_cond_expr)

                return [irt.SplitStmt(then_stmts, else_stmts)]

            return transform_expr(stmt.f_cond_expr, cond_ctx)

        elif stmt.is_a(lal.CaseStmt):
            # todo
            return []

        elif stmt.is_a(lal.LoopStmt):
            return [irt.LoopStmt(transform_stmts(stmt.f_stmts))]

        elif stmt.is_a(lal.WhileLoopStmt):
            return transform_expr(stmt.f_spec.f_expr, lambda cond: [
                irt.LoopStmt(
                    [irt.AssumeStmt(cond)] + transform_stmts(stmt.f_stmts)
                )
            ])

        elif stmt.is_a(lal.ForLoopStmt):
            # todo
            return []

        elif stmt.is_a(lal.ExceptionHandler):
            # todo ?
            return []

        unimplemented(stmt)

    def transform_decls(decls):
        res = []
        for decl in decls:
            res.extend(transform_decl(decl))
        return res

    def transform_stmts(stmts):
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

    def __init__(self, unit):
        """
        :param lal.AbstractNode unit: Any lal node.
        """
        super(ConvertUniversalTypes, self).__init__()
        self.bool = unit.p_bool_type
        self.universal_int = unit.p_universal_int_type
        self.universal_real = unit.p_universal_real_type
        self.eval = ConstExprEvaluator(unit).eval

    def has_universal_type(self, expr):
        """
        :param irt.Expr expr: A Basic IR expression.

        :return: True if the expression is either of universal int type, or
            universal real type.

        :rtype: bool
        """
        return expr.data.type_hint in [self.universal_int, self.universal_real]

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
                self.eval(expr),
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
        assume.expr = self.try_convert_expr(assume.expr, self.bool)

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


AdaTrue = 'True'
AdaFalse = 'False'


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
        ops.And: lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) and ConstExprEvaluator.to_bool(y)
        ),
        ops.Or: lambda x, y: ConstExprEvaluator.from_bool(
            ConstExprEvaluator.to_bool(x) or ConstExprEvaluator.to_bool(y)
        ),

        ops.Neq: lambda x, y: ConstExprEvaluator.from_bool(x != y),
        ops.Eq: lambda x, y: ConstExprEvaluator.from_bool(x == y),
        ops.Lt: lambda x, y: ConstExprEvaluator.from_bool(x < y),
        ops.Le: lambda x, y: ConstExprEvaluator.from_bool(x <= y),
        ops.Ge: lambda x, y: ConstExprEvaluator.from_bool(x >= y),
        ops.Gt: lambda x, y: ConstExprEvaluator.from_bool(x > y),
        ops.DotDot: lambda x, y: ConstExprEvaluator.Range(x, y),

        ops.Plus: lambda x, y: x + y,
        ops.Minus: lambda x, y: x - y
    }

    UnOps = {
        ops.Not: lambda x: ConstExprEvaluator.from_bool(
            not ConstExprEvaluator.to_bool(x)
        ),
        ops.Neg: lambda x: -x,
        ops.GetFirst: lambda x: x.first,
        ops.GetLast: lambda x: x.last
    }

    def __init__(self, unit):
        """
        :param lal.AbstractNode unit: Any lal node.
        """
        super(ConstExprEvaluator, self).__init__()
        self.bool = unit.p_bool_type
        self.int = unit.p_int_type
        self.universal_int = unit.p_universal_int_type
        self.universal_real = unit.p_universal_real_type

    @staticmethod
    def to_bool(x):
        """
        :param str x: The boolean to convert.
        :return: The representation of the corresponding boolean literal.
        :rtype: bool
        """
        return x == AdaTrue

    @staticmethod
    def from_bool(x):
        """
        :param bool x: The representation of a boolean literal to convert.
        :return: The corresponding boolean.
        :rtype: str
        """
        return AdaTrue if x else AdaFalse

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
    :param lal.AnalysisContext ctx: The lal analysis context.
    :return: A Typer for Ada's standard types.
    :rtype: types.Typer[lal.BaseTypeDecl]
    """
    node = ctx.get_from_file('standard.ads').root
    bool_type = node.p_bool_type
    int_type = node.p_int_type

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


def new_context():
    """
    Creates a new program extraction context.

    Programs extracted with the same context have compatible standard types.

    Note that the context must be kept alive as long as long as the
    programs that were extracted with this context are intended to be used.

    :return: A new libadalang analysis context.

    :rtype: lal.AnalysisContext
    """
    return lal.AnalysisContext()


def extract_programs(ctx, ada_file):
    """
    :param lal.AnalysisContext ctx: The libadalang context.

    :param str ada_file: A path to the Ada source file from which to extract
        programs.

    :return: a Basic IR Program for each subprogram body that exists in the
        given source code.

    :rtype: iterable[irt.Program]
    """
    unit = ctx.get_from_file(ada_file)

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

    converter = ConvertUniversalTypes(unit.root)
    for prog in progs:
        prog.visit(converter)

    return progs


def default_typer(ctx):
    """
    :param lal.AnalysisContext ctx: The lal analysis context.
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
