"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt
from lalcheck.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from lalcheck.utils import Bunch
from lalcheck.constants import ops, lits
from lalcheck import types


_lal_op_type_2_symbol = {
    (lal.OpLt, 2): irt.bin_ops[ops.Lt],
    (lal.OpLte, 2): irt.bin_ops[ops.Le],
    (lal.OpEq, 2): irt.bin_ops[ops.Eq],
    (lal.OpNeq, 2): irt.bin_ops[ops.Neq],
    (lal.OpGte, 2): irt.bin_ops[ops.Ge],
    (lal.OpGt, 2): irt.bin_ops[ops.Gt],
    (lal.OpPlus, 2): irt.bin_ops[ops.Plus],
    (lal.OpMinus, 2): irt.bin_ops[ops.Minus],
    (lal.OpMinus, 1): irt.un_ops[ops.Neg],
    (lal.OpNot, 1): irt.un_ops[ops.Not],
}


def _gen_ir(subp):
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
            if expr.f_attribute.text == 'Access':
                return transform_expr(expr.f_prefix, lambda prefix: ctx(
                    irt.UnExpr(
                        irt.un_ops[ops.Address],
                        prefix,
                        type_hint=expr.p_expression_type
                    )
                ))

        unimplemented(expr)

    def transform_decl(decl):
        if decl.is_a(lal.TypeDecl, lal.EnumTypeDecl):
            return []
        if decl.is_a(lal.ObjectDecl):
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
    def __init__(self):
        super(ConvertUniversalTypes, self).__init__()

    def visit_assign(self, assign):
        assign.expr.visit(self, assign.var.data.type_hint),

    def visit_assume(self, assume):
        assume.expr.visit(self, assume.expr.data.type_hint)

    def visit_binexpr(self, binexpr, expected_type):
        univ_int = binexpr.data.type_hint.p_universal_int_type

        if binexpr.data.type_hint == univ_int:
            new_data = dict(**binexpr.data)
            new_data['type_hint'] = expected_type
            binexpr.data = Bunch(**new_data)

        lhs_hint = binexpr.lhs.data.type_hint
        rhs_hint = binexpr.rhs.data.type_hint

        if lhs_hint == rhs_hint == univ_int:
            in_expected_type = binexpr.data.type_hint
        else:
            in_expected_type = lhs_hint if rhs_hint == univ_int else rhs_hint

        binexpr.lhs.visit(self, in_expected_type)
        binexpr.rhs.visit(self, in_expected_type)

    def visit_unexpr(self, unexpr, expected_type):
        univ_int = unexpr.data.type_hint.p_universal_int_type

        if unexpr.data.type_hint == univ_int:
            new_data = dict(**unexpr.data)
            new_data['type_hint'] = expected_type
            unexpr.data = Bunch(**new_data)

        e_hint = unexpr.expr.data.type_hint

        if e_hint == univ_int:
            in_expected_type = unexpr.data.type_hint
        else:
            in_expected_type = e_hint

        unexpr.expr.visit(self, in_expected_type)

    def visit_lit(self, lit, expected_type):
        univ_int = lit.data.type_hint.p_universal_int_type

        if lit.data.type_hint == univ_int:
            new_data = dict(**lit.data)
            new_data['type_hint'] = expected_type
            lit.data = Bunch(**new_data)


@types.typer
def int_range_typer(hint):
    if hint.is_a(lal.TypeDecl):
        if hint.f_type_def.is_a(lal.SignedIntTypeDef):
            rng = hint.f_type_def.f_range.f_range
            frm = int(rng.f_left.text)
            to = int(rng.f_right.text)
            return types.IntRange(frm, to)


@types.typer
def enum_typer(hint):
    if hint.is_a(lal.EnumTypeDecl):
        literals = hint.findall(lal.EnumLiteralDecl)
        return types.Enum([lit.f_enum_identifier.text for lit in literals])


def access_typer(inner_typer):
    @types.typer
    def typer(hint):
        if hint.p_is_access_type:
            accessed_type = hint.f_type_def.f_subtype_indication.f_name
            tpe = inner_typer.from_hint(accessed_type.p_referenced_decl)
            if tpe:
                return types.Pointer(tpe)

    return typer


def standard_typer_of(ctx):
    def decl_finder(kind, name):
        def find_node(n):
            return n.is_a(kind) and n.f_type_id.text == name

        return find_node

    std = ctx.get_from_file('standard.ads')
    bool_decl = std.root.find(decl_finder(lal.EnumTypeDecl, 'Boolean'))
    integer_decl = std.root.find(decl_finder(lal.TypeDecl, 'Integer'))

    @types.typer
    def typer(hint):
        if hint == bool_decl:
            return types.Boolean()
        elif hint == integer_decl:
            return types.IntRange(-2 ** 31, 2 ** 31 - 1)

    return typer


def new_context():
    """
    Creates a new program extraction context.

    Programs extracted with the same context have compatible standard types.

    Note that the context must be kept alive as long as long as the
    programs that were extracted with this context are intended to be used.
    """
    return lal.AnalysisContext()


def extract_programs(ctx, ada_file):
    """
    Given a context and the path to a file containing an Ada program, extracts
    all the functions and procedures present file as an iterable of
    Basic IR Programs.
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

    converter = ConvertUniversalTypes()
    for prog in progs:
        prog.visit(converter)

    return progs


def default_typer(ctx):
    standard_typer = standard_typer_of(ctx)

    @types.delegating_typer
    def typer():
        return (standard_typer |
                int_range_typer |
                enum_typer |
                access_typer(typer))

    return typer
