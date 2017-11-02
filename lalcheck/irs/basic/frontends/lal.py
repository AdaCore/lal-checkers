"""
Provides a libadalang frontend for the Basic IR.
"""

import libadalang as lal

from lalcheck.irs.basic import tree as irt
from lalcheck import types


_lal_op_type_2_symbol = {
    (lal.OpLt, 2): irt.bin_ops['<'],
    (lal.OpLte, 2): irt.bin_ops['<='],
    (lal.OpEq, 2): irt.bin_ops['=='],
    (lal.OpNeq, 2): irt.bin_ops['!='],
    (lal.OpGte, 2): irt.bin_ops['>='],
    (lal.OpGt, 2): irt.bin_ops['>'],
    (lal.OpPlus, 2): irt.bin_ops['+'],
    (lal.OpMinus, 2): irt.bin_ops['-'],
    (lal.OpMinus, 1): irt.un_ops['-'],
    (lal.OpNot, 1): irt.un_ops['!'],
}


def _gen_ir(subp):
    var_decls = {}

    def prepare_sem_query(node):
        def closest_xref_entrypoint(n):
            if n.p_xref_entry_point:
                return n
            elif n.parent is not None:
                return closest_xref_entrypoint(n.parent)
            else:
                return None

        closest = closest_xref_entrypoint(node)

        if closest is not None and not hasattr(closest, "is_resolved"):
            closest.p_resolve_names
            closest.is_resolved = True

    def ref_val(node):
        prepare_sem_query(node)
        return node.p_ref_val

    def type_val(node):
        prepare_sem_query(node)
        return node.p_type_val

    def transform_operator(lal_op, arity):
        return _lal_op_type_2_symbol[type(lal_op), arity]

    def unimplemented(node):
        raise NotImplementedError(
            'Cannot transform "{}" ({})'.format(node.text, type(node))
        )

    def transform_expr(expr):
        if expr.is_a(lal.ParenExpr):
            return transform_expr(expr.f_expr)
        if expr.is_a(lal.BinOp):
            return irt.BinExpr(
                transform_expr(expr.f_left),
                transform_operator(expr.f_op, 2),
                transform_expr(expr.f_right),
                type_hint=type_val(expr)
            )
        elif expr.is_a(lal.UnOp):
            return irt.UnExpr(
                transform_operator(expr.f_op, 1),
                transform_expr(expr.f_expr),
                type_hint=type_val(expr)
            )
        elif expr.is_a(lal.Identifier):
            ref = ref_val(expr)
            if ref.is_a(lal.ObjectDecl):
                return var_decls[ref, expr.text]
            elif ref.is_a(lal.EnumLiteralDecl):
                return irt.Lit(
                    expr.text,
                    type_hint=ref.parent.parent
                )
        elif expr.is_a(lal.IntLiteral):
            return irt.Lit(
                int(expr.f_tok.text),
                type_hint=type_val(expr)
            )

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
                default_val = transform_expr(decl.f_default_expr)
                return [
                    irt.AssignStmt(var_decls[decl, var_id.text], default_val)
                    for var_id in decl.f_ids
                ]

        unimplemented(decl)

    def transform_stmt(stmt):
        if stmt.is_a(lal.AssignStmt):
            return [irt.AssignStmt(
                var_decls[ref_val(stmt.f_dest), stmt.f_dest.text],
                transform_expr(stmt.f_expr)
            )]

        elif stmt.is_a(lal.IfStmt):
            cond = transform_expr(stmt.f_cond_expr)
            not_cond = irt.UnExpr(
                irt.un_ops['!'],
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

        elif stmt.is_a(lal.CaseStmt):
            # todo
            return []

        elif stmt.is_a(lal.LoopStmt):
            return [irt.LoopStmt(transform_stmts(stmt.f_stmts))]

        elif stmt.is_a(lal.WhileLoopStmt):

            cond = transform_expr(stmt.f_spec.f_expr)
            stmts = [irt.AssumeStmt(cond)] + transform_stmts(stmt.f_stmts)

            return [irt.LoopStmt(stmts)]

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

    return progs


def default_typer(ctx):
    standard_typer = standard_typer_of(ctx)

    @types.delegating_typer
    def typer():
        return (standard_typer |
                int_range_typer |
                enum_typer)

    return typer
