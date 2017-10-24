"""
Provides the AST constructors of the "Basic" intermediate representation.
"""

from lalcheck.utils import Bunch


def _visitable(name):
    """
    A class decorator that adds to a node class an entry point for visitors of
    this node. The given method name will be called on the visitor with the
    visited node.
    """
    def enter_visitor(node, visitor, *args):
        return getattr(visitor, name)(node, *args)

    def wrapper(cls):
        setattr(cls, "visit", enter_visitor)
        return cls

    return wrapper


class Node(object):
    """
    The base class for any AST node.
    """
    def __init__(self, **data):
        """
        Initializes the node with the given user data.
        """
        self.data = Bunch(**data)

    def children(self):
        """
        Returns a iterable of the children of this node.
        """
        raise NotImplementedError

    def pretty_print(self, opts):
        """
        Returns a human-readable string representation of this node.
        """
        raise NotImplementedError


@_visitable("visit_program")
class Program(Node):
    """
    The node which represents a "program".

    It consists of a list of statements that are to be executed one after
    the other.
    """
    def __init__(self, stmts, **data):
        Node.__init__(self, **data)
        self.stmts = stmts

    def children(self):
        for stmt in self.stmts:
            yield stmt

    def pretty_print(self, opts):
        return "Program:\n{}".format(pretty_print_stmts(self.stmts, opts))


@_visitable("visit_ident")
class Identifier(Node):
    """
    An identifier, like "x", "True", etc.
    """
    def __init__(self, name, **data):
        Node.__init__(self, **data)
        self.name = name

    def children(self):
        return iter(())

    def pretty_print(self, opts):
        return str(self.name)


class Stmt(Node):
    """
    The base class for all statements.
    """
    pass


@_visitable("visit_assign")
class AssignStmt(Stmt):
    """
    Represents the assign statement, i.e. [identifier] = [expr].
    """
    def __init__(self, var, expr, **data):
        Node.__init__(self, **data)
        self.var = var
        self.expr = expr

    def children(self):
        yield self.var
        yield self.expr

    def pretty_print(self, opts):
        return "{} = {}".format(
            self.var.pretty_print(opts),
            self.expr.pretty_print(opts)
        )


@_visitable("visit_split")
class SplitStmt(Stmt):
    """
    A control-flow statement representing a nondeterministic choice of
    execution path, i.e. the two branches are visited.
    """
    def __init__(self, fst_stmts, snd_stmts, **data):
        Node.__init__(self, **data)
        self.fst_stmts = fst_stmts
        self.snd_stmts = snd_stmts

    def children(self):
        for stmt in self.fst_stmts + self.snd_stmts:
            yield stmt

    def pretty_print(self, opts):
        indents = opts.indents()
        return "split:\n{}\n{}|:\n{}".format(
            pretty_print_stmts(self.fst_stmts, opts),
            indents,
            pretty_print_stmts(self.snd_stmts, opts),
        )


@_visitable("visit_loop")
class LoopStmt(Stmt):
    """
    Represents a nondeterministic loop.
    """
    def __init__(self, stmts, **data):
        Node.__init__(self, **data)
        self.stmts = stmts

    def children(self):
        for stmt in self.stmts:
            yield stmt

    def pretty_print(self, opts):
        return "loop:\n{}".format(pretty_print_stmts(self.stmts, opts))


@_visitable("visit_read")
class ReadStmt(Stmt):
    """
    Represents the havoc operation on a variable.
    """
    def __init__(self, var, **data):
        Node.__init__(self, **data)
        self.var = var

    def children(self):
        yield self.var

    def pretty_print(self, opts):
        return "read({})".format(self.var.pretty_print(opts))


@_visitable("visit_use")
class UseStmt(Stmt):
    """
    Represents the fact that a variable was used at this point.
    """
    def __init__(self, var, **data):
        Node.__init__(self, **data)
        self.var = var

    def children(self):
        yield self.var

    def pretty_print(self, opts):
        return "use({})".format(self.var.pretty_print(opts))


@_visitable("visit_assume")
class AssumeStmt(Stmt):
    """
    Represents the fact that an expression is assumed to be true at this point.
    """
    def __init__(self, expr, **data):
        Node.__init__(self, **data)
        self.expr = expr

    def children(self):
        yield self.expr

    def pretty_print(self, opts):
        return "assume({})".format(self.expr.pretty_print(opts))


class Expr(Node):
    """
    Base class for expression nodes.
    """
    pass


@_visitable("visit_binexpr")
class BinExpr(Expr):
    """
    Represents a binary operation, i.e. ([expr] [op] [expr])
    """
    def __init__(self, lhs, bin_op, rhs, **data):
        Node.__init__(self, **data)
        self.lhs = lhs
        self.bin_op = bin_op
        self.rhs = rhs

    def children(self):
        yield self.lhs
        yield self.rhs

    def pretty_print(self, opts):
        return "{} {} {}".format(
            self.lhs.pretty_print(opts),
            self.bin_op.pretty_print(opts),
            self.rhs.pretty_print(opts)
        )


@_visitable("visit_unexpr")
class UnExpr(Expr):
    """
    Represents an unary operation, i.e. ([op] [expr])
    """
    def __init__(self, un_op, expr, **data):
        Node.__init__(self, **data)
        self.un_op = un_op
        self.expr = expr

    def children(self):
        yield self.expr

    def pretty_print(self, opts):
        return "{}{}".format(
            self.un_op.pretty_print(opts),
            self.expr.pretty_print(opts)
        )


@_visitable("visit_lit")
class Lit(Expr):
    """
    Represents a literal value.
    """
    def __init__(self, val, **data):
        Node.__init__(self, **data)
        self.val = val

    def children(self):
        return iter(())

    def pretty_print(self, opts):
        return str(self.val)


class Operator(object):
    """
    Holds an operator. The address of this object uniquely identifies the
    operator that is held, unlike its string representation.
    """
    def __init__(self, sym):
        self.sym = sym

    def pretty_print(self, _):
        return str(self.sym)


bin_ops = {
    '+': Operator('+'),
    '-': Operator('-'),
    '<': Operator('<'),
    '<=': Operator('<='),
    '==': Operator('=='),
    '!=': Operator('!='),
    '>=': Operator('>='),
    '>': Operator('>')
}

un_ops = {
    '!': Operator('!'),
    '-': Operator('-')
}


def pretty_print_stmts(stmts, opts):
    """
    Returns a human-readable string representation of the iterable of
    statements.
    """
    indents = opts.indents(1)
    return "\n".join(map(
        lambda stmt: indents + stmt.pretty_print(opts.indented()),
        stmts
    ))


class PrettyPrintOpts(object):
    """
    An object that holds the pretty-printing context.
    """
    def __init__(self, indent):
        self.indent = indent

    def indents(self, offset=0):
        return "  " * (self.indent + offset)

    def indented(self):
        return PrettyPrintOpts(self.indent + 1)
