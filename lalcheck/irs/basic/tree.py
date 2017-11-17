"""
Provides the tree constructors of the Basic intermediate representation.
"""

from lalcheck.utils import Bunch
from lalcheck.constants import ops


def _visitable(name):
    """
    A class decorator that adds to a node class an entry point for visitors of
    this node. The given method name will be called on the visitor with the
    visited node.

    :param str name: The name of the method used to visit this type of node.

    :return: A function which can be called on a class to augment it with
        visiting capabilities.

    :rtype: type -> type
    """
    def enter_visitor(node, visitor, *args):
        """
        :param Node node: The visited node.
        :param visitors.Visitor visitor: The node visitor.
        :param *object args: Additional arguments for the visitor.
        :return: Whatever the visitor returns.
        :rtype: object
        """
        return getattr(visitor, name)(node, *args)

    def wrapper(cls):
        """
        :param type cls: The class being augmented with visiting capabilities.

        :return: The same class, with an additional method allowing visitors
            to visit its instances.

        :rtype: type
        """
        cls.visit = enter_visitor
        return cls

    return wrapper


class Node(object):
    """
    The base class for any Basic tree node.
    """
    def __init__(self, **data):
        """
        Initializes the node with the given user data.

        :param **object data: User-defined data.
        """
        self.data = Bunch(**data)

    def children(self):
        """
        :return: The children of this node.
        :rtype: iterable[Node]
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
        """
        :param list[Stmt] stmts: The list of statements this program consists
            of.

        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.stmts = stmts

    def children(self):
        for stmt in self.stmts:
            yield stmt


@_visitable("visit_ident")
class Identifier(Node):
    """
    An identifier, like "x", "True", etc.
    """
    def __init__(self, name, **data):
        """
        :param str name: The name of this identifier.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.name = name

    def children(self):
        return iter(())


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
        """
        :param Identifier var: The identifier being assigned.
        :param Expr expr: The expression assigned to the identifier.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.var = var
        self.expr = expr

    def children(self):
        yield self.var
        yield self.expr


@_visitable("visit_split")
class SplitStmt(Stmt):
    """
    A control-flow statement representing a nondeterministic choice of
    execution path, i.e. the two branches are visited.
    """
    def __init__(self, fst_stmts, snd_stmts, **data):
        """
        :param list[Stmt] fst_stmts: The list of the statements appearing in
            the first branch.

        :param list[Stmt] snd_stmts: The list of statements appearing in
            the second branch.

        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.fst_stmts = fst_stmts
        self.snd_stmts = snd_stmts

    def children(self):
        for stmt in self.fst_stmts + self.snd_stmts:
            yield stmt


@_visitable("visit_loop")
class LoopStmt(Stmt):
    """
    Represents a nondeterministic loop.
    """
    def __init__(self, stmts, **data):
        """
        :param list[Stmt] stmts: The list of statements appearing in the body
            of this loop.

        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.stmts = stmts

    def children(self):
        for stmt in self.stmts:
            yield stmt


@_visitable("visit_read")
class ReadStmt(Stmt):
    """
    Represents the havoc operation on a variable.
    """
    def __init__(self, var, **data):
        """
        :param Identifier var: The variable being read.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.var = var

    def children(self):
        yield self.var


@_visitable("visit_use")
class UseStmt(Stmt):
    """
    Represents the fact that a variable was used at this point.
    """
    def __init__(self, var, **data):
        """
        :param Identifier var: The variable being used.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.var = var

    def children(self):
        yield self.var


@_visitable("visit_assume")
class AssumeStmt(Stmt):
    """
    Represents the fact that an expression is assumed to be true at this point.
    """
    def __init__(self, expr, **data):
        """
        :param Expr expr: The expression being assumed.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.expr = expr

    def children(self):
        yield self.expr


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
        """
        :param Expr lhs: The left-hand side of the binary expression.
        :param Operator bin_op: The binary operator.
        :param Expr rhs: The right-hand side of the binary expression.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.lhs = lhs
        self.bin_op = bin_op
        self.rhs = rhs

    def children(self):
        yield self.lhs
        yield self.rhs


@_visitable("visit_unexpr")
class UnExpr(Expr):
    """
    Represents an unary operation, i.e. ([op] [expr])
    """
    def __init__(self, un_op, expr, **data):
        """
        :param Operator un_op: The unary operator

        :param Expr expr: The expression which the unary operator is applied
            on.

        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.un_op = un_op
        self.expr = expr

    def children(self):
        yield self.expr


@_visitable("visit_lit")
class Lit(Expr):
    """
    Represents a literal value.
    """
    def __init__(self, val, **data):
        """
        :param object val: Any value.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.val = val

    def children(self):
        return iter(())


class Operator(object):
    """
    Holds an operator. The address of this object uniquely identifies the
    operator that is held, unlike its string representation.
    """
    def __init__(self, sym):
        """
        :param object sym: The symbol associated with this operator.
        """
        self.sym = sym

    def __str__(self):
        """
        :return: A representation of the operator.
        :rtype: str
        """
        return str(self.sym)


bin_ops = {
    ops.Plus: Operator(ops.Plus),
    ops.Minus: Operator(ops.Minus),
    ops.Lt: Operator(ops.Lt),
    ops.Le: Operator(ops.Le),
    ops.Eq: Operator(ops.Eq),
    ops.Neq: Operator(ops.Neq),
    ops.Ge: Operator(ops.Ge),
    ops.Gt: Operator(ops.Gt),
    ops.DotDot: Operator(ops.DotDot)
}

un_ops = {
    ops.Not: Operator(ops.Not),
    ops.Neg: Operator(ops.Neg),
    ops.Address: Operator(ops.Address),
    ops.Deref: Operator(ops.Deref),
    ops.GetFirst: Operator(ops.GetFirst),
    ops.GetLast: Operator(ops.GetLast)
}