"""
Provides the tree constructors of the Basic intermediate representation.
"""

from ai.utils import Bunch


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
    An identifier, like "x", etc.
    """
    def __init__(self, var, **data):
        """
        :param Variable var: The variable object associated to this identifier.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.var = var

    def children(self):
        yield self.var


@_visitable("visit_var")
class Variable(Node):
    """
    Represents a unique variable object, that can be referred to by multiple
    identifiers.
    """
    def __init__(self, name, **data):
        """
        :param string name: The name of the variable
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
    def __init__(self, id, expr, **data):
        """
        :param Identifier id: The identifier being assigned.
        :param Expr expr: The expression assigned to the identifier.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.id = id
        self.expr = expr

    def children(self):
        yield self.id
        yield self.expr


@_visitable("visit_split")
class SplitStmt(Stmt):
    """
    A control-flow statement representing a nondeterministic choice of
    execution path, i.e. the two branches are visited.
    """
    def __init__(self, branches, **data):
        """
        :param list[list[Stmt]] branches: The branches of the split statement,
            where each branch is a list of statements. At least two branches
            are expected.

        :param **object data: User-defined data.
        """
        assert len(branches) >= 2

        Node.__init__(self, **data)
        self.branches = branches

    def children(self):
        for branch in self.branches:
            for stmt in branch:
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


@_visitable("visit_label")
class LabelStmt(Stmt):
    """
    Represents a point in the program which can be jumped to.
    """
    def __init__(self, name, **data):
        """
        :param str name: The name of the label.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.name = name

    def children(self):
        return iter(())


@_visitable("visit_goto")
class GotoStmt(Stmt):
    """
    Represents a jump statement.
    """
    def __init__(self, label, **data):
        """
        :param LabelStmt label: The label to jump to.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.label = label

    def children(self):
        yield self.label


@_visitable("visit_read")
class ReadStmt(Stmt):
    """
    Represents the havoc operation on a variable.
    """
    def __init__(self, id, **data):
        """
        :param Identifier id: The identifier being read.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.id = id

    def children(self):
        yield self.id


@_visitable("visit_use")
class UseStmt(Stmt):
    """
    Represents the fact that a variable was used at this point.
    """
    def __init__(self, id, **data):
        """
        :param Identifier id: The identifier being used.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.id = id

    def children(self):
        yield self.id


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


@_visitable("visit_funcall")
class FunCall(Expr):
    """
    Represents a function call.
    """
    def __init__(self, fun_id, args, **data):
        """
        :param object fun_id: An object identifying the function called.
        :param list[Expr] args: The arguments passed.
        :param **object data: User-defined data.
        """
        Node.__init__(self, **data)
        self.fun_id = fun_id
        self.args = args

    def children(self):
        for arg in self.args:
            yield arg


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
