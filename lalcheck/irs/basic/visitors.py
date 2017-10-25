"""
Defines the interface for Basic IR tree visitors,
as well as a few abstract visitors.
"""


class Visitor(object):
    """
    Base abstract visitor.
    """
    def visit_program(self, prgm, *args):
        return

    def visit_ident(self, ident, *args):
        return

    def visit_assign(self, assign, *args):
        return

    def visit_split(self, splitstmt, *args):
        return

    def visit_loop(self, loopstmt, *args):
        return

    def visit_assume(self, assumestmt, *args):
        return

    def visit_read(self, read, *args):
        return

    def visit_use(self, use, *args):
        return

    def visit_binexpr(self, binexpr, *args):
        return

    def visit_unexpr(self, unexpr, *args):
        return

    def visit_lit(self, lit, *args):
        return


class ImplicitVisitor(Visitor):
    """
    Abstract visitor which automatically visits children of a node if
    the handler method for that node is not overriden.
    """
    def visit_children(self, node, *args):
        for child in node.children():
            child.visit(self, *args)

    def visit_program(self, prgm, *args):
        self.visit_children(prgm, *args)

    def visit_ident(self, ident, *args):
        self.visit_children(ident, *args)

    def visit_assign(self, assign, *args):
        self.visit_children(assign, *args)

    def visit_split(self, splitstmt, *args):
        self.visit_children(splitstmt, *args)

    def visit_loop(self, loopstmt, *args):
        self.visit_children(loopstmt, *args)

    def visit_assume(self, assumestmt, *args):
        self.visit_children(assumestmt, *args)

    def visit_read(self, read, *args):
        self.visit_children(read, *args)

    def visit_use(self, use, *args):
        self.visit_children(use, *args)

    def visit_binexpr(self, binexpr, *args):
        self.visit_children(binexpr, *args)

    def visit_unexpr(self, unexpr, *args):
        self.visit_children(unexpr, *args)

    def visit_lit(self, lit, *args):
        self.visit_children(lit, *args)


class CFGNodeVisitor(Visitor):
    """
    Abstract visitor which only visits node that can be held by a program's
    control-flow graph's nodes, i.e. statements.
    """
    def err(self, node):
        raise LookupError("Cannot visit '{}'".format(node))

    def visit_program(self, prgm, *args):
        self.err(prgm)

    def visit_ident(self, ident, *args):
        self.err(ident)

    def visit_assign(self, assign, *args):
        return

    def visit_split(self, splitstmt, *args):
        self.err(splitstmt)

    def visit_loop(self, loopstmt, *args):
        self.err(loopstmt)

    def visit_assume(self, assumestmt, *args):
        return

    def visit_read(self, read, *args):
        return

    def visit_use(self, use, *args):
        return

    def visit_binexpr(self, binexpr, *args):
        self.err(binexpr)

    def visit_unexpr(self, unexpr, *args):
        self.err(unexpr)

    def visit_lit(self, lit, *args):
        self.err(lit)


def findall(node, predicate):
    """
    Traverses the given node and eturns an iterable containing any child
    node that satisfies the predicate.
    """
    res = [node] if predicate(node) else []
    for x in node.children():
        res.extend(findall(x, predicate))
    return res
