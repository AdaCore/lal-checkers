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

    def visit_var(self, var, *args):
        return

    def visit_assign(self, assign, *args):
        return

    def visit_split(self, splitstmt, *args):
        return

    def visit_loop(self, loopstmt, *args):
        return

    def visit_label(self, labelstmt, *args):
        return

    def visit_goto(self, gotostmt, *args):
        return

    def visit_assume(self, assumestmt, *args):
        return

    def visit_read(self, read, *args):
        return

    def visit_use(self, use, *args):
        return

    def visit_funcall(self, funcall, *args):
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

    def visit_var(self, var, *args):
        self.visit_children(var, *args)

    def visit_assign(self, assign, *args):
        self.visit_children(assign, *args)

    def visit_split(self, splitstmt, *args):
        self.visit_children(splitstmt, *args)

    def visit_loop(self, loopstmt, *args):
        self.visit_children(loopstmt, *args)

    def visit_label(self, labelstmt, *args):
        self.visit_children(labelstmt, *args)

    def visit_goto(self, gotostmt, *args):
        self.visit_children(gotostmt, *args)

    def visit_assume(self, assumestmt, *args):
        self.visit_children(assumestmt, *args)

    def visit_read(self, read, *args):
        self.visit_children(read, *args)

    def visit_use(self, use, *args):
        self.visit_children(use, *args)

    def visit_funcall(self, funcall, *args):
        self.visit_children(funcall, *args)

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

    def visit_var(self, var, *args):
        self.err(var)

    def visit_assign(self, assign, *args):
        return

    def visit_split(self, splitstmt, *args):
        self.err(splitstmt)

    def visit_loop(self, loopstmt, *args):
        self.err(loopstmt)

    def visit_label(self, labelstmt, *args):
        self.err(labelstmt)

    def visit_goto(self, gotostmt, *args):
        self.err(gotostmt)

    def visit_assume(self, assumestmt, *args):
        return

    def visit_read(self, read, *args):
        return

    def visit_use(self, use, *args):
        return

    def visit_funcall(self, funcall, *args):
        self.err(funcall)

    def visit_lit(self, lit, *args):
        self.err(lit)


def findall(node, predicate):
    """
    :param tree.Node node: The base parent node.

    :param tree.Node -> bool predicate: The filtering predicate.

    :return: All the child nodes of the given node that satisfy the given
        predicate.

    :rtype: list[tree.Node]
    """
    res = [node] if predicate(node) else []
    for x in node.children():
        res.extend(findall(x, predicate))
    return res
