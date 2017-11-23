class Purpose(object):
    """
    A purpose object is to be attached to a node of the program in order
    to provide more involved information about it. This information can
    be used by checkers to provide more precise results.
    """
    def __str__(self):
        return "{}{}".format(
            self.__class__.__name__,
            str(self.__dict__)
        )

    @classmethod
    def is_purpose_of(cls, node):
        """
        :param tree.Node node: The Basic IR Node.
        :return: True if the node has a purpose of this type.
        """
        if 'purpose' in node.data:
            return isinstance(node.data.purpose, cls)

        return False


class DerefCheck(Purpose):
    """
    Attached to a node that was created for the purpose of checking a
    dereference.
    """
    def __init__(self, derefed_expr):
        """
        :param tree.Expr derefed_expr: The derefed expression.
        """
        self.expr = derefed_expr
