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


class ExistCheck(Purpose):
    """
    Attached to a node that was created for the purpose of checking that
    a field exists in a given record instance.
    """
    def __init__(self, accessed_expr, field_name, discr_name):
        """
        :param tree.Expr accessed_expr: The expression being accessed.
        :param str field_name: The name of the accessed field.
        :param str discr_name: The name of the discriminant which is tested
            in this check.
        """
        self.accessed_expr = accessed_expr
        self.field_name = field_name
        self.discr_name = discr_name


class ContractCheck(Purpose):
    """
    Attached to a node that was created for the purpose of checking that a
    Pre/Post condition is satisfied.
    """
    def __init__(self, contract_name, orig_call):
        """
        :param str contract_name: "Precondition" or "Postcondition"
        :param lal.AdaNode orig_call: The original call node associated to the
            contract check.
        """
        self.contract_name = contract_name
        self.orig_call = orig_call


class SyntheticVariable(Purpose):
    """
    Attached to an identifier that was created synthetically. (For example,
    to hold a temporary value.)
    """
    pass
