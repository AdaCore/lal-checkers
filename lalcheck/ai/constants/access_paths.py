class AccessPath(object):
    pass


class Null(AccessPath):
    def __str__(self):
        return "null"


class Subprogram(AccessPath):
    """
    Represents an access to a subprogram.
    """
    def __init__(self, subp_obj, out_indices, does_return):
        """
        :param object subp_obj: An object identifying the subprogram accessed.
        :param list[int] out_indices: The indices of the parameters that are
            "out".
        :param bool does_return: True iff the subprogram returns a value.
        """
        self.subp_obj = subp_obj
        self.out_indices = out_indices
        self.does_return = does_return

    def __str__(self):
        return "Subp_{}".format(self.subp_obj)


class Var(AccessPath):
    def __init__(self, var_obj):
        self.var_obj = var_obj

    def __str__(self):
        return "Var_{}".format(self.var_obj)


class Field(AccessPath):
    def __init__(self, field_obj):
        self.field_obj = field_obj

    def __str__(self):
        return "Get_{}".format(self.field_obj)
