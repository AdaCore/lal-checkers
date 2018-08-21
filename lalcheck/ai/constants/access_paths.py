class AccessPath(object):
    pass


class Null(AccessPath):
    def __str__(self):
        return "null"


class Subprogram(AccessPath):
    """
    Represents an access to a subprogram.
    """
    def __init__(self, subp_obj):
        """
        :param object subp_obj: An object identifying the subprogram accessed.
        """
        self.subp_obj = subp_obj

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
