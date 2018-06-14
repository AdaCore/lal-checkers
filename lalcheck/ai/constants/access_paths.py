class AccessPath(object):
    pass


class Null(AccessPath):
    def __str__(self):
        return "null"


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
