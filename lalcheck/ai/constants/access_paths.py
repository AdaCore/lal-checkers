class AccessPath(object):
    pass


class Null(AccessPath):
    def __str__(self):
        return "null"


class Subprogram(AccessPath):
    """
    Represents an access to a subprogram.
    """
    class CallInterface(object):
        """
        Represents the call interface of a subprogram.
        """

        IGNORE_STACK = 0
        PASS_STACK = 1
        RETURN_STACK = 2

        def __init__(self, out_indices, stack_action, does_return):
            """
            :param list[int] out_indices: The indices of the parameters that
                are "out".

            :param int stack_action: Whether the subprogram:
                - completely ignores the stack: IGNORE_STACK
                - receives the stack but in read-only: PASS_STACK
                - receives the stack and can write to it: RETURN_STACK

            :param bool does_return: True iff the subprogram returns a value.
            """
            self.out_indices = out_indices
            self.stack_action = stack_action
            self.does_return = does_return

        def takes_stack(self):
            """
            Returns True iff this subprogram needs to be passed the stack.
            :rtype: bool
            """
            return self.stack_action != self.IGNORE_STACK

        def returns_stack(self):
            """
            Returns True iff this subprogram returns the stack.
            :rtype: bool
            """
            return self.stack_action == self.RETURN_STACK

    def __init__(self, subp_obj, interface):
        """
        :param object subp_obj: An object identifying the subprogram accessed.
        :param Subprogram.CallInterface interface: The call interface of this
            subprogram.
        :return:
        """
        self.subp_obj = subp_obj
        self.interface = interface

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
