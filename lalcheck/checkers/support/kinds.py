"""
Contains all kinds of messages that checkers can output.
"""


class MessageKind(object):
    @classmethod
    def name(cls):
        """
        Returns the name of this kind.
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def description(cls):
        """
        Returns a short (one liner) description of the kind.
        :rtype: str
        """
        raise NotImplementedError


class NullDereference(MessageKind):
    @classmethod
    def name(cls):
        return "null dereference"

    @classmethod
    def description(cls):
        return "dereference of an access that could be null"


class InvalidDiscriminant(MessageKind):
    @classmethod
    def name(cls):
        return "invalid discriminant"

    @classmethod
    def description(cls):
        return ("access to a field of a variant record that holds"
                " the wrong determinant")


class InvalidContract(MessageKind):
    @classmethod
    def name(cls):
        return "invalid contract"

    @classmethod
    def description(cls):
        return ("user-provided contract (precondition, postcondition or "
                "assertion) might be violated")


class AlwaysTrue(MessageKind):
    @classmethod
    def name(cls):
        return "always true"

    @classmethod
    def description(cls):
        return "boolean expression is always true"


class DeadCode(MessageKind):
    @classmethod
    def name(cls):
        return "dead code"

    @classmethod
    def description(cls):
        return "code is unreachable"


class DuplicateCode(MessageKind):
    @classmethod
    def name(cls):
        return "duplicate code"

    @classmethod
    def description(cls):
        return "code is duplicated at several places."


class SameOperands(MessageKind):
    @classmethod
    def name(cls):
        return "same operands"

    @classmethod
    def description(cls):
        return "a binary operation uses the same two operands"
