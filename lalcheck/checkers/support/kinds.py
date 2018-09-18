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


class AccessCheck(MessageKind):
    @classmethod
    def name(cls):
        return "access check"

    @classmethod
    def description(cls):
        return "dereference of a possibly null reference"


class DiscriminantCheck(MessageKind):
    @classmethod
    def name(cls):
        return "discriminant check"

    @classmethod
    def description(cls):
        return "a field for the wrong variant/discriminant is accessed"


class ContractCheck(MessageKind):
    @classmethod
    def name(cls):
        return "contract check"

    @classmethod
    def description(cls):
        return "user contract (pragma Assert, pre/postcondition) could fail"


class TestAlwaysTrue(MessageKind):
    @classmethod
    def name(cls):
        return "test always true"

    @classmethod
    def description(cls):
        return "test is always 'true'"


class TestAlwaysFalse(MessageKind):
    @classmethod
    def name(cls):
        return "test always false"

    @classmethod
    def description(cls):
        return "test is always 'false'"


class DeadCode(MessageKind):
    @classmethod
    def name(cls):
        return "dead code"

    @classmethod
    def description(cls):
        return "code is unreachable"


class CodeDuplicated(MessageKind):
    @classmethod
    def name(cls):
        return "code duplicated"

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
