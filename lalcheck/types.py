"""
This file defines the Typer interface as well as a collection of common types.

Typically, a typer is used to convert type hints provided by the frontend to
these middle-end types that are supported straightaway by lalcheck, i.e. that
already have a default interpretation (see interpretations.py).
"""


class Typer(object):
    """
    A Typer can be used to create a middle-end Type from a type hint typically
    provided by the frontend from semantic information.
    """
    def from_hint(self, hint):
        """
        Given a type hint (which can be anything), returns a Type instance
        if this typer can make something of the hint, otherwise None.
        """
        raise NotImplementedError

    def __or__(self, other):
        """
        Creates a new typer by combining two typers. This new Typer will try
        to type the given hint using the first typer. It it failed (returned
        None), it will type it using the second typer.
        """
        @typer
        def f(hint):
            x = self.from_hint(hint)
            return x if x is not None else other.from_hint(hint)

        return f


def typer(fun):
    """
    A useful decorator to use on a function to turn it into a Typer object.
    The decorated function must receive a type hint as parameter and return
    a Type instance.
    """
    class AnonymousTyper(Typer):
        def from_hint(self, hint):
            return fun(hint)

    return AnonymousTyper()


def delegating_typer(fun):
    """
    A useful decorator to use on a function to turn it into a Typer object.
    The decorated function must not receive any parameter and must return
    another Typer object.
    """
    class AnonymousTyper(Typer):
        def from_hint(self, hint):
            return fun().from_hint(hint)

    return AnonymousTyper()


class Type(object):
    """
    Base class for types.
    """
    def is_a(self, tpe):
        """
        Returns true if this type is an instance of the given Type class.
        """
        return isinstance(self, tpe)


class Boolean(Type):
    """
    Represents the default Boolean type.
    """
    pass


class IntRange(Type):
    """
    Represents an integer range.
    """
    def __init__(self, frm, to):
        self.frm, self.to = frm, to


class Enum(Type):
    """
    Represents an enum type.
    """
    def __init__(self, lits):
        self.lits = lits


class Pointer(Type):
    """
    Given a type, represents the type of pointers on that type.
    """
    def __init__(self, elem_type):
        self.elem_type = elem_type
