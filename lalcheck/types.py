"""
This file defines the Typer interface as well as a collection of common types.

Typically, a typer is used to convert type hints provided by the frontend to
these middle-end types that are supported straightaway by lalcheck, i.e. that
already have a default interpretation (see interpretations.py).
"""
from utils import Transformer


Typer = Transformer
"""
Typer[T] is equivalent to Transformer[T, Type]
"""

typer = Transformer.as_transformer
delegating_typer = Transformer.from_transformer_builder
memoizing_typer = Transformer.make_memoizing


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


class Product(Type):
    """
    Represents a product type.
    """
    def __init__(self, elem_types):
        """
        :param list[Type] elem_types: types of the elements of the product.
        """
        self.elem_types = elem_types


class Array(Type):
    """
    Represents an array (possibly multidimensional)
    """
    def __init__(self, index_types, component_type):
        self.index_types = index_types
        self.component_type = component_type
