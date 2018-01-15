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


class ASCIICharacter(IntRange):
    """
    Represents an ASCII Character.
    """
    def __init__(self):
        super(ASCIICharacter, self).__init__(0, 255)


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


class FunOutput(Product):
    """
    Represents the type of function outputs.
    """
    def __init__(self, out_indices, out_types):
        """
        :param tuple[int] out_indices: indices of the out parameters.
        :param list[Type] out_types: types of the out values.
        """
        super(FunOutput, self).__init__(out_types)
        self.out_indices = out_indices

    def get_return_type(self):
        """
        Both lists have the same size if the function does not have
        a return value, otherwise the last element of the list of out types
        is the type of the return value.

        :rtype: Type | None
        """
        return (None if len(self.out_indices) == len(self.elem_types)
                else self.elem_types[-1])


class Array(Type):
    """
    Represents an array (possibly multidimensional)
    """
    def __init__(self, index_types, component_type):
        self.index_types = index_types
        self.component_type = component_type
