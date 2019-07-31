import libadalang as lal
from lalcheck.ai.constants import ops
from lalcheck.ai.irs.basic.visitors import ImplicitVisitor as IRImplicitVisitor
from funcy.calc import memoize
from lalcheck.ai.utils import dataclass


class Mode(object):
    """
    Represent the mode of an Ada variable.
    """
    Local = 0
    Global = 1
    Out = 2

    @staticmethod
    def from_lal_mode(mode):
        """
        Returns the equivalent mode.
        :param lal.Mode mode: The lal mode.
        :rtype: int
        """
        if mode.is_a(lal.ModeIn, lal.ModeDefault):
            return Mode.Local
        elif mode.is_a(lal.ModeOut, lal.ModeInOut):
            return Mode.Out


def closest(node, *tpe):
    """
    Returns the closest parent of this node which is of the given type.
    :param lal.AdaNode node: The node from which to start
    :param type* tpe: The types to look for.
    :rtype: lal.AnaNode | None
    """

    if node.is_a(*tpe):
        return node
    elif node.parent is not None:
        return closest(node.parent, *tpe)
    else:
        return None


def is_access_attribute(attribute_text):
    """
    Returns true if the given attribute text denotes an "access" attribute,
    such as "access", "unrestricted_access" or "address".

    :param str attribute_text: The text to check (in lower case).
    :rtype: bool
    """
    return attribute_text in ('access', 'address', 'unrestricted_access')


def eval_as_real(node):
    """
    Evaluates the given libadalang node to a real value.

    :type node: lal.AdaNode
    :rtype: float
    """
    try:
        # todo: use p_eval_as_real once it is available libadalang.
        return float(node.text)
    except ValueError:
        raise NotImplementedError


class ValueHolder(object):
    """
    Holds a value.
    """
    def __init__(self, init=None):
        """
        :param T init: The value to initialize the held value with.
        """
        self.value = init


@dataclass
class CustomTypeHint(object):
    """
    Represents a type that lives in the same space as types of Ada expressions
    that come from libadalang.
    """
    def is_a(self, tpe):
        return isinstance(self, tpe)


class ExtendedCallReturnType(CustomTypeHint):
    def __init__(self, out_indices, out_types, ret_type=None):
        self.out_indices = out_indices
        self.out_types = out_types
        self.ret_type = ret_type


class PointerType(CustomTypeHint):
    def __init__(self, elem_type):
        self.elem_type = elem_type


class StackType(CustomTypeHint):
    def __init__(self):
        pass


class RecordField(object):
    """
    Data structure that holds information about the field of a record.
    """
    def __init__(self, decl, record_decl, name, index, conditions=[]):
        """
        :param lal.ComponentDecl | lal.DiscriminantSpec decl: The node where
            the field is declared.

        :param lal.TypeDecl record_decl: The record declaration in which this
            field is defined.

        :param lal.DefiningName name: The defining name of the field.

        :param int index: The index of the field inside the record.

        :param list[(lal.Identifier, lal.AlternativesList)] conditions:
            The conditions that must hold in order for this field to exist,
            where a condition is a tuple with (the selector discriminant,
            the value(s) that it must have for the condition to hold).
        """
        self.decl = decl
        self.record_decl = record_decl
        self.name = name
        self.index = index
        self.conds = conditions

    def is_referred_by(self, name):
        """
        Returns True if the given identifier refers to this record field.

        :param lal.Identifier name: The identifier to test.
        :rtype: bool
        """
        return name.p_xref(True) == self.name

    def field_type_expr(self):
        """
        Returns the type of this field, be it a component decl or a
        discriminant spec.

        :rtype: lal.BaseTypeDecl
        """
        if self.decl.is_a(lal.ComponentDecl):
            return self.decl.f_component_def.f_type_expr
        else:
            return self.decl.f_type_expr


@memoize
def record_fields(record_decl):
    """
    Returns an iterable of the fields of the given record, where a field is
    identified by the pair (its declaration, its name).

    :param lal.TypeDecl record_decl: The record whose fields to list.
    :rtype: list[RecordField]
    """
    res = []
    i = ValueHolder(0)

    def add_component_fields(component_list, conds=[]):
        """
        :param lal.ComponentList component_list:
        :param list[(lal.Identifier, lal.AlternativesList)] conds:
        :return:
        """
        for component in component_list.f_components:
            if not component.is_a(lal.NullComponentDecl):
                for name in component.f_ids:
                    res.append(RecordField(
                        component, record_decl, name, i.value, conds)
                    )
                    i.value += 1

        variant_part = component_list.f_variant_part
        if variant_part is not None:
            for variant in variant_part.f_variant:
                add_component_fields(variant.f_components, conds + [(
                    variant_part.f_discr_name,
                    variant.f_choices
                )])

    if record_decl.f_discriminants is not None:
        for discr in record_decl.f_discriminants.f_discr_specs:
            for name in discr.f_ids:
                res.append(RecordField(discr, record_decl, name, i.value))
                i.value += 1

    tp_def = record_decl.f_type_def

    if tp_def.is_a(lal.DerivedTypeDef):
        record_def = tp_def.f_record_extension
    else:
        record_def = tp_def.f_record_def

    add_component_fields(record_def.f_components)
    return res


def get_subp_identity(subp):
    """
    :param lal.BaseSubpBody | lal.BasicSubpDecl subp: The subprogram for which
        to retrieve a unique identity object.
    """
    if subp.is_a(lal.GenericSubpInternal):
        # This case needs to come first because a GenericSubpInternal is a
        # BasicSubpDecl.
        return subp.p_generic_instantiations[0]
    elif subp.is_a(lal.BasicSubpDecl, lal.GenericSubpDecl):
        try:
            body = subp.p_body_part
            if body is not None:
                return body
        except lal.PropertyError:
            pass

    return subp


def sorted_by_position(nodes):
    """
    Sorts the given list of lal nodes such that a node X in the resulting list
    occurs after another node Y in the list iff X appears after Y in the source
    code.

    :param list[lal.AdaNode] nodes: The nodes to sort.
    :rtype: list[lal.AdaNode]
    """
    return sorted(nodes, key=lambda node: (node.sloc_range.start,
                                           node.sloc_range.end))


@memoize
def proc_parameters(proc):
    """
    Returns the parameters from the given subprogram that have the "Out" mode.

    :param lal.SubpBody | lal.SubpDecl proc: The procedure for which to
        retrieve the parameters.

    :rtype: iterable[(int, lal.Identifier, lal.ParamSpec)]
    """
    spec = proc.f_subp_spec

    def gen():
        i = 0

        if spec.f_subp_params is not None:
            for param in spec.f_subp_params.f_params:
                for name in param.f_ids:
                    yield (i, name, param)
                    i += 1

    return list(gen())


def get_field_info(field_id):
    """
    Computes the index of the given record's field. Example:
    type Foo is record
      x, y : Integer;
      z : Integer;
    end record;

    => "x" has index 0, "y" has index 1, "z" has index 2

    :param lal.Identifier field_id: An identifier referring to a record's
        field.
    :rtype: RecordField
    """
    ref = field_id.p_referenced_decl(True)
    record_decl = closest(ref, lal.TypeDecl)

    return next(
        field
        for field in record_fields(record_decl)
        if field.is_referred_by(field_id)
    )


def is_record_field(ident):
    """
    Returns True if the given name refers to the field of a record.
    :param lal.Identifier ident: The identifier to check.
    :rtype: bool
    """
    ref = ident.p_referenced_decl(True)
    return (ref is not None and
            ref.is_a(lal.ComponentDecl, lal.DiscriminantSpec))


def is_access_to_subprogram(tpe):
    """
    Returns True iff the given type is an access to subprogram type.
    :param lal.AdaNode tpe: The type to check
    :rtype: bool
    """
    if tpe is not None and tpe.is_a(lal.TypeDecl):
        return tpe.f_type_def.is_a(lal.AccessToSubpDef)

    return False


def is_array_type_decl(tpe):
    """
    Returns True iff the given type is an array type decl.
    :param lal.AdaNode tpe: The type to check.
    :rtype: bool
    """
    if tpe is not None and tpe.is_a(lal.TypeDecl):
        return tpe.f_type_def.is_a(lal.ArrayTypeDef)

    return False


def get_array_index_types(array_def):
    """
    Given an array type definition, returns a tuple containing the type
    expression of all its indices.

    :param lal.ArrayTypeDef array_def: The array type of which to retrieve
        the type of the indices.
    :rtype: tuple[lal.SubtypeIndication]
    """
    indices = array_def.f_indices
    return (
        tuple(indices.f_list) if indices.is_a(lal.ConstrainedArrayIndices)
        else tuple(tpe.f_subtype_indication for tpe in indices.f_types)
    )


def collect_assigned_variables(nodes):
    """
    Returns the set of IR variables that appear on the left-hand side of
    an assign statement inside the given list of nodes and their children.

    :param nodes: list[irt.Node]
    :rtype: set[irt.Variable]
    """
    assigned_vars = set()

    class Visitor(IRImplicitVisitor):
        def visit_assign(self, assign):
            # Add the variable to the set but do not recursively traverse the
            # RHS since it cannot contain further assignments.
            assigned_vars.add(assign.id.var)

    v = Visitor()
    for node in nodes:
        node.visit(v)

    return assigned_vars


ADA_TRUE = 'True'
ADA_FALSE = 'False'


class NotConstExprError(ValueError):
    def __init__(self):
        super(NotConstExprError, self).__init__()


class ConstExprEvaluator(IRImplicitVisitor):
    """
    Used to evaluate expressions statically.
    See eval.
    """

    class Range(object):
        def __init__(self, first, last):
            self.first = first
            self.last = last

    Ops = {
        (ops.AND, 2): lambda x, y: x & y if type(x) == type(y) == int else (
            ConstExprEvaluator.from_bool(
                ConstExprEvaluator.to_bool(x) and ConstExprEvaluator.to_bool(y)
            )
        ),
        (ops.OR, 2): lambda x, y: x | y if type(x) == type(y) == int else (
            ConstExprEvaluator.from_bool(
                ConstExprEvaluator.to_bool(x) or ConstExprEvaluator.to_bool(y)
            )
        ),

        (ops.NEQ, 2): lambda x, y: ConstExprEvaluator.from_bool(x != y),
        (ops.EQ, 2): lambda x, y: ConstExprEvaluator.from_bool(x == y),
        (ops.LT, 2): lambda x, y: ConstExprEvaluator.from_bool(x < y),
        (ops.LE, 2): lambda x, y: ConstExprEvaluator.from_bool(x <= y),
        (ops.GE, 2): lambda x, y: ConstExprEvaluator.from_bool(x >= y),
        (ops.GT, 2): lambda x, y: ConstExprEvaluator.from_bool(x > y),
        (ops.DOT_DOT, 2): lambda x, y: ConstExprEvaluator.Range(x, y),

        (ops.PLUS, 2): lambda x, y: x + y,
        (ops.MINUS, 2): lambda x, y: x - y,

        (ops.NOT, 1): lambda x: ConstExprEvaluator.from_bool(
            not ConstExprEvaluator.to_bool(x)
        ),
        (ops.NEG, 1): lambda x: -x,
        (ops.GET_FIRST, 1): lambda x: x.first,
        (ops.GET_LAST, 1): lambda x: x.last
    }

    def __init__(self, bool_type, int_type, float_type, char_type,
                 u_int_type, u_real_type):
        """
        :param lal.AdaNode bool_type: The standard boolean type.
        :param lal.AdaNode int_type: The standard int type.
        :param lal.AdaNode float_type: The standard float type.
        :param lal.AdaNode char_type: The standard char type.
        :param lal.AdaNode u_int_type: The standard universal int type.
        :param lal.AdaNode u_real_type: The standard universal real type.
        """
        super(ConstExprEvaluator, self).__init__()
        self.bool = bool_type
        self.int = int_type
        self.float = float_type
        self.char = char_type
        self.universal_int = u_int_type
        self.universal_real = u_real_type

    @staticmethod
    def to_bool(x):
        """
        :param str x: The boolean to convert.
        :return: The representation of the corresponding boolean literal.
        :rtype: bool
        """
        return x == ADA_TRUE

    @staticmethod
    def from_bool(x):
        """
        :param bool x: The representation of a boolean literal to convert.
        :return: The corresponding boolean.
        :rtype: str
        """
        return ADA_TRUE if x else ADA_FALSE

    def eval(self, expr):
        """
        Evaluates an expression, returning the value it evaluates to.

        :param irt.Expr expr: A Basic IR expression to evaluate.
        :rtype: int | str | ConstExprEvaluator.Range
        :raise NotConstExprError: if the expression is not a constant.
        :raise NotImplementedError: if implementation is incomplete.
        """
        return self.visit(expr)

    @memoize
    def visit(self, expr):
        """
        To use instead of node.visit(self). Performs memoization, so as to
        avoid evaluating expression referred to by constant symbols multiple
        times.

        :param irt.Expr expr: The IR Basic expression to evaluate

        :return: The value of this expression.

        :rtype: int | str | ConstExprEvaluator.Range
        """
        return expr.visit(self)

    def visit_ident(self, ident):
        raise NotConstExprError

    def visit_funcall(self, funcall):
        try:
            op = ConstExprEvaluator.Ops[funcall.fun_id, len(funcall.args)]
            return op(*(
                self.visit(arg) for arg in funcall.args
            ))
        except KeyError:
            raise NotConstExprError

    def visit_lit(self, lit):
        return lit.val
