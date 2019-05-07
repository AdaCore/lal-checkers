"""
First pass that discovers all the subprogram bodies in the given units and
calculates useful information about them.
"""

import libadalang as lal
from funcy.calc import memoize

from utils import get_subp_identity, sorted_by_position, is_access_attribute


class AnalysisConfiguration(object):
    """
    Contains the configuration of the analysis to be done.
    """

    NO_GLOBAL = 'NO_GLOBAL'
    UP_LEVEL = 'UP_LEVEL'
    UP_LEVEL_LOCAL_DECL = 'UP_LEVEL_LOCAL_DECL'

    def __init__(self, discover_spills, global_var_predicate):
        """
        :param bool discover_spills: True if the analysis should discover
            local variables that are accessed at some point inside the
            subprogram they are declared in using the 'Access attribute.

        :param str global_var_predicate: The predicate to use to determine if
            a variable declaration should be considered global to a given
            subprogram. Can be one of:
            - NO_GLOBAL: Never consider any given variable as a global variable
                of the given procedure.
            - UP_LEVEL: Consider as global any variables declared outside the
                given procedure.
            - UP_LEVEL_LOCAL_DECL: Consider as global any variables declared
                outside the given procedure but that are local to another
                procedure.
        """
        self.discover_spills = discover_spills
        self.global_var_predicate = global_var_predicate


_default_configuration = AnalysisConfiguration(
    True,
    AnalysisConfiguration.UP_LEVEL_LOCAL_DECL
)


class SubpAnalysisData(object):
    """
    Contains information about a subprogram:

    - explicit_global_vars: The set of global variables that are explicitly
        referenced in its body.
    - local_vars: The set of variables that are declared inside the subprogram.
    - vars_to_spill: The set of local variables which 'Access is being taken
        at some point in the subprogram.
    - out_calls: The set of subprogram that are called by this one.
    - all_global_vars: The set of global variables that are ultimately used by
        this subprogram, including through a call to another subprogram.
    """
    def __init__(self):
        self.explicit_global_vars = set()
        self.local_vars = set()
        self.vars_to_spill = set()
        self.out_calls = set()
        self.all_global_vars = None  # Not yet computed

    def __repr__(self):
        return "SubpAnalysisData({}, {}, {}, {}, {})".format(
            self.explicit_global_vars,
            self.local_vars,
            self.vars_to_spill,
            self.out_calls,
            self.all_global_vars if self.all_global_vars else set()
        )


def _base_accessed_var(expr):
    """
    The base variable being accessed in a "'Access" expression. E.g, in:

    "a.b(2).c'Access"

    The base variable being accessed is "a". That is because the memory model
    used for the analysis uses one memory location for each local variable,
    independently of its actual size/representation (e.g. a integer variable
    or an array of records with 30 fields both hold a single location in the
    "stack"). Consequently, pointers on this memory are expressions (instead
    of absolute addresses) that point to indicate a memory location PLUS some
    more precision, such as "The 3rd field of the record at memory location 2".
    See lalcheck.ai.constants.access_paths for more information about the
    language for pointers.

    :param lal.Expr expr: The accessed expression (the prefix of the 'Access
        attribute ref).
    :rtype: lal.DefiningName
    """
    if expr.is_a(lal.Identifier):
        return expr.p_xref(True)
    elif expr.is_a(lal.DottedName):
        return _base_accessed_var(expr.f_prefix)
    elif (expr.is_a(lal.AttributeRef)
            and expr.f_attribute.text.lower() == 'model'):
        return _base_accessed_var(expr.f_prefix)
    else:
        return None


def compute_global_vars(subpdata):
    """
    For each subprogram tracked in subpdata, computes the set of global
    variables that are ultimately being used by each subprogram (see
    all_global_vars attribute).

    Solves the following set of equations using a fix-point algorithm:

    Globals[subp] = ExplicitGlobals[subp] u
        U{x in OutCalls[subp]} (Globals[x]) \ Locals[subp]

    :param dict[lal.SubpBody, SubpAnalysisData] subpdata: A mapping from the
        tracked subprograms to their SubpAnalysisData instances.
    """
    def compute_size():
        return sum(len(x.all_global_vars) for _, x in subpdata.iteritems())

    for _, userdata in subpdata.iteritems():
        userdata.all_global_vars = userdata.explicit_global_vars.copy()

    old_count = 0
    count = compute_size()

    while old_count != count:
        for subp, userdata in subpdata.iteritems():
            for called in userdata.out_calls:
                called_userdata = subpdata.get(called, None)
                if called_userdata is not None:
                    userdata.all_global_vars.update(
                        called_userdata.all_global_vars
                    )
            userdata.all_global_vars.difference_update(userdata.local_vars)

        old_count = count
        count = compute_size()

    for subp, userdata in subpdata.iteritems():
        # Guarantee fixed iteration order
        userdata.all_global_vars = sorted_by_position(userdata.all_global_vars)
        for called in userdata.out_calls:
            called_userdata = subpdata.get(called, None)
            if called_userdata is not None:
                # All variables that are accessed as globals by subprograms
                # that are called inside ours must be spilled.
                userdata.vars_to_spill.update(called_userdata.all_global_vars)


def _solve_renamings(ref):
    """
    Returns the original definition (declaration or body) of the given
    subprogram reference. If it is already a subprogram body or declaration,
    we return itself. If it is a renaming of another subprogram, we return
    the actual definition by traversing the renaming chain.

    :param lal.BasicSubpDecl ref: A subprogram declaration that may be a
        renaming declaration.

    :rtype: lal.BaseSubpBody | lal.BasicSubpDecl
    """
    old_ref = ref
    try:
        while ref.is_a(lal.SubpRenamingDecl):
            renamed = ref.f_renames.f_renamed_object
            ref = renamed.p_referenced_decl(True)
            if ref is None or ref == old_ref:
                return None
            old_ref = ref

        if ref.is_a(lal.BaseSubpBody, lal.BasicSubpDecl,
                    lal.GenericSubpInstantiation):
            return ref
    except lal.PropertyError:
        return None


def _get_ref_decl(node):
    """
    Finds the declaration referenced by the given node. Wraps p_referenced_decl
    by always returning None in case of failure instead of a raising an
    exception.

    :param lal.AdaNode node: The node to resolve.
    :rtype: lal.BasicDecl|None
    """
    try:
        return node.p_referenced_decl(True)
    except lal.PropertyError:
        return None


def _is_up_level(decl, subp):
    """
    Given a declaration decl for which a reference was found in the given
    subprogram subp (meaning it is visible from somewhere inside subp),
    returns true if decl lies in an outer scope of subp.
    :rtype: bool
    """
    parents = decl.parent_chain
    return subp not in parents


def _is_up_level_local_decl(decl, subp):
    """
    Given a declaration decl for which a reference was found in the given
    subprogram subp (meaning it is visible from somewhere inside subp),
    returns true if decl:
     - lies in an outer scope of subp.
     - does not belong to a package, but a subprogram body.
    :rtype: bool
    """
    parents = decl.parent_chain
    if subp in parents:
        return False

    try:
        actual_parent = next(
            p for p in parents
            if p.is_a(lal.BasePackageDecl, lal.BaseSubpBody)
        )
        return actual_parent.is_a(lal.BaseSubpBody)
    except StopIteration:
        return False


def _is_concrete_object_decl(node):
    return (node.is_a(lal.ObjectDecl, lal.ParamSpec)
            and not node.parent.is_a(lal.GenericFormalObjDecl))


def is_expr(node):
    """
    Returns True iff the given node is part of an expression.

    :type node: lal.AdaNode
    :rtype: bool
    """
    try:
        return node.p_expression_type is not None
    except lal.PropertyError:
        return False


def traverse_unit(unit, config=_default_configuration):
    """
    Analyzes the given compilation unit using the given analysis configuration.

    Computes useful information about each subprogram body defined in the unit,
    which are returned as a mapping from SubpBody to SubpAnalysisData. See
    SubpAnalysisData class to know what information is computed on subprograms.

    :param lal.CompilationUnit unit: The compilation unit to analyze.
    :param AnalysisConfiguration config: The configuration of the analysis.
    :rtype: dict[lal.SubpBody, SubpAnalysisData]
    """

    if config.global_var_predicate == AnalysisConfiguration.NO_GLOBAL:
        def global_var_predicate(*_): return False
    elif config.global_var_predicate == AnalysisConfiguration.UP_LEVEL:
        global_var_predicate = memoize(_is_up_level)
    elif (config.global_var_predicate
            == AnalysisConfiguration.UP_LEVEL_LOCAL_DECL):
        global_var_predicate = memoize(_is_up_level_local_decl)

    subpdata = {}

    def traverse_childs(node, *args, **kwargs):
        for child in node:
            if child is not None:
                traverse(child, *args, **kwargs)

    def traverse(node, subp=None):
        if node.is_a(lal.BaseSubpBody):
            subp = node
            subpdata[subp] = SubpAnalysisData()
        elif node.is_a(lal.GenericSubpInstantiation):
            # For a function which is an instantiation of a generic subprogram,
            # we conservatively say it calls all subprograms that we
            # instantiate it with, and has access to all object declarations
            # that we instantiate it with.
            subpdata[node] = inst_data = SubpAnalysisData()

            generic_subp = _get_ref_decl(node.f_generic_subp_name)
            if generic_subp is not None:
                # Add a synthetic call to the generic subprogram, to make sure
                # that global variables of the generic subprogram are also
                # global variables of the instantiations.
                inst_data.out_calls.add(get_subp_identity(generic_subp))

            if config.global_var_predicate != AnalysisConfiguration.NO_GLOBAL:
                for assoc in node.f_params:
                    actual = assoc.f_r_expr
                    ref = _get_ref_decl(actual)

                    if ref is not None:
                        if _is_concrete_object_decl(ref):
                            inst_data.explicit_global_vars.add(
                                actual.p_xref(True)
                            )
                        elif ref.is_a(lal.BaseSubpBody, lal.BasicSubpDecl):
                            inst_data.out_calls.add(get_subp_identity(ref))

        elif node.is_a(lal.BasePackageDecl, lal.PackageBody,
                       lal.GenericPackageDecl, lal.PackageBodyStub):
            subp = None
        elif subp is not None:
            if node.is_a(lal.ObjectDecl, lal.ParamSpec):
                subpdata[subp].local_vars.update(node.f_ids)
            elif node.is_a(lal.Identifier) and is_expr(node):
                ref = _get_ref_decl(node)

                if ref is not None and _is_concrete_object_decl(ref):
                    if global_var_predicate(ref, subp):
                        # For now, "globals" are only the variables that
                        # are defined in an up-level procedure.
                        subpdata[subp].explicit_global_vars.add(
                            node.p_xref(True)
                        )
            elif node.is_a(lal.AttributeRef) and config.discover_spills:
                if is_access_attribute(node.f_attribute.text.lower()):
                    accessed = _base_accessed_var(node.f_prefix)
                    if accessed is not None:
                        ref_decl = accessed.p_basic_decl
                        if ref_decl.is_a(lal.BasicSubpDecl,
                                         lal.BaseSubpBody,
                                         lal.GenericSubpInstantiation):
                            # Access is done on a subprogram.
                            subpdata[subp].out_calls.add(
                                get_subp_identity(ref_decl)
                            )
                        else:
                            # Access is done on a variable.
                            subpdata[subp].vars_to_spill.add(accessed)

            if config.global_var_predicate != AnalysisConfiguration.NO_GLOBAL:
                if node.is_a(lal.Name):
                    # Do not find use a fallback here, since the is_call
                    # property cannot use it anyway.
                    ref = _get_ref_decl(node)

                    if ref is not None and ref.is_a(
                            lal.BaseSubpBody, lal.BasicSubpDecl,
                            lal.SubpBodyStub, lal.GenericSubpInstantiation):
                        actual = _solve_renamings(ref)
                        if actual is not None:
                            subpdata[subp].out_calls.add(
                                get_subp_identity(actual)
                            )

        traverse_childs(node, subp)

    traverse_childs(unit)
    compute_global_vars(subpdata)

    return subpdata
