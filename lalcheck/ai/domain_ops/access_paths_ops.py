import boolean_ops
from itertools import product as cartesian_product


def deref(ptr_dom, elem_dom):
    path_dom = ptr_dom.dom

    def try_access(ptr, memory):
        try:
            return ptr.access(memory)
        except path_dom.NullDeref:
            return elem_dom.bottom
        except path_dom.TopValue:
            return elem_dom.top
        except path_dom.BottomValue:
            return elem_dom.bottom

    def do(ptrs, memory):
        return reduce(
            elem_dom.join,
            [try_access(ptr, memory) for ptr in ptrs],
            elem_dom.bottom
        )

    return do


def inv_deref(ptr_dom, mem_dom):
    path_dom = ptr_dom.dom

    def try_inv_access(ptr, memory, elem):
        try:
            ptr.inv_access(memory, elem)
        except (path_dom.NullDeref, path_dom.TopValue, path_dom.BottomValue):
            pass

    def do(elem, addr_constr, mem_constr):
        memory = (mem_constr[0].copy(), mem_constr[1])
        for addr in addr_constr:
            try_inv_access(addr, memory, elem)

        return addr_constr, mem_dom.meet(mem_constr, memory)

    return do


def updated(ptr_dom):
    path_dom = ptr_dom.dom

    def try_update(ptr, memory, elem):
        try:
            ptr.update(memory, elem)
        except (path_dom.NullDeref, path_dom.TopValue, path_dom.BottomValue):
            pass

    def do(mem, ptr, val):
        updated_mem = (mem[0].copy(), mem[1])
        for addr in ptr:
            try_update(addr, updated_mem, val)

        return updated_mem

    return do


def inv_updated(*_):
    raise NotImplementedError


def call(sig):
    """
    Returns a function which implements the call to a subprogram access.
    :param lalcheck.ai.interpretations.Signature sig: The signature of
        subprogram called.
    :rtype: (list, *object) -> tuple
    """
    ptr_dom = sig.input_domains[0]
    out_doms = tuple(sig.input_domains[i] for i in sig.out_param_indices)
    ret_dom = sig.output_domain
    res_doms = out_doms + ((ret_dom,) if ret_dom is not None else ())

    def do_single(fun_path, args, stack):
        """
        Calls the given subprogram with the given arguments. Takes care of
        passing the pointers on the captures and the stack.

        :param lalcheck.ai.domains.AccessPathsLattice.Subprogram fun_path: The
            access path to the subprogram to call.
        :param *object args: The arguments to call the subprogram with.
        :param object stack: The stack argument.
        :rtype: object
        """
        actual_args = [arg for arg in args]
        for var in fun_path.vars:  # Append captures
            actual_args.append(ptr_dom.build([var]))

        actual_args.append(stack)

        res = fun_path.defs[0](*actual_args)
        return res

    def do(fun_ptrs, *args):
        """
        Implements the call to the given subprogram pointer with the given
        arguments.

        The subprogram pointer may represent multiple subprograms, in which
        case each of them are called independently and the final return value
        is the join of all independent return values.

        :param list fun_ptrs: An element of the pointer domain representing
            the subprogram pointers.
        :param *object args: The arguments of the call, as element of their
            respective domains.
        :return: tuple
        """
        results = (
            do_single(path, args[:-1], args[-1])
            for path in fun_ptrs
        )
        return tuple(
            reduce(res_dom.join, res, res_dom.bottom)
            for res_dom, res in zip(res_doms, zip(*results))
        )

    return do


def inv_call():
    """
    Returns a function which implements the inverse of the call to a subprogram
    pointer.

    Note: unimplemented.
    """
    def do(_, *constrs):
        return constrs[0] if len(constrs) == 1 else constrs

    return do


def var_address(ptr_dom, elem_dom, var_index):
    path_dom = ptr_dom.dom

    def do(stack):
        return ptr_dom.build([
            path_dom.Address(stack[1] + var_index, elem_dom)
        ])

    return do


def inv_var_address(ptr_dom, var_index):
    path_dom = ptr_dom.dom

    def do(ptrs, stack_constr):
        assert ptr_dom.size(ptrs) == 1 and all(
            isinstance(addr, path_dom.Address) for addr in ptrs
        )
        return stack_constr[0], next(iter(ptrs)).val - var_index

    return do


def subp_address(ptr_dom, subp, defs):
    """
    Returns a function which can construct a representation of a pointer on
    subprogram "subp" given a list of captures (pointers on variables).

    :param domains.Powerset ptr_dom: The pointer domain.
    :param object subp: The object identifying the subprogram accessed.
    :param (function, function) defs: The forward and backward implementations
        of the subprogram.
    :rtype: (*list) -> list
    """
    path_dom = ptr_dom.dom

    def do(*capture_ptrs):
        """
        Given a list of captures as elements of the pointer domain, constructs
        an element of the pointer domain representing an access on subprogram
        "subp" with the given captures.

        :param *list capture_ptrs: Elements of the pointer domain representing
            the variables that are captured by this subprogram.
        :rtype: list
        """
        return ptr_dom.build(
            path_dom.Subprogram(subp, defs, capture_paths)
            for capture_paths in cartesian_product(*capture_ptrs)
        )

    return do


def inv_subp_address():
    """
    Returns a function which computes the inverse of taking the access on a
    subprogram.

    Note: unimplemented.
    """
    def do(_, *constrs):
        return constrs[0] if len(constrs) == 1 else constrs

    return do


def field_address(ptr_dom, elem_dom, field_index):
    path_dom = ptr_dom.dom

    def do(prefix):
        return ptr_dom.build(
            path_dom.ProductGet(path, field_index, elem_dom)
            for path in prefix
        )

    return do


def inv_field_address(field_index):
    def do(addr, prefix_constr):
        return prefix_constr

    return do


def eq(ptr_dom):
    def do(a, b):
        if ptr_dom.is_empty(a) or ptr_dom.is_empty(b):
            return boolean_ops.none
        else:
            meet = ptr_dom.meet(a, b)
            if (ptr_dom.size(meet) == ptr_dom.size(a) ==
                    ptr_dom.size(b) == 1):
                return boolean_ops.true
            elif ptr_dom.size(meet) == 0:
                return boolean_ops.false
            else:
                return boolean_ops.both
    return do


def inv_eq(ptr_dom):
    def do(res, a_constr, b_constr):
        if (ptr_dom.eq(a_constr, ptr_dom.bottom) or
                ptr_dom.eq(b_constr, ptr_dom.bottom) or
                boolean_ops.Boolean.eq(res, boolean_ops.none)):
            return None

        if boolean_ops.Boolean.eq(res, boolean_ops.true):
            meet = ptr_dom.meet(a_constr, b_constr)
            return None if ptr_dom.is_empty(meet) else (meet, meet)
        elif boolean_ops.Boolean.eq(res, boolean_ops.false):
            meet = ptr_dom.meet(a_constr, b_constr)
            meet_size = ptr_dom.size(meet)

            if meet_size == ptr_dom.size(a_constr) == 1:
                b_constr = ptr_dom.split(b_constr, a_constr)
            elif meet_size == ptr_dom.size(b_constr) == 1:
                a_constr = ptr_dom.split(a_constr, b_constr)

            if ptr_dom.is_empty(a_constr) or ptr_dom.is_empty(b_constr):
                return None
            else:
                return a_constr, b_constr
        else:
            return a_constr, b_constr

    return do


def neq(ptr_dom):
    do_eq = eq(ptr_dom)

    def do(a, b):
        return boolean_ops.not_(do_eq(a, b))

    return do


def inv_neq(ptr_dom):
    do_inv_eq = inv_eq(ptr_dom)

    def do(res, a_constr, b_constr):
        return do_inv_eq(boolean_ops.not_(res), a_constr, b_constr)

    return do


def lit(ptr_dom):
    path_dom = ptr_dom.dom

    def do(_):
        return ptr_dom.build([path_dom.Null()])

    return do
