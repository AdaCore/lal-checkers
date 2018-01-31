import boolean_ops


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

    def try_set(ptr, memory, elem):
        try:
            ptr.set(memory, elem)
        except (path_dom.NullDeref, path_dom.TopValue, path_dom.BottomValue):
            pass

    def do(elem, addr_constr, mem_constr):
        memory = (mem_constr[0].copy(), mem_constr[1])
        for addr in addr_constr:
            try_set(addr, memory, elem)

        return addr_constr, mem_dom.meet(mem_constr, memory)

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
