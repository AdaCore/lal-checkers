def getter(index, dom):
    def do(stack):
        offset = index + stack[1]
        if offset in stack[0]:
            return stack[0][offset][1]
        else:
            return dom.top

    return do


def inv_getter(index, dom):
    def do(res, stack_constr):
        return stack_constr

    return do


def updater(index, dom):
    def do(stack, value):
        new = stack[0].copy()
        new[index + stack[1]] = (dom, value)
        return new, stack[1]

    return do


def inv_updater(index, dom):
    def do(res, stack_constr, value_constr):
        return stack_constr, value_constr

    return do


def offseter(index):
    def do(state):
        return state[0], state[1] + index

    return do


def inv_offseter(index):
    def do(res, state_constr):
        return state_constr

    return do


def copy_offset(frm, to):
    return to[0], frm[1]


def inv_copy_offset(res, frm_constr, to_constr):
    return frm_constr, to_constr


def builder(val):
    raise NotImplementedError
