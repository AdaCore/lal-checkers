def getter(index, dom):
    """
    Returns a function which, when given an element of the random access memory
    domain, returns the value that is stored at the given index (using its
    offset).

    :param int index: The address to lookup.
    :param lalcheck.ai.domains.AbstractDomain dom: The abstract domain that
        represents the value stored at the location which is accessed.
    :rtype: ((dict, int)) -> object
    """
    def do(stack):
        """
        Given an element of the abstract memory domain, returns the value that
        is stored at location (index + offset), where offset is the dynamic
        offset stored in the representation of the abstract memory.

        :type stack: (dict, int)
        :rtype: object
        """
        offset = index + stack[1]
        if offset in stack[0]:
            return stack[0][offset][1]
        else:
            return dom.top

    return do


def inv_getter(index, dom):
    """
    Returns a function which performs the inverse operation of accessing a
    value at a given index in a given memory representation.

    :param int index: The index to consider.
    :param lalcheck.ai.domains.AbstractDomain dom: The domain of the element
        that is expected to live at the considered location.
    :rtype: (object, (dict, int)) -> (dict, int)
    """
    def do(res, stack_constr):
        """
        Given a value that is expected to live at location (index + offset)
        and a constraint on the memory representation, returns a new memory
        representation in which the value at this location is closest possible
        to the expected value while satisfying the constraint.

        :param object res: The value that is expected to live at the considered
            location.
        :param (dict, int) stack_constr: A constraint on the memory
            representation.
        :rtype: (dict, int)
        """
        new_stack = stack_constr[0].copy()
        offset = index + stack_constr[1]
        old_elem = new_stack.get(offset, (dom, dom.top))[1]
        new_stack[offset] = (dom, dom.meet(old_elem, res))
        return new_stack, stack_constr[1]

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
