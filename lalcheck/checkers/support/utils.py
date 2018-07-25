import libadalang as lal


def same_as_parent(binop):
    """
    Checks whether binop is a BinOp with the same structure as its parent (same
    operator).

    :rtype: bool
    """
    par = binop.parent
    return (binop.is_a(lal.BinOp)
            and par.is_a(lal.BinOp)
            and binop.f_op.is_a(type(par.f_op)))


def closest_enclosing(node, *tpes):
    """
    Given a libadalang node n, returns its closest enclosing libadalang
    node of one of the given types which directly or indirectly contains n.

    :param lal.AdaNode node: The node from which to start the search.
    :param *type tpes: The kind of node to look out for.
    :rtype: lal.AdaNode|None
    """
    while node.parent is not None:
        node = node.parent
        if node.is_a(*tpes):
            return node
    return None


def format_text_for_output(text, max_char_count=40, ellipsis='...'):
    """
    Formats the given text so that it can be included in an output message.
    - A newline is replaced by a space, and every consecutive space (or tab or
      newline, etc.) is removed so as to discard the indentation.
    - The text length is capped by the given maximal character count, and an
      ellipsis replaces whatever exceeds this threshold.

    :param string text: The text to format.
    :param int max_char_count: The maximal length of the final string.
    :param string ellipsis: The ellipsis representation to use.
    :rtype: string
    """
    next_newline = text.find('\n')
    while next_newline != -1:
        text = text[:next_newline] + text[next_newline:].lstrip(' \t\n')
        next_newline = text.find('\n', next_newline)

    return (text if len(text) <= max_char_count
            else text[:max_char_count-len(ellipsis)] + ellipsis)
