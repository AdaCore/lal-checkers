import libadalang as lal
from lalcheck.ai.irs.basic.tree import AssumeStmt
from lalcheck.ai.domain_capabilities import Capability


def collect_assumes_with_purpose(cfg, purpose_type):
    """
    Given the control-flow graph of the program, finds the program points
    which have an assume statement whose purpose is the given type.

    :param Digraph cfg: The control-flow graph of the program.
    :param type purpose_type: The purpose to collect.
    :rtype: (lalcheck.tools.digraph.Digraph.Node,
             lalcheck.ai.irs.basic.tree.Expr,
             purpose_type)
    """
    # Retrieve nodes in the CFG that correspond to program statements.
    nodes_with_ast = (
        (node, node.data.node)
        for node in cfg.nodes
        if 'node' in node.data
    )

    # Collect those that are assume statements and that have a purpose tag
    # of the desired kind.
    return [
        (node, assume.expr, assume.data.purpose)
        for node, assume in nodes_with_ast
        if isinstance(assume, AssumeStmt)
        if purpose_type.is_purpose_of(assume)
    ]


def orig_bool_expr_statically_equals(expr, values):
    """
    Returns True iff the given expression has an orig_node which is a valid
    Ada static expression which evaluates to one of the given boolean values.

    :param lalcheck.ai.irs.basic.tree.Expr expr: The expression to consider.
    :param iterable[bool] values: A sequence of boolean values.
    :rtype: bool
    """
    if 'orig_node' not in expr.data:
        return False

    try:
        if not expr.data.orig_node.p_is_static_expr:
            return False

        value = expr.data.orig_node.p_eval_as_int
        return any(value == int(v) for v in values)
    except (lal.PropertyError, lal.NativeException):
        return False


def eval_expr_at(analysis, node, expr):
    """
    Evaluates the given expression "expr" at the given program point "node".
    Returns a list of results, where a result is a pair holding:
     1. The program trace leading to that node from which evaluating the
        expression will return the result in the second element.
     2. The result of the evaluation as the set of concrete elements which the
        expression can evaluate to. This means that the domain must support
        concretization (if not, an empty list is returned).

    :param abstract_semantics.AnalysisResults analysis: The results of the
        abstract semantics analysis.
    :param lalcheck.tools.digraph.Digraph.Node node: The program point at which
        to evaluate the given expression.
    :param lalcheck.ai.irs.basic.tree.Expr expr: The expression to evaluate.
    :rtype: list[(frozenset[Digraph.Node], frozenset[object])]
    """
    expr_domain = analysis.evaluator.model[expr].domain
    if not Capability.HasConcretize(expr_domain):
        return []

    return [
        (frozenset(trace) | {node}, expr_domain.concretize(value))
        for anc in analysis.cfg.ancestors(node)
        for trace, value in analysis.eval_at(anc, expr).iteritems()
        if value is not None
    ]


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


def relevant_tokens(node):
    """
    Returns the list of tokens of the given node without taking into account
    trivia tokens (whitespaces, newlines, etc.)

    :param lal.AdaNode node: The node for which to retrieve the tokens.
    :rtype: list[lal.Token]
    """
    return [t for t in node.tokens if not t.is_trivia]


def tokens_info(node):
    """
    Returns a list describing the tokens that make up the node. Each token
    is represented by a pair containing its kind and its text.

    :param lal.AdaNode node: The node for which to retrieve information about
        its tokens.
    :rtype: list[(str, str)]
    """
    return tuple((t.kind, t.text) for t in relevant_tokens(node))


def token_count(node):
    """
    Returns the number of tokens in the given node, discarding empty trivia
    tokens.

    :type node: lal.AdaNode
    :rtype: int
    """
    return sum(1 for t in node.tokens if not t.is_trivia)


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
