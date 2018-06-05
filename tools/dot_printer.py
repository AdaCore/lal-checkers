from __future__ import division


def _clamp(value, v_min, v_max):
    """
    Returns "value", or min/max if it's out of range.
    """
    if value < v_min:
        return v_min
    elif value > v_max:
        return v_max
    else:
        return value


class _Color(object):
    """
    Handy holder to compute colors.
    """
    def __init__(self, red, green, blue, alpha=255):
        self.red = _clamp(red, 0, 255)
        self.green = _clamp(green, 0, 255)
        self.blue = _clamp(blue, 0, 255)
        self.alpha = _clamp(alpha, 0, 255)

    def __add__(self, color):
        return _Color(self.red + color.red,
                      self.green + color.green,
                      self.blue + color.blue,
                      self.alpha + color.alpha)

    def __mul__(self, scalar):
        return _Color(self.red * scalar,
                      self.green * scalar,
                      self.blue * scalar,
                      self.alpha)

    def __repr__(self):
        return '#{:02x}{:02x}{:02x}{:02x}'.format(int(self.red),
                                                  int(self.green),
                                                  int(self.blue),
                                                  int(self.alpha))


_WHITE = _Color(255, 255, 255)
_BLACK = _Color(0, 0, 0)
_BLUE_AQUA = _Color(0, 128, 255)

_BACKGROUND = _Color(0, 0, 0, 0)  # _WHITE * 0.1
_LINES = _WHITE * 0.3
_TITLE = _WHITE * 0.2 + _BLUE_AQUA * 0.4
_REGULAR_LABEL = _BLACK * 0.8  # _WHITE * 0.5
_OTHER_LABEL = _WHITE * 0.3


def _edge(node, child):
    return '{} -> {} [color="{}"];'.format(
        id(node),
        id(child),
        _LINES
    )


def _colored(text, color):
    return '<font color="{}" face="Sans">{}</font>'.format(color, text)


def _table(title, rows):
    result = [
        '<table color="#404040" cellborder="0">',
        '<tr><td colspan="2"><b>{}</b></td></tr>'.format(
            _colored(title, _TITLE)
        )
    ]
    for row in rows:
        if len(row) == 1:
            result.append(
                '<hr/><tr><td align="center" colspan="2">{}</td></tr>'.format(
                    _colored(row[0], _REGULAR_LABEL)
                )
            )
        elif len(row) == 2:
            result.append(
                ('<hr/><tr><td align="left">{}</td>'
                 '<td align="left">{}</td></tr>').format(
                    _colored(row[0], _REGULAR_LABEL),
                    _colored(row[1], _REGULAR_LABEL)
                )
            )
    result.append('</table>')
    return ''.join(result)


class DataPrinter(object):
    """
    See gen_dot for explanations.
    """
    def __init__(self, data_key, printer):
        """
        Constructs a new DataPrinter object from a key string and a
        printer function.
        """
        self.data_key = data_key
        self.printer = printer

    def test(self, node):
        """
        Tests whether the given node contains this DataPrinter's key in its
        data.
        """
        return self.data_key in node.data

    def __call__(self, node):
        """
        Given a digraph node, returns a string representation of the value
        associated to this DataPrinter's key inside this node's data.
        """
        return self.printer(node.data[self.data_key])


def gen_dot(digraph, data_printers):
    """
    Generates a DOT representation of the given digraph. Since nodes in a
    digraph can contain arbitrary data, displaying them is done through the
    user of data printers.
    When rendering a node, data printers will check for the existence of the
    data key inside the node and render its value accordingly.

    :param lalcheck.digraph.Digraph digraph: The digraph for which to generate
        a dot representation.
    :param list[DataPrinter] data_printers: The data printers to use.
    """
    visited = set()
    result = []

    def to_dot(node):
        if node in visited:
            return []
        visited.add(node)

        label = []
        rows = []

        for data_printer in data_printers:
            if data_printer.test(node):
                rows.append(data_printer(node))

        label.extend(_table(node.name, rows))

        for succ in digraph.successors(node):
            to_dot(succ)
            result.append(_edge(node, succ))

        result.append(
            '{} [label=<{}>, shape=rectangle, penwidth=0];'.format(
                id(node),
                ''.join(label),
            )
        )

    while True:
        left = set(digraph.nodes) - visited
        if len(left) > 0:
            to_dot(left.pop())
        else:
            break

    return ('digraph g {' +
            'graph [rankdir="TB", ' +
            'splines=true, bgcolor="{}", fontname="Sans"];'.format(
                _BACKGROUND
            ) +
            '\n'.join(result) + '}')
