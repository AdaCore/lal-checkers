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
    def __init__(self, red, green, blue):
        self.red = _clamp(red, 0, 255)
        self.green = _clamp(green, 0, 255)
        self.blue = _clamp(blue, 0, 255)

    def __add__(self, color):
        return _Color(self.red + color.red,
                      self.green + color.green,
                      self.blue + color.blue)

    def __mul__(self, scalar):
        return _Color(self.red * scalar,
                      self.green * scalar,
                      self.blue * scalar)

    def __repr__(self):
        return '#{:02x}{:02x}{:02x}'.format(int(self.red),
                                            int(self.green),
                                            int(self.blue))


WHITE = _Color(255, 255, 255)
BLUE_AQUA = _Color(0, 128, 255)

BACKGROUND = WHITE * 0.1
LINES = WHITE * 0.3
TITLE = WHITE * 0.2 + BLUE_AQUA * 0.4
REGULAR_LABEL = WHITE * 0.5
OTHER_LABEL = WHITE * 0.3


def _edge(node, child):
    return '{} -> {} [color="{}"];'.format(
        id(node),
        id(child),
        LINES
    )


def _colored(text, color):
    return '<font color="{}" face="Sans">{}</font>'.format(color, text)


def _table(title, rows):
    result = [
        '<table color="#404040" cellborder="0">',
        '<tr><td colspan="2"><b>{}</b></td></tr>'.format(
            _colored(title, TITLE)
        )
    ]
    for row in rows:
        if len(row) == 1:
            result.append(
                '<hr/><tr><td align="center" colspan="2">{}</td></tr>'.format(
                    _colored(row[0], REGULAR_LABEL)
                )
            )
        elif len(row) == 2:
            result.append(
                ('<hr/><tr><td align="left">{}</td>'
                 '<td align="left">{}</td></tr>').format(
                    _colored(row[0], REGULAR_LABEL),
                    _colored(row[1], REGULAR_LABEL)
                )
            )
    result.append('</table>')
    return ''.join(result)


def gen_dot(cfg, data_printers):
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

        for succ in cfg.successors(node):
            to_dot(succ)
            result.append(_edge(node, succ))

        result.append(
            '{} [label=<{}>, shape=rectangle, penwidth=0];'.format(
                id(node),
                ''.join(label),
            )
        )

    to_dot(cfg.start_node)

    return ('digraph g {' +
            'graph [rangkdir="LR", ' +
            'splines=true, bgcolor="{}", fontname="Sans"];'.format(
                BACKGROUND
            ) +
            '\n'.join(result) + '}')
