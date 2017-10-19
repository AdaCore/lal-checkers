from utils import Bunch


class CFG(object):
    class Node(object):
        def __init__(self, name, **data):
            self.name = name
            self.data = Bunch(**data)

        def __hash__(self):
            return self.data.__hash__()

        def __repr__(self):
            return "{}{}".format(self.name, repr(self.data))

    class Edge(object):
        def __init__(self, frm, to):
            self.frm = frm
            self.to = to

        def __hash__(self):
            return (self.frm, self.to).__hash__()

        def __repr__(self):
            return "({} -> {})".format(repr(self.frm), repr(self.to))

    class DataPrinter(object):
        def __init__(self, data_key, printer):
            self.data_key = data_key
            self.printer = printer

        def test(self, node):
            return hasattr(node.data, self.data_key)

        def __call__(self, node):
            return self.printer(node.data[self.data_key])

    def __init__(self, nodes, edges, start_node):
        self.nodes = nodes
        self.edges = edges
        self.start_node = start_node

    def successors(self, node):
        return (e.to for e in self.edges if e.frm == node)

    def ancestors(self, node):
        return (e.frm for e in self.edges if e.to == node)

    def __repr__(self):
        return "({}, {}, {})".format(
            self.nodes, self.edges, self.start_node
        )
