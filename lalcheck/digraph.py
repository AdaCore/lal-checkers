from utils import Bunch


class Digraph(object):
    """
    Represents a simple directed graph.

    Can be used to represent the control-flow graph of a program.
    """

    class Node(object):
        """
        A node of the digraph. Can contain arbitrary data.
        """
        def __init__(self, name, **data):
            """
            Constructs a new node from a name and bunch of arbitrary data.

            Nodes are independent structures that are not associated to any
            digraph. They can be reused in any graph.
            """
            self.name = name
            self.data = Bunch(**data)

        def __hash__(self):
            return self.data.__hash__()

        def __repr__(self):
            return "{}{}".format(self.name, repr(self.data))

    class Edge(object):
        """
        An edge (or arc) of the digraph.

        Edges are independent structures that are not associated to any
        digraph. They can be reused in any graph.
        """
        def __init__(self, frm, to):
            self.frm = frm
            self.to = to

        def __hash__(self):
            return (self.frm, self.to).__hash__()

        def __repr__(self):
            return "({} -> {})".format(repr(self.frm), repr(self.to))

    def __init__(self, nodes, edges):
        """
        Constructs a new digraph from the given iterable of nodes and edges.
        """
        self.nodes = nodes
        self.edges = edges

    def successors(self, node):
        """
        Returns an iterable of all the nodes that are direct successors
        of the given node .
        """
        return (e.to for e in self.edges if e.frm == node)

    def ancestors(self, node):
        """
        Returns an iterable of all the nodes that are direct predecessors
        of the given node .
        """
        return (e.frm for e in self.edges if e.to == node)

    def __repr__(self):
        return "({}, {})".format(self.nodes, self.edges)
