from lalcheck.ai.utils import Bunch
from funcy.calc import memoize
from collections import deque


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

    class HierarchicalOrdering(object):
        """
        Represents a hierarchical ordering of a graph. It is constructed
        from an iterable of elements, where an element is either a node of the
        original graph, or a hierarchical ordering itself.
        """
        def __init__(self, elements):
            """
            :param iterable elements: The elements of this ordering.
            """
            self.elements = elements

        def __iter__(self):
            for elem in self.elements:
                yield (elem, isinstance(elem, Digraph.Node))

        def iter_nodes(self):
            """
            Iterates over all the nodes that compose this hierarchical
            ordering.

            :rtype: iterable[Digraph.Node]
            """
            for elem, is_node in self:
                if is_node:
                    yield elem
                else:
                    for x in elem.iter_nodes():
                        yield x

        def flatten(self):
            """
            Returns a new hierarchical ordering which is composed of the same
            nodes as this one, but which is flat.

            :rtype: Digraph.HierarchicalOrdering
            """
            return Digraph.HierarchicalOrdering(tuple(self.iter_nodes()))

    def __init__(self, nodes, edges):
        """
        Constructs a new digraph from the given iterable of nodes and edges.
        """
        self.nodes = nodes
        self.edges = edges

    @memoize
    def successors(self, node):
        """
        Returns an iterable of all the nodes that are direct successors
        of the given node .
        """
        return frozenset(e.to for e in self.edges if e.frm == node)

    @memoize
    def ancestors(self, node):
        """
        Returns an iterable of all the nodes that are direct predecessors
        of the given node .
        """
        return frozenset(e.frm for e in self.edges if e.to == node)

    @memoize
    def is_leaf(self, node):
        return len(self.successors(node)) == 0

    @memoize
    def is_root(self, node):
        return len(self.ancestors(node)) == 0

    @memoize
    def leafs(self):
        return frozenset(node for node in self.nodes if self.is_leaf(node))

    @memoize
    def roots(self):
        return frozenset(node for node in self.nodes if self.is_root(node))

    def __repr__(self):
        return "({}, {})".format(self.nodes, self.edges)

    def subgraph(self, nodes):
        """
        Returns a subgraph of this graph, such that all edges connecting nodes
        that are no longer in the subgraph are removed.

        :param iterable[Digraph.Node] nodes: The nodes from the original graph
            to keep in the subgraph.
        :rtype: Digraph
        """
        return Digraph(
            nodes,
            [e for e in self.edges if e.frm in nodes and e.to in nodes]
        )

    def strongly_connected_components(self):
        """
        Returns the list of strongly connected components sorted in topological
        order, as a tuple. Each strongly connected component is a subgraph of
        the original graph.

        Note: This is an implementation of Tarjan's algorithm, translated from
        the wikipedia page: https://en.wikipedia.org/wiki/Tarjan%27s_strongly_
        connected_components_algorithm.

        :rtype: tuple[Digraph]
        """

        class Info(object):
            next_index = 0

            def __init__(self):
                self.index = Info.next_index
                self.lowlink = Info.next_index
                Info.next_index += 1
                self.on_stack = True

        components = deque()
        node_info = {}
        node_stack = []

        def connect(v):
            v_info = node_info[v] = Info()
            node_stack.append(v)

            for w in self.successors(v):
                w_info = node_info.get(w)
                if w_info is None:
                    connect(w)
                    w_info = node_info[w]
                    v_info.lowlink = min(v_info.lowlink, w_info.lowlink)
                elif w_info.on_stack:
                    v_info.lowlink = min(v_info.lowlink, w_info.index)

            if v_info.lowlink == v_info.index:
                component = deque()
                while True:
                    w = node_stack.pop()
                    node_info[w].on_stack = False
                    component.appendleft(w)
                    if v == w:
                        component_graph = self.subgraph(tuple(component))
                        components.appendleft(component_graph)
                        return

        for node in self.nodes:
            if node not in node_info:
                connect(node)

        return tuple(components)

    def weak_topological_ordering(self):
        """
        Returns a weak topological ordering of this graph, as a hierarchical
        ordering.

        Note: This algorithm was inferred from the high-level implementation
        described here: http://pages.cs.wisc.edu/~elder/stuff/bourdoncle.pdf

        :rtype: Digraph.HierarchicalOrdering
        """
        return Digraph.HierarchicalOrdering(tuple(
            scc.nodes[0]
            if len(scc.nodes) == 1
            else Digraph.HierarchicalOrdering(
                scc.nodes[:1] +
                (scc.subgraph(scc.nodes[1:])
                 .weak_topological_ordering().elements)
            )
            for scc in self.strongly_connected_components()
        ))

    def flat_topological_ordering(self):
        """
        Returns a flat topological ordering of this graph.

        :rtype: Digraph.HierarchicalOrdering
        """
        return self.weak_topological_ordering().flatten()
