from typing import Any, Iterator, Iterable, TypeVar, Generic
from collections import defaultdict

from .base import Properties, PropertyDict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

E = TypeVar("E")
N = TypeVar("N")


class MultiEdge(Generic[E, N]):
    """A tuple representing undirected multigraph edges.

    ``MultiEdge(s, (n1, n2)) == MultiEdge(s, (n2, n1))`` since internally, the
    inputs ``n1`` and ``n2`` are always sorted by their string representation.

    Type Variables
    --------------
    E
        Type for distinguishing edge identifier
    N
        Type for node identifier
    """

    __slots__ = ["e", "n"]

    def __init__(self, e: E, n: tuple[N, N]) -> None:
        """Instantiate a new multiedge.

        Parameters
        ----------
        e : E
            A unique edge identifier
        n : tuple[N, N]
            A pair of nodes that are connected by the e
        """
        # Sort nodes by their string representation
        if repr(n[0]) >= repr(n[1]):
            self.n = (n[1], n[0])
        else:
            self.n = n

        self.e = e

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiEdge):
            return False

        if self.e == other.e and self.n == other.n:
            return True

        return False

    def __hash__(self) -> int:
        return hash((self.e, self.n))

    def __repr__(self) -> str:
        return f"MultiEdge({self.e}, {self.n})"


class UndirectedMultiGraph(Generic[E, N]):
    """A base class for undirected multigraphs.

    Type Aliases
    ------------
    Properties : defaultdict[str, Any]
        Default property value is ``None``.

    Attributes
    ----------
    nodes : PropertyDict[Node]
        A dictionary storing node properties.
    edges : PropertyDict[Edge]
        A dictionary storing edge properties.
    """

    __slots__ = ("graph", "nodes", "edges")

    def __init__(self) -> None:
        self.graph: defaultdict[N, set[tuple[E, N]]] = defaultdict(set)
        self.nodes: PropertyDict[N] = PropertyDict()
        self.edges: PropertyDict[MultiEdge[E, N]] = PropertyDict()

    def __len__(self) -> int:
        return len(self.nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UndirectedMultiGraph):
            return False

        return (
            self.graph == other.graph
            and self.nodes == other.nodes
            and self.edges == other.edges
        )

    def __iter__(self) -> Iterator[N]:
        return iter(self.nodes.keys())

    def __getitem__(self, key: N | MultiEdge[E, N]) -> Properties:
        if isinstance(key, MultiEdge):
            try:
                return self.edges[key]
            except KeyError:
                raise KeyError(f"no edge with key {repr(key)} exists")

        try:
            return self.nodes[key]
        except KeyError:
            raise KeyError(f"no node with key {repr(key)} exists")

    def copy(self) -> Self:
        # Return correct class
        other = type(self)()
        other.graph = self.graph.copy()
        other.nodes = self.nodes.copy()
        other.edges = self.edges.copy()

        return other

    def add_node(self, key: N, **kwargs: Any) -> None:
        self.nodes[key] = defaultdict(lambda: None, kwargs)

    def remove_node(self, key: N) -> None:
        """Remove node from graph.

        This function also removes edges that involve the removed node.

        Parameters
        ----------
        key : N
            The node to remove
        """
        del self.nodes[key]

        # Also remove edges that involve the node
        del self.graph[key]
        for edge in list(self.edges.keys()):
            if edge.n[0] == key or edge.n[1] == key:
                del self.edges[edge]

    def add_edge(self, e: E, n: tuple[N, N], **kwargs: Any) -> None:
        if n[0] not in self.nodes:
            raise ValueError(f"Node {repr(n[0])} not in graph.")
        if n[1] not in self.nodes:
            raise ValueError(f"Node {repr(n[1])} not in graph.")

        self.edges[MultiEdge(e, n)] = defaultdict(lambda: None, kwargs)
        self.graph[n[0]].add((e, n[1]))
        # Add the reverse also since we have an undirected graph
        self.graph[n[1]].add((e, n[0]))

    def remove_edge(self, key: MultiEdge[E, N]) -> None:
        """Remove edge from graph.

        Parameters
        ----------
        key : MultiEdge[E, N]
            The edge to remove
        """
        del self.edges[key]
        self.graph[key.n[0]].remove((key.e, key.n[1]))
        self.graph[key.n[1]].remove((key.e, key.n[0]))

    def incident_edges(self, key: N) -> Iterable[MultiEdge[E, N]]:
        if key not in self.nodes:
            raise ValueError(f"Node {repr(key)} not in graph.")

        for e, n in self.graph[key]:
            yield MultiEdge(e, (n, key))

    def subgraph(self, nodes: Iterable[N]) -> Self:
        if not all(n in self.nodes for n in nodes):
            raise ValueError(
                "All specified nodes need to be present in graph."
            )

        # Allow super-classes to return the right class
        other = type(self)()
        for n in nodes:
            other.graph[n] = set(
                (e_, n_) for e_, n_ in self.graph[n] if n_ in nodes
            )
        for n in nodes:
            other.nodes[n] = self.nodes[n]
        for e, prop in self.edges.items():
            if e.n[0] in nodes and e.n[1] in nodes:
                other.edges[e] = prop

        return other
