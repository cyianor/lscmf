from typing import Any, Iterator, Iterable, TypeVar, Generic
from collections import defaultdict
from copy import deepcopy

from .base import Properties, PropertyDict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

V = TypeVar("V")


class Edge(Generic[V], tuple[V, V]):
    """A sorted tuple representing undirected graph edges.

    ``Edge(n1, n2) == Edge(n2, n1)`` since internally, the inputs ``n1``
    and ``n2`` are always sorted by their string representation.
    """

    def __new__(cls, n1: V, n2: V) -> Self:
        # Sort nodes by their string representation
        if str(n1) >= str(n2):
            tmp = n2
            n2 = n1
            n1 = tmp

        return tuple.__new__(Edge, (n1, n2))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> Self:
        cls = self.__class__
        other = cls.__new__(
            cls, deepcopy(self[0], memo), deepcopy(self[1], memo)
        )
        memo[id(self)] = other
        return other


class UndirectedGraph(Generic[V]):
    """A base class for undirected graphs.

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
        self.graph: defaultdict[V, set[V]] = defaultdict(set)
        self.nodes: PropertyDict[V] = PropertyDict()
        self.edges: PropertyDict[Edge[V]] = PropertyDict()

    def __len__(self) -> int:
        return len(self.nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UndirectedGraph):
            return False

        return (
            self.graph == other.graph
            and self.nodes == other.nodes
            and self.edges == other.edges
        )

    def __iter__(self) -> Iterator[V]:
        return iter(self.nodes.keys())

    def __getitem__(self, key: V | Edge[V]) -> Properties:
        if isinstance(key, Edge):
            try:
                return self.edges[key]
            except KeyError:
                raise KeyError(f"no edge with key {key} exists")

        try:
            return self.nodes[key]
        except KeyError:
            raise KeyError(f"no node with key {key} exists")

    def copy(self) -> Self:
        # Return correct class
        other = type(self)()
        other.graph = self.graph.copy()
        other.nodes = self.nodes.copy()
        other.edges = self.edges.copy()

        return other

    def add_node(self, key: V, **kwargs: Any) -> None:
        self.nodes[key] = defaultdict(lambda: None, kwargs)

    def remove_node(self, key: V) -> None:
        """Remove node from graph.

        This function also removes edges that involve the removed node.

        Parameters
        ----------
        key : Node
            The node to remove
        """
        del self.nodes[key]

        # Also remove edges that involve the node
        del self.graph[key]
        for n in self.nodes:
            if Edge(key, n) in self.edges:
                del self.edges[Edge(key, n)]

    def add_edge(self, n1: V, n2: V, **kwargs: Any) -> None:
        if n1 not in self.nodes:
            raise ValueError(f"Node {n1} not in graph.")
        if n2 not in self.nodes:
            raise ValueError(f"Node {n2} not in graph.")

        self.edges[Edge(n1, n2)] = defaultdict(lambda: None, kwargs)
        self.graph[n1].add(n2)
        # Add the reverse also since we have an undirected graph
        self.graph[n2].add(n1)

    def remove_edge(self, key: Edge[V]) -> None:
        """Remove edge from graph.

        Parameters
        ----------
        key : Edge
            The edge to remove
        """
        del self.edges[key]
        self.graph[key[0]].remove(key[1])
        self.graph[key[1]].remove(key[0])

    def incident_edges(self, key: V) -> Iterable[Edge[V]]:
        if key not in self.nodes:
            raise ValueError(f"Node {key} not in graph.")

        for n in self.graph[key]:
            yield Edge(n, key)

    def subgraph(self, nodes: Iterable[V]) -> Self:
        if not all(n in self.nodes for n in nodes):
            raise ValueError(
                "All specified nodes need to be present in graph."
            )

        # Allow super-classes to return the right class
        other = type(self)()
        for n in nodes:
            other.graph[n] = set(n_ for n_ in self.graph[n] if n_ in nodes)
        for n in nodes:
            other.nodes[n] = self.nodes[n]
        for e, prop in self.edges.items():
            if e[0] in nodes and e[1] in nodes:
                other.edges[e] = prop

        return other
